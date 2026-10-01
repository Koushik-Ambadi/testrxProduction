from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import collection_contract, model_path
from .integrity import index_contract, load_release, static_release_checks
from .store import InvocationStore

logger = logging.getLogger("testrx.production")


class TestrxApplication:
    def __init__(self, config: dict[str, Any] | None = None, *, client=None,
                 encoder=None, reranker=None, store=None, verify_integrity: bool = True):
        if config is None:
            self.config, self.release_lock, checks = load_release()
        else:
            self.config = config
            self.release_lock, checks = {}, []
        if config is not None and verify_integrity:
            config_path = Path(os.getenv("TESTRX_CONFIG", Path(__file__).parents[2] / "config.yaml")).resolve()
            disk_config, self.release_lock, _ = load_release(config_path)
            lock_path = config_path.parent / disk_config["integrity"]["lock_file"]
            checks = static_release_checks(config, self.release_lock, lock_path)
            if config != disk_config:
                checks.append({"name": "loaded configuration matches deployed configuration",
                               "status": "FAIL", "expected": disk_config, "actual": config})
        failures = [item["name"] for item in checks if item["status"] == "FAIL"]
        if failures:
            logger.error("TESTRX release integrity failed: %s", ", ".join(failures))
            raise RuntimeError("TESTRX release integrity failed: " + ", ".join(failures))
        self.integrity_checks = checks
        logger.info("TESTRX release integrity passed: release=%s checks=%d",
                    self.config["index_release_id"], len(checks))
        self.retrieval = self.config["retrieval"]
        self.collection = self.config["vector_store"]["collection"]
        if client is None:
            url = os.getenv("QDRANT_URL")
            if not url:
                raise ValueError("QDRANT_URL is required")
            self.client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"), timeout=30)
        else:
            self.client = client
        self._validate_collection()
        if encoder is None:
            spec = self.config["models"]["bi_encoder"]
            self.encoder = SentenceTransformer(
                str(model_path(spec)), device=os.getenv("MODEL_DEVICE", "cpu"),
                model_kwargs={"local_files_only": True},
            )
        else:
            self.encoder = encoder
        if reranker is None:
            spec = self.config["models"]["cross_encoder"]
            self.reranker = CrossEncoder(
                str(model_path(spec)), device=os.getenv("MODEL_DEVICE", "cpu"),
                automodel_args={"local_files_only": True},
            )
        else:
            self.reranker = reranker
        dimension = getattr(self.encoder, "get_sentence_embedding_dimension", lambda: None)()
        expected_dimension = self.config["models"]["bi_encoder"]["dimension"]
        if dimension is not None and int(dimension) != expected_dimension:
            raise ValueError("Loaded bi-encoder dimension differs from frozen vector contract")
        self.store = store or InvocationStore()

    def _validate_collection(self) -> None:
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        if not hasattr(vectors, "size"):
            raise ValueError("TESTRX collection must use one unnamed dense vector")
        expected = self.config["models"]["bi_encoder"]["dimension"]
        if vectors.size != expected or vectors.distance != models.Distance.COSINE:
            raise ValueError("Qdrant vector size/distance conflicts with frozen config")
        points, _ = self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key="record_type", match=models.MatchValue(value="contract")
            )]), limit=1, with_payload=True, with_vectors=False,
        )
        expected_contract = self._contract()
        if not points or points[0].payload != expected_contract:
            raise ValueError("Qdrant collection release contract conflicts with frozen config")
        chunks, _ = self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key="record_type", match=models.MatchValue(value="chunk")
            )]), limit=1, with_payload=True, with_vectors=False,
        )
        required = set(self.config["vector_store"]["payload_fields"])
        if not chunks or not required.issubset(chunks[0].payload or {}):
            raise ValueError("Qdrant chunk payload does not match frozen payload schema")

    def _contract(self) -> dict[str, Any]:
        if not self.release_lock:
            return collection_contract(self.config)
        return index_contract(self.config, self.release_lock)

    def retrieve(self, query: str, *, top_k: int | None = None,
                 candidate_k: int | None = None) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        top_k = self.retrieval["top_k"] if top_k is None else int(top_k)
        candidate_k = self.retrieval["candidate_k"] if candidate_k is None else int(candidate_k)
        if top_k < 1 or candidate_k < top_k:
            raise ValueError("candidate_k must be at least top_k, and top_k must be positive")

        started = time.perf_counter_ns()
        query_vector = self.encoder.encode(
            [self.config["models"]["bi_encoder"]["query_prefix"] + query], batch_size=1,
            normalize_embeddings=self.config["models"]["bi_encoder"]["normalize_embeddings"], convert_to_numpy=True,
            show_progress_bar=False,
        )[0].tolist()
        embedded = time.perf_counter_ns()
        found = self.client.query_points(
            collection_name=self.collection, query=query_vector, limit=candidate_k,
            with_payload=True, search_params=models.SearchParams(exact=True),
            query_filter=models.Filter(must=[models.FieldCondition(
                key="record_type", match=models.MatchValue(value="chunk")
            )]),
        ).points
        searched = time.perf_counter_ns()
        candidates = [{"point": point, "payload": point.payload, "dense_score": float(point.score),
                       "dense_rank": rank} for rank, point in enumerate(found, 1)]
        pairs = [(query, candidate["payload"]["text"]) for candidate in candidates]
        scores = self.reranker.predict(
            pairs, batch_size=self.config["models"]["cross_encoder"]["batch_size"],
            show_progress_bar=False,
        ) if pairs else []
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
        reranked = sorted(candidates, key=lambda c: (-round(c["rerank_score"], 12), c["dense_rank"]))
        for rank, candidate in enumerate(reranked, 1):
            candidate["rerank_rank"] = rank
        selected = reranked[:top_k]
        finished = time.perf_counter_ns()
        timings = {
            "query_embedding_ns": embedded - started,
            "vector_search_ns": searched - embedded,
            "cross_encoder_ns": finished - searched,
            "total_retrieval_ns": finished - started,
        }
        context_parts = []
        output_chunks = []
        selected_ranks: dict[str, int] = {}
        for rank, candidate in enumerate(selected, 1):
            payload = candidate["payload"]
            meta = payload["metadata"]
            selected_ranks[payload["chunk_id"]] = rank
            pages = f"p. {meta['page_start']}" if meta["page_start"] == meta["page_end"] else f"pp. {meta['page_start']}–{meta['page_end']}"
            context_parts.append(f"[{pages}; { ' > '.join(meta['section_path']) }]\n{payload['text']}")
            output_chunks.append({
                "chunk_id": payload["chunk_id"], "rank": rank,
                "dense_rank": candidate["dense_rank"], "dense_score": candidate["dense_score"],
                "rerank_rank": candidate["rerank_rank"], "rerank_score": candidate["rerank_score"],
                "text": payload["text"], "metadata": meta,
            })
        context = "\n\n".join(context_parts)
        prompt = [
            {"role": "system", "content": self.config["prompt"]["system"]},
            {"role": "user", "content": self.config["prompt"]["template"].format(
                query=query, context=context
            )},
        ]
        invocation_id = str(uuid.uuid4())
        response = {
            "invocation_id": invocation_id,
            "config_id": self.config["config_id"],
            "index_release_id": self.config["index_release_id"],
            "query": query, "candidate_k": candidate_k, "top_k": top_k,
            "timings": timings, "prompt": prompt, "chunks": output_chunks,
            "ranking_trace": [{
                "chunk_id": c["payload"]["chunk_id"], "dense_rank": c["dense_rank"],
                "dense_score": c["dense_score"], "rerank_rank": c["rerank_rank"],
                "rerank_score": c["rerank_score"],
                "selected_rank": selected_ranks.get(c["payload"]["chunk_id"]),
            } for c in candidates],
        }
        self.store.record({
            **response, "created_at": datetime.now(timezone.utc).isoformat(),
            "encoder_id": self.config["models"]["bi_encoder"]["id"],
            "reranker_id": self.config["models"]["cross_encoder"]["id"],
        }, [{
            "invocation_id": invocation_id, "chunk_id": c["payload"]["chunk_id"],
            "dense_rank": c["dense_rank"], "dense_score": c["dense_score"],
            "rerank_rank": c["rerank_rank"], "rerank_score": c["rerank_score"],
            "selected_rank": next((x["rank"] for x in output_chunks if x["chunk_id"] == c["payload"]["chunk_id"]), None),
            "payload": c["payload"]["metadata"],
        } for c in candidates])
        return response


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    candidate_k: int | None = Field(default=None, ge=1)


class AnswerRequest(BaseModel):
    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.testrx = TestrxApplication()
    yield


api = FastAPI(title="TESTRX application", version="1.0.0", lifespan=lifespan)


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/prompt")
def prompt(request: RetrieveRequest):
    try:
        return api.state.testrx.retrieve(
            request.query, top_k=request.top_k, candidate_k=request.candidate_k
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@api.post("/invocations/{invocation_id}/answer")
def record_answer(invocation_id: str, request: AnswerRequest):
    try:
        api.state.testrx.store.record_answer(invocation_id, request.answer)
        return {"invocation_id": invocation_id, "stored": True}
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
