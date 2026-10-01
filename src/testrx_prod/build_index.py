from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from testrx_retriever.configuration import ChunkingConfig
from testrx_retriever.parsing.parser import parse_manual, write_outputs
from testrx_retriever.retrieval.hierarchical_chunking import build_chunker
from testrx_retriever.retrieval.tokenization import RegexTokenizer

from .config import model_path
from .integrity import canonical_sha256, index_contract, load_release, vectors_sha256


class ReleaseBuildError(RuntimeError):
    pass


def _check(report: dict, name: str, expected, actual, detail: str = "") -> None:
    item = {"name": name, "status": "PASS" if expected == actual else "FAIL",
            "expected": expected, "actual": actual}
    if detail:
        item["detail"] = detail
    report["checks"].append(item)


def _save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _require_pass(report: dict, stage: str) -> None:
    failed = [item["name"] for item in report["checks"] if item["status"] == "FAIL"]
    if failed:
        raise ReleaseBuildError(f"{stage} failed checks: {', '.join(failed)}")


def build(pdf: Path, *, config_path: Path | None = None, lock_path: Path | None = None,
          report_path: Path | None = None, replace: bool = False) -> dict:
    config_path = Path(config_path or os.getenv("TESTRX_CONFIG", Path(__file__).parents[2] / "config.yaml"))
    report_path = Path(report_path or os.getenv("TESTRX_BUILD_REPORT", Path(tempfile.gettempdir()) / "testrx-index-build-report.json"))
    report = {"run_id": str(uuid.uuid4()), "started_at": datetime.now(timezone.utc).isoformat(),
              "status": "RUNNING", "checks": [], "errors": []}
    client = None
    try:
        config, lock, static_checks = load_release(config_path)
        report.update({"config_id": config["config_id"], "index_release_id": config["index_release_id"]})
        report["checks"].extend(static_checks)
        if lock_path:
            configured_lock = Path(config["integrity"]["lock_file"])
            configured_lock = configured_lock if configured_lock.is_absolute() else config_path.parent / configured_lock
            _check(report, "provided and configured release locks match",
                   canonical_sha256(json.loads(configured_lock.read_text(encoding="utf-8"))),
                   canonical_sha256(json.loads(Path(lock_path).read_text(encoding="utf-8"))))
        _require_pass(report, "static release integrity")

        digest = hashlib.sha256()
        with pdf.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        source_sha = digest.hexdigest().upper()
        _check(report, "input PDF checksum", lock["source"]["sha256"], source_sha)
        _check(report, "config and lock source checksum", config["source"]["sha256"], lock["source"]["sha256"])
        _require_pass(report, "source identity")

        document = parse_manual(pdf)
        _check(report, "parser version", lock["parser"]["implementation"]["version"], document.parser_version)
        _check(report, "document schema", lock["parser"]["implementation"]["document_schema_version"], document.schema_version)
        _check(report, "canonical document checksum", lock["parser"]["canonical_document_sha256"],
               canonical_sha256(document.to_dict()))
        validation_dir = Path(os.getenv("TESTRX_PARSE_OUTPUT", Path(tempfile.gettempdir()) / "testrx-parsing"))
        validation, warnings = write_outputs(document, validation_dir)
        _check(report, "parser validation", "PASS", "FAIL" if validation["overall_status"] == "FAIL" else "PASS",
               f"warnings={warnings['summary']['warnings']}")
        _require_pass(report, "parsing")

        chunks = build_chunker(ChunkingConfig(
            strategy=config["chunking"]["strategy"], max_tokens=config["chunking"]["max_tokens"]
        ), RegexTokenizer()).chunk_document(document.to_dict())
        _check(report, "chunk count", lock["chunking"]["chunk_count"], len(chunks))
        _check(report, "ordered chunk text and metadata checksum",
               lock["chunking"]["ordered_chunks_sha256"],
               canonical_sha256([chunk.to_dict() for chunk in chunks]))
        _require_pass(report, "chunk generation")

        encoder_spec = config["models"]["bi_encoder"]
        encoder = SentenceTransformer(
            str(model_path(encoder_spec)), device=os.getenv("MODEL_DEVICE", "cpu"),
            model_kwargs={"local_files_only": True},
        )
        vectors = encoder.encode(
            [encoder_spec["document_prefix"] + chunk.text for chunk in chunks],
            batch_size=encoder_spec["batch_size"],
            normalize_embeddings=encoder_spec["normalize_embeddings"],
            convert_to_numpy=True, show_progress_bar=False,
        )
        expected_shape = (len(chunks), encoder_spec["dimension"])
        _check(report, "document embedding shape", expected_shape, tuple(vectors.shape))
        _check(report, "document embedding checksum (rounded 1e-6)",
               lock["chunking"]["document_vectors_sha256"], vectors_sha256(vectors))
        _require_pass(report, "embedding")

        url = os.getenv("QDRANT_URL")
        _check(report, "QDRANT_URL configured", True, bool(url))
        _require_pass(report, "vector store connection")
        client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"), timeout=60)
        collection = config["vector_store"]["collection"]
        existing = collection in {item.name for item in client.get_collections().collections}
        _check(report, "collection is new or explicitly replaceable", False if not replace else True,
               existing if not replace else True)
        _require_pass(report, "collection safety")
        if existing:
            client.delete_collection(collection)
            report["checks"].append({"name": "existing collection removed by explicit --replace",
                                     "status": "PASS", "expected": "explicit replacement",
                                     "actual": collection})
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=encoder_spec["dimension"], distance=models.Distance.COSINE),
        )
        points = [models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{config['index_release_id']}:{chunk.chunk_id}")),
            vector=vector.tolist(),
            payload={"record_type": "chunk", "chunk_id": chunk.chunk_id, "text": chunk.text,
                     "metadata": {key: value for key, value in chunk.to_dict().items()
                                  if key not in {"chunk_id", "text"}}},
        ) for chunk, vector in zip(chunks, vectors)]
        points.append(models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, config["index_release_id"] + ":contract")),
            vector=[1.0] + [0.0] * (encoder_spec["dimension"] - 1),
            payload=index_contract(config, lock),
        ))
        client.upload_points(collection_name=collection, points=points, batch_size=64, wait=True)
        chunk_point_ids = [point.id for point in points[:-1]]
        stored_vectors = client.retrieve(collection, ids=chunk_point_ids,
                                         with_payload=False, with_vectors=True)
        vectors_by_id = {str(point.id): point.vector for point in stored_vectors}
        ordered_stored_vectors = [vectors_by_id[str(point_id)] for point_id in chunk_point_ids]
        _check(report, "Qdrant stored vector checksum (rounded 1e-6)",
               lock["chunking"]["document_vectors_sha256"], vectors_sha256(ordered_stored_vectors))
        info = client.get_collection(collection)
        actual_params = {"dimension": info.config.params.vectors.size,
                         "distance": info.config.params.vectors.distance.value}
        _check(report, "Qdrant vector schema", {"dimension": encoder_spec["dimension"], "distance": "Cosine"},
               actual_params)
        stored_contract, _ = client.scroll(collection, scroll_filter=models.Filter(must=[
            models.FieldCondition(key="record_type", match=models.MatchValue(value="contract"))
        ]), limit=1, with_payload=True, with_vectors=False)
        _check(report, "Qdrant release contract stored", index_contract(config, lock),
               stored_contract[0].payload if stored_contract else None)
        count = client.count(collection, exact=True).count
        _check(report, "Qdrant point count", len(chunks) + 1, count)
        _require_pass(report, "Qdrant publication")
        report["status"] = "PASS"
        report["published_chunks"] = len(chunks)
        report["collection"] = collection
        return report
    except Exception as error:
        report["status"] = "FAIL"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
        if not any(item["status"] == "FAIL" for item in report["checks"]):
            report["checks"].append({"name": "build execution", "status": "FAIL",
                                     "expected": "release build completed", "actual": str(error)})
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        _save_report(report_path, report)
        for check in report["checks"]:
            print(f"{check['status']}: {check['name']}")
        print(f"{report['status']}: report={report_path}")
        if client is not None:
            client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and publish the frozen TESTRX Qdrant index release")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--lock", type=Path, help="Optional independently supplied copy of the dev release lock")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--replace", action="store_true", help="Replace the configured collection after integrity checks pass")
    args = parser.parse_args()
    try:
        build(args.pdf, config_path=args.config, lock_path=args.lock,
              report_path=args.report, replace=args.replace)
    except Exception as error:
        print(f"RELEASE BUILD FAILED: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
