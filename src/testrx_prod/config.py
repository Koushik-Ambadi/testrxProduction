from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or os.getenv("TESTRX_CONFIG", Path(__file__).parents[2] / "config.yaml"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "1.0" or config.get("freeze", {}).get("status") != "frozen":
        raise ValueError("Unsupported or unfrozen TESTRX production config")
    retrieval = config["retrieval"]
    if retrieval["candidate_k"] < retrieval["top_k"] or retrieval["top_k"] < 1:
        raise ValueError("Invalid frozen candidate_k/top_k contract")
    if config["chunking"] != {
        "strategy": "hierarchical_max_tokens", "max_tokens": 384,
        "tokenizer": "testrx_regex_tokenizer", "tokenizer_version": "1.0",
    }:
        raise ValueError("Production chunking contract differs from the frozen release")
    if config["retrieval"]["algorithm"] != "cosine_similarity_exact" \
            or config["retrieval"]["distance"] != "cosine" \
            or config["retrieval"]["exact_search"] is not True:
        raise ValueError("Production vector search differs from the frozen exact-cosine contract")
    if config["parser"]["document_schema_version"] != "1.0":
        raise ValueError("Unsupported canonical document schema")
    if config["parser"]["implementation"] != "native_pdf":
        raise ValueError("Unsupported frozen parser implementation")
    if config["models"]["bi_encoder"]["algorithm"] != "sentence_transformer" \
            or config["models"]["cross_encoder"]["algorithm"] != "sentence_transformer_cross_encoder":
        raise ValueError("Model algorithms differ from the frozen runtime implementation")
    if config["models"]["bi_encoder"]["query_prefix"] or config["models"]["bi_encoder"]["document_prefix"]:
        raise ValueError("Unsupported encoder prefixes in this frozen release")
    return config


def model_path(spec: dict[str, Any]) -> Path:
    root = Path(os.getenv("MODEL_ROOT", "/models"))
    path = root / spec["path"]
    if not path.is_dir():
        raise FileNotFoundError(f"Shared model artifact is missing: {path}")
    return path


def collection_contract(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_type": "contract",
        "contract_version": config["vector_store"]["contract_version"],
        "config_id": config["config_id"],
        "index_release_id": config["index_release_id"],
        "source_sha256": config["source"]["sha256"],
        "chunking": config["chunking"],
        "encoder_id": config["models"]["bi_encoder"]["id"],
        "encoder_revision": config["models"]["bi_encoder"]["revision"],
        "encoder_parameters": {
            "query_prefix": config["models"]["bi_encoder"]["query_prefix"],
            "document_prefix": config["models"]["bi_encoder"]["document_prefix"],
            "normalize_embeddings": config["models"]["bi_encoder"]["normalize_embeddings"],
            "batch_size": config["models"]["bi_encoder"]["batch_size"],
        },
        "reranker_id": config["models"]["cross_encoder"]["id"],
        "reranker_revision": config["models"]["cross_encoder"]["revision"],
        "reranker_batch_size": config["models"]["cross_encoder"]["batch_size"],
        "retrieval": config["retrieval"],
        "application_packages": config["runtime"]["application_packages"],
        "payload_schema_version": config["vector_store"]["payload_schema_version"],
        "payload_fields": config["vector_store"]["payload_fields"],
        "dimension": config["models"]["bi_encoder"]["dimension"],
        "distance": config["retrieval"]["distance"],
    }
