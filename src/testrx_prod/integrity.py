"""Integrity primitives shared by the dev release-freeze and production build."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np


CORE_FILES = (
    "__init__.py",
    "configuration.py",
    "parsing/__init__.py",
    "parsing/domain.py",
    "parsing/figures.py",
    "parsing/inspection.py",
    "parsing/normalize.py",
    "parsing/parser.py",
    "parsing/structure.py",
    "parsing/tables.py",
    "parsing/validation.py",
    "retrieval/chunking.py",
    "retrieval/hierarchical_chunking.py",
    "retrieval/tokenization.py",
)
RUNTIME_PACKAGES = (
    "pdfplumber", "pdfminer.six", "Pillow", "pypdfium2", "pypdf", "numpy",
    "sentence-transformers", "transformers", "tokenizers", "huggingface-hub",
    "torch", "safetensors", "regex", "PyYAML", "scikit-learn", "scipy",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def config_sha256(config: dict[str, Any]) -> str:
    normalized = json.loads(json.dumps(config))
    normalized.get("integrity", {}).pop("expected_lock_sha256", None)
    return canonical_sha256(normalized)


def component_files(package_root: Path) -> dict[str, str]:
    return {name: sha256_file(package_root / name) for name in CORE_FILES}


def production_runtime_files(production_root: Path) -> dict[str, str]:
    paths = [production_root / name for name in ("Dockerfile", "pyproject.toml", "requirements.txt")]
    paths.extend((production_root / "src/testrx_prod").glob("*.py"))
    return {path.relative_to(production_root).as_posix(): sha256_file(path)
            for path in sorted(paths)}


def tree_manifest(root: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or {".cache", ".git", "__pycache__"}.intersection(path.parts):
            continue
        files[path.relative_to(root).as_posix()] = sha256_file(path)
    return {"files": files, "sha256": canonical_sha256(files), "file_count": len(files)}


def runtime_versions() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in RUNTIME_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python_minor": f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}",
            "packages": versions}


def add_check(checks: list[dict[str, Any]], name: str, expected: Any,
              actual: Any, *, detail: str = "") -> None:
    checks.append({"name": name, "status": "PASS" if expected == actual else "FAIL",
                   "expected": expected, "actual": actual, "detail": detail})


def core_package_root() -> Path:
    return Path(__file__).parents[1] / "testrx_retriever"


def index_contract(config: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    return {
        **lock["contract"],
        "release_lock_sha256": config["integrity"]["expected_lock_sha256"],
        "canonical_document_sha256": lock["parser"]["canonical_document_sha256"],
        "ordered_chunks_sha256": lock["chunking"]["ordered_chunks_sha256"],
        "chunk_count": lock["chunking"]["chunk_count"],
        "document_vectors_sha256": lock["chunking"]["document_vectors_sha256"],
        "build_runtime": lock["runtime"],
        "model_artifact_sha256": {
            role: value["artifact"]["sha256"] for role, value in lock["models"].items()
        },
    }


def vectors_sha256(vectors: Any) -> str:
    rounded = np.asarray(vectors, dtype=np.float32).round(decimals=6)
    return canonical_sha256(rounded.tolist())


def static_release_checks(config: dict[str, Any], lock: dict[str, Any],
                          lock_path: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    add_check(checks, "release lock checksum", config["integrity"]["expected_lock_sha256"],
              canonical_sha256(lock))
    add_check(checks, "production configuration checksum", lock["production_config_sha256"],
              config_sha256(config))
    from .config import collection_contract
    add_check(checks, "runtime/index contract", lock["contract"], collection_contract(config))
    expected_files = lock["parser"]["code_files"] | lock["chunking"]["code_files"]
    actual_files = component_files(core_package_root())
    for name, expected in expected_files.items():
        add_check(checks, f"source implementation {name}", expected, actual_files.get(name))
    expected_runtime_code = lock["production_runtime_code_files"]
    actual_runtime_code = production_runtime_files(Path(__file__).parents[2])
    add_check(checks, "production runtime implementation", expected_runtime_code, actual_runtime_code)
    expected_runtime = lock["runtime"]
    actual_runtime = runtime_versions()
    add_check(checks, "Python major/minor", expected_runtime["python_minor"], actual_runtime["python_minor"])
    for name, expected in expected_runtime["packages"].items():
        add_check(checks, f"runtime dependency {name}", expected, actual_runtime["packages"].get(name))
    for name, expected in lock["application_packages"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        add_check(checks, f"application dependency {name}", expected, actual)
    model_root = Path(os.getenv("MODEL_ROOT", "/models"))
    for role, expected in lock["models"].items():
        spec = config["models"][role]
        add_check(checks, f"{role} model identity", {
            "id": expected["id"], "revision": expected["revision"],
        }, {"id": spec["id"], "revision": spec["revision"]})
        path = model_root / spec["path"]
        actual = tree_manifest(path) if path.is_dir() else {"files": {}, "sha256": "MISSING", "file_count": 0}
        add_check(checks, f"{role} artifact checksum", expected["artifact"], actual)
    return checks


def load_release(config_path: str | Path | None = None):
    from .config import load_config

    resolved = Path(config_path or os.getenv("TESTRX_CONFIG", Path(__file__).parents[2] / "config.yaml")).resolve()
    config = load_config(resolved)
    lock_path = Path(config["integrity"]["lock_file"])
    if not lock_path.is_absolute():
        lock_path = resolved.parent / lock_path
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = static_release_checks(config, lock, lock_path)
    return config, lock, checks
