from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from qdrant_client import models

from testrx_prod.application import TestrxApplication
from testrx_prod.config import collection_contract, load_config
from testrx_prod.store import InvocationStore


class FakeClient:
    def __init__(self, config):
        self.config = config
        self.scroll_count = 0
        self.payloads = [
            {"record_type": "chunk", "chunk_id": "c1", "text": "Manual says use Menu A.",
             "metadata": {"page_start": 2, "page_end": 2, "section_path": ["1 Setup"]}},
            {"record_type": "chunk", "chunk_id": "c2", "text": "Manual says use Menu B.",
             "metadata": {"page_start": 4, "page_end": 5, "section_path": ["2 Use"]}},
        ]

    def get_collection(self, name):
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(
            size=384, distance=models.Distance.COSINE
        ))))

    def scroll(self, *args, **kwargs):
        self.scroll_count += 1
        payload = collection_contract(self.config) if self.scroll_count == 1 else self.payloads[0]
        return [SimpleNamespace(payload=payload)], None

    def query_points(self, **kwargs):
        return SimpleNamespace(points=[
            SimpleNamespace(payload=payload, score=0.9 - i * 0.1)
            for i, payload in enumerate(self.payloads)
        ])


class FakeEncoder:
    def encode(self, *args, **kwargs):
        return [[0.0] * 384]


class FakeReranker:
    def predict(self, pairs, **kwargs):
        return [0.2, 0.9]


class ProductionApplicationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).parents[1] / "config.yaml")
        self.temp = tempfile.TemporaryDirectory()
        self.store = InvocationStore(Path(self.temp.name) / "audit.sqlite3")
        self.app = TestrxApplication(
            self.config, client=FakeClient(self.config), encoder=FakeEncoder(),
            reranker=FakeReranker(), store=self.store, verify_integrity=False,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_frozen_contract_and_rerank_trace_return_prompt(self):
        result = self.app.retrieve("Which menu should I use?")
        self.assertEqual(result["candidate_k"], 10)
        self.assertEqual(result["top_k"], 5)
        self.assertEqual(result["chunks"][0]["chunk_id"], "c2")
        self.assertEqual(result["chunks"][0]["dense_rank"], 2)
        self.assertEqual(result["chunks"][0]["rerank_rank"], 1)
        self.assertEqual(len(result["ranking_trace"]), 2)
        self.assertEqual(result["ranking_trace"][0]["selected_rank"], 2)
        self.assertEqual(result["ranking_trace"][1]["selected_rank"], 1)
        self.assertIn("[pp. 4–5; 2 Use]", result["prompt"][1]["content"])
        self.assertEqual(result["config_id"], self.config["config_id"])

    def test_k_override_validation_and_normalized_query_log(self):
        result = self.app.retrieve("A question", candidate_k=2, top_k=1)
        self.assertEqual(result["top_k"], 1)
        with self.assertRaisesRegex(ValueError, "candidate_k"):
            self.app.retrieve("A question", candidate_k=1, top_k=2)
        with self.store.connect() as db:
            invocation = db.execute("SELECT * FROM invocation WHERE invocation_id=?",
                                    (result["invocation_id"],)).fetchone()
            rankings = db.execute("SELECT * FROM ranking WHERE invocation_id=? ORDER BY dense_rank",
                                  (result["invocation_id"],)).fetchall()
        self.assertEqual(invocation["candidate_k"], 2)
        self.assertEqual(len(rankings), 2)
        self.assertEqual(rankings[1]["rerank_rank"], 1)
        self.store.record_answer(result["invocation_id"], "Menu B")
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT answer FROM invocation WHERE invocation_id=?",
                                        (result["invocation_id"],)).fetchone()[0], "Menu B")

    def test_collection_contract_mismatch_fails_startup(self):
        bad = self.config.copy()
        bad["config_id"] = "other-release"
        with self.assertRaisesRegex(ValueError, "release contract"):
            TestrxApplication(bad, client=FakeClient(self.config), encoder=FakeEncoder(),
                              reranker=FakeReranker(), store=self.store, verify_integrity=False)


if __name__ == "__main__":
    unittest.main()
