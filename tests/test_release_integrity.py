from __future__ import annotations

import json
from pathlib import Path
import unittest

import yaml

from testrx_prod.integrity import static_release_checks


ROOT = Path(__file__).parents[1]


class ReleaseIntegrityTests(unittest.TestCase):
    def test_config_and_bundled_lock_are_linked_and_code_matches(self):
        config_path = ROOT / "config.yaml"
        lock_path = ROOT / "release-lock.json"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        checks = static_release_checks(config, lock, lock_path)
        selected = [check for check in checks if check["name"] in {
            "release lock checksum", "production configuration checksum",
            "runtime/index contract", "production runtime implementation",
        } or check["name"].startswith("source implementation ")]
        self.assertTrue(selected)
        self.assertTrue(all(check["status"] == "PASS" for check in selected), selected)
        self.assertEqual(lock["parser"]["canonical_document_sha256"].__len__(), 64)
        self.assertEqual(lock["chunking"]["ordered_chunks_sha256"].__len__(), 64)
        self.assertEqual(lock["chunking"]["document_vectors_sha256"].__len__(), 64)


if __name__ == "__main__":
    unittest.main()
