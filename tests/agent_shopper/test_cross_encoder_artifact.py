"""Unit tests for scripts/prepare_cross_encoder_artifact.py's packaging
logic (manifest generation, checksum verification, missing/corrupted-file
detection, overwrite-refusal semantics). Runs entirely against a tiny fake
"HF cache snapshot" directory built in a temp dir -- never touches the real
256MB checkpoint or the real Hugging Face cache."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_cross_encoder_artifact.py"

_spec = importlib.util.spec_from_file_location("prepare_cross_encoder_artifact", SCRIPT_PATH)
_artifact_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _artifact_mod
_spec.loader.exec_module(_artifact_mod)


def _write_fake_snapshot(snapshot_dir: Path, contents: dict[str, bytes]) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        (snapshot_dir / name).write_bytes(data)


def _fake_source_contents() -> dict[str, bytes]:
    return {
        "model.safetensors": b"fake-weights-bytes-not-a-real-checkpoint",
        "config.json": b'{"model_type": "bert"}',
        "tokenizer.json": b'{"version": "1.0"}',
        "tokenizer_config.json": b"{}",
        "special_tokens_map.json": b"{}",
        "vocab.txt": b"[PAD]\n[UNK]\n",
    }


class ComputeManifestTest(unittest.TestCase):
    def test_manifest_shape_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = Path(tmp) / "snapshot"
            _write_fake_snapshot(snapshot_dir, _fake_source_contents())
            manifest = _artifact_mod._compute_manifest(snapshot_dir, "fake/model", "deadbeef")

            self.assertEqual(manifest["model_id"], "fake/model")
            self.assertEqual(manifest["revision"], "deadbeef")
            self.assertEqual(manifest["license"], "Apache-2.0")
            self.assertEqual(len(manifest["files"]), len(_artifact_mod.REQUIRED_FILES))
            paths = {f["path"] for f in manifest["files"]}
            self.assertEqual(paths, set(_artifact_mod.REQUIRED_FILES))
            for f in manifest["files"]:
                self.assertGreater(f["size_bytes"], 0)
                self.assertEqual(len(f["sha256"]), 64)
            self.assertEqual(len(manifest["tree_sha256"]), 64)

    def test_tree_hash_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = Path(tmp) / "snapshot"
            _write_fake_snapshot(snapshot_dir, _fake_source_contents())
            m1 = _artifact_mod._compute_manifest(snapshot_dir, "fake/model", "deadbeef")
            m2 = _artifact_mod._compute_manifest(snapshot_dir, "fake/model", "deadbeef")
            self.assertEqual(m1["tree_sha256"], m2["tree_sha256"])

    def test_tree_hash_changes_on_bit_flip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = Path(tmp) / "snapshot"
            contents = _fake_source_contents()
            _write_fake_snapshot(snapshot_dir, contents)
            original = _artifact_mod._compute_manifest(snapshot_dir, "fake/model", "deadbeef")

            corrupted = dict(contents)
            corrupted["model.safetensors"] = b"F" + corrupted["model.safetensors"][1:]  # single byte flipped
            _write_fake_snapshot(snapshot_dir, corrupted)
            flipped = _artifact_mod._compute_manifest(snapshot_dir, "fake/model", "deadbeef")

            self.assertNotEqual(original["tree_sha256"], flipped["tree_sha256"])

    def test_tree_hash_changes_on_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = Path(tmp) / "snapshot"
            contents = _fake_source_contents()
            _write_fake_snapshot(snapshot_dir, contents)
            original = _artifact_mod._compute_manifest(snapshot_dir, "fake/model", "deadbeef")

            truncated = dict(contents)
            truncated["model.safetensors"] = truncated["model.safetensors"][:5]
            _write_fake_snapshot(snapshot_dir, truncated)
            shrunk = _artifact_mod._compute_manifest(snapshot_dir, "fake/model", "deadbeef")

            self.assertNotEqual(original["tree_sha256"], shrunk["tree_sha256"])
            original_size = next(f["size_bytes"] for f in original["files"] if f["path"] == "model.safetensors")
            shrunk_size = next(f["size_bytes"] for f in shrunk["files"] if f["path"] == "model.safetensors")
            self.assertNotEqual(original_size, shrunk_size)

    def test_missing_required_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = Path(tmp) / "snapshot"
            contents = _fake_source_contents()
            del contents["vocab.txt"]
            _write_fake_snapshot(snapshot_dir, contents)
            with self.assertRaises(SystemExit):
                _artifact_mod._compute_manifest(snapshot_dir, "fake/model", "deadbeef")


class OverwriteRefusalTest(unittest.TestCase):
    def test_existing_tree_hash_is_read_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "artifact"
            output_dir.mkdir()
            self.assertIsNone(_artifact_mod._existing_tree_hash(output_dir))

            (output_dir / "manifest.json").write_text('{"tree_sha256": "abc123"}', encoding="utf-8")
            self.assertEqual(_artifact_mod._existing_tree_hash(output_dir), "abc123")

    def test_corrupted_manifest_json_reads_as_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "artifact"
            output_dir.mkdir()
            (output_dir / "manifest.json").write_text("{not valid json", encoding="utf-8")
            self.assertIsNone(_artifact_mod._existing_tree_hash(output_dir))


class ModelSlugTest(unittest.TestCase):
    def test_strips_org_prefix(self) -> None:
        self.assertEqual(_artifact_mod._model_slug("cross-encoder/ms-marco-TinyBERT-L-6"), "ms-marco-TinyBERT-L-6")

    def test_no_prefix_passes_through(self) -> None:
        self.assertEqual(_artifact_mod._model_slug("bert-base-uncased"), "bert-base-uncased")


if __name__ == "__main__":
    unittest.main()
