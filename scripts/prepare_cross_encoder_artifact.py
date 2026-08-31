"""Phase 3 packaging: copy the exact validated frozen cross-encoder checkpoint
(cross-encoder/ms-marco-TinyBERT-L-6 @ defbb7d2405cfb2a0f9db418cd8a377c97469552
-- the substitute that produced TechnicalScore 0.5989 at alpha=0.30, see
README.md's "What we tried" and agent_shopper/cross_encoder_reranker.py's
module docstring) out of this machine's user-specific Hugging Face cache and
into a self-contained, repository-relative staging directory that can travel
with a submission archive (scripts/build_submission_archive.py) and be loaded
with local_files_only=True on a machine that has never seen this cache.

This script only ever reads from the local HF cache with local_files_only=True
(huggingface_hub.snapshot_download refuses to hit the network in that mode --
it raises LocalEntryNotFoundError instead) and copies real file bytes (never
symlinks) for exactly the files agent_shopper.cross_encoder_reranker's
sentence-transformers CrossEncoder loader needs. It does not modify, quantise,
or convert the model in any way -- byte-for-byte copy, checksummed.

Usage:
    python3 scripts/prepare_cross_encoder_artifact.py [--force]
    python3 scripts/prepare_cross_encoder_artifact.py --model <id> --revision <rev> --output <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = "cross-encoder/ms-marco-TinyBERT-L-6"
DEFAULT_REVISION = "defbb7d2405cfb2a0f9db418cd8a377c97469552"

# Exactly what agent_shopper.cross_encoder_reranker.FrozenCrossEncoderScorer's
# sentence_transformers.CrossEncoder(...) load needs -- no README/model-card/
# extra-format files, no .gitattributes, no cache lock/ref metadata.
REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)

LICENSE_ATTRIBUTION_TEMPLATE = """\
Model: {model_id}
Revision (commit hash): {revision}
Source: https://huggingface.co/{model_id}

License: Apache License, Version 2.0 (SPDX: Apache-2.0)

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Attribution:

This checkpoint is a frozen, pre-trained pointwise cross-encoder for the
MS MARCO Passage Ranking task, published by the sentence-transformers /
cross-encoder project. It builds on the base architecture
"nreimers/TinyBERT_L-6_H-768_v2" (6-layer, 768-hidden BertForSequenceClassification)
and was fine-tuned by the model publisher on the "sentence-transformers/msmarco"
training data (see the model card at the Source URL above for training details).

This model is used here strictly frozen / inference-only -- no gradient
updates, no fine-tuning, no further training was performed by this project
(see agent_shopper/cross_encoder_reranker.py's module docstring). Weight
files are copied byte-for-byte from the publisher's checkpoint; see
manifest.json in this directory for per-file SHA-256 checksums and the
overall deterministic tree hash.
"""


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_snapshot_dir(model: str, revision: str) -> Path:
    """Resolve the local HF cache snapshot directory for (model, revision),
    strictly offline -- never downloads. Raises with a clear message if the
    checkpoint isn't already cached locally at that exact revision."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(f"huggingface_hub is required to resolve the local cache: {exc}")

    try:
        snapshot_dir = snapshot_download(repo_id=model, revision=revision, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 -- surface exactly why offline resolution failed
        raise SystemExit(
            f"Could not resolve '{model}' @ '{revision}' from the local Hugging Face cache "
            f"with local_files_only=True (never attempted a download): {type(exc).__name__}: {exc}\n"
            "This means the exact validated checkpoint is not unambiguously available on this "
            "machine at that revision -- packaging must stop here rather than silently substitute "
            "a different revision or download one."
        )
    return Path(snapshot_dir)


def _model_slug(model: str) -> str:
    return model.split("/", 1)[-1]


def _compute_manifest(source_dir: Path, model: str, revision: str) -> dict:
    files = []
    missing = []
    for name in REQUIRED_FILES:
        src = source_dir / name
        if not src.is_file():
            missing.append(name)
            continue
        resolved = src.resolve()  # follow the cache's symlink to the real blob
        files.append({
            "path": name,
            "size_bytes": resolved.stat().st_size,
            "sha256": _sha256_of_file(resolved),
        })
    if missing:
        raise SystemExit(
            f"Required checkpoint file(s) missing from the resolved snapshot {source_dir}: "
            f"{', '.join(missing)}. Refusing to package a partial/ambiguous artefact."
        )

    tree_lines = sorted(f"{f['path']}:{f['sha256']}" for f in files)
    tree_hash = hashlib.sha256("\n".join(tree_lines).encode("utf-8")).hexdigest()

    return {
        "model_id": model,
        "revision": revision,
        "license": "Apache-2.0",
        "source_url": f"https://huggingface.co/{model}",
        "attribution": (
            "Base architecture nreimers/TinyBERT_L-6_H-768_v2; fine-tuned by the "
            "cross-encoder/sentence-transformers project on MS MARCO Passage Ranking "
            "(sentence-transformers/msmarco). Frozen, inference-only use in this project -- "
            "no fine-tuning or gradient updates performed here."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": files,
        "tree_sha256": tree_hash,
    }


def _existing_tree_hash(output_dir: Path) -> str | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8")).get("tree_sha256")
    except (json.JSONDecodeError, OSError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", default=None, help="Defaults to models/cross_encoder/<model-slug>/ under the repo root.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing, differing artefact.")
    args = parser.parse_args()

    slug = _model_slug(args.model)
    output_dir = Path(args.output).resolve() if args.output else (ROOT / "models" / "cross_encoder" / slug)

    print(f"Resolving '{args.model}' @ '{args.revision}' from the local Hugging Face cache (offline, no download)...")
    source_dir = _resolve_snapshot_dir(args.model, args.revision)
    print(f"  resolved: {source_dir}")

    manifest = _compute_manifest(source_dir, args.model, args.revision)

    existing_hash = _existing_tree_hash(output_dir)
    if existing_hash is not None and existing_hash != manifest["tree_sha256"] and not args.force:
        raise SystemExit(
            f"Refusing to overwrite existing artefact at {output_dir}: its manifest tree hash "
            f"({existing_hash}) does not match the freshly-resolved source ({manifest['tree_sha256']}). "
            "Pass --force to overwrite explicitly."
        )
    if existing_hash == manifest["tree_sha256"]:
        print(f"Existing artefact at {output_dir} already matches (tree hash {existing_hash[:12]}...) -- re-copying anyway for idempotency.")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for file_entry in manifest["files"]:
        src = (source_dir / file_entry["path"]).resolve()
        dst = output_dir / file_entry["path"]
        shutil.copyfile(src, dst)  # real bytes, never a symlink into the user cache
        total_bytes += file_entry["size_bytes"]

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "LICENSE-ATTRIBUTION.txt").write_text(
        LICENSE_ATTRIBUTION_TEMPLATE.format(model_id=args.model, revision=args.revision), encoding="utf-8"
    )

    print(f"\nPackaged {len(manifest['files'])} files to {output_dir}")
    for f in manifest["files"]:
        print(f"  {f['path']:28s} {f['size_bytes']:>12,d} bytes  sha256={f['sha256'][:16]}...")
    print(f"\nTotal artefact size: {total_bytes / (1024 * 1024):.1f} MB ({total_bytes:,} bytes)")
    print(f"Tree hash: {manifest['tree_sha256']}")
    print(f"Manifest:  {output_dir / 'manifest.json'}")
    print(f"License:   {output_dir / 'LICENSE-ATTRIBUTION.txt'}")


if __name__ == "__main__":
    main()
