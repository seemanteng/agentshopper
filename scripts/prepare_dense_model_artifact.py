"""Packaging: copy the exact validated dense-route embedding model
(sentence-transformers/all-MiniLM-L6-v2 @ revision
1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -- the model every dense-route
number in README.md's "What we tried" was measured with) out of this
machine's user-specific Hugging Face cache and into a self-contained,
repository-relative staging directory that can travel with a submission
archive (scripts/build_submission_archive.py) and be loaded with
local_files_only=True on a machine that has never seen this cache.

This is the same offline-packaging gap README.md's "Deployment readiness"
and "Limitations" sections flagged for the dense route (the frozen
cross-encoder already has this via scripts/prepare_cross_encoder_artifact.py;
the dense encoder did not). Mirrors that script's approach exactly: read-only
against the local HF cache with local_files_only=True (snapshot_download
refuses to hit the network in that mode -- it raises LocalEntryNotFoundError
instead), real file bytes copied (never symlinks), no modification,
quantisation, or conversion of the model -- byte-for-byte copy, checksummed.

Usage:
    python3 scripts/prepare_dense_model_artifact.py [--force]
    python3 scripts/prepare_dense_model_artifact.py --model <id> --revision <rev> --output <dir>
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

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# Exactly what agent_shopper.dense_index.DenseIndex's
# sentence_transformers.SentenceTransformer(...) load needs for this
# Transformer+Pooling+Normalize pipeline -- no README/model-card, no
# .gitattributes, no cache lock/ref metadata. The Normalize module (stage 2
# of modules.json) is parameterless and persists no config file of its own.
REQUIRED_FILES = (
    "modules.json",
    "config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "model.safetensors",
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

This checkpoint is a frozen, pre-trained sentence-embedding model published
by the sentence-transformers project, built on the base architecture
"nreimers/MiniLM-L6-H384-uncased" (6-layer, 384-hidden distilled BERT) and
fine-tuned by the model publisher for general-purpose sentence similarity.

This model is used here strictly frozen / inference-only -- no gradient
updates, no fine-tuning, no further training was performed by this project
(see agent_shopper/dense_index.py's module docstring). Weight files are
copied byte-for-byte from the publisher's checkpoint; see manifest.json in
this directory for per-file SHA-256 checksums and the overall deterministic
tree hash.
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
            "Base architecture nreimers/MiniLM-L6-H384-uncased; fine-tuned by the "
            "sentence-transformers project for general-purpose sentence similarity. "
            "Frozen, inference-only use in this project -- no fine-tuning or gradient "
            "updates performed here."
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
    parser.add_argument("--output", default=None, help="Defaults to models/dense/<model-slug>/ under the repo root.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing, differing artefact.")
    args = parser.parse_args()

    slug = _model_slug(args.model)
    output_dir = Path(args.output).resolve() if args.output else (ROOT / "models" / "dense" / slug)

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
        dst.parent.mkdir(parents=True, exist_ok=True)  # e.g. 1_Pooling/config.json
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
