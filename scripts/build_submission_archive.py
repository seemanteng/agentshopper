"""Phase E packaging: build the actual submission ZIP from an explicit
allowlist (no "everything except X" globbing), so its contents match what
scripts/run_local_eval.py and scripts/smoke_test_cross_encoder_subprocess.py
verify against in the acceptance run.

Two variants:

  --variant cross-encoder (default)
      The real shipped default: agent_shopper/config.py's cross-encoder
      block is packaged unmodified (FROZEN_CROSS_ENCODER_ENABLED=True,
      alpha=0.30, packaged TinyBERT path, local_files_only=True), plus the
      packaged checkpoint itself under models/cross_encoder/.

  --variant baseline
      A genuine, zero-env-var rollback artefact: the STAGED COPY (never the
      working tree) of agent_shopper/config.py has its cross-encoder block's
      fallback defaults patched back to the pre-promotion values
      (FROZEN_CROSS_ENCODER_ENABLED=False, alpha=0.0) before zipping, and
      the checkpoint directory is omitted entirely (not needed when
      disabled). This is not "set an env var" -- it's a different archive
      whose own defaults are already off.

data/catalog.jsonl is deliberately excluded from both (organizer-supplied
at judging time, per this repo's own "distributed separately" convention --
see README.md's Setup section) -- our own clean-room reproduction run copies
the already-downloaded local catalog into the extracted directory purely for
verification, outside the archive itself.

Usage:
    python3 scripts/build_submission_archive.py [--variant cross-encoder|baseline] [--output PATH]
"""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (source path relative to ROOT, whether it's a directory) -- explicit, not a
# glob exclusion. Directories are copied recursively, minus __pycache__/.pyc.
ALLOWLIST = (
    ("starter", True),
    ("agent_shopper", True),
    ("evaluator", True),
    ("scripts/run_local_eval.py", False),
    ("scripts/prepare_cross_encoder_artifact.py", False),
    ("requirements.txt", False),
    ("README.md", False),
)

CROSS_ENCODER_MODEL_DIR = "models/cross_encoder/ms-marco-TinyBERT-L-6"

_IGNORE_PATTERNS = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")


def _stage_allowlist(staging_dir: Path) -> None:
    for rel_path, is_dir in ALLOWLIST:
        src = ROOT / rel_path
        dst = staging_dir / rel_path
        if not src.exists():
            raise SystemExit(f"Allowlisted path missing from repo: {rel_path}")
        if is_dir:
            shutil.copytree(src, dst, ignore=_IGNORE_PATTERNS)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)


def _patch_baseline_config(staged_config_path: Path) -> None:
    """Flip the cross-encoder block's fallback defaults back to pre-promotion
    values in the STAGED copy only -- never touches the working tree's real
    agent_shopper/config.py."""
    text = staged_config_path.read_text(encoding="utf-8")

    enabled_pattern = r'FROZEN_CROSS_ENCODER_ENABLED = os\.environ\.get\("AGENT_SHOPPER_FROZEN_CROSS_ENCODER", "1"\)'
    enabled_replacement = 'FROZEN_CROSS_ENCODER_ENABLED = os.environ.get("AGENT_SHOPPER_FROZEN_CROSS_ENCODER", "0")'
    if not re.search(enabled_pattern, text):
        raise SystemExit("build_submission_archive.py: could not find FROZEN_CROSS_ENCODER_ENABLED default to patch for the baseline variant -- config.py's shape changed, update this script's pattern.")
    text = re.sub(enabled_pattern, enabled_replacement, text)

    alpha_pattern = r'FROZEN_CROSS_ENCODER_ALPHA = _env_float\("AGENT_SHOPPER_CROSS_ENCODER_ALPHA", 0\.30\)'
    alpha_replacement = 'FROZEN_CROSS_ENCODER_ALPHA = _env_float("AGENT_SHOPPER_CROSS_ENCODER_ALPHA", 0.0)'
    if not re.search(alpha_pattern, text):
        raise SystemExit("build_submission_archive.py: could not find FROZEN_CROSS_ENCODER_ALPHA default to patch for the baseline variant -- config.py's shape changed, update this script's pattern.")
    text = re.sub(alpha_pattern, alpha_replacement, text)

    staged_config_path.write_text(text, encoding="utf-8")


def _zip_directory(staging_dir: Path, output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(staging_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging_dir))


def build(variant: str, output_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix=f"submission-{variant}-") as tmp:
        staging_dir = Path(tmp) / "staged"
        staging_dir.mkdir()
        print(f"Staging allowlisted files for variant '{variant}'...")
        _stage_allowlist(staging_dir)

        if variant == "cross-encoder":
            model_src = ROOT / CROSS_ENCODER_MODEL_DIR
            if not model_src.is_dir() or not (model_src / "manifest.json").is_file():
                raise SystemExit(
                    f"Packaged checkpoint not found at {model_src} -- run "
                    "scripts/prepare_cross_encoder_artifact.py first."
                )
            model_dst = staging_dir / CROSS_ENCODER_MODEL_DIR
            model_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(model_src, model_dst)
            (staging_dir / "SUBMISSION_VARIANT.txt").write_text(
                "cross-encoder -- frozen TinyBERT cross-encoder reranking enabled by default "
                "(alpha=0.30, K=100), local-files-only, zero AGENT_SHOPPER_* env vars required.\n",
                encoding="utf-8",
            )
        elif variant == "baseline":
            _patch_baseline_config(staging_dir / "agent_shopper" / "config.py")
            (staging_dir / "SUBMISSION_VARIANT.txt").write_text(
                "baseline -- cross-encoder reranking disabled by default (rollback artefact); "
                "reproduces the 0.5674 heuristic-only TechnicalScore.\n",
                encoding="utf-8",
            )
        else:
            raise SystemExit(f"Unknown variant: {variant}")

        print(f"Zipping to {output_path}...")
        _zip_directory(staging_dir, output_path)

    size_bytes = output_path.stat().st_size
    print(f"\nBuilt {output_path}")
    print(f"Exact size: {size_bytes:,} bytes ({size_bytes / (1024 * 1024):.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["cross-encoder", "baseline"], default="cross-encoder")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = Path(args.output).resolve() if args.output else ROOT / f"submission-{args.variant}.zip"
    build(args.variant, output_path)


if __name__ == "__main__":
    main()
