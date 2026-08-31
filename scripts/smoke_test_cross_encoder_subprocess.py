"""Phase 5 native-crash isolation for the packaged frozen cross-encoder.

A process-level native crash (the SIGBUS that the originally-specified
cross-encoder/ms-marco-MiniLM-L-6-v2 checkpoint produces on this development
machine -- see agent_shopper/cross_encoder_reranker.py's module docstring and
README.md's "What we tried") cannot be caught by ordinary Python exception
handling. This script's parent process launches every real model load/score
call in a CHILD process instead, so a crash there can never take down the
parent verifier -- it only ever shows up as a negative `returncode` (the
terminating signal number), which the parent classifies and reports.

This is a diagnostic/verification tool, run manually, not part of the unit
test suite (tests/agent_shopper/test_cross_encoder_reranker.py has a small,
fast unit test for _classify()'s pure logic only -- it never spawns a real
subprocess or loads a real model).

Usage (parent -- runs the full suite against the packaged checkpoint):
    python3 scripts/smoke_test_cross_encoder_subprocess.py [--model-path PATH]

Usage (child -- invoked internally by the parent; only run this manually as
a diagnostic if you know what you're doing):
    python3 scripts/smoke_test_cross_encoder_subprocess.py --child cold-load --model-path PATH
    python3 scripts/smoke_test_cross_encoder_subprocess.py --child warm-batches --model-path PATH --sizes 20,50,100 --repeats 100
    python3 scripts/smoke_test_cross_encoder_subprocess.py --child order-invariance --model-path PATH --sizes 20,50,100
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run directly (not via -m), matches this repo's other scripts

CHILD_TIMEOUT_SECONDS = 60
# 100 repeats x 3 sizes of real inference is much slower than one cold load
# (README.md documents ~4.2s mean per ~110-candidate batch on this machine) --
# the warm-batches child needs a much larger budget than cold-load/order-
# invariance do.
WARM_BATCHES_TIMEOUT_SECONDS = 1800


def _make_fake_products(n: int):
    """Deterministic, catalog-free fixture products -- real inference, no
    dependency on data/catalog.jsonl. Diverse enough text that scores aren't
    all identical (which would trivially "pass" an order-invariance check)."""
    from agent_shopper.models import Product

    materials = ["cotton", "leather", "wool", "denim", "silk", "linen", "polyester", "canvas"]
    colors = ["black", "blue", "red", "green", "white", "brown", "grey", "navy"]
    products = []
    for i in range(n):
        products.append(Product(
            parent_asin=f"SMOKE{i:04d}",
            title=f"{colors[i % len(colors)].capitalize()} {materials[i % len(materials)]} item {i}",
            features=[f"{materials[i % len(materials)]} construction", "durable"],
            description=[f"A well-made {materials[i % len(materials)]} product, style {i}."],
            price=19.99 + i,
            categories=["Clothing", "Accessories"],
            details={"Material": materials[i % len(materials)]},
            average_rating=4.0,
            rating_number=10 + i,
            store="SmokeTestBrand",
        ))
    return products


# --- Child-side entry points -------------------------------------------------

def _child_cold_load(model_path: str) -> dict:
    from agent_shopper.cross_encoder_reranker import FrozenCrossEncoderScorer

    scorer = FrozenCrossEncoderScorer(model_name_or_path=model_path, local_files_only=True)
    products = _make_fake_products(10)
    start = time.monotonic()
    scores = scorer.score("comfortable everyday item", products)
    elapsed = time.monotonic() - start
    return {"ok": True, "n_scored": len(scores), "load_seconds": scorer.last_load_seconds, "score_seconds": elapsed}


def _child_warm_batches(model_path: str, sizes: list[int], repeats: int) -> dict:
    from agent_shopper.cross_encoder_reranker import FrozenCrossEncoderScorer

    scorer = FrozenCrossEncoderScorer(model_name_or_path=model_path, local_files_only=True)
    # Prime the load once (cold), excluded from the "warm" latency numbers.
    scorer.score("warm-up query", _make_fake_products(2))

    results = {}
    for size in sizes:
        products = _make_fake_products(size)
        latencies = []
        for _ in range(repeats):
            start = time.monotonic()
            scorer.score("comfortable everyday item under $50", products)
            latencies.append(time.monotonic() - start)
        sorted_lat = sorted(latencies)
        pct = lambda p: sorted_lat[min(len(sorted_lat) - 1, int(p * len(sorted_lat)))]  # noqa: E731
        results[str(size)] = {
            "n": len(latencies), "mean": statistics.fmean(latencies),
            "p50": pct(0.5), "p95": pct(0.95), "max": sorted_lat[-1],
        }
    return {"ok": True, "by_size": results}


def _child_order_invariance(model_path: str, sizes: list[int]) -> dict:
    from agent_shopper.cross_encoder_reranker import FrozenCrossEncoderScorer

    scorer = FrozenCrossEncoderScorer(model_name_or_path=model_path, local_files_only=True)
    tol = 1e-4
    per_size = {}
    all_ok = True
    for size in sizes:
        products = _make_fake_products(size)
        reversed_products = list(reversed(products))
        forward = scorer.score("comfortable everyday item under $50", products)
        backward = scorer.score("comfortable everyday item under $50", reversed_products)
        mismatches = [
            asin for asin in forward
            if abs(forward[asin] - backward.get(asin, float("nan"))) > tol
        ]
        ok = not mismatches and set(forward) == set(backward)
        all_ok = all_ok and ok
        per_size[str(size)] = {"ok": ok, "mismatches": mismatches}
    return {"ok": all_ok, "by_size": per_size}


def _run_child() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", required=True, choices=["cold-load", "warm-batches", "order-invariance"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sizes", default="20,50,100")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sizes = [int(s) for s in args.sizes.split(",") if s]

    try:
        if args.child == "cold-load":
            result = _child_cold_load(args.model_path)
        elif args.child == "warm-batches":
            result = _child_warm_batches(args.model_path, sizes, args.repeats)
        else:
            result = _child_order_invariance(args.model_path, sizes)
    except Exception as exc:  # noqa: BLE001 -- report, don't mask; parent classifies this as an "ordinary exception"
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}))
        sys.exit(1)

    print(json.dumps(result))
    sys.exit(0)


# --- Parent-side orchestration -----------------------------------------------

def _classify(returncode: int | None, timed_out: bool) -> str:
    """Pure classification logic, unit-tested directly (no real subprocess)
    in tests/agent_shopper/test_cross_encoder_reranker.py."""
    if timed_out:
        return "timeout"
    if returncode is None:
        return "unknown"
    if returncode == 0:
        return "clean_success"
    if returncode < 0:
        try:
            return f"signal:{signal.Signals(-returncode).name}"
        except ValueError:
            return f"signal:{-returncode}"
    return "ordinary_exception"


def _run_child_subprocess(child_mode: str, model_path: str, extra_args: list[str] | None = None, timeout_seconds: int = CHILD_TIMEOUT_SECONDS) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", child_mode, "--model-path", model_path]
    cmd += extra_args or []
    env = {k: v for k, v in os.environ.items() if not k.startswith("AGENT_SHOPPER_")}
    timed_out = False
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout_seconds)
        returncode = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        stdout, stderr = (exc.stdout or ""), (exc.stderr or "")

    classification = _classify(returncode, timed_out)
    payload = None
    if not timed_out and returncode is not None and returncode >= 0 and stdout.strip():
        try:
            payload = json.loads(stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            payload = None
    return {
        "classification": classification, "returncode": returncode,
        "stderr_tail": stderr[-500:] if stderr else "", "payload": payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--child", choices=["cold-load", "warm-batches", "order-invariance"], default=None,
                         help=argparse.SUPPRESS)  # internal, used for the child re-invocation
    parser.add_argument("--model-path", default=str(ROOT / "models" / "cross_encoder" / "ms-marco-TinyBERT-L-6"))
    parser.add_argument("--sizes", default="20,50,100")
    parser.add_argument("--repeats", type=int, default=100)
    args, _unknown = parser.parse_known_args()

    if args.child is not None:
        _run_child()
        return

    print(f"Model path: {args.model_path}")
    print("=" * 100)

    print("\n5 independent cold-load subprocesses:")
    cold_results = []
    for i in range(5):
        result = _run_child_subprocess("cold-load", args.model_path)
        cold_results.append(result)
        load_s = (result["payload"] or {}).get("load_seconds") if result["payload"] else None
        print(f"  run {i + 1}: {result['classification']:20s} returncode={result['returncode']} load_seconds={load_s}")
        if result["classification"] not in ("clean_success",):
            print(f"    stderr tail: {result['stderr_tail']}")

    print(f"\n1 child running {args.repeats} repeated warm scoring batches at sizes {args.sizes} (up to {WARM_BATCHES_TIMEOUT_SECONDS}s budget)...")
    warm_result = _run_child_subprocess(
        "warm-batches", args.model_path, ["--sizes", args.sizes, "--repeats", str(args.repeats)],
        timeout_seconds=WARM_BATCHES_TIMEOUT_SECONDS,
    )
    print(f"  classification: {warm_result['classification']}")
    if warm_result["payload"]:
        for size, stats in warm_result["payload"]["by_size"].items():
            print(f"    size={size:>4s}  n={stats['n']:3d}  mean={stats['mean']:.3f}s  p50={stats['p50']:.3f}s  "
                  f"p95={stats['p95']:.3f}s  max={stats['max']:.3f}s")
    elif warm_result["classification"] != "clean_success":
        print(f"    stderr tail: {warm_result['stderr_tail']}")

    print("\nOrder invariance (original vs. reversed candidate order) at sizes 20/50/100:")
    order_result = _run_child_subprocess("order-invariance", args.model_path, ["--sizes", args.sizes])
    print(f"  classification: {order_result['classification']}")
    if order_result["payload"]:
        for size, stats in order_result["payload"]["by_size"].items():
            print(f"    size={size:>4s}  {'PASS' if stats['ok'] else 'FAIL (' + str(stats['mismatches']) + ')'}")

    print("\n" + "=" * 100)
    all_cold_clean = all(r["classification"] == "clean_success" for r in cold_results)
    warm_clean = warm_result["classification"] == "clean_success"
    order_clean = order_result["classification"] == "clean_success" and bool(
        order_result["payload"] and all(s["ok"] for s in order_result["payload"]["by_size"].values())
    )
    signal_terminations = [r["classification"] for r in cold_results if r["classification"].startswith("signal:")]
    if warm_result["classification"].startswith("signal:"):
        signal_terminations.append(warm_result["classification"])
    if order_result["classification"].startswith("signal:"):
        signal_terminations.append(order_result["classification"])

    if signal_terminations:
        print(f"RESULT: FAIL -- signal termination(s) detected: {signal_terminations}")
    elif all_cold_clean and warm_clean and order_clean:
        print("RESULT: PASS -- all cold-load, warm-scoring, and order-invariance checks succeeded, zero signal terminations.")
    else:
        print("RESULT: FAIL -- see classifications above (ordinary exception or timeout, not a signal).")


if __name__ == "__main__":
    main()
