> Relocated from README.md's "Deployment readiness" section to keep the
> top-level README readable. This is the full verification trail for the
> shipped frozen cross-encoder reranker: model identity, offline packaging,
> stability/latency measurements, the resolved organizer policy questions,
> and submission archive packaging. See [README.md](../README.md) for the
> summarized version.

# Deployment Readiness — Full Verification Log

**Validated substitute model identity.** `cross-encoder/ms-marco-TinyBERT-L-6`
@ revision `defbb7d2405cfb2a0f9db418cd8a377c97469552` (base architecture
`nreimers/TinyBERT_L-6_H-768_v2`, MS MARCO Passage Ranking fine-tune,
Apache-2.0 licensed). This is the exact checkpoint every validated number in
this README's cross-encoder entry was measured with, not an inference from
the originally-intended model's name.

**Why the originally intended model was rejected.** `cross-encoder/ms-marco-MiniLM-L-6-v2`
reliably crashes this development machine's process with SIGBUS during
inference — an OS-level signal, not a catchable Python exception, so no
in-process fallback design can save it (root-caused: same-shaped random-weight
BERT and this project's own `all-MiniLM-L6-v2` dense encoder both work,
`ms-marco-MiniLM-L-4-v2` also crashes, `ms-marco-TinyBERT-L-6` and
`ms-marco-electra-base` don't — see the earlier "What we tried" entry for the
full trace). `cross-encoder/ms-marco-MiniLM-L-6-v2` does not appear anywhere
in `agent_shopper/config.py` as an active or fallback default.

**Frozen, not fine-tuned.** No gradient updates, no optimizer, no LoRA/
adapter/projection-head training were performed by this project on this or
any other checkpoint. `FrozenCrossEncoderScorer._ensure_loaded` calls
`.eval()` and sets every parameter's `requires_grad_(False)`; `.score()`
runs inference under `torch.inference_mode()` — enforced by
`tests/agent_shopper/test_cross_encoder_reranker.py`'s
`FrozenCrossEncoderScorerFrozenContractTest`, which asserts this against a
real `torch.nn.Module` without downloading the real checkpoint.

**Model size.** 256.3 MB (267,839,500-byte `model.safetensors` plus five
small tokenizer/config files, 268,784,478 bytes total) — checksummed per-file
(SHA-256) and by an overall deterministic tree hash in
`models/cross_encoder/ms-marco-TinyBERT-L-6/manifest.json`. Note:
`model.safetensors` alone (255 MB) is over GitHub's own 100 MB hard
per-file push-block limit — see "Packaging and remaining blockers" below.

**Offline packaging method.** `scripts/prepare_cross_encoder_artifact.py`
copies real file bytes (never symlinks into any developer's Hugging Face
cache) from the local cache snapshot at the pinned revision, offline
(`local_files_only=True`, refuses to download), into
`models/cross_encoder/ms-marco-TinyBERT-L-6/` — a path resolved in
`agent_shopper/config.py` via `Path(__file__).resolve().parent.parent`,
i.e. relative to the repository/module location, never the current working
directory. `agent_shopper/cross_encoder_reranker.py`'s `_ensure_loaded` sets
`HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` itself (via `setdefault`, never
overriding a value already present) whenever `local_files_only=True` — so a
judge's run needs **zero** `AGENT_SHOPPER_*` or `HF_*` environment variables
to load the exact validated checkpoint fully offline. Reproduce with:

```bash
python3 scripts/prepare_cross_encoder_artifact.py
```

**Stability.** `scripts/smoke_test_cross_encoder_subprocess.py` isolates
every real model load/score call in a child process (a native crash can't be
caught by ordinary Python exception handling, so the parent never trusts
in-process error handling alone). Against the packaged checkpoint: 5/5
independent cold-load subprocesses clean (load time 0.14–0.20s each, zero
signal terminations), 300 warm scoring batches (100 repeats × sizes 20/50/100)
all clean, and candidate-order invariance (original vs. reversed order) PASS
at all three sizes.

**Fail-open circuit breaker.** A process-scoped safety net on top of the
per-turn try/except in `FrozenCrossEncoderReranker.rerank` (which alone only
covers a *raised* exception, not a hang). Every scoring attempt is bounded
by `AGENT_SHOPPER_CROSS_ENCODER_TIMEOUT_SECONDS` (default 20s — roughly
4-5x the worst real per-turn latency measured below); any failure or timeout
trips a module-level breaker that keeps every later turn of every session in
that process on the heuristic path without retrying, rather than repeating
the same failed attempt every turn. Deliberately process-scoped, not
session-scoped like the existing LLM reranker's `state.llm_disabled` — a
local checkpoint failure is almost certainly a persistent environment issue,
not a transient blip worth retrying per-session. Fault-injected against a
nonexistent checkpoint path on the real pipeline: the breaker tripped on
turn 1 (11.56s, well under the timeout), turns 2-3 dropped to 0.6s/0.46s
(no repeated attempts), and the session completed normally throughout on
the heuristic path — the exact 0.5674-reproducing floor, not a degraded or
inconsistent result. The successful path is unaffected: re-run against the
real, working checkpoint, every turn still shows `last_used_cross_encoder=
True` with `alpha=0.30`, `depth=100`, no failure reason, and the breaker
stays closed.

**Cold-load and warm inference latency.** Measured on this machine (Apple
M1, 8 cores, 8 GB RAM, CPU only — no CUDA/MPS available):

| Stage | Time |
|---|---|
| `import agent_shopper.cross_encoder_reranker` | 2.04s |
| `import torch` | 1.60s |
| `import sentence_transformers` | 3.93s |
| Cold model load (`CrossEncoder(...)` + `.eval()` + `requires_grad_(False)`) | 0.23s |
| First inference call (5 candidates, includes lazy load) | 0.98s |
| Warm inference, 5 candidates | 0.16s |
| Peak RSS during load + 100-candidate score | ≈644 MB |

Warm per-turn scoring latency by candidate-batch size (100 repeats each, own
synthetic fixture data, steady-state after model warm-up):

| Candidates | Mean | p50 | p95 | Max |
|---|---|---|---|---|
| 20 | 0.285s | 0.274s | 0.391s | 0.453s |
| 50 | 0.815s | 0.812s | 1.099s | 1.514s |
| 100 | 1.486s | 1.428s | 1.820s | 3.036s |

For comparison, the original per-turn latency measured against real,
diverse session text (`scripts/replay_cross_encoder_offline.py`'s Phase 7
run, up to 110 real candidates per union): mean 4.221s, p50 4.284s, p95
4.535s — higher than the synthetic-batch numbers above, likely because real
per-turn text varies every call (no benefit from repeated-identical-batch
warm paths) — reported as the more representative number for session
overhead below.

10-turn session overhead (using the 26.2% eligible-turn proportion Phase 7
measured — 259 of 989 scored turns had the target reachable in the K=100
union):

- Best case (0 eligible turns): +0s
- Typical case (~2.6 eligible turns): +~11s
- Worst case (10/10 eligible turns): +~42–45s

**Full-evaluation runtime and 800-session projection.** The complete
200-session public-set evaluation, run against the extracted submission
archive itself (not the working tree — see below), with the cross-encoder
enabled by default and zero cross-encoder-specific environment variables:
**4948.1s (≈82.5 minutes)**. Linear ×4 projection for the private 800-session
set: **≈19,792s (≈5.5 hours)**, sequential, single-process — a straight-line
extrapolation only; no judging-side parallelism is imposed by the organizer
(see below), so this is a reasonable estimate of real wall-clock time, not
a bound against some external limit. The heuristic-only rollback path is
much faster: 693.2s (≈11.6 minutes) for 200 sessions, ≈46 minutes projected
for 800.

**Re-measured against the corrected archive**: a full 200-session run against
the actual extracted, dense-model-fixed submission archive (zero
`AGENT_SHOPPER_*`/`HF_*` environment variables, `HF_HOME` pointed at a
freshly-created empty directory) reproduced the headline metrics **bit-exactly**
-- TechnicalScore 0.598867, HitRate@10 0.705, MRR 0.439224, MTTC 5.27, all
identical to the figures already cited throughout this README -- confirming
the packaging fix changed nothing about actual eval outcomes, only how the
model gets loaded. Wall-clock time for that specific run was 6892s (≈114.9
minutes), well above the 82.5-minute figure above; this was measured while
this machine was also running unrelated concurrent work (a separate
diagnostic script, other background tasks) and reflects CPU contention, not
a real regression in per-turn latency -- the 82.5-minute figure from an
uncontended run remains the more representative estimate for the
runtime/800-session-projection numbers above.

**Timeout and size-limit status — resolved.** Originally flagged as three
open organizer questions; all three are now answered by
`docs/final_evaluation_faq.md` (published by the organizer after this
project's initial fork, pulled in during this pass — see its "Hardware,
Runtime, and Timeouts" and "Data, Catalog, and Derived Artifacts" sections
for the full text) and the updated `docs/submission_rules.md`:

- **Timeout:** none. "There is no standardized organizer-provided CPU, RAM,
  GPU, startup-time, or per-response limit because teams run the final
  evaluation in their own environments... the current evaluator does not
  impose a separate explicit per-response timeout." The 82.5-minute/5.5-hour
  figures above are informational, not a risk against an external cap.
- **Archive/package size:** "There is currently no track-specific
  package-size limit." GitHub's own 100 MB hard per-file push-block limit
  (independent of anything organizer-specific) still applies if pushing
  through the normal git protocol — `model.safetensors` (255 MB) needed Git
  LFS for that reason, not an organizer size rule.
- **Checkpoint bundling:** explicitly allowed — "legally usable pretrained
  embedding, reranking, and language models" and "precomputed local
  artifacts" are named as permitted offline preprocessing. The organizer's
  stated *preference*, however, is "documented and reproducible download
  instructions **rather than** committed directly to the repository" for
  large assets — the Git-LFS commit already made for
  `models/cross_encoder/ms-marco-TinyBERT-L-6/` isn't forbidden by this, but
  it also isn't the pattern the organizer asked for; worth reconsidering a
  download-on-setup approach (`scripts/prepare_cross_encoder_artifact.py`
  already does the offline-cache-to-local-directory half of this — it would
  need a download step added, or the checkpoint published as a release
  asset instead of an LFS blob) if this comes up again.

**The bigger revision this resolves:** evaluation is not organizer-hosted or
network-disabled at all — "teams will run the unmodified official evaluator
in their own environments," network access and external APIs are allowed
there, and "an offline fallback is not mandatory." The offline-packaging,
circuit-breaker, and dense-route hardening work in this README remain
genuinely good engineering (a team's own network can still be flaky
mid-run, and failing fast/gracefully instead of hanging or crashing is
strictly better either way) — but none of it was mitigating an organizer-
imposed constraint, because that constraint doesn't exist. Treat every
"protects against a network-disabled judging environment" framing below as
"protects against the team's own environment having a bad network day,"
which is a real but much lower-stakes risk than originally assumed.

**Packaging and remaining blockers.** Two submission archives were built
via `scripts/build_submission_archive.py` from an explicit file allowlist
(not "everything except X"):

| Archive | Size | Default configuration |
|---|---|---|
| `submission-cross-encoder.zip` | 332,614,380 bytes (317.21 MB) | Cross-encoder enabled, α=0.30, K=100, packaged TinyBERT checkpoint included, plus the packaged dense-route checkpoint (below) |
| `submission-baseline.zip` | 83,601,164 bytes (79.73 MB) | Cross-encoder disabled in the staged config copy — rollback artefact, reproduces the 0.5674 heuristic baseline; still includes the packaged dense-route checkpoint, since the dense route is unrelated to the cross-encoder toggle |

**A real bug this verification step caught:** the allowlist originally
didn't include `models/dense/` at all -- once `DENSE_MODEL_NAME` was
repointed at that packaged local path (see the dense-route entry in "What
we tried"), neither archive actually contained it, so a judge's run would
have silently lost the entire dense retrieval route (`DenseIndex` degrades
gracefully on a missing model rather than crashing -- so this would **not**
have surfaced as an error, just a quiet drop in Hit Rate@10 versus every
number documented in this README). Caught by this same extract-and-verify
step, not inferred -- fixed by adding an unconditional dense-checkpoint
copy to `build_submission_archive.py` for both variants, then re-verified
by extracting the rebuilt archives into a fresh directory and confirming
`DenseIndex` loads and returns real search results from each extracted
copy with zero environment variables and no dev-cache reachable.

Both archives are independently verified by **extracting them into a fresh
temporary directory and running every check exclusively against the
extracted copy** (never the working tree): the full 200-session evaluation
for both variants, a dedicated zero-environment-variable, network-blocked,
real-Hugging-Face-cache-independent-for-its-own-checkpoint diagnostic run
(see "Whether the cross-encoder is submission-default" below), and now also
the dense-route load-and-search check above. The two ZIP archives
themselves are rebuildable verification artefacts and were never committed
(gitignored, per `scripts/build_submission_archive.py`'s own docstring).
Both underlying checkpoints (`models/cross_encoder/ms-marco-TinyBERT-L-6/`
via Git LFS, `models/dense/all-MiniLM-L6-v2/` as a plain file) *are*
committed -- see "resolved" above for the organizer's stated preference for
a download-based approach instead, which the cross-encoder checkpoint
doesn't currently follow (the dense checkpoint sidesteps this by not
needing LFS at all).

**A related finding, since fixed and, as of this pass, fully closed (see
the dense-route entry in "What we tried" for the full story):** the
pre-existing dense-retrieval route (`agent_shopper/dense_index.py`)
participates in every turn's retrieval and originally had no
`local_files_only` support and no error handling at all around model
loading — worse than "would attempt a real network call," a load failure or
a slow network-disabled retry loop (observed directly during this project's
own offline-verification testing) would propagate uncaught through
`Agent()` construction, which is called once and reused for every session —
a whole-submission risk, not a per-turn one. First fixed to mirror the
cross-encoder's own offline-enforcement and graceful-degradation pattern
exactly (`config.DENSE_MODEL_LOCAL_FILES_ONLY`, default on), which made a
load failure fast and contained instead of a hang or crash but still left
the model itself unpackaged. That remaining gap is now closed the same way
the cross-encoder's was: `scripts/prepare_dense_model_artifact.py` packages
`sentence-transformers/all-MiniLM-L6-v2` @ revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` into a self-contained,
checksummed local artefact at `models/dense/all-MiniLM-L6-v2/`
(`config.DENSE_MODEL_NAME` now defaults to this packaged path), and it's
been verified to load with zero `AGENT_SHOPPER_*`/`HF_*` environment
variables and, separately, with `HF_HOME` pointed at a freshly-created empty
directory — no developer cache reachable at all — followed by a real
`DenseIndex` build-and-search call against the loaded model to confirm the
whole pipeline works end to end, not just the load call in isolation. At
87.3 MB, this checkpoint is under GitHub's 100 MB per-file push-block limit,
so — unlike the cross-encoder checkpoint — it's committed as a plain file
with no Git LFS dependency at all (`.gitattributes`'s LFS rule is scoped to
`models/cross_encoder/` specifically for this reason).

**Submission default vs. experimental.** The frozen cross-encoder is the
**submission default** (`AGENT_SHOPPER_FROZEN_CROSS_ENCODER` defaults to
enabled; opt out with `AGENT_SHOPPER_FROZEN_CROSS_ENCODER=0` to reproduce
the 0.5674 heuristic-only baseline exactly — confirmed via a fresh full
200-session run of `submission-baseline.zip`, bit-identical to the
`post-revert-confirm` reference to 6 decimal places on every headline
metric). All three organizer questions that previously withheld a
"deployment-ready" claim are now resolved (see "resolved" above) and don't
block it: no timeout, no package-size limit, and checkpoint bundling is
allowed. What's honestly still open, so this isn't oversold as fully
certified either: the organizer's stated *preference* for download-based
large-asset delivery over a committed LFS blob isn't followed yet for the
cross-encoder checkpoint (the dense checkpoint sidesteps this entirely by
not needing LFS in the first place, per above). The dense route's own
embedding-*cache* rebuild cost (not the model — see "What we tried") is a
separate, smaller, still-open item: see "Limitations."

**Reproduction command** (exact validated configuration, zero cross-encoder-
specific environment variables required — shown here explicitly for
clarity, not because they're needed):

```bash
python3 scripts/prepare_cross_encoder_artifact.py   # once, to (re)build models/cross_encoder/ms-marco-TinyBERT-L-6/
python3 scripts/run_local_eval.py --label "packaged-offline-cross-encoder-alpha-030"
```

Expected result (reproduced bit-exactly against the extracted submission
archive in a zero-`AGENT_SHOPPER_*`-env-var, network-blocked run): HitRate@10
0.705, MRR 0.439224, MTTC 5.27, TechnicalScore 0.598867, 6 miss→hit / 0
hit→miss versus the heuristic baseline, ≤10 recommendations every turn.
