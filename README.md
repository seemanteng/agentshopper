# Agent Shopper

A conversational shopping search & recommendation agent built for the
**TechJam 2026 "Shopping Copilot" challenge**. Agent Shopper finds one
target product out of a 50,000-item Amazon `Clothing_Shoes_and_Jewelry`
catalog through up to 10 turns of natural-language dialogue, combining
dual-track intent routing, multi-route retrieval, a slot-accumulation +
intent-override dialog state machine, and adaptive, context-distilled
orchestration.

This repository is a fork of the official
[`techjam-conversational-search`](https://github.com/TechJam2026/techjam-conversational-search)
participant kit. The competition harness (`evaluator/`, `data/`, `docs/`)
is unmodified; the full agent implementation lives in the new
[`agent_shopper/`](agent_shopper/) package, wired in through
[`starter/agent.py`](starter/agent.py).

## How this addresses the problem statement

The brief calls for four pillars; each maps to a small set of modules:

| Pillar | What it means here | Modules |
|---|---|---|
| **I. Intent Routing & Hybrid Pipeline** | Every turn is scored on two separate signals — how decisive the shopper's *language* is, and how *specified* the request already is from accumulated slots — combined with hysteresis so the track doesn't flip on a marginal score change. Buying hard-gates retrieval to a structured category/attribute/price filter once enough slots are known; a slot stated with hard-requirement language ("no more than $50", "must be leather") is enforced as an exact filter immediately, even on the browsing track. Three routes (keyword/BM25, structured category filter, TF-IDF vector) are fused by weighted Reciprocal Rank Fusion, then reranked by a pluggable heuristic-or-LLM stage that treats catalog text as untrusted data. | [`intent.py`](agent_shopper/intent.py), [`bm25_index.py`](agent_shopper/bm25_index.py), [`tfidf_index.py`](agent_shopper/tfidf_index.py), [`category_index.py`](agent_shopper/category_index.py), [`retrieval.py`](agent_shopper/retrieval.py), [`reranker.py`](agent_shopper/reranker.py) |
| **II. Multi-Turn Scenario Evolution** | A `SessionState` accumulates slots (category/material/color/size/style/brand/budget/feature/use_case) turn by turn, distinguishing plain Information Accumulation from an Intent Override (explicit contradiction language, a conflicting category, or a non-overlapping budget) that erases and rewrites the affected slots. A category override only clears style/feature when the new category is in a different department (jewelry/footwear/apparel/accessories/bags) — a same-department swap ("shoes"→"boots") keeps them, and a cleared attribute is reopened for clarification. Slots stated with hard-requirement language are marked as such, so relaxing an over-constrained filter prefers dropping a soft preference first and asks the shopper which requirement to adjust only once nothing soft is left. A letter clothing size and a numeric shoe/waist size are never folded into one OR-accumulated value. When the candidate pool is over-general, retrieval is cut off in favor of an information-gain-selected clarifying question — while still returning a best-effort recommendation the same turn. | [`slots.py`](agent_shopper/slots.py), [`dialog_state.py`](agent_shopper/dialog_state.py), [`dialog_policy.py`](agent_shopper/dialog_policy.py) |
| **III. Dynamic Context Programming** | The harness's `reset()` `user_profile` is distilled once into soft priors (a decisiveness prior, a preference-tag boost, a rating-floor hint) — this *is* the "long-term profile" signal, since each session is an isolated single user. Session slots and shown-item history are distilled every turn into a bounded, token-cheap representation, which also drives a soft repeat-penalty against re-surfacing an already-shown, unconverted item. A small decision-table orchestrator re-selects routing/reranking/clarification strategy at runtime based on turn budget, pool size, and progress — skipping the LLM call when the turn's output is a clarifying question anyway or turns are running out, stopping clarification only once no untried attribute is left worth asking about, and relaxing an over-constrained filter when stuck. | [`context.py`](agent_shopper/context.py), [`orchestrator.py`](agent_shopper/orchestrator.py) |
| **IV. Evaluation** | Scored against the official `evaluator/local_evaluator.py` (Hit Rate@10, MRR, MTTC, TechnicalScore) on the 200-session public dev set. Every tunable in `config.py` was validated this way before being locked in — see "What we tried" below for the specific before/after numbers. | [`scripts/run_local_eval.py`](scripts/run_local_eval.py) |

The full per-turn algorithm lives in `dialog_policy.process_turn`; `agent.py`
is intentionally a thin `reset()`/`respond()` shim over it.

### LLM usage: pluggable and optional

A paid LLM is **not required**. The pipeline runs entirely free/local by
default — hand-rolled BM25, TF-IDF/cosine, rule-based intent & slot
extraction, and a weighted heuristic reranker. If `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY` is set, the semantic-ranking stage upgrades to a real
LLM judge call (`reranker.LLMReranker`), with automatic fallback to the
heuristic path on any failure, a circuit breaker after repeated failures,
and the last turn always using the fast local path regardless. Force the
free path explicitly with `AGENT_SHOPPER_FORCE_HEURISTIC=1`.

## Setup and installation

```bash
git clone <this-repo-url> agent-shopper
cd agent-shopper
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Catalog is distributed separately (not committed) -- download + verify:
curl -L -o data/catalog.jsonl.gz \
  https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz
curl -L -o data/SHA256SUMS \
  https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/SHA256SUMS
(cd data && shasum -a 256 -c <(grep catalog.jsonl.gz SHA256SUMS))
gunzip -k data/catalog.jsonl.gz   # -> data/catalog.jsonl, 50,000 rows
```

Optional, only if you want the LLM-upgraded reranker locally:

```bash
cp .env.example .env   # if present; otherwise just export directly
export OPENAI_API_KEY=sk-...        # or ANTHROPIC_API_KEY=...
```

## Steps to reproduce results

```bash
# Unit tests (agent_shopper's own tests + the kit's own evaluator tests)
python -m unittest discover -p "test_*.py"

# Full local evaluation against the 200-session public dev set
python3 scripts/run_local_eval.py --label "my-run"
# Force the free/local path only (fast, no API cost):
AGENT_SHOPPER_FORCE_HEURISTIC=1 python3 scripts/run_local_eval.py --label "heuristic-only"
```

`scripts/run_local_eval.py` wraps the official `evaluator.local_evaluator`,
prints Hit Rate@10 / MRR / MTTC / TechnicalScore overall and broken down by
`scenario_type` (buying/browsing/intent_override/boundary) and by
`difficulty_bucket`/`category_bucket`, and appends each run to a local
`eval_runs.jsonl` for comparing tuning iterations.

### Results on the public dev set (200 sessions)

| | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| Stock weak-BM25 starter agent (organizer baseline) | 0.125 | 0.068 | 9.81 | 0.107 |
| Agent Shopper, pre-dense-route (heuristic-only, no LLM) | 0.650 | 0.400 | 5.75 | 0.550 |
| Agent Shopper, + dense route (heuristic-only, no LLM) | 0.670 | 0.408 | 5.70 | 0.563 |
| **Agent Shopper (heuristic-only, no LLM, + `MATERIALS` fix)** | 0.675 | 0.410 | 5.66 | **0.567** |

That's a ~5.4x Hit Rate@10, ~6.0x MRR, and ~42% turn-efficiency improvement
over the baseline, without spending a single LLM call. Per-scenario
breakdown from the same run:

| scenario_type | n | Hit Rate@10 | MRR | MTTC |
|---|---|---|---|---|
| buying | 80 | 0.713 | 0.402 | 5.13 |
| browsing | 80 | 0.688 | 0.440 | 5.44 |
| intent_override | 30 | 0.533 | 0.368 | 7.80 |
| boundary | 10 | 0.700 | 0.365 | 5.30 |

`intent_override` still trails the other three tracks — expected, since
it's the deliberately adversarial track (a preference reversal mid-session)
and it's a much smaller sample (30 sessions, so a couple of misses move the
rate a lot) — but it's moved the most of any track across two separate
fixes: the dense route (HitRate@10 0.367→0.500, MRR 0.266→0.349, MTTC
8.73→8.03, a clean +4/-0 session-level hit-flip) and, on top of that, the
`MATERIALS` extraction-gap fix just above (HitRate@10 0.500→0.533, MRR
0.349→0.368, MTTC 8.03→7.80, a clean +1/-0 flip with buying/browsing/
boundary bit-identical to 6 decimal places) — see "What we tried" for the
full story on both, and why three other, cheaper attempts at this same
problem were tried and reverted first. Re-run `scripts/run_local_eval.py`
to reproduce these numbers and
see the current numbers as the code evolves — they're logged to
`eval_runs.jsonl`, not hand-copied, so this table can drift stale but the
logged history never does.

The LLM-upgraded reranker path (`OPENAI_API_KEY`/`ANTHROPIC_API_KEY` set,
`AGENT_SHOPPER_FORCE_HEURISTIC` unset) hasn't been benchmarked here since it
costs real API calls across 200 sessions x up to 10 turns; run it yourself
to compare.

### What we tried (and the eval evidence behind what shipped)

Every behavior in `config.py` was validated by an A/B run of
`scripts/run_local_eval.py` on the public dev set before being locked in as
the single default — no toggles or alternate code paths were kept around
once a comparison was decided. In order of impact:

- **Per-attribute clarify exhaustion** (stop clarifying only once no
  untried, well-scoring attribute is left, instead of a blunt "2
  consecutive unhelpful replies about anything" global cutoff) — the
  single biggest win: TechnicalScore 0.4207→0.4855, HitRate@10 0.48→0.565,
  MRR 0.329→0.369, MTTC 6.90→6.38, improved in *every* scenario bucket
  including `boundary` (0.40→0.60), the case the blunt cutoff was
  ostensibly protecting.
- **Reranker weights learned via k-fold-validated logistic regression**,
  replacing `HEURISTIC_RERANK_WEIGHTS`' 5 hand-picked values with
  coefficients fit by the new `scripts/train_reranker_weights.py`. This is
  the first config change validated by cross-validation rather than a
  single A/B run, specifically because fitting on all 200 sessions and
  evaluating on the same 200 would produce a flattering number that might
  not transfer to the hidden 800-session set — the risk this whole "What we
  tried" section already lives with (`intent_override` alone is n=30). A
  stratified 5-fold CV — fit `LogisticRegression(class_weight="balanced")`
  per fold on that fold's logged candidate features + is-target label, then
  re-run the *actual interactive* evaluator loop (not an offline score) on
  the held-out fold with the fitted weights vs. the hand-tuned default — won
  on **5/5 folds** (TechnicalScore delta +0.028 to +0.128, mean +0.061, net
  +16 hit-turn flips, zero folds net-negative) before being trusted; see
  `reranker_cv_runs.jsonl` for the fold-by-fold numbers. Refit on the full
  200 sessions and shipped as the new default: TechnicalScore
  0.4927→0.5499, HitRate@10 0.575→0.650, MRR 0.376→0.400, MTTC 6.38→5.75,
  improving `buying` and `browsing` (160/200 sessions) on every metric. The
  one real, accepted trade-off: `intent_override`'s MRR got worse
  (0.322→0.266, hit rate flat) — the learned weights favor `rating` roughly
  5x more heavily than any of the other 4 features (fitted values: `bm25`
  1.69, `vector` 1.94, `attr_match` 1.98, `rating` 9.03, `price_fit` 0.26),
  which plausibly interacts worse with `intent_override`'s preference-
  reversal dynamic than the old, flatter hand-tuned weighting did — a
  candidate for a future, `intent_override`-specific look rather than a
  reason to revert a change that won cleanly everywhere else.
- **Soft repeat-penalty on already-shown, unconverted items** (a
  multiplicative rank demotion in the heuristic reranker, never a hard
  exclusion, since MRR/MTTC only score the *first* hit turn) — stacked on
  top of the above: TechnicalScore 0.4855→0.4927, HitRate@10 0.565→0.575,
  MRR 0.369→0.376, MTTC 6.38→6.375. Per-session diff: 3 miss→hit flips (all
  `buying`), 7 rank improvements, 0 rank regressions, against one accepted
  hit→miss loss (`public_0046`, `intent_override`).
- **Running the heuristic reranker (attribute match / rating / price fit)
  on small pools instead of leaving them in raw fused-RRF order** —
  measurement-neutral on this dataset (identical aggregate numbers), but
  free (no LLM cost) and strictly more informed, so it stays on.
- **Per-department clarify-attribute relevance weighting** (favor asking
  about "size" over "color" for footwear, reusing the same department
  table used for category-compatibility) — TechnicalScore 0.4186→0.4207,
  MRR 0.3197→0.3286, HitRate@10 unchanged, no scenario_type regression.
- **Diagnostic (evidence-based) constraint relaxation** — tried replacing
  the static drop-lowest-priority-slot-first relaxation with one that
  test-retrieves each droppable slot and drops whichever restores the most
  candidates. Measurement-neutral: the zero-pool buying-gate case it
  targets is rare enough in the public set that it never changed an
  outcome. Reverted in favor of the simpler static version rather than
  keeping an unused code path.
- **Rank-impact-aware clarify selection** — tried weighting
  `choose_clarify_attribute`'s score by whether the pool's *current top
  contenders* (by pre-rerank fused score) would plausibly be separated by
  an attribute's answer, not just whether it splits the wider pool.
  **Regressed**: TechnicalScore 0.4927→0.4784, 4 hit→miss flips vs. 1
  miss→hit, 7 rank regressions vs. 2 improvements. Likely cause: pre-rerank
  fused score is a noisy proxy for "who's currently winning" — it hasn't
  been through attribute-match/rating/price-fit scoring yet, so the factor
  ended up penalizing attributes the already-validated entropy signal was
  correctly favoring. Reverted.
- **Negative/exclusion constraints** ("no leather", "anything but red")
  and **provisional (capped) recommendations on broad-pool clarify turns**
  were both prototyped and measured; the latter confirmed a real,
  predicted trade-off (TechnicalScore 0.4207→0.3623, HitRate@10 0.48→0.395
  — the official evaluator scores strictly on whether the target ASIN
  appears in that turn's `recommendations`, so shrinking the list on
  exactly the turns where a guess is broadest trades away scored chances
  by design). Both removed rather than kept as disabled code paths.
- **Stripping the evaluator's own fixed dialogue scaffolding** (e.g.
  "Actually, ignore my earlier preference. What I need is: X.", "Those
  options are not quite right yet. Ask me about one specific attribute.")
  out of the BM25/TF-IDF query text before tokenizing, instead of letting it
  dilute every turn's query. Motivated by `scripts/diagnose_intent_override.py`
  (new diagnostic script, kept): 8 of the 12 `intent_override` sessions that
  never recalled their target were recallable *before* the override turn,
  dropping out specifically once this boilerplate entered that turn's
  query. The fix confirmed the diagnosis in the track it targeted
  (`intent_override` HitRate@10 0.367→0.400, MTTC 8.73→8.43) but
  **regressed** `buying` (HitRate@10 0.713→0.663) and `boundary`'s MRR
  (0.440→0.285) enough to net-regress TechnicalScore 0.5499→0.5371 overall.
  Reverted; the diagnostic script stays since it's read-only tooling, not a
  behavior change.
- **Feature-richer reranker retrain**: extended the learned reranker's
  5-feature CV pipeline (see the earlier reranker-weights entry above) with
  two more signals -- `category` (`route_scores["category"]`, real
  retrieval signal the reranker never looked at) and a learned
  `preference_tag` match score, replacing the old fixed
  +0.05-per-tag-hit-capped-at-0.15 boost -- then refit and re-validated with
  the same 5-fold CV. **Regressed**: 2/5 folds won, mean TechnicalScore
  delta -0.0090, net hit flips -2. Root cause: `category` and `attr_match`
  both measure slot match and are largely redundant, so the fit split
  weight between them unstably across folds (`attr_match` 0.20-0.67 vs. the
  shipped 5-feature model's 1.98, `category` 2.39-3.14). `preference_tag`
  alone was stable and consistently positive (0.59-0.93 across all 5
  folds) -- a real signal, just not enough on its own to offset the
  collinearity damage from bundling it with `category`. Reverted in full
  (including the dead-code cleanup of `_preference_tag_boost` ->
  `_preference_tag_score` it would have enabled) rather than keeping the
  restructuring at zero-weight, since that would silently differ from the
  last actually-validated behavior. Worth retrying with `preference_tag`
  alone, without `category`, as a future follow-up.
- **`preference_tag` retried in isolation** (no `category` this time),
  same 5-fold CV. **Still net-negative, still reverted**: 3/5 folds won,
  mean TechnicalScore delta -0.0042, net hit flips -1 -- closer to neutral
  than the bundled attempt (-0.0090) as the collinearity theory predicted,
  but fold 2 alone lost 3 hit-turns and pulled the net negative.
  `preference_tag` was again stable and positive in every fold of both
  runs (0.56-0.93) -- it's real, learnable signal, it just isn't enough on
  its own, on this 200-session dev set, to beat the current model. Reverted
  the same way as the bundled attempt (full revert, not zero-weighted).
  Two clean, disciplined negative results on the same feature now close
  out this line of investigation without more evidence (e.g. the private
  800-session set, which isn't available for local iteration).
- **Override-scoped boilerplate stripping** (`agent_shopper.context`'s
  `_strip_boilerplate`, gated by `config.OVERRIDE_QUERY_STRIP_ENABLED`): a
  second attempt at the blanket boilerplate-stripping idea above, narrowed
  to only strip on turns where `state.has_overridden()` is already True,
  instead of every turn regardless of scenario. Mechanically confirmed
  correct (traced query_text before/after: noise tokens do drop out, e.g.
  `'shoes leather actually ignore earlier preference what'` →
  `'shoes leather'`), but **still a net negative**, just contained entirely
  within `intent_override` sessions this time instead of leaking into
  buying/boundary: 4/30 sessions changed outcome (2 miss→hit at ranks 6 and
  8, 2 hit→miss that had been ranked **1st and 3rd**), netting HitRate@10
  unchanged (11/30 both ways) and MRR down (0.2656→0.2308). 5-fold CV
  confirms: 2 wins/1 loss/2 ties, mean TechnicalScore delta -0.0011, net
  hit flips 0 (`scripts/train_override_query_strip.py`, logged to
  `override_query_strip_cv_runs.jsonl`). Also found along the way: only
  18/30 `intent_override` sessions ever set `state.has_overridden()` at
  all -- `_apply_override` only fires on a value-to-*different*-value
  change, and several of the scripted override `new_value`s (e.g. "Faux
  Fur", "Hand Wash Only", "100% Acrylic") never extracted into a comparable
  slot in the first place, so the override never registered as a
  contradiction. Reverted (`OVERRIDE_QUERY_STRIP_ENABLED` defaults off);
  the untouched extraction gap is a separate, real finding worth its own
  follow-up.
- **Dense semantic retrieval** (`agent_shopper/dense_index.py`, a new
  `dense` route alongside `keyword`/`vector`/`category`): motivated by
  `scripts/diagnose_retrieval.py`'s per-route Recall@10/50/100 breakdown
  (new instrumentation, kept), which showed `keyword` and `vector` are
  highly overlapping and both cap around 15-21% R@100 -- fusing two
  lexical routes that mostly agree adds little, and pointed at a genuine
  vocabulary/paraphrase gap rather than a ranking problem. Two cheaper,
  dependency-free attempts at `intent_override` specifically (both entries
  above) were tried and reverted first. Added a frozen
  `sentence-transformers/all-MiniLM-L6-v2` encoder, embedding
  title/attributes/category/description as four separate fields (not one
  concatenated document -- same reasoning `BM25_FIELD_WEIGHTS` already
  applies) and combined via hand-set `config.DENSE_FIELD_WEIGHTS`, fused
  into the existing RRF pipeline via new `config.ROUTE_WEIGHTS["dense"]`
  entries. **Shipped, clean win, not yet CV-tuned**: overall TechnicalScore
  0.5499→0.5634 (HitRate@10 0.650→0.670, MRR 0.400→0.408, MTTC 5.755→5.70).
  `intent_override` moved the most and cleanest of any track -- HitRate@10
  0.367→0.500, MRR 0.266→0.349, MTTC 8.73→8.03, a session-level +4/-0 hit
  flip (two landing at rank 1) confirmed by direct with/without-dense
  comparison -- while buying/browsing/boundary stayed **exactly** flat (not
  a single session flipped either way). `ROUTE_WEIGHTS`'s dense weight was
  then checked against two hand-reasoned alternatives (`scripts/
  sweep_dense_weights.py`, deliberately a small set rather than a grid --
  re-running against the same 200 public sessions to pick a winner is its
  own overfitting risk): a "dense-up" variant (more weight shifted from
  `vector`) scored worse (0.5565), and a "dense-down" variant scored worse
  still (0.5563) and specifically dropped `intent_override` back to 0.433 --
  confirming the shipped weight is doing the real work, not an accident.
  `DENSE_FIELD_WEIGHTS` (the title/attributes/category/description split)
  remains an unswept hand-set prior. Embeddings are cached to disk (`data/
  .dense_cache_*.npz`, ~245MB for the full 50k catalog) after a ~27-minute
  one-time CPU build; a warm `Agent()` construction after that loads the
  cache in ~4s (vs. ~10s for BM25, ~21s for TF-IDF -- comparable order of
  magnitude, not a new bottleneck) and per-query dense search latency
  benchmarks at ~92ms, in line with the existing bm25 (89ms) and vector
  (142ms) routes, not slower. The 27-minute cost is a one-time catalog
  build, never a per-query cost -- but the 245MB cache file itself is
  currently local-only (gitignored); a fresh clone without it would pay
  that 27 minutes again on first run, which is worth addressing before
  this is judged/deployed anywhere the cache doesn't travel with it.
- **Calibrated override-probability model** (`agent_shopper/override_model.py`,
  `agent_shopper.dialog_policy._override_features`): a small logistic
  regression predicting P(this turn is an intent reversal) from 5 features
  (explicit reversal language, department change, budget conflict,
  attribute-contradiction count, is-first-turn), fit on real per-turn traces
  of the full 200-session public set (`scripts/train_override_model.py`)
  with ground-truth labels from `sample["behavior"]["override"]["turn"]` --
  not the contradiction-language regex re-applied to itself, which would
  have been circular. 5-fold CV: AUC/precision/recall all 1.0 on every fold
  (expected -- `contradiction_language` alone nearly determines this
  scripted dataset's labels; see the fitted weights' comment in
  `config.py`). Computed every turn and threaded through
  `DistilledSession.override_probability`, currently unconsumed pending a
  future confidence-weighted-slots pass -- deliberately NOT used to drive
  any hand-coded intervention directly (see the reverted boilerplate-
  stripping entry above for why a single hard-coded reaction to this
  signal already failed once).
- **LambdaMART reranker** (query-grouped `lightgbm.LGBMRanker` alternative
  to `HeuristicReranker`'s 5-feature linear weighted sum, over a 14-feature
  set including override_probability and its interaction with rating):
  built, then removed after validation, not kept gated off. **5-fold CV:
  DO NOT ADOPT** -- 2 wins/3 losses/0 ties, mean TechnicalScore delta
  -0.0188, net hit flips -3. `intent_override` HitRate@10 was unchanged in
  3/5 folds, worse in one (0.667->0.500), better in one (0.833->1.000) --
  no systematic improvement despite the override-aware features, consistent
  with this project's established pattern of plausible-looking reranker
  features not surviving CV on a dataset this small (~160 training
  sessions/fold, ~24 true override-turn rows total across all folds). Two
  real bugs were caught and fixed *during* validation, before the negative
  result made them moot: (1) LightGBM segfaulting when imported after
  `torch` in the same process on macOS/ARM, fixed by import-ordering torch
  first -- silent enough that the first CV attempt looked like "no crash"
  until traced; (2) a self-referential `unittest.mock.patch` in the CV
  script's fold-comparison helper that recursed into itself and got
  silently swallowed by the evaluator's per-turn exception handling,
  producing a nonsensical 0.0 HitRate@10 with no visible error until
  reproduced outside that try/except. Worth retrying with lighter
  hyperparameters (fewer trees/leaves, more regularization) if the
  private/held-out set ever becomes available for a second opinion.
- **Closing the `intent_override` slot-extraction gap** (`agent_shopper/
  slots.py`'s `MATERIALS`): the untouched extraction gap flagged in the
  boilerplate-stripping entry above -- 7/30 `intent_override` sessions'
  override message extracted zero slots at all (`extract_slots()` is a
  closed vocabulary, and the evaluator's `new_value` is drawn from the
  target product's own raw `features`/`details` catalog text, e.g. "Faux
  Fur", "100% Acrylic", "Water Resistant"), leaving the pre-override slot
  value stale and permanently excluding the target from gated buying-track
  sessions for the rest of the session. First pass added 4 `MATERIALS`
  words ("textile", "faux fur", "acrylic", "synthetic") plus a new,
  previously-dead `FEATURES` vocabulary/slot (16 words: "water resistant",
  "hand wash", "lightweight", etc. -- `SlotSet.feature` was already fully
  wired downstream but `extract_slots()` never populated it). **Net
  negative, reverted**: `intent_override` HitRate@10 unchanged (0.500,
  one clean miss->hit at `public_0072` cancelled by one hit->miss at
  `public_0084`) and MRR/MTTC both got worse (0.349->0.319, 8.03->8.20),
  while regressing 2 unrelated sessions elsewhere (`public_0035` boundary
  and `public_0138` browsing, both hit->miss). Root cause, found by
  per-session before/after diff and turn-by-turn tracing: `customer_reply()`
  draws ordinary attribute-disclosure text from the same catalog-snippet
  mechanism (`intent_card()`'s `hard_constraints`/`soft_preferences`) as
  the override's `new_value`, so any new vocabulary word doesn't just fire
  on override turns -- it also newly fires on disclosure turns in
  buying/browsing/boundary sessions that previously extracted nothing
  there, perturbing the heuristic reranker's attribute-match score
  unpredictably. Isolating each word's effect via the same diff showed only
  2 of the 4 `MATERIALS` additions had any measured effect at all:
  **"faux fur" and "acrylic" shipped alone** (`public_0072` miss->hit at
  rank 3, `public_0125` rank 10->3, zero other sessions changed --
  confirmed bit-identical buying/browsing/boundary metrics to 6 decimal
  places): `intent_override` HitRate@10 0.500->0.533, MRR 0.349->0.368,
  MTTC 8.03->7.80, overall TechnicalScore 0.5634->0.5674. "textile" and
  "synthetic" and the whole `FEATURES` vocabulary were dropped entirely
  rather than kept at partial credit, since neither produced a single
  confirmed win on the sessions they targeted. `public_0068` ("Imported")
  and 3 other sessions remain unextracted on purpose -- see the code
  comment on `slots.MATERIALS`.

## Development tools, APIs, libraries, and data

- **Dev tools:** Claude Code, standard Python tooling (`venv`, `unittest`).
- **APIs (optional, only when a key is configured):** OpenAI (Responses API,
  structured outputs) or Anthropic (Messages API) for the LLM reranking
  stage. No paid API is required to run or evaluate the agent.
- **Libraries:** `numpy`, `scikit-learn` (`TfidfVectorizer`/cosine
  similarity), `pydantic` (structured LLM output schemas), `python-dotenv`;
  `openai`/`anthropic` only imported when a key is present. BM25 is
  hand-rolled (no `rank_bm25` dependency). `sentence-transformers`/`torch`
  power the dense route (`agent_shopper/dense_index.py`) -- a frozen,
  non-fine-tuned encoder. The model import is deferred to first actual use
  (not module import time), but any real turn still needs it loaded to
  embed that turn's query text, even when the 50k-product catalog
  embeddings themselves come from a warm on-disk cache. `lightgbm` was
  used for a LambdaMART reranker attempt (see "What we tried") -- CV
  rejected it, and both the dependency and the code were removed rather
  than kept gated off.
- **Data:** the official TechJam frozen catalog (50,000 products from
  Amazon Reviews 2023's `Clothing_Shoes_and_Jewelry` category) and the
  200-session public dev set, both provided by the competition organizer.

## Limitations and what we'd improve with more time

- **Slot extraction is keyword/regex-based**, not learned — it covers the
  vocabulary we hand-curated (materials, colors, styles, use-cases,
  category words) and will miss paraphrases outside that list. A small
  local NLI or embedding-similarity slot matcher would generalize better
  without violating the "no fine-tuning of base LLMs" constraint.
- **No negative/exclusion constraint handling** ("no leather", "anything
  but red") — prototyped and measured (see "What we tried"), but removed
  rather than shipped disabled, since it has no eval track record either
  way. A real version would need closer-to-real parsing (dependency
  parsing or a small classifier) to attribute a negation to the right noun
  phrase reliably enough to trust as a hard filter.
- **Clarify-attribute relevance weighting is a hand-tuned per-department
  table**, not learned from the catalog itself. It reuses the same
  5-department bucketing `slots.CATEGORY_DEPARTMENT` uses for
  category-compatibility, so it's cheap to maintain, but the specific
  weights (e.g. "size" matters 1.3x more for footwear) are asserted, not
  fit. A real catalog-driven relevance signal (e.g. co-occurrence of
  use_case and attribute vocabulary in product text) would generalize
  better than a fixed table.
- **The `vector` route is still TF-IDF only**; dense embeddings were added
  as a separate `dense` route (`agent_shopper/dense_index.py`) alongside
  it rather than replacing it, deliberately keeping the "light, no required
  model download" TF-IDF path intact as a fallback. `ROUTE_WEIGHTS["dense"]`
  was checked against two alternatives and the shipped value won (see "What
  we tried"); `DENSE_FIELD_WEIGHTS` (the title/attributes/category/
  description split) remains an unswept hand-set prior. Dense embeddings'
  classic failure mode (pulling in semantically-similar-but-wrong-category
  items) hasn't been specifically
  stress-tested beyond the aggregate eval numbers in "What we tried".
- **No browsing-track diversity pass** — the competition's TechnicalScore
  has no diversity term at all, so an MMR-lite subcategory/store penalty
  could only ever demote the true purchased item's rank in exchange for
  variety nobody scores here. Worth adding for a real deployment, not for
  this rubric.
- **Constraint strength is inferred from a hand-curated hard-language
  regex** (`_HARD_CONSTRAINT_RE` in `dialog_policy.py`), not a learned
  classifier, and deliberately excludes generic-looking phrasing that
  happens to match the competition simulator's own scripted turn-1 message
  ("a key requirement is: ..."). A bisected eval run against the public dev
  set caught a real regression from including it: that phrasing fires on
  nearly every buying session's first turn, and hard-filtering the fused
  pool off a single, possibly-noisy, turn-1 slot extraction dropped buying-
  scenario Hit Rate@10 from 0.51 to 0.41 — worse than not hard-filtering at
  all. Left out; see the regex's own comment for the full trace. A cleaner
  fix would score extraction *confidence* directly rather than a boolean
  regex match, so a real hard requirement doesn't need to resemble any
  particular template to be trusted.
- **The information-gain clarify-attribute selection uses fixed price/
  vocabulary buckets**, not a fully general entropy calculation over
  arbitrary attribute values — good enough for this catalog's scale, but a
  more principled per-attribute cardinality estimate would generalize
  better to a different catalog. We also tried (and reverted, see "What we
  tried") weighting it by whether the pool's current top contenders would
  actually be separated by an attribute's answer — the idea is sound, but
  needs a post-rerank ranking signal to work from, not the pre-rerank
  fused score, to avoid fighting the already-validated entropy signal.
- **LLM reranking is (at most) one call per turn** over up to 20 candidates
  with truncated title/features plus a short keyword-prioritized
  description snippet (not the full description) — it's now skipped
  entirely on turns whose primary output is a clarifying question, and the
  system prompt explicitly instructs the model to treat candidate text as
  untrusted data rather than instructions. We don't currently stream partial
  context across turns to the LLM beyond the distilled slot summary, so a
  genuinely adaptive prompt-compression strategy (e.g. only re-describing
  what changed since the last LLM call) could reduce token usage further.
- **No cross-session personalization** is implemented beyond the harness's
  own `reset()` profile, since the competition spec defines each session as
  an isolated single-user interaction with no cross-session history
  available — this is a scope decision, not an oversight, but a production
  deployment would obviously want real cross-session memory.
- Token usage reporting (`usage` in the response) is currently a static
  `{0, 0}` for the LLM path; wiring through the actual `response.usage`
  from the provider call would make the reported cost/feasibility numbers
  exact rather than absent.

## Team member contributions

Solo project — architecture, implementation, and evaluation by the
repository owner.
