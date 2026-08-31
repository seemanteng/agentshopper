> Relocated from README.md's "What we tried" section to keep the top-level
> README readable. This is the full experiment ledger: every change that was
> attempted, the eval evidence for or against it, and why it was kept or
> reverted. See [README.md](../README.md) for the summarized results and
> architecture. Referenced from README's "How this was validated" section.

# What We Tried (and the Eval Evidence Behind What Shipped)

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
  build, never a per-query cost -- and, since the model packaging below
  removed the *network* dependency that one-time build used to carry (it no
  longer needs to download or find a cached copy of the encoder to start),
  a fresh clone is now guaranteed to complete that build offline rather than
  possibly stalling or failing on a network-disabled machine -- it just
  still costs the same ~27 CPU-minutes once. The 245MB *embeddings* cache
  file itself (not the model -- see below) remains local-only (gitignored)
  and isn't itself packaged/committed, so that CPU cost is still paid again
  on a fresh clone; see "Limitations" for this narrower remaining item.

  **Offline-safety hardening (later pass, prompted by the cross-encoder
  promotion work):** unlike the cross-encoder, this route had no
  `local_files_only` support and no error handling around model loading at
  all -- a load failure (or, empirically observed during this project's own
  offline-verification testing, a network-disabled environment retrying
  with exponential backoff rather than failing fast) would propagate
  uncaught through `Agent()` construction, which is called once and reused
  for every session. That's a whole-submission risk, not a per-session one:
  every one of the 800 hidden sessions would fail, not just this route's
  contribution. Fixed to mirror `cross_encoder_reranker.py`'s own pattern
  exactly: `config.DENSE_MODEL_LOCAL_FILES_ONLY` (default on) passes
  `local_files_only=True` to `SentenceTransformer(...)` and sets
  `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`, and `DenseIndex` now degrades
  gracefully on any load failure (empty field matrices, `search()` returns
  `[]`, never retries a known-failed load) instead of crashing -- the same
  "never let it crash the turn/session" contract the cross-encoder's own
  `CrossEncoderUnavailable` handling already has, just applied at
  construction time rather than per-call. Verified against the real
  pipeline: a broken model path no longer crashes or hangs `Agent()`
  construction, sessions complete normally on keyword+vector+category
  alone, and the working path is unaffected (confirmed `_model_load_failed
  is False`, identical recommendation counts). At the time this hardening
  pass landed, it deliberately did **not** solve the separate packaging gap
  (the model itself wasn't vendored yet, so a network-disabled *and*
  not-already-cached judging environment would still lose this route) --
  it only made that failure fast and contained instead of a slow hang or a
  fatal crash. **That packaging gap is now closed** (see "Deployment
  readiness"'s "A related finding, since fixed and, as of this pass, fully
  closed" and `scripts/prepare_dense_model_artifact.py`) -- this graceful-
  degradation path remains as a belt-and-suspenders fallback for anyone who
  overrides `AGENT_SHOPPER_DENSE_MODEL` away from the packaged default, not
  the primary way the shipped configuration is expected to work.
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
- **Clearing stale point-constraint slots on an unattributed override**
  (third attempt at the `intent_override` extraction-gap problem above, this
  time touching no vocabulary at all): when contradiction language fires but
  `extract_slots()` returns nothing this turn (5/30 sessions:
  `public_0003/0023/0038/0068/0186`), the hypothesis was that the
  pre-override slot value survives stale and permanently excludes the true
  target from a gated buying-track pool (`retrieval.retrieve`'s exact
  AND-filter). The fix cleared every currently-droppable point-constraint
  slot (reusing `orchestrator.droppable_slots_by_tier`, never `category`)
  whenever this fired, relying on the override message's own words already
  being tokenized into that turn's `query_text` regardless
  (`context.build_query_text` always includes the raw message). **Bit-
  identical eval result on every metric and every scenario_type** --
  traced turn-by-turn (not just the aggregate number) before concluding
  this: 4 of the 5 candidate sessions already hit on turn 1, before the
  override turn was ever reached; the remaining one (`public_0023`) is on
  the **browsing** track with only `use_case` filled -- browsing never
  gates to an exact category filter, so there was no stale hard filter to
  break in the first place, and manually clearing it (confirmed by a
  one-off patched trace) still didn't produce a hit -- that session's miss
  is an unrelated BM25/dense recall failure. The specific mechanism this
  targeted (stale slot + gated hard filter after an unattributed override)
  provably doesn't co-occur anywhere in the 200-session public set. Reverted
  in full rather than kept as an unproven, zero-measured-effect code path,
  per this project's standing rule -- even though it was provably harmless
  (every other override path is bit-for-bit untouched) and may still matter
  on the hidden 800-session set's different session mix, keeping unproven
  behavior around isn't this project's convention. Worth retrying if a
  future diagnostic finds an actual gated-buying-track session where an
  unattributed override's stale slot demonstrably blocks the target.
- **LLM reranker benchmark** (`gpt-4o-mini` via `OPENAI_API_KEY`,
  `scripts/run_llm_benchmark.py`): the previously-unbenchmarked LLM path
  (see "LLM usage" above) turns out to be a clean, repeatable *regression*
  against the shipped heuristic reranker, not the semantic-judgment upgrade
  it was hoped to be. 3 full 200-session runs (needed because, unlike every
  other change in this section, the LLM path is non-deterministic) plus one
  heuristic-only baseline: TechnicalScore 0.5674 -> 0.5301 mean of 3 runs
  (stdev 0.0021, range 0.5277-0.5313 -- the run-to-run spread is ~18x
  smaller than the gap to heuristic, so this is a real effect, not noise),
  HitRate@10 0.6750->0.6317, MRR 0.4103->0.3726, MTTC 5.66->5.88. Every
  scenario_type regressed; `intent_override` -- the track semantic judgment
  was hoped to help most, per this conversation's own reasoning about where
  an LLM's contextual understanding could beat hand-engineered features --
  lost the most (MRR 0.3678->0.2580, HitRate@10 0.5333->0.4889). Only
  ~24% of turns are actually LLM-eligible overall (route breakdown: 69%
  `clarify_skip`, 24% `eligible`, 7% `last_turn`, <1% `tight_pool` --
  `scripts/estimate_llm_cost.py`), so the loss is concentrated in real LLM
  judgments, not diluted by the heuristic fallback majority.

  Root-caused with `scripts/llm_rerank_diagnostics.py`'s two checks, run
  against `intent_override` sessions specifically (the biggest loser, and
  the track this project's own prior work already flagged as most sensitive
  to context/prompt construction -- see the override-related entries
  above): **severe candidate-order sensitivity, not prompt injection**.
  Replaying 40 real LLM calls with candidates reordered (`--position-bias`)
  flipped the top pick 92.5% of the time -- but so a stochasticity control
  matters here: replaying the *same* (unshuffled) order twice already flips
  30.0% of the time on its own, since `gpt-4o-mini` is called with no
  temperature/seed pinning. The attributable order effect is the difference,
  +62.5 percentage points -- reordering alone, holding content fixed,
  changes the model's answer most of the time. Candidates reach the LLM in
  fused-RRF-score order (`reranker.py`'s `RERANK_CANDIDATE_LIMIT` slice), so
  this is a live, reproducible failure mode of the shipped prompt
  construction, not a one-off. Separately, a manual skim of a 57-call trace
  sample (`--dump-trace`, weighted toward `intent_override`) found no sign
  of prompt-injection susceptibility -- `relevance_score` judgments tracked
  genuine title/feature/description content, not marketing-copy imperatives
  in the (explicitly untrusted-by-instruction) candidate text.

  Total cost across the cost estimate, all 3 benchmark runs, and both
  diagnostics: well under $1. The LLM reranker remains available
  (`OPENAI_API_KEY`/`ANTHROPIC_API_KEY` set, `AGENT_SHOPPER_FORCE_HEURISTIC`
  unset) but the heuristic path stays the shipped default. The natural next
  step, if this is revisited, is an order-robustness fix before anything
  else -- e.g. averaging judgments across a couple of shuffled orderings, or
  moving from listwise to pairwise/pointwise scoring -- since the current
  regression looks driven almost entirely by this one mechanism rather than
  the model's underlying judgment quality.

- **Frozen pointwise cross-encoder reranking pilot** (`agent_shopper/
  cross_encoder_reranker.py`, `scripts/replay_cross_encoder_offline.py`,
  `scripts/cv_cross_encoder.py`): motivated by `scripts/diagnose_retrieval.py`'s
  Oracle recall-ceiling diagnostic, which found 30 of the 65 current misses
  had their target within the fused top-100 pool (Hybrid-Union Oracle
  HitRate@10 82.5% at K=100 vs. the 67.5% shipped baseline) but that
  replacing the heuristic outright would exclude 8 sessions it currently
  hits from beyond position 100 in the raw fused order — pointing at a
  hybrid addition, not a replacement. Built as a frozen, local, pointwise
  scorer specifically to avoid the LLM reranker's diagnosed failure mode
  (candidate-order sensitivity from listwise judging, see that entry below)
  — this scores each query-candidate pair independently and never sees
  another candidate's text, and candidate-order invariance is directly unit-
  and integration-tested (including a real-model spot check, not just
  fake-scorer logic tests).

  **Environment-blocking finding, not a code bug**: the originally-specified
  `cross-encoder/ms-marco-MiniLM-L-6-v2` reliably crashes this development
  machine's process with SIGBUS during inference — not a catchable Python
  exception, so no fallback design can save it. Root-caused before writing
  around it: rules out sandbox restrictions, thread/BLAS config, SDPA vs.
  eager attention, mmap loading, and checkpoint corruption (safetensors file
  verified intact, no NaN/Inf in any of its 106 tensors, vocab/embedding
  sizes match). Isolated by comparison — a random-weight BERT of the same
  shape works; the project's own existing dense encoder
  `sentence-transformers/all-MiniLM-L6-v2` (**identical config**: 6 layers,
  384 hidden, 12 heads) works; `cross-encoder/ms-marco-MiniLM-L-4-v2` (same
  family) also crashes; `cross-encoder/ms-marco-TinyBERT-L-6` and
  `cross-encoder/ms-marco-electra-base` (different architectures) don't —
  a real, reproducible numerical/BLAS-level issue specific to this
  checkpoint's weight values on this torch 2.11.0 build, not something
  application code can catch or fix. `AGENT_SHOPPER_CROSS_ENCODER_MODEL`
  stays defaulted to the originally-specified `ms-marco-MiniLM-L-6-v2`; every
  number below was measured with `cross-encoder/ms-marco-TinyBERT-L-6`
  (verified working, same MS MARCO training data, 6 layers/768 hidden — a
  larger, slower substitute) via that override, not the default.

  **Phase 7 (offline replay, real model, full 200-session public set)**:
  989 scored turns, 259 with the target reachable in the K=100 hybrid union.
  Gate: **PASS**. `alpha=0.0` reproduced the heuristic baseline exactly
  (sanity check). `alpha=0.30`: 25 top-10 gains / 2 losses (net +23), mean
  reciprocal-rank delta +0.060, positive or flat in every scenario_type,
  candidate-order invariance PASS on a 20-turn real-model spot check.

  **Phase 8 (interactive 5-fold CV, stratified by scenario_type, seed=42)**:
  all three nonzero alphas cleared the adoption bar (5/5 folds non-negative,
  no severe intent_override/boundary regression). `alpha=0.30`: mean
  TechnicalScore delta +0.0315, **zero hit→miss flips across all 5 folds**
  (6 flips, all upward). `alpha=0.50`: mean delta +0.0378 (numerically
  highest) but 3 real hit→miss regressions against 9 gains, with its extra
  gain concentrated in `buying` at the cost of giving back `intent_override`'s
  improvement. `alpha=0.30` chosen over the numerically-higher `alpha=0.50`
  for exactly this reason — a clean, broad, zero-regression win over a
  marginally-higher but noisier one, matching this project's standing
  preference throughout this section.

  **Phase 9 (full 200-session confirmation, `alpha=0.30`)**: HitRate@10
  0.675→**0.705** (+6 sessions), MRR 0.410→**0.439**, MTTC 5.66→**5.27**,
  TechnicalScore 0.5674→**0.5989** (≈**0.60**, +0.0315 — matching the CV's
  mean prediction almost exactly, a meaningful internal-consistency check
  against the CV result being fold-partition noise). Session-level diff: 6 miss→hit
  (`public_0002` intent_override, `public_0005`/`0058`/`0107` buying,
  `public_0059`/`0115` browsing), **0 hit→miss** (confirms none of the 8
  baseline hits the Oracle diagnostic flagged as originating below fused
  rank 100 were lost), 46 sessions with an improved (lower) rank, 27 with a
  regressed-but-still-hit rank, 15 earlier first-hit turns vs. 2 later.
  Scenario breakdown matches the CV's aggregate prediction almost exactly:
  buying 0.713→0.750, browsing 0.688→0.713, intent_override 0.533→0.567,
  boundary unchanged at 0.700 (n=10, no signal either way, consistent with
  the Oracle diagnostic's own finding there).

  **Latency/packaging** (substitute model; the intended smaller MiniLM
  would likely be faster): per-turn scoring (up to 110 candidates) mean
  4.221s / p50 4.284s / p95 4.535s cold (Phase 7); a real 10-turn session
  would add roughly 10-40s of total latency, not the 40-75 minutes the
  batch CV/offline runs took evaluating hundreds of simulated sessions.
  Checkpoint size: `ms-marco-TinyBERT-L-6` ≈256MB on disk vs. the intended
  `ms-marco-MiniLM-L-6-v2` ≈88MB. At the time this pilot was run, neither was
  vendored in this repository and network-free judging would have meant
  downloading it manually and pointing `AGENT_SHOPPER_CROSS_ENCODER_MODEL`
  at the local directory with `AGENT_SHOPPER_CROSS_ENCODER_LOCAL_ONLY=1` --
  see the "Updated verdict" below and "Deployment readiness" for how this
  was actually resolved (the substitute checkpoint is now packaged and
  committed, no manual download needed).

  **Original verdict (superseded, kept for the record): kept as an explicit,
  off-by-default experimental pilot, not shipped enabled.** The CV/
  confirmation evidence for the *mechanism* (frozen pointwise scorer +
  hybrid-union RRF fusion) was genuinely strong and internally consistent —
  but it was only validated against a heavier substitute model, not the
  originally-specified checkpoint, because that checkpoint is unable to run
  at all on this development machine for reasons that remain unresolved.
  Shipping the untested default (`ms-marco-MiniLM-L-6-v2`) enabled by
  default would risk the exact same crash in an unknown judging environment;
  shipping the validated substitute enabled by default without first
  packaging and offline-verifying it would mean silently deploying a
  3x-larger checkpoint and ~4s/turn of added latency nobody explicitly
  signed off on for production, with no guarantee it would even load without
  a developer's Hugging Face cache present.

  **Updated verdict: promoted to the shipped default.** Once the substitute
  checkpoint was packaged into a self-contained, checksummed local artefact
  (`models/cross_encoder/ms-marco-TinyBERT-L-6/`) and every deployment gate
  below was independently re-verified — offline loading with zero
  `AGENT_SHOPPER_*` environment variables, no dependency on any developer's
  cache, 5/5 subprocess cold-load tests clean with zero signal terminations,
  candidate-order invariance against the packaged checkpoint, and a full
  200-session reproduction from a network-blocked, freshly-extracted copy of
  the submission archive itself (not just the working tree) — the remaining
  objection (packaging risk) was resolved rather than merely accepted. See
  the "Deployment readiness" subsection immediately below for the complete
  evidence trail and exact numbers. The three organizer questions this
  section originally reported as open (archive-size limit,
  checkpoint-bundling/Git-LFS permission, numeric timeout) were genuinely
  unresolved at the time and reported as blockers rather than silently
  assumed either way -- they were later resolved by an organizer FAQ this
  project's fork hadn't received yet (see "Deployment readiness"'s
  "resolved" note for the full, verbatim policy). Both the config flag and
  every re-validation script
  (`scripts/replay_cross_encoder_offline.py`, `scripts/cv_cross_encoder.py`,
  `scripts/prepare_cross_encoder_artifact.py`,
  `scripts/smoke_test_cross_encoder_subprocess.py`,
  `scripts/build_submission_archive.py`) are kept, fully tested, and
  documented.

- **"Never-recalled" retrieval diagnostic and two reverted retrieval-side
  attempts** (`scripts/diagnose_never_retrieved.py`, new, kept as a
  read-only diagnostic tool): of the 65 heuristic-only misses, 26 are never
  recalled by *any* route at *any* depth up to 200 (retrieve()'s hard cap)
  — a retrieval-side failure no reranker change of any kind can fix,
  distinct from the other 39 (recalled at least once, never ranked
  top-10). A battery of read-only, offline, one-change-at-a-time
  counterfactuals (re-querying `retrieval.retrieve()`/`bm25_index`/
  `tfidf_index`/`dense_index`/`filter_products` directly with the real
  recorded query/slots/plan from a single interactive pass, never a second
  full re-simulation) found a striking structural pattern: **all 26 have
  `gate_to_category=False`** on every turn, and 21 of them structurally
  satisfy the accumulated slots' filter criteria (`filter_products`, no
  depth cap) — the target was simply never *looked for* outside whatever
  keyword/vector/dense already happened to surface, because the ungated
  branch's category signal (`_soft_category_ranking`) only re-ranks
  already-surfaced candidates, never independently recalls a filter-
  matching one. Two candidate fixes were built and measured against a
  freshly-captured heuristic-only baseline (`scripts/run_local_eval.py`,
  bit-identical to `post-revert-confirm`: TechnicalScore 0.5674, HitRate@10
  0.675) — **both reverted as net-negative**:

  - *Category as an independent recall route on ungated turns* (reused the
    gated branch's own `filter_products`+`rank_by_rating`, soft/RRF-
    weighted rather than a hard gate, capped at `ROUTE_SEARCH_LIMIT`):
    TechnicalScore 0.5674→0.5650, **zero of the 26 targets actually
    recovered** (0 miss→hit), 1 new hit→miss (`public_0035`), rank
    regressed on 7 sessions vs. improved on 3. The diagnostic's own
    `route_category_at_500` counterfactual measured candidate
    *availability* only (no depth cap, no reranking) — it never proved the
    real, rating-heavy `HeuristicReranker` would actually place a newly-
    available candidate in the top-10, and empirically it didn't. Exactly
    the candidate-availability-vs-real-recovery gap this project's own
    Hybrid-Union-Oracle methodology (`scripts/diagnose_retrieval.py`)
    already warns about.
  - *Raise per-route search depth 200→500*
    (`config.RETRIEVAL_ROUTE_SEARCH_LIMIT`, `AGENT_SHOPPER_ROUTE_SEARCH_LIMIT`):
    TechnicalScore 0.5674→0.5589 (worse), intent_override HitRate@10
    collapsed 0.533→0.433, 43 of 200 sessions changed, only 3 miss→hit
    (**none of them among the diagnosed 26** — `public_0008`/`0146`/`0175`
    are unrelated sessions) against 4 new hit→miss and a 8:28 rank-
    improve:regress ratio. A system-wide depth increase touches every
    turn's RRF fusion, not just the 26 targeted ones, and the churn was
    net-harmful.

  Both changes were built, isolated (one at a time, per this project's own
  discipline), measured with a full before/after session-level diff, and
  reverted the same session they were tried — nothing was shipped or left
  behind as a disabled toggle. The diagnostic script and its full per-
  session evidence (`diagnose_never_retrieved.log`, gitignored) are kept;
  the 26-session retrieval gap itself remains open, a genuine, unsolved
  limitation — see "Limitations and what we'd improve with more time."

- **Attempt #3: a smaller, capped, low-weight structured-match injection —
  stopped before implementation, on corrected diagnostic evidence**
  (`scripts/diagnose_conservative_injection.py`, new, kept as a read-only
  diagnostic tool). The actual lesson from #1/#2 wasn't "the mechanism was
  wrong" — it was that their *validating counterfactuals* only checked
  candidate availability (unlimited depth, no reranking), never whether the
  real, rating-heavy `HeuristicReranker` would actually place a newly-
  available candidate in a real top-10 under the real `ROUTE_SEARCH_LIMIT`.
  So before writing a single line of production code this time, the
  counterfactual itself was corrected: for each of the 26 sessions, inject
  a capped (5 or 10) set of not-already-surfaced `filter_products` matches
  ranked by rating — exactly what a real implementation would do — into the
  *real* candidates from the *real* `retrieve()` call, and run the result
  through the *real*, unmodified `HeuristicReranker().rerank()` (its actual
  `ctx`, captured the same monkeypatch way `scripts/diagnose_retrieval.py`
  already does), checking real top-10 membership, not pool membership.

  **Result: 0 of 26 recovered, at both cap=5 and cap=10.** Per the
  design's own stop condition, implementation was never attempted — no
  production code was touched. Why: of the 26, all had at least one
  ungated, slots-filled (i.e. "applicable") turn, but the target entered
  the rating-ranked injected set at all on only **2** of them — a rating-
  based top-K cap excludes the vast majority of targets that structurally
  match the slots but aren't highly rated, which is most of them. (This is
  also why the earlier, uncapped `route_category_at_500` counterfactual
  measured 21/26 "recoverable" — an artifact of having no depth cap at
  all, not a realistic estimate.) Even the 2 sessions where the target did
  enter the injected set still didn't survive `HeuristicReranker`'s own
  scoring into the real top-10 — assumed at the time to be `rating`
  (~9x the other raw weights) dominating the 5-feature composite against a
  fresh keyword/vector score of 0 for a never-lexically-surfaced candidate.

  **Follow-up spot-check, because that was an inference from the weights,
  never actually measured** (`scripts/diagnose_reranker_weight_gap.py`):
  computed the real per-feature weighted-contribution breakdown for both
  comparable sessions (`public_0040`, `public_0180`) using
  `HeuristicReranker`'s own `_feature_vector`. The assumption didn't hold —
  in both cases `bm25` (lexical absence), not `rating`, was the dominant
  contributor to the gap, and `rating`'s own contribution was small or
  actually favored the target (`public_0040`: `bm25`=+1.43, `attr_match`=
  -1.32, `rating`=+0.18; `public_0180`: `bm25`=+1.03, `vector`=+0.53,
  `rating`=-0.99 — `rating`'s confidence-shrunk scale compresses real gaps
  more than a route swinging from "found" to "not found" does, so its
  larger raw weight moves the composite less in practice). n=2 is too
  small to generalize the reranking-stage finding, but large enough to
  retire "rating dominates" as the accepted explanation. The real
  bottleneck is two layers: `rating`'s actual leverage is in *which*
  candidates the injection selects in the first place (`rank_by_rating`
  excludes 24 of 26 targets before reranking is ever reached), not in the
  final scoring of whichever ones survive that cut. This 26-session gap
  remains open; a structured-match signal that can survive
  `HeuristicReranker`'s own feature weighting (not just enter the pool)
  would need either a different injection-selection criterion (not
  `rating`) or a fabricated bm25/vector-comparable score, not an
  independent rating-based selection — a different, unexplored design, not
  a smaller version of #1/#2.
