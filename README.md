# MiniJev

The smallest useful implementation of Jev: a tinygrad model that judges a query
before a larger system handles it.

MiniJev predicts two things from a short UTF-8 query:

1. `confidence` in `[0, 1]`
2. `route_logits` over four handling paths: `local`, `rag`, `cloud`, `review`

The model is deliberately an embedding, a masked mean pool, one hidden layer and
two heads. It is a supervised decision model, not a text generator.

## Architecture in one picture

The model produces scores. A policy spends them. Nothing else decides a route.

```text
query ──▶ ByteTokenizer ──▶ TinyJev ──▶ route_logits[4]
                                            │
                                   softmax → route_probs[4]
                                            │
                        confidence.confidence_from(route_probs)
                                            │
                              policies.Policy.decide(scores)
                                            │
                                            ▼
                        Decision(route, confidence, confidence_measure,
                                 route_logits, route_probs, escalated_from)
```

The model emits logits and nothing else. Confidence is computed from them, and the
policy spends both. There is exactly one source for each number.

Three entry points share those modules. They are **not** three copies of the code
and there are no `training/` and `inference/` directory trees:

| Entry point | Command |
|---|---|
| training | `python -m minijev.training` |
| inference | `python -m minijev.inference` |
| evaluation | `python -m minijev.evaluation` |

Supporting CLIs: `python -m minijev.data` (split, stats), `python -m minijev.synthetic`
(generate), `python -m minijev.frame_product` (slot cross-product), and
`python -m minijev.convert` (external datasets).

## Slot cross-product boundary corpus

The hand-written frame bank remains the explanatory reference set. For a tighter
estimate of boundary separation, `minijev.frame_product` expands the same deciding
slots (`source`, `derivation`, `steps`, `risk`, and `underspecified`) over a
controlled Chinese predicate/object vocabulary. The route is always recomputed by
`frames.route_from_frame`; it is never copied into a generated frame.

The checked-in generation command produces 1,280 balanced frames (320 per route),
with 960 training frames and 320 held-out benchmark frames. All variants of a
frame stay in one split, and the manifest records route and slot counts:

```powershell
python -m minijev.frame_product --output-dir data --target-per-route 320 `
  --benchmark-fraction 0.25 --variants 3 --seed 17
```

This writes `data/frame_product_train.jsonl`,
`data/frame_product_benchmark.jsonl`, and
`data/frame_product_manifest.json`. The product benchmark is a new measurement
instrument; `data/hard_benchmark.jsonl` remains the original hand-written set and
is not replaced by generated rows. The boundary evaluator detects the generated
metadata and derives a disjoint set of minimal contrast pairs automatically; the
default manifest contains 137 such pairs.

## Phase 1 boundary

- tinygrad only for model computation
- deterministic UTF-8 byte tokenizer, no vocabulary training step
- masked mean pooling encoder
- a single route head; confidence is derived from it, not a second head
- no PyTorch, text generation, or LSTM; the pooled encoder remains the default and
  the small Transformer is an explicit comparison mode (`--encoder-type transformer`)
- **no reinforcement learning** — see "Reinforcement learning" below

## Quickstart

```powershell
$env:PYTHONPATH = "D:\jev\tinygrad"
cd D:\jev\MiniJev

# Quality gate
ruff check . ; ruff format --check . ; python -m pytest tests -q

# Generate data, split it by semantic group, train
python -m minijev.synthetic --output data\synthetic.jsonl --per-route 250 --seed 42
python -m minijev.data split data\synthetic.jsonl --output-dir data --seed 0
python -m minijev.training --epochs 100 --batch-size 32 --patience 15

# Route a query with the trained weights
python -m minijev.inference "请总结这份合同的风险"

# Evaluate, with the majority-class baseline alongside
python -m minijev.evaluation data\test.jsonl --checkpoint models\minijev_phase1.safetensors --baseline
```

## Splitting: the part that makes numbers mean something

`minijev.data split` groups rows by `semantic_group_id` before splitting, so every
template lands in exactly one split. `assert_no_group_leak` runs on every split and
`tests/test_split.py` proves the guard can fail.

This matters because the first version of the dataset did the opposite. Of 1,036
rows it had only 315 unique queries, and 117 of the 129 distinct test queries also
appeared in training. The reported 98% measured template memorisation. The archived
version of that data is in `data/legacy_v1/` for comparison; do not train on it.

A consequence worth knowing: honest template-level splitting leaves **few distinct
test templates**, because 77 groups split three ways is a thin test set. That is a
property of the data, not of the splitter.

## Boundary cases are frames, not labels

A route label is an **assertion**: `(query, "rag")` does not say *why* the answer is
rag, so it cannot be checked, varied, or argued with. `minijev/frames.py` replaces the
assertion with a **frame** — six slots describing the request — and *derives* the route
by one inspectable rule.

### Why not 主谓宾

Annotating each query like 主谓宾 is the right instinct — structure beats holistic
judgement — but 主谓宾 is **not the discriminating structure for routing**. Take the two
cases the spec names:

```
分析产品手册中的安装步骤        -> rag
分析这套系统为什么存在并发问题   -> cloud
```

Both are `[谓] 分析 + [宾] 名词短语`. Identical 主谓宾, opposite routes. What differs is
the **状语** (where the answer must come from) and the **补语** (whether the answer
exists or must be built):

```
来源=外部手册, 答案=已存在(取回)   -> rag
来源=给定文本, 答案=需构造(推理)   -> cloud
```

So the frame carries `predicate` 谓语, `object` 宾语, `source` 状语, `derivation` 补语,
`steps` 步数, `risk` 情态 and `underspecified` 省略. Only five of these can change a
route, and those five are the *deciding slots*.

### The rule

```python
risk HIGH            -> review    # cost of being wrong dominates
underspecified       -> review    # nothing to act on
CONSTRUCT or steps>1 -> cloud     # the answer must be built
external source      -> rag       # the answer must be fetched
otherwise            -> local     # the answer is already at hand
```

**Nothing in the bank stores a route.** Routes are computed, so a frame and its route
cannot drift apart, and changing the policy re-derives the whole bank. Every row records
`route_rule`, the one line naming the slot that decided it, so an error can be traced to
a slot instead of to a whole sentence.

### The benchmark

`data/hard_benchmark.jsonl` holds **46 frames / 138 rows**, 11–12 frames per route, and
is never read by the training path. It is built from **contrast pairs** whose two halves
are held out together, because a pair only measures separation when the model is asked
both halves. Some pairs are **minimal**: exactly one deciding slot differs, so a failure
is attributable to that slot alone.

`minijev.boundary` scores a checkpoint on it and reports the metric that matters — the
**collapse rate**, the fraction of pairs answered with the *same* route. A model can
score well on the benchmark overall and still collapse every pair; that is the failure
the benchmark exists to expose.

```powershell
python -m minijev.frames audit
python -m minijev.frames benchmark --output data\hard_benchmark.jsonl --variants 3
python -m minijev.boundary --checkpoint models\exp4_pooled_merged.safetensors --only-minimal
```

### What it measured

Four arms, identical data/seed/policy/tokenizer, argmax so the policy cannot flatter the
encoder. The majority-class baseline is 0.261.

| arm | synthetic | literary | MASSIVE | boundary acc | both correct | **collapsed** |
|---|---:|---:|---:|---:|---:|---:|
| pooled / merged | 0.533 | 0.889 | 0.207 | **0.522** | **31.6%** | **15.8%** |
| transformer / merged | **0.623** | 0.889 | **0.272** | 0.515 | 21.1% | **36.8%** |
| pooled / +massive | 0.392 | 0.861 | 0.788 | 0.312 | 0.0% | 68.4% |
| transformer / +massive | 0.523 | **0.972** | **0.802** | 0.254 | 10.5% | 68.4% |

Three findings, all of which corrected an earlier belief:

1. **The Transformer does not win where it was required to.** It is level on boundary
   accuracy (0.515 vs 0.522) but collapses **twice as many** contrast pairs
   (36.8% vs 15.8%). Its advantage on the synthetic test set is real and does not
   transfer to boundary discrimination, so `pooled` stays the default.
2. **Merging every corpus is wrong.** MASSIVE's derived rubric has **zero cloud rows**,
   and adding it drives pooled's cloud recall from 0.58 to **0.12** and the Transformer's
   to **0.00**, with the `steps 2+` slot at zero. An absent class is learned as a class
   that does not exist. The Transformer resists this better on literary (0.972 vs 0.861)
   but not on boundaries.
3. **Both encoders fix about half the slots.** pooled reads `source_class`
   (external 0.67 / internal 0.52); the Transformer nails `underspecified` (1.00) but is
   worse on external sources (0.47). Neither reads the whole frame.

Two invariants are enforced by tests, because both were real failures:

- **the benchmark shares no query text with the training corpus.** Two underspecified
  frames had reused ambiguous phrasings verbatim from the template bank, so those rows
  had already been trained on;
- **both halves of every contrast pair are held out**, or the pair cannot be scored.

### Where the construction can still improve

- **The training side of boundary data is thin** — 30 frames / 90 rows, against a
  benchmark of 46 *unseen* frames. Half the error is "has not seen enough boundary
  cases", not "wrong encoder".
- **All 76 frames are hand-written.** The next step is slot cross-product generation
  (valid `predicate × object × source × derivation × steps` combinations) to take the
  bank from 76 to 1000+, which is the only way to tighten the collapse-rate interval.
- **The strongest available idea: supervise the frame slots directly.** Since the route
  is a function of the slots, a multi-task head predicting `source`/`derivation`/`steps`/
  `risk` may need far fewer examples than learning the route from surface text. It is
  the one proposal that actually uses the new annotation, and it is untested here.
- The benchmark is synthetic, and its author is also the system being measured. Until it
  is replaced by human annotation, only internal comparisons are meaningful.

## Broad domains: news, philosophy, literature, academic

The idea being tested is that a handful of broad domains covers most of the grammar and
most of the situations a router meets, so a decision learned on news and philosophy holds
up on an academic abstract.

**Merging is only safe when every corpus covers every route.** That is measured, not
assumed: MASSIVE's derived rubric contains **zero cloud rows**, and merging it drove cloud
recall from 0.58 to 0.12 for the pooled model and to 0.00 for the transformer. An absent
class is learned as a class that does not exist. So `minijev.domains` pairs every passage
with the same four requests — bounded extraction (local), provenance lookup (rag),
multi-step interpretation (cloud), rights or ethics judgement (review) — and each passage
therefore contributes exactly one row per route.

```
domain audit: news 25/25/25/25, philosophy 25/25/25/25, literature 25/25/25/25, academic 25/25/25/25
```

The four domains do not share text: news, philosophy and academic read authored synthetic
passages from `data/external/domains/passages.json` (no real people, institutions or
journals), and literature reuses the public-domain classical excerpts. `domain_audit()`
recomputes the proportions and a test fails if any domain tilts.

## Context: the same request, a different state

Every other benchmark here scores a single sentence. That is the limitation worth naming:
a router decides *inside a conversation*, and much of what determines the route is
established earlier.

A **context pair** is two cases with an **identical query** whose preceding turns differ,
and whose routes therefore differ:

```
query:  这两种能不能一起用
ctx A:  我在整理两种清洁剂的成分表。          -> local   (given text, bounded)
ctx B:  我母亲在吃降压药，医生又开了安眠药。   -> review  (high stakes)
```

A model that ignores context **must** answer both halves identically, so it collapses every
pair — 100% collapse is the signature of an encoder that cannot see state. The probe holds
the query text fixed and varies only the state, so a collapse cannot be blamed on wording.
It covers all six route combinations, two pairs each, and is route-balanced.

Turns are joined with an explicit separator id (`turn_sep_id`, `vocab_size` 258), not a
newline, so a boundary cannot be confused with content. The separator lives in the config
and travels in the checkpoint sidecar: a single-turn checkpoint keeps 257 rows and still
loads, and `--context` is what opts into the extra row.

## Auxiliary supervision: predict the deciding slots

The route is a deterministic function of five slot values, so supervising those slots
directly is a denser version of the same question: instead of asking the encoder to infer
"rag" from surface text, it is asked where the answer must come from, whether it already
exists, how many steps it needs and how costly a mistake is.

```powershell
python -m minijev.training --train data\broad_train.jsonl --aux-weight 0.5 ...
```

Three properties are enforced by tests, because each is a way the idea fails silently:

- **the labels are the real ones** — `slot_index` is asserted against `slot_value` for
  every framed row, and every label must be reachable, so no head is trained on noise;
- **the decision path never reads the heads** — `JevOutputs.slot_logits` is separate from
  `route_logits` and `None` on a plain model, so an aux checkpoint and a plain one answer
  identically;
- **it is not a capacity trick** — the heads add 780 parameters to 20,868 (<4%), asserted,
  because the encoder comparison already taught that a bigger model is not a better idea.

Rows without a frame are masked out rather than given a fake label, so template data and
framed data share one corpus.

## Confidence is derived, not learned

Every decision carries the full distribution over the four routes (`route_probs`),
and `confidence` is a **statistic of that distribution's shape** — concentrated
means a clear read, flat means the model cannot tell the routes apart. This is the
same contract TypeSafe uses, and it matters that confidence is *derived* rather
than trained:

> A head trained to predict a dataset label cannot know whether **this** prediction
> is right. That is a different question from "how contested is this kind of query",
> and it is the question a router's safety branch has to answer.

The first version of MiniJev had such a head. It was trained to regress the
annotator-agreement label, and because 96.5% of that label was a single value, it
collapsed to a constant. Measured over the 226-row test split:

| signal | spread across the test set |
|---|---:|
| `max(route_probs)` | 0.642 |
| `1 − H/log(4)` (the default) | **0.944** |
| the removed confidence head | **0.018** |

The distribution was carrying a real signal the whole time, and the head was
overwriting it with a constant. The derived measure also tracks correctness:
**mean confidence 0.700 when the route was right, 0.445 when it was wrong.**

So the head is gone. `JevSample.confidence` survives as the annotated agreement
label, where it does honest work as a **per-sample training weight**
(`--no-agreement-weights` disables that) — it is a weight, never a prediction
target.

### Choosing a measure

`minijev.confidence.CONFIDENCE_MEASURES` provides four, and `Decision` records
which one produced its number, so two decisions are only comparable when they used
the same measure:

| measure | what it says | when to use |
|---|---|---|
| `normalized_entropy` *(default)* | `1 − H/log(K)`, flatness scaled by option count | the general case; comparable across different `K` |
| `top_probability` | the largest probability | quickest to explain, but **not** comparable across `K` |
| `margin` | top minus runner-up | "is there an obvious winner" |
| `effective_options` | perplexity `exp(H)` — a *count*, not a confidence | easiest number to justify to a non-ML reader |

`distribution_shape()` returns all of them plus the raw shape, so you are never
bound by our default.

### Thresholds vary with risk

A single confidence floor cannot express that a wrong `local` answer is cheap and a
wrong `cloud` decision is expensive. `RiskLadderPolicy` gives each route its own
gate and its own escalation target, ordered by **what is at stake** rather than by
position in the chain:

```python
DEFAULT_LADDER = {
    Route.LOCAL:  RouteGate(accept=0.30, escalate_to=Route.RAG),     # cheap, recoverable
    Route.RAG:    RouteGate(accept=0.35, escalate_to=Route.CLOUD),   # grounding may miss
    Route.CLOUD:  RouteGate(accept=0.55, escalate_to=Route.REVIEW),  # expensive to get wrong
    Route.REVIEW: RouteGate(accept=0.00, escalate_to=Route.REVIEW),  # terminal, safest
}
```

A read that does not clear its route's gate steps one rung towards more care, and
`review` is terminal so a genuinely unknown answer stays with a human instead of
bouncing. `--caution 2.0` scales every gate together, which is the only sane way to
tune four thresholds at once.

These gates are **reasoned starting points, not fitted values**. Where to sit on the
accuracy/caution curve depends on what a misroute costs, which is a product
decision. Measured on the MASSIVE test split (2,974 rows), picking the current
defaults over the shipped-first-guess alternative:

| ladder | accuracy | macro-F1 | escalated |
|---|---:|---:|---:|
| no escalation (`argmax`) | 0.7724 | 0.5585 | 0.0% |
| this default | 0.7683 | **0.6117** | 3.1% |
| earlier guess (`local` strictest) | 0.6809 | 0.5903 | 22.1% |

The first guess gated `local` hardest, which is backwards: it treated the cheapest
action as the one needing most confidence. Getting the ordering wrong cost 9
accuracy points and escalated a fifth of all traffic.

`FlatThresholdPolicy` keeps the one-number version around so a single-threshold
baseline stays measurable. `tests/test_policies.py` pins the difference down with
three reads of *identical* confidence that the ladder treats three different ways.

### Watching for the failure

`DecisionStats` and the confidence audit in every training log report the spread,
the escalation rate and the mean effective options. A useful signal shows up during
training: on the synthetic split, confidence climbs from 0.003 to 0.770 while the
escalation rate falls from 0.971 to 0.000 — the model learns, so the safety net
stops firing. Neither number is meaningful if the spread stays at zero.

### Transformer comparison mode

The byte-level pooled encoder is the production baseline. The optional Transformer
encoder is a controlled ablation for the demonstrated boundary failure, not a silent
replacement. Its projection from pooled hidden states is part of the checkpoint, so
training and inference reuse the same weights:

```powershell
python -m minijev.training --encoder-type transformer --num-layers 2 `
  --num-heads 4 --ffn-dim 128 --output models\minijev_transformer.safetensors `
  --metrics models\minijev_transformer_metrics.json
```

Keep the split, seed, policy and tokenizer settings fixed when comparing it with
`--encoder-type pooled`; report hard-case and cross-domain results separately.

## External data: MASSIVE

MASSIVE 1.1 `zh-CN` is downloaded under `data/external/massive/1.1/`. It annotates
*intents*, not routing requirements, so `minijev.convert` holds an explicit rubric:

```powershell
python -m minijev.convert audit   --data data\external\massive\1.1\data\zh-CN.jsonl
python -m minijev.convert convert --data data\external\massive\1.1\data\zh-CN.jsonl `
  --output-dir data\external\massive\derived_v2
```

**Read this before quoting any MASSIVE number.** The rubric is a pilot rubric, not
ground truth, and MASSIVE structurally cannot evidence all four routes:

| route | test rows | note |
|---|---:|---|
| local | 1,326 | device control, on-device writes, built-in knowledge |
| rag | 1,479 | external services, live or stored-data lookups |
| cloud | **0** | **MASSIVE contains no reasoning-heavy intent at all** |
| review | 169 | one intent (`general_quirky`), ambiguous requests |

So MASSIVE supports the `local`↔`rag` boundary and a thin `review` tail. Use
`--labels supported` (or `local,rag`) to report the two classes it can actually
evidence; a four-class macro average would be dragged down by a class with no
samples. `cloud` needs a different source — see `Open_Source_Datasets_Survey.md`.

The rubric version history is in the header comment of `minijev/convert.py`.
v1 sent every non-device intent to `rag` and used `cloud` for the three
`recommendation_*` intents; v2 fixes both and leaves `cloud` empty rather than
inventing samples for it.

## RouterBench integration

RouterBench records model quality, cost, latency and (when exported) availability; it does not contain MiniJev's four
route labels. The converter uses an explicit policy: sort models by cost, choose
the cheapest model whose quality reaches `--quality-threshold`, map the first,
second and later qualifying tiers to `local`, `rag` and `cloud`, and map a row
with no qualifying model to `review`. Every observed model is retained as a
`candidates` counterfactual with model, quality, cost, latency and availability. These labels are a
cost/quality experiment, not human ground truth, so the threshold is reported
with every run.

RouterBench is MIT licensed. Given a local JSONL, JSON, CSV or Parquet export:

```powershell
$env:PYTHONPATH = "D:\jev\tinygrad;D:\jev\MiniJev"
python -m minijev.convert routerbench --data path\routerbench.jsonl --audit --quality-threshold 0.8
python -m minijev.convert routerbench --data path\routerbench.jsonl `
  --output data\external\routerbench\routerbench_v1.jsonl --quality-threshold 0.8

# Build and audit a utility table for offline contextual-bandit work.
python -m minijev.convert routerbench --data path\routerbench.jsonl --audit `
  --latency-weight 0.1 --latency-scale-ms 1000 `
  --error-cost local=2 --error-cost cloud=5
```

The converter accepts nested `models`/`model_scores` records and wide columns
such as `model|performance`, `model|score`, `model|total_cost` and latency. It
refuses rows without a quality field rather than treating missing measurements
as a failed model. The raw RouterBench files are not committed here.

Eligibility is part of the counterfactual contract: a model below the quality bar
or explicitly unavailable is marked `eligible: false` and scores `-inf` in the
reward table, and a row where nothing qualifies gets a **synthesised `review`
candidate** carrying declared human-review cost/latency priors. Availability that
was not exported remains `null` and is reported as missing; it is never silently
treated as success or failure.

`minijev.rewards` computes `utility-v2`:

```text
reward = quality_weight * quality
       - cost_weight * cost / cost_scale
       - latency_weight * latency_ms / latency_scale_ms
       - error_cost(route) * (1 - quality)
```

The converter writes this reward into each candidate, records the full reward
configuration in row metadata, and `OfflineRouterEnv` reuses that configuration
unless the caller explicitly supplies another one. `--audit` adds finite-action,
availability and best-route coverage before an offline bandit run is attempted.
Human-review cost and latency remain declared priors in `minijev.convert.py`; they
must be replaced by measured values before a reward-trained policy is trusted.

## Literary data supplement

`python -m minijev.literary` creates a deterministic domain supplement under
`data/external/literary/`. It uses short traditional/classical excerpts as the
state and varies the requested operation: local extraction, provenance lookup
(`rag`), multi-step interpretation (`cloud`) and copyright/legal review
(`review`). The label belongs to the requested operation, not to the literary
work. Each row carries a `semantic_group_id`, work metadata and a provenance
note, so the normal group-aware splitter keeps paraphrases together:

```powershell
python -m minijev.literary --variants 3
python -m minijev.data split data\external\literary\literary_v1.jsonl `
  --output-dir data\external\literary\splits --seed 17
```

Each group carries up to three distinct phrasings; earlier versions emitted the
same text for variants two and three, which is pinned down by
`tests/test_literary.py`.

This is a controlled domain probe. Replace or augment it with licensed literary
queries and human labels before using it for product claims, especially for
copyright and other high-risk review decisions.

For a local UTF-8 copy of a licensed text such as *The Stranger*, generate the same
four operation labels without committing the novel to the repository:

```powershell
python -m minijev.literary --text-file path\the_stranger.txt `
  --title "局外人" --author "加缪" --chunk-size 420 `
  --output data\external\literary\the_stranger_v1.jsonl
```

The generator chunks non-empty paragraphs, records the block index and keeps every
block balanced across `local`, `rag`, `cloud` and `review`. These labels describe
the requested operation, so copyright and legal examples still need human review.

## Reinforcement learning

Policy-gradient training is still deliberately postponed. The reward data layer is
now usable and auditable, while the learning algorithm remains a later step:

- `minijev.objectives.policy_gradient_loss` — implemented and tested, called by
  nothing on the training path.
- `minijev.rewards` — pluggable `compute_reward`, a `RouterEnv` protocol, an
  `OfflineRouterEnv` that replays recorded rewards, and a `MockRouterEnv`.
- `minijev.schema.Candidate` — the counterfactual table. A sample can record what
  *every* route was worth, not just the one taken.
- `Decision.route_probs` and `Decision.log_prob()` — a policy-gradient step can
  sample a route and score it. The route head emits **logits, never a one-hot**.

RouterBench conversion now supplies the counterfactual table. Purely supervised
rows still carry only one `route` label, so they remain unsuitable for offline RL.
Use converted rows only after checking `reward_audit` and replacing placeholder
review and error-cost values with product measurements.

`MockRouterEnv` exists to make the trap concrete. Its reward depends only on the
route, never on the query, so any policy optimised against it collapses to the
single best route — the "routing collapse" failure mode from the literature. Call
`rewards.constant_route_check(env, queries)` before trusting an RL run; it returns
the collapsing route, or `None` when the environment can distinguish queries.

`RewardWeights.error_costs` adds the product-specific part of the utility:

```text
reward = quality_weight * quality
       - cost_weight * cost / cost_scale
       - latency_weight * latency_ms / latency_scale_ms
       - error_cost(route) * (1 - quality)
```

It is empty by default because route error costs are not technical constants. Set it
only after measuring real misroutes, and record the chosen mapping with the experiment.

The intended path is supervised → offline contextual bandit (LinUCB or Thompson
sampling) → policy gradient only if the bandit actually beats the supervised
baseline. Not PPO/GRPO first.

## Encoders: the Transformer is an ablation arm, not the default

`--encoder-type` selects `pooled` (default) or `transformer`. The rule for promoting
the Transformer was stated before the experiment: it must **stably beat pooled on the
boundary cases**. Measured under identical data, seed, policy and tokenizer:

| arm | params | synthetic test | literary test | hard cases (12) |
|---|---:|---:|---:|---:|
| pooled | 20,868 | 0.486 | 0.972 | 8/12 |
| pooled-640 *(parameter-matched control)* | 60,612 | 0.493 | 0.972 | 8/12 |
| transformer | 62,532 | **0.536** | **1.000** | 8/12 |

With MASSIVE mixed into training (12,425 rows) the Transformer's robustness gap is
much wider — pooled's literary score collapses from 0.972 to 0.528 while the
Transformer holds 0.944 — and on the synthetic test the Transformer reaches 0.693
against 0.493. **But on the boundary cases the Transformer does not win**: 8/12 vs
8/12 on the merged corpus, 5/12 vs 6/12 with MASSIVE. The promotion rule is therefore
**not satisfied, and `pooled` stays the default.**

Two things to carry forward:

- the parameter-matched control rules out the obvious confound. Widening pooled to
  60,612 parameters made validation macro-F1 *worse* (0.452 vs 0.467), so the
  Transformer's gain is not capacity;
- **12 boundary rows cannot decide this.** One row is 8.3 points, so "stably beats" is
  not measurable with the current instrument. Enlarge the hard test set to 60–100
  rows before re-running the comparison, and suspend the encoder question until then.

## Source layout

```text
MiniJev/
├── minijev/
│   ├── __init__.py     # narrow public facade: Router, Decision, and shapes
│   ├── config.py       # JevConfig and filesystem paths (pure data)
│   ├── schema.py       # Route, JevSample, Candidate, Decision  ← the data contract
│   ├── confidence.py   # distribution shape -> a scalar confidence
│   ├── tokenizer.py    # deterministic UTF-8 byte tokenizer
│   ├── model.py        # TinyJev (one head) + checkpoint save/load with a shape sidecar
│   ├── routers.py      # Router ABC + neural / rule / prior + registry + DecisionStats
│   ├── policies.py     # per-route risk ladder, flat baseline, caution sweep
│   ├── objectives.py   # supervised loss + the reserved RL loss
│   ├── rewards.py      # reward function, environments, collapse check
│   ├── data.py         # load / save / group-aware split / describe
│   ├── convert.py      # external sources -> schema, with rubrics
│   ├── synthetic.py    # deterministic generators (every template is a group)
│   ├── frames.py       # structured boundary frames; the route is derived, not stored
│   ├── boundary.py     # boundary-benchmark scoring, incl. contrast-pair collapse rate
│   ├── literary.py     # literary domain: built-in excerpts or a local text file
│   ├── training.py     # training entry point
│   ├── inference.py    # inference entry point (loads a checkpoint by default)
│   └── evaluation.py   # metrics, calibration audit, caution sweep
├── tests/              # pytest suite
├── data/               # JSONL datasets; external/ holds downloaded sources
└── models/             # checkpoints and metric reports; history/ is pre-refactor
```

Four rules keep the boundaries honest:

1. `model.py` does not know what a route means; it emits logits.
2. Confidence is computed only in `confidence.py`.
3. A route is chosen only in `policies.py`.
4. `training.py` and `inference.py` each depend on the shared modules and never on
   each other.

## Checkpoints

`safetensors` stores tensors and nothing else, so `save_checkpoint` also writes
`<name>.config.json` with the shapes. Without it, a checkpoint trained at a
non-default width cannot be loaded — the loader would build a network of the wrong
shape.

Checkpoints in `models/history/` predate both the sidecar and the removal of the
confidence head, so they no longer load. They are kept as evidence, not as weights.

`minijev.inference` loads weights by default and raises `CheckpointMissing` if they
are absent. It will not silently run random weights; pass `--allow-untrained`
if you only want to smoke-test the execution path.

## Code quality

```powershell
ruff check . ; ruff format --check . ; python -m pytest
```

Four-space indentation and a 150-character line limit, matching the surrounding
tinygrad repository.

## Next construction steps

1. **The boundary cases are now in the benchmark, and the pooled model fails them
   as designed.** The synthetic generator emits 28 hard-case pairs (the spec's §6.1
   same-keyword samples: `分析产品手册中的安装步骤` is rag while `分析这套系统为什么存在并发问题`
   is cloud). The model scores 0.493 argmax on the 20-group test, and on the hard
   pairs specifically 0.583 — it drags "分析..." reads towards cloud. This is the
   signal the spec asked for; the two-layer Transformer is now available as an
   isolated comparison mode, while pooled remains the default.
2. **Domain transfer does not happen.** A synthetic-trained model scores 0.306 on
   the literary test (vs 0.972 when trained on literary rows), and 0.182 on MASSIVE.
   Byte-level mean pooling learns surface form per domain; every new domain needs
   its own rows. The merged model (synthetic + literary) keeps 0.972 on literary
   and beats the synthetic-only model on the synthetic test (0.450 vs 0.393), so
   merging helps — collect data per domain and merge.
3. **Calibrate the ladder on real data.** The accept thresholds are reasoned
   defaults, not fitted values. `minijev.evaluation --sweep` produces the curve;
   what it needs is a validation set where the route actually matters.
4. Bring in the real RouterBench export and audit at 0.7 / 0.8 / 0.9 before
   composing the training set, as `data/external/routerbench/README.md` says.
   HF is unreachable from this machine; the fixture exercises the whole path.
5. Add the counterfactual table to a dataset so `OfflineRouterEnv` has something to
   replay, and try LinUCB against the supervised baseline — the RouterBench
   converter now emits a replayable candidates table, including the review
   synthesis, so this step only needs data.
6. Train both encoders on the same merged synthetic, MASSIVE and local-literary
   splits; compare hard cases, cross-domain accuracy and calibration before making
   any architectural default change.
