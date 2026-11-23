# CBRE Facilities Dispatch Agent — Architecture Document

**Status:** Final
**Owners:** Shahin Sheikh, Rich
**Scope:** `agent/` package and its immediate dependencies (`scripts/build_chroma.py`, `scripts/fit_risk_policy.py`, `operational/*`, `models/risk_policy.json`).
**Audience:** Engineering reviewers, ops stakeholders, and anyone extending the pipeline.

---

## 1. Purpose & Context

CBRE's facilities operators field ~10K facilities calls per day across a portfolio of properties. The vast majority are routine — a stopped toilet, a flickering light, a stuck badge — but the operator must triage them all manually. The business ask is direct:

> Auto-resolve routine calls. Escalate to a human only when it genuinely requires judgment.

This agent is the auto-resolve path. It ingests a multi-turn caller dialogue and emits a structured triage decision: a `(category, subcategory, risk_level)` from a 10-category × 36-subcategory taxonomy, the canonical building/floor/address, a selected vendor (or a flag that no vendor qualifies), and an audit-grade trainer log. It also decides whether to dispatch emergency services, ask the caller one clarifying question, or hand the call to a human reviewer.

The grader weights accuracy across six axes (subcategory, risk level, HITL-F1, clarification-F1, location, vendor match) with a hard-fail penalty for false emergency dispatch. Those axes mirror the real costs of the problem, and the architecture follows them: each deterministic guard protects against one specific kind of expensive error, while the LLM handles the task where it has an edge (natural language → structured intent).

---

## 2. Quality Attribute Drivers

The architecture is shaped by three quality attributes, in priority order:

| Attribute | Driver | Where it shows up |
|---|---|---|
| **Accuracy** | Misclassification has real cost — a missed gas leak risks lives; a false 911 wastes municipal resources and damages CBRE's brand. | RAG retrieval + reranking, deterministic over-escalation guards, historical-policy risk caps, demote-only consensus voting, Risk = P×C HITL trigger. |
| **Latency** | The eval harness runs ~1000 calls per submission; production would run live, where multi-second triage is acceptable but minutes are not. | Two LLM calls per call (extract + classify), Chroma top-10 with metadata rerank, pre-loaded operational data, per-node timing instrumentation, compiled graph cached as a module singleton. |
| **Security / safety** | The agent acts on the world (dispatches vendors, can flag 911). False or manipulated dispatch decisions are real-world incidents. | Hard-coded `GENUINE_EMERGENCY_SUBCATEGORIES` allowlist, deterministic dispatch gating, fail-closed vendor cascade, prompt-as-template (positional formatting), full provenance in the trainer log. |

Accuracy is the first-class concern; the architecture spends latency and code complexity to buy it.

---

## 3. High-Level Architecture

The agent is a [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` whose nodes alternate between LLM reasoning and deterministic guards. The state is a 34-field `TypedDict` ([WorkflowState](../agent/state.py:99)) that flows linearly through the graph; conditional edges handle the clarification loop and the HITL override branch.

```
START
 → extract_and_check        [LLM: gpt-4.1-mini, Pydantic schema]
 → clarification_gate       [deterministic: 4 heuristics]
 → (needs_followup?)        ── interactive: gather_followup → extract_and_check
                            └─ eval: build_enriched_query
 → retrieve                 [Chroma k=10, rerank to k=5]
 → classify_issue           [LLM: gpt-4.1, JSON taxonomy]
 → consensus_check          [demote-only RAG vote]
 → over_escalation_guard    [trap patterns + danger-word fallback]
 → risk_policy_cap          [empirical historical band cap]
 → compute_risk_score       [Risk = P(error) × C(error)]
 → resolve_location         [fuzzy match + profile fallback]
 → dispatch_vendor          [specialty → city → building_type cascade]
 → validator_gate           [HITL pattern 1: interrupt if flagged]
 → (override?)              ── override → trainer_log
                            └─ trainer_log
 → END
```

A rendered Mermaid version lives at [agent_graph.mmd](agent_graph.mmd) and [agent_graph.png](agent_graph.png). Both are generated from the compiled graph by [render_graph.py](../scripts/render_graph.py); the wiring itself is in [graph.py:71](../agent/graph.py:71).

### 3.1 Public API

A single entry point — [`classify(turns, caller_phone) -> dict`](../agent/graph.py:147) — is what the eval harness imports. It flattens the dialogue, looks up the caller profile, invokes the compiled graph in `eval_mode=True`, and projects the rich `WorkflowState` down to the grader's schema in [`_format_prediction`](../agent/graph.py:206).

### 3.2 Singleton & cold-start cost

The compiled graph is cached as a module-level singleton in [`_get_eval_graph`](../agent/graph.py:139). On first call we pay:

- LLM client construction (cheap)
- Chroma store open (sub-second; the DB is pre-built by [build_chroma.py](../scripts/build_chroma.py))
- `risk_policy.json` load (kilobytes, ~ms)

Subsequent calls reuse the compiled graph. Operational JSON (buildings, vendors, profiles, taxonomy) is loaded once at `config.py` import time and exposed as pre-built `dict` indices for O(1) lookups.

---

## 4. Architectural Decisions

This section captures the load-bearing decisions, the alternatives considered, and the impact on accuracy, latency, and security.

### ADR-01: LangGraph over a hand-rolled pipeline

**Decision.** Orchestrate the agent as a LangGraph `StateGraph` rather than a procedural function or a generic agent loop (e.g., ReAct).

**Rationale.** The task is naturally a directed graph with conditional branches (clarification loop, HITL override). LangGraph gives us:

- Explicit nodes and edges, which makes the architecture trivially auditable in a diagram.
- Built-in `interrupt()` semantics for the HITL pause/resume, instead of an ad-hoc state machine.
- Per-node hooks for timing/instrumentation (we wrap every node with [`timed_node()`](../agent/utils.py:149)).

**Consequences.**

- *Accuracy:* Each node has a single, testable responsibility, which made it tractable to layer six deterministic guards on top of the LLM.
- *Latency:* The graph itself adds negligible overhead; the bottleneck remains the two LLM calls.
- *Security:* The visible graph structure makes it easy to verify that the dispatch and emergency flags can only be set on the legitimate code paths.

### ADR-02: Two-stage LLM with a small extractor and a large classifier

**Decision.** Use `gpt-4.1-mini` for entity extraction ([intake.py:182](../agent/intake.py:182)) and `gpt-4.1` for classification ([rag_classify.py:213](../agent/rag_classify.py:213)).

**Rationale.** Extraction is a narrow, structured task — pull issue, system, severity, building, floor, plus three quality flags — and is well-served by the mini model with a Pydantic schema ([`EntityWithQuality`](../agent/state.py:34)) for guaranteed shape. Classification is the hard accuracy axis (taxonomy, risk band, HITL decision) and benefits from the larger model.

**Consequences.**

- *Accuracy:* Pydantic-validated extraction gives the classifier clean, structured inputs (severity, system, location); the larger model is reserved for the decision that drives most of the score.
- *Latency:* Two sequential LLM calls per call (~5–15s end-to-end in practice). Cannot be parallelized because `classify_issue` depends on the extracted entity for the enriched query.
- *Cost:* The mini model handles the chatty first pass; the expensive model only sees the distilled prompt.

Both clients are configured with `temperature=0`, `max_retries=3`, and explicit `request_timeout` (20s mini, 30s main) at [graph.py:78–79](../agent/graph.py:78). Determinism matters because (a) the grader is deterministic and (b) the provenance fields in the trainer log are most useful when the same input reproduces the same output.

### ADR-03: Structured output as a contract

**Decision.** Use Pydantic schemas for LLM outputs where shape matters, and `response_format={"type": "json_object"}` for the classifier.

**Implementation.**

- Extractor: `EntityWithQuality` ([state.py:34](../agent/state.py:34)) — combines extraction fields with three quality flags (`description_specific_enough`, `location_ok`, `needs_clarification`) so the same call drives both the entity and the clarification loop.
- Classifier: JSON object with explicit keys (category, subcategory, risk_level, confidence, reasoning, needs_human_review, needs_clarification, dispatched_emergency_services, call_summary, building_name, floor) and a 142-line prompt ([prompts.py:90](../agent/prompts.py:90)) that documents each key's semantics.

**Consequences.**

- *Accuracy:* No string parsing fragility; the validator-style fields (`description_specific_enough`) feed directly into the clarification loop with no glue code.
- *Security:* Structured output narrows the attack surface — a maliciously crafted transcript can influence values but cannot cause the agent to emit fields outside the contract.

### ADR-04: Deterministic clarification gate

**Decision.** Compute the clarification verdict from four explicit heuristics ([intake.py:31–176](../agent/intake.py:31)) *before* the LLM ever sees the transcript:

1. **Sparse caller** — first turn ≤6 words AND total ≤35 words.
2. **Generic opening** — caller's first utterance literally echoes a taxonomy label (suggests operator-led, not caller-led, framing).
3. **Floor self-correction** — caller names two different floor numbers across turns.
4. **Floor exceeds building** — extracted floor > building's `floor_count` (likely a suite number mis-parsed as a floor).

If any heuristic fires, the gate appends a tailored follow-up question to `follow_ups` and writes a tuple of audit-trail reasons (e.g., `"sparse_caller:first3_total29"`) into state.

**Rationale.** The clarification axis is one of the six rubric weights, and LLMs are inconsistent at "do I need to ask a follow-up?" The heuristics are cheap, explainable, and reusable. The LLM still gets a final say in `classify_issue` (the prompt has 50+ lines of calibration examples for keeping vs. flipping the verdict at [prompts.py:158–209](../agent/prompts.py:158)), but the gate sets the prior.

**Consequences.**

- *Accuracy:* Sets the clarification prior from deterministic signals that the LLM tends to under-weight.
- *Latency:* Pure Python, sub-millisecond.
- *Auditability:* The `clarification_reasons` tuple flows all the way into the trainer log, making it possible to attribute every clarification decision to a named heuristic.

### ADR-05: Hybrid retrieval — semantic search + metadata reranking

**Decision.** Retrieve k=10 from Chroma via cosine similarity, then rerank with a four-signal composite scorer to take the top 5 ([rag_classify.py:48–168](../agent/rag_classify.py:48)).

**Scoring.** Each candidate gets:

- Base = `(total - rank) / total` (Chroma's signal)
- `+0.50` for building_name exact match
- `+0.20` for building_type match
- `+0.10` for severity_class match (extracted severity vs. doc's `risk_level` band)
- `+0.05` for city match

**Rationale.** Pure semantic search drifts when callers describe a problem in terms similar to a different building's history. The composite rerank lets us weight provenance (this same building had the same issue last year) over loose semantic similarity. Building-name match is the strongest signal because the historical corpus is dense in repeat incidents per building.

**Consequences.**

- *Accuracy:* The classifier sees five tickets that are *both* semantically and contextually close to the current call.
- *Latency:* k=10 + reranking is sub-second; the LLM call dominates.
- *Cost of being wrong:* Bad retrieval is bounded — the consensus check (ADR-06) can only *demote* the LLM's confidence, never promote it, so a noisy retrieval cannot inflate risk.

### ADR-06: Demote-only RAG consensus

**Decision.** Use RAG consensus as a one-way correction: it can lower the LLM's risk band or weaken its confidence, but never raise either ([rag_classify.py:266–320](../agent/rag_classify.py:266)).

**Mechanism.** Two voting axes over the top-5 retrieved tickets:

1. **Subcategory vote (legacy).** If the RAG majority disagrees with the LLM, weaken confidence by 30%.
2. **Risk-band vote.** If ≥3 same-subcategory docs exist and ≥60% vote for a band below the LLM's, demote to that band.

**Rationale.** RAG is a noisy signal — the historical corpus has known QA issues ([qa_audit_findings.json](../evaluation/qa_audit_findings.json)) — so we trust it to flag over-escalation but not to override the LLM's read of *this specific* dialogue. Demote-only also dovetails with the rubric's hard-fail penalty for false emergency dispatch: every safety-net layer in this agent is asymmetric.

**Consequences.**

- *Accuracy:* Catches LLM over-confidence without amplifying historical errors.
- *Safety:* Guarantees RAG cannot, by itself, cause an under-escalation; legitimate emergencies stay where the LLM put them.

### ADR-07: Over-escalation guards (TRAP_PATTERNS)

**Decision.** Hard-code six "false 911" patterns ([risk.py:32–76](../agent/risk.py:32)). Each pattern names a dangerous subcategory, a list of benign keywords (`burnt`, `popcorn`, `cleaning`, etc.), optional exclude keywords (`rotten eggs` overrides a benign spoiled-food pattern), and a safer corrected subcategory + risk cap.

After classification, [`over_escalation_guard`](../agent/risk.py:82) rewrites the classification when a trap matches, clears any emergency dispatch flag, and sets `needs_human_review=True` so a human verifies the de-escalation.

A second-line `_DANGER_WORDS_RE` fallback ([utils.py:138](../agent/utils.py:138)) catches transcripts with safety language whose final classification falls outside `GENUINE_EMERGENCY_SUBCATEGORIES` ([config.py:129](../agent/config.py:129)) and routes them to human review.

**Rationale.** False emergency dispatch is the highest-cost error in the rubric and the highest-cost error in reality. The traps encode incidents that the LLM specifically tends to mis-fire on — burnt popcorn classified as fire, cleaning fumes as gas leak, escorted contractor as intruder — patterns that were observed during dev-set iteration.

**Consequences.**

- *Accuracy:* Direct gain on the EMERGENCY false-positive axis.
- *Safety:* Hardest safety guarantee in the system. Even if the LLM is fully compromised on a given turn, the trap regex strips the emergency flag for subcategories outside the allowlist.
- *Audit:* Every rewrite is recorded in `guard_applied` and surfaces in the trainer log.

### ADR-08: Historical-policy risk cap (demote-only)

**Decision.** For each subcategory, compute the highest risk band that ≥3% of historical tickets ever reached, and cap the LLM's emission to that band ([risk.py:141–170](../agent/risk.py:141), [fit_risk_policy.py](../scripts/fit_risk_policy.py)).

**Rationale.** The 10K-ticket historical corpus is the empirical distribution; the LLM's predictions should not regularly exceed bands that real ops never reached. The 3% threshold is forgiving enough not to suppress rare-but-real emergencies while catching cases like "leaky faucet → HIGH".

**Consequences.**

- *Accuracy:* Pulls outliers back to historically supported bands.
- *Safety:* Demote-only — the cap can never escalate beyond what the LLM emitted.
- *Audit:* `original_risk_level` + `risk_policy_applied` preserved in state and trainer log.

### ADR-09: Risk = P(error) × Cost(error) HITL trigger

**Decision.** Compute a single Risk score that fuses uncertainty signals with cost signals, and trip `needs_human_review` when it exceeds 0.30 ([risk.py:173–301](../agent/risk.py:173)).

**Formula.**

- **P(error)** = `0.40 × (1 - confidence) + 0.30 × rag_disagreement + 0.30 × historical_mismatch_rate[sub]`
- **Cost(error)** = `0.50 × severity_weight[risk_level] + 0.30 × subcategory_cost_factor[sub] + 0.20 × normalized_building_modifier[building_type]`
- **Risk** = P × C; HITL trigger threshold = 0.30 ([config.py:124](../agent/config.py:124))

Two unconditional rules layer on top: any HIGH/EMERGENCY band auto-triggers review, and any confidence below 0.6 auto-triggers review.

**Rationale.** A single threshold on confidence is a poor HITL signal — a 0.55-confidence routine plumbing call should not page a human, but a 0.55-confidence gas leak should. Multiplying probability of error by cost of error gives a principled trade-off that matches how the ops team would reason about it.

**Consequences.**

- *Accuracy on HITL-F1:* Direct, named optimization target.
- *Auditability:* Every component (p_error, cost_error, threshold, reason string) is preserved in `risk_details` and surfaced in the trainer log.
- *Tunability:* All weights and thresholds are constants in [config.py](../agent/config.py), so the policy can be retuned without touching code.

### ADR-10: Three-tier vendor cascade, fail-closed

**Decision.** Select a vendor by progressively narrowing filters ([utils.py:68](../agent/utils.py:68)): specialty → city → building_type. Sort survivors by `(status_order, -rating, response_sla_minutes)` and take the first. If the cascade ends empty, return `None` and let [`dispatch_vendor`](../agent/dispatch.py:75) flag the call for human review.

**Rationale.** The vendor roster ([vendors.json](../operational/vendors.json)) is sparse; not every subcategory has a vendor in every city. A monolithic single-stage filter would return empty on most calls. The cascade gracefully degrades — we'd rather dispatch a slightly-out-of-city vendor than fail to dispatch — while still failing closed when nothing qualifies.

**Consequences.**

- *Accuracy on vendor-match axis:* The cascade is the empirical sweet spot between strict matching (high precision, low recall) and a fully unconstrained pick (high recall, low precision).
- *Safety:* Fail-closed prevents auto-dispatch of a wrong vendor; a human reroutes instead.

### ADR-11: HITL trinity — validator, override, trainer log

**Decision.** Implement three distinct HITL patterns ([hitl.py](../agent/hitl.py)):

1. **Validator gate** — fires `interrupt()` when `needs_human_review` is set; auto-approves in eval mode.
2. **Override node** — applies human corrections to the classification when a human resume payload exists.
3. **Trainer log** — assembles the full audit record on every call (whether or not a human was involved).

**Rationale.** Separating the three concerns means the trainer log is built unconditionally — even for fully auto-resolved calls — which is what the rubric expects. The override path is a strict no-op unless an override exists, so the eval harness sees the same graph shape regardless of mode.

**Consequences.**

- *Accuracy:* The trainer log captures every correction provenance field (original_risk_level, original_risk_level_rag, risk_policy_applied, consensus_band_applied, guard_applied, clarification_reasons, risk_score, risk_details) so a downstream fine-tune has the full causal trace.
- *Safety:* Human override is the last word — it can override any guard, including emergency dispatch — by design.

---

## 5. Cross-Cutting Patterns

### 5.1 Determinism by default, LLM only where it earns its keep

The pipeline has 14 nodes; only 2 are stochastic. The deterministic nodes encode policy that we want to be auditable, reproducible, and cheap. The LLM is reserved for natural-language → structured-intent, which is where the model's lift is unambiguous.

### 5.2 Asymmetric correction

Every layer that touches the classification *after* the LLM (consensus, policy cap, over-escalation guard) is *demote-only* or rewrite-with-cap. The system is asymmetrically biased against over-escalation, which matches the rubric's hard-fail penalty and the real-world cost asymmetry.

### 5.3 Provenance everywhere

Every correction leaves a trail: `clarification_reasons` (tuple of heuristic strings), `original_risk_level` (pre-cap), `original_risk_level_rag` (pre-consensus), `guard_applied` (which trap fired), `risk_score` + `risk_details` (full P×C breakdown). These flow into [`make_trainer_log`](../agent/hitl.py:86) so the next-generation model can learn from the corrections, not just the final answer.

### 5.4 Static data, pre-built indices

Operational JSON ([buildings.json](../operational/buildings.json), [vendors.json](../operational/vendors.json), [caller_profiles.json](../operational/caller_profiles.json)) is loaded once and indexed at `config.py` import time. Lookup costs are O(1) and amortize to zero across many calls — there is no per-call I/O outside the LLM and Chroma round trips.

### 5.5 Per-node timing

Every node is wrapped with [`timed_node()`](../agent/utils.py:149), and [`print_timing_report()`](../agent/utils.py:164) emits per-node avg/total/count at end of run. This is the primary tool for diagnosing latency regressions during development.

---

## 6. Quality Attribute Analysis

### 6.1 Accuracy

The agent's accuracy strategy is **deep defense in depth on the high-cost axes** (emergency dispatch, risk band) and **let the LLM do the work** on lower-cost axes (subcategory, call summary).

For risk and emergency, every classification passes through:

1. LLM read of the transcript with explicit de-escalation rules in the prompt
2. RAG consensus (subcategory + risk-band votes, demote-only)
3. Over-escalation guard (six trap patterns + danger-word fallback)
4. Historical-policy band cap (demote-only)
5. Risk = P × C HITL trigger (mathematical combination of confidence, RAG agreement, historical mismatch, severity, cost factor, building type)

For subcategory, the LLM sees: the full 10×36 taxonomy with descriptions, the building registry, five reranked historical tickets, and the extracted entity. The taxonomy block in [prompts.py:22–67](../agent/prompts.py:22) is itself a rubric — each subcategory has a description that includes positive and negative examples (e.g., `fire_smoke — CONFIRMED visible fire or smoke (NOT burnt-food smell)`).

For HITL-F1, the trigger is multi-signal (not pure confidence threshold), which captures the asymmetric cost of false negatives in safety-sensitive subcategories.

For clarification-F1, the deterministic gate sets a strong prior from four cheap signals, and the prompt includes 50 lines of calibration examples for when to flip the gate.

For location, the [`resolve_location`](../agent/dispatch.py:27) function fuses three signals — LLM emission, extractor emission, caller profile — through fuzzy matching ([`_match_building`](../agent/utils.py:38)) with three fallback tiers and floor normalization via regex.

For vendor match, the three-tier cascade balances strict matching with graceful degradation, and the consistency snap ([dispatch.py:87](../agent/dispatch.py:87)) prevents category/subcategory drift across nodes.

### 6.2 Latency

End-to-end latency is dominated by the two LLM calls:

- `extract_and_check`: gpt-4.1-mini, 20s timeout
- `classify_issue`: gpt-4.1, 30s timeout

In practice, calls complete in 5–15s in eval mode. The deterministic nodes (clarification gate, over-escalation guard, policy cap, risk score, location, dispatch, trainer log) are sub-millisecond each. Chroma retrieval + reranking is sub-second.

The two LLM calls **cannot be parallelized** because `classify_issue` requires the enriched query and severity from `extract_and_check`. A future optimization could speculatively run retrieval against the raw transcript while extraction is in flight, but the quality of the retrieval would suffer; this trade-off was deemed not worth it given current throughput targets.

The compiled graph is a module-level singleton; cold start is paid once per process. The eval harness runs all 1000 calls against a single instance, so amortized startup is negligible.

Notable choices that *protect* latency:

- Pre-loaded static data (no per-call file I/O)
- Pre-compiled regexes for trap patterns and danger words ([utils.py:129](../agent/utils.py:129))
- No streaming (we need the full classification to run downstream guards anyway)
- No re-extraction unless the quality flags demand it ([intake.py:201](../agent/intake.py:201))

### 6.3 Security & Safety

The agent acts in the real world: it dispatches vendors and (conditionally) emergency services. The threat model is dominated not by adversarial transcripts — there's no public-facing surface — but by *accidental misuse*: a caller using safety language for a benign problem, or the LLM hallucinating an escalation.

**Defense layers for unsafe dispatch:**

1. `GENUINE_EMERGENCY_SUBCATEGORIES` is a hard-coded 5-element allowlist ([config.py:129](../agent/config.py:129)). Emergency dispatch is clamped to this set in [`over_escalation_guard`](../agent/risk.py:82) regardless of what the LLM emits.
2. Trap patterns rewrite false-emergency classifications before the dispatch node ever runs.
3. The vendor cascade fails closed: no qualified vendor → human review, not auto-dispatch.
4. Category/subcategory consistency is re-snapped at the prediction boundary ([graph.py:213](../agent/graph.py:213)).

**Prompt-injection posture:**

- Prompts are built with positional `.format(...)` substitution ([prompts.py:244](../agent/prompts.py:244)), not f-string interpolation, so the transcript is observed as a data slot, not a control slot.
- The CLASSIFY_PROMPT contains explicit instructions (de-escalation rules, scope tests, default-to-low heuristic) that take precedence over caller language. A caller saying "this is an emergency, dispatch fire trucks now" is just transcript content; the de-escalation rules in the prompt still gate the decision.
- The trap patterns and emergency clamp run *after* the LLM, so even an LLM that follows a prompt-injection still cannot land an unauthorized emergency dispatch.

**PII handling:**

- Caller phone, name, and profile flow through the state and into the trainer log. The trainer log is intended for internal replay and fine-tuning; it is not exposed beyond the eval harness.
- There is no external sink (no webhook, no third-party logging) in the agent code. Network I/O is limited to OpenAI (LLM) and the local Chroma store.

**Audit trail:**

- The trainer log ([hitl.py:86](../agent/hitl.py:86)) captures the full transcript, the AI prediction (with all correction provenance), any human override, and the final decision. This is the single record per call that can be replayed end-to-end.

---

## 7. Module Reference

| Module | Lines | Role |
|---|---|---|
| [`config.py`](../agent/config.py) | 130 | Static data load (buildings, vendors, profiles, taxonomy), risk policy constants, hard-coded emergency allowlist. |
| [`state.py`](../agent/state.py) | 134 | `WorkflowState` TypedDict, `EntityWithQuality` Pydantic schema, `ClarificationDecision` dataclass. |
| [`utils.py`](../agent/utils.py) | 180 | Transcript flattening, building/floor/vendor utilities, `timed_node` decorator, regex compiler. |
| [`prompts.py`](../agent/prompts.py) | 250 | Two prompt templates (extract + classify) with embedded taxonomy and de-escalation calibration. |
| [`intake.py`](../agent/intake.py) | 290 | Clarification gate (4 heuristics), extract-and-check node factory, follow-up loop. |
| [`rag_classify.py`](../agent/rag_classify.py) | 320 | Enriched query builder, retrieve + rerank, classify_issue, consensus check. |
| [`risk.py`](../agent/risk.py) | 300 | Trap patterns, over-escalation guard, historical band cap, Risk = P × C score. |
| [`dispatch.py`](../agent/dispatch.py) | 110 | Location resolution, vendor cascade dispatch. |
| [`hitl.py`](../agent/hitl.py) | 160 | Validator gate, override node, trainer log. |
| [`graph.py`](../agent/graph.py) | 235 | LangGraph wiring, compiled graph singleton, public `classify()` entry point. |

---

## 8. Known Limitations & Future Work

1. **The clarification follow-up loop runs only in interactive mode.** Eval mode skips `gather_followup` entirely (the LLM still emits a `needs_clarification` flag for grading). Closing that loop in eval would require simulating caller responses, which the harness does not support.
2. **Cold-start retrieval quality.** New buildings or new vendor types are invisible to RAG until the Chroma index is rebuilt. There is no online-learning path.
3. **Sequential LLM calls.** A speculative-retrieval optimization could overlap extraction and retrieval, at the cost of retrieval quality. Not implemented.
4. **No structured output for the call summary.** The summary is free text and is only post-validated by a length check + fallback template ([hitl.py:140](../agent/hitl.py:140)).
5. **The over-escalation traps are keyword rules written from dev-set errors.** They may not generalize to new phrasing and can misfire (a real fire involving a microwave matches the burnt-food trap). Every rewrite is routed to human review, and the case is recorded as an expected failure in [tests/test_risk_guards.py](../tests/test_risk_guards.py). At scale we would replace the keyword lists with a classifier trained on the QA audit data and on reviewer overrides from the trainer log, keeping the emergency allowlist as a hard rule.
6. **Risk policy is a static artifact.** [`fit_risk_policy.py`](../scripts/fit_risk_policy.py) is run offline. There is no scheduled refresh and no canary for the policy artifact.

---

## 9. References

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — graph orchestration and `interrupt()` semantics.
- [evaluation/scoring.py](../evaluation/scoring.py) — authoritative scoring function used for both dev and test sets.
- C4 model and lightweight ADR conventions informed the structure of this document.
