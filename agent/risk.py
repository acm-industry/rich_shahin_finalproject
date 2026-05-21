"""
Risk-band correction + HITL trigger nodes.

Maps to the brief's deliverables on:
  - **Risk-level assignment** (LOW/MEDIUM/HIGH/EMERGENCY) — the four nodes here
    are the layered correction chain that runs after the LLM emits a band.
  - **Over-escalation traps** + **false-911 hard cost** — over_escalation_guard
    uses keyword-based TRAP_PATTERNS to demote dangerous-sounding-but-benign
    classifications (burnt popcorn, cleaning fumes, empty elevator, etc.) and
    clamps dispatched_emergency_services on subcategories outside the
    canonical 5-emergency set.
  - **Per-subcategory base-risk table derived from history** — make_risk_policy_cap
    reads `models/risk_policy.json` (the historical empirical band distribution
    per subcategory) and caps the LLM's emission at the max-plausible band.
  - **Risk = P(error) × Cost(error) HITL trigger** (the brief's named formula) —
    make_compute_risk_score computes Risk as a deterministic blend of three
    P(error) signals and three Cost(error) factors, all clamped to [0,1], and
    fires `needs_human_review` additively when Risk >= RISK_HITL_THRESHOLD.
"""
from __future__ import annotations

from .config import (
    BUILDING_TYPE_COST_MODIFIER,
    GENUINE_EMERGENCY_SUBCATEGORIES,
    RISK_HITL_THRESHOLD,
    RISK_LEVEL_SEVERITY,
)
from .state import WorkflowState
from .utils import _DANGER_WORDS_RE, _compile_kw_regex


# ── Over-escalation trap patterns ────────────────────────────────────────────
# Each entry: when LLM picks the "dangerous" subcategory AND a keyword from
# `keywords` appears in the transcript (and no `exclude_keywords` overrides),
# rewrite to the safer `correction.subcategory` and cap risk at `max_risk`.
# Pre-compiled regexes (_kw_re, _exc_re) are attached at module load.

TRAP_PATTERNS = [
    {
        "dangerous": "fire_smoke",
        "keywords": ["burnt", "popcorn", "toast", "microwave", "food", "cooking", "burnt food", "smoke smell"],
        "correction": {"subcategory": "air_quality", "category": "HVAC", "max_risk": "LOW"},
    },
    {
        "dangerous": "gas_chemical",
        "keywords": ["cleaning", "wax", "paint", "filter", "chemical smell", "floor wax", "polish", "solvent"],
        "correction": {"subcategory": "air_quality", "category": "HVAC", "max_risk": "LOW"},
    },
    {
        "dangerous": "gas_chemical",
        "keywords": ["spoiled food", "rotten food", "garbage smell", "food smell", "trash smell", "decomposing food"],
        "exclude_keywords": ["rotten eggs", "gas", "propane", "natural gas", "evacuate", "sick", "headache", "911"],
        "correction": {"subcategory": "waste_odor", "category": "JANITORIAL", "max_risk": "LOW"},
    },
    {
        "dangerous": "entrapment",
        "keywords": ["doors open", "got out", "exited", "empty", "no one", "nobody", "already out", "walked out", "open doors"],
        "correction": {"subcategory": "malfunction", "category": "ELEVATOR", "max_risk": "MEDIUM"},
    },
    {
        "dangerous": "active_threat",
        "keywords": ["left", "escorted", "gone", "departed", "peaceful", "walked away", "no longer", "already left"],
        "correction": {"subcategory": "suspicious_person", "category": "SECURITY", "max_risk": "MEDIUM"},
    },
    {
        "dangerous": "structural",
        "keywords": ["tile", "ceiling tile", "one tile", "stain", "water spot", "single tile", "sagging tile"],
        "correction": {"subcategory": "roof_leak", "category": "PLUMBING", "max_risk": "MEDIUM"},
    },
]

# Compile each pattern's keyword + exclude regexes once at module load.
for _trap in TRAP_PATTERNS:
    _trap["_kw_re"] = _compile_kw_regex(_trap["keywords"])
    if _trap.get("exclude_keywords"):
        _trap["_exc_re"] = _compile_kw_regex(_trap["exclude_keywords"])


# ── Nodes ────────────────────────────────────────────────────────────────────


def over_escalation_guard(state: WorkflowState):
    """Keyword-driven over-escalation defense (the brief's false-911 protection).

    Three behaviors:
      1. TRAP_PATTERN match → rewrite subcategory + category, cap risk at the
         pattern's max_risk, clear dispatched_emergency_services, set
         needs_human_review=True so a human verifies the de-escalation.
      2. Danger-word fallback → if any of {fire, gas, trapped, threat,
         chemical, collapse, smoke, entrapment} appears in transcript AND
         final subcategory isn't in GENUINE_EMERGENCY_SUBCATEGORIES, flag for
         human review.
      3. Emergency-services clamp → if dispatched_emergency_services=True but
         subcategory isn't in GENUINE_EMERGENCY_SUBCATEGORIES or risk isn't
         HIGH/EMERGENCY, clear the dispatch flag (still routes for review).
    """
    classification = state["classification"]
    transcript = state["transcript"]
    subcategory = (classification.get("subcategory") or "").lower()

    for trap in TRAP_PATTERNS:
        if subcategory == trap["dangerous"]:
            # Check exclude keywords first — if any match, skip this trap.
            if trap.get("_exc_re") and trap["_exc_re"].search(transcript):
                continue
            if trap["_kw_re"].search(transcript):
                # Rewrite to the safer subcategory + cap risk at trap.max_risk.
                correction = trap["correction"]
                classification["subcategory"] = correction["subcategory"]
                classification["category"] = correction["category"]
                risk = (classification.get("risk_level") or "LOW").upper()
                max_risk = correction["max_risk"]
                risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "EMERGENCY": 3}
                if risk_order.get(risk, 0) > risk_order.get(max_risk, 0):
                    classification["risk_level"] = max_risk
                classification["dispatched_emergency_services"] = False
                classification["guard_applied"] = (
                    f"De-escalated from {trap['dangerous']} to {correction['subcategory']}"
                )
                # De-escalation from dangerous → human must verify.
                classification["needs_human_review"] = True
                break

    # Danger-word fallback for cases the explicit trap didn't catch.
    if _DANGER_WORDS_RE.search(transcript):
        final_sub = (classification.get("subcategory") or "").lower()
        if final_sub not in GENUINE_EMERGENCY_SUBCATEGORIES:
            classification["needs_human_review"] = True

    # Emergency-services clamp.
    subcategory = (classification.get("subcategory") or "").lower()
    if classification.get("dispatched_emergency_services"):
        if subcategory not in GENUINE_EMERGENCY_SUBCATEGORIES:
            classification["dispatched_emergency_services"] = False
        elif (classification.get("risk_level") or "").upper() not in ("HIGH", "EMERGENCY"):
            classification["dispatched_emergency_services"] = False

    return {"classification": classification}


def make_risk_policy_cap(policy: dict):
    """Per-subcategory historical band cap (demote-only).

    Each subcategory has a "max plausible band" derived from the empirical
    distribution of final_risk_level in historical_records.json. The cap is
    the highest band where empirical mass is at least `threshold_pct`
    (default 3% — i.e., a band has to have appeared on at least ~3% of that
    subcategory's historical tickets to count). When the LLM emits a band
    that exceeds the cap, demote to the cap. LLM under-estimates pass
    through unchanged.
    """
    caps = policy["caps"]
    band_ord = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "EMERGENCY": 3}

    def risk_policy_cap(state: WorkflowState):
        classification = state["classification"]
        sub = (classification.get("subcategory") or "").lower()
        cap = caps.get(sub, "EMERGENCY")
        llm = (classification.get("risk_level") or "MEDIUM").upper()

        if band_ord.get(llm, 1) > band_ord.get(cap, 3):
            classification["original_risk_level"] = llm
            classification["risk_level"] = cap
            classification["risk_policy_applied"] = (
                f"capped {llm}→{cap} (P[≥{llm}] < {policy.get('threshold_pct', 3)}% for {sub})"
            )

        return {"classification": classification}

    return risk_policy_cap


def make_compute_risk_score(policy: dict):
    """Risk = P(error) × Cost(error) HITL trigger (the brief's named formula).

    P(error) is a weighted blend of three deterministic signals:
      - 0.40 × (1 - llm_confidence)           [LLM's own uncertainty]
      - 0.30 × rag_disagreement_strength      [bands the RAG-consensus demoted by]
      - 0.30 × historical_mismatch_rate[sub]  [P(intake_band != final_band) per sub]

    Cost(error) is a weighted blend of three deterministic factors:
      - 0.50 × severity_weight[risk_level]                       [band severity]
      - 0.30 × subcategory_cost_factor[sub]                      [historical $ cost]
      - 0.20 × normalized_building_modifier[building_type]       [criticality]

    Both clamped to [0, 1]. Risk = P × C. Trigger needs_human_review when
    Risk >= RISK_HITL_THRESHOLD, ADDITIVELY (never overrides existing False).
    All existing hard rules preserved (band HIGH/EMERGENCY → review,
    confidence<0.6 → review).
    """
    mismatch_rates = policy.get("intake_vs_final_mismatch_rate", {})
    cost_factors = policy.get("subcategory_cost_factor", {})
    band_ord = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "EMERGENCY": 3}

    # Normalize the BUILDING_TYPE_COST_MODIFIER constants to [0, 1] range.
    # Min=0.7 (parking), Max=1.3 (medical). office (1.0) lands at 0.5.
    _bm_min = min(BUILDING_TYPE_COST_MODIFIER.values())
    _bm_max = max(BUILDING_TYPE_COST_MODIFIER.values())
    _bm_range = _bm_max - _bm_min if _bm_max > _bm_min else 1.0

    def compute_risk_score(state: WorkflowState):
        classification = state["classification"]

        # ── P(error) components ──
        confidence = classification.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence_uncertainty = max(0.0, min(1.0, 1.0 - confidence))

        # RAG-band disagreement strength = bands the consensus demoted by.
        # consensus_check sets original_risk_level_rag only when it demoted.
        rag_disagreement = 0.0
        original_rag = classification.get("original_risk_level_rag")
        if original_rag:
            current_band = (classification.get("risk_level") or "MEDIUM").upper()
            diff = band_ord.get(original_rag.upper(), 1) - band_ord.get(current_band, 1)
            rag_disagreement = min(1.0, max(0.0, diff / 3.0))

        subcat = (classification.get("subcategory") or "").lower()
        historical_mismatch = mismatch_rates.get(subcat, 0.0)

        # P(error) — weighted blend of independent signals
        p_error = (
            0.40 * confidence_uncertainty +
            0.30 * rag_disagreement +
            0.30 * historical_mismatch
        )
        p_error = round(min(1.0, max(0.0, p_error)), 4)

        # ── Cost(error) components ──
        risk_level = (classification.get("risk_level") or "MEDIUM").upper()
        severity_weight = RISK_LEVEL_SEVERITY.get(risk_level, 0.4)

        sub_cost = cost_factors.get(subcat, 0.5)

        building_type = (state.get("_building_type") or "").lower()
        bm_raw = BUILDING_TYPE_COST_MODIFIER.get(building_type, 1.0)
        building_modifier_norm = (bm_raw - _bm_min) / _bm_range  # [0, 1]

        cost_error = (
            0.50 * severity_weight +
            0.30 * sub_cost +
            0.20 * building_modifier_norm
        )
        cost_error = round(min(1.0, max(0.0, cost_error)), 4)

        # ── Risk = P × C ──
        risk = round(p_error * cost_error, 4)

        # ── Preserve existing HITL behavior ──
        needs_human = bool(classification.get("needs_human_review", False))
        needs_clarification = state.get("needs_clarification", False) or bool(
            classification.get("needs_clarification", False)
        )

        # Hard rules (kept as-is — additive with the new Risk trigger):
        if risk_level in ("HIGH", "EMERGENCY"):
            needs_human = True
        if confidence < 0.6:
            needs_human = True

        # Risk-based additive trigger.
        if risk >= RISK_HITL_THRESHOLD:
            needs_human = True

        # Build a compact reason string for trainer-log auditability.
        reason_parts = []
        if risk >= RISK_HITL_THRESHOLD:
            reason_parts.append(
                f"Risk={risk:.2f} (P_err={p_error:.2f} × Cost={cost_error:.2f})"
            )
        if risk_level in ("HIGH", "EMERGENCY"):
            reason_parts.append(f"band={risk_level}")
        if confidence < 0.6:
            reason_parts.append(f"low conf {confidence:.2f}")

        return {
            "risk_score": risk,
            "risk_details": {
                "p_error": p_error,
                "cost_error": cost_error,
                "p_error_components": {
                    "confidence_uncertainty": round(confidence_uncertainty, 4),
                    "rag_disagreement": round(rag_disagreement, 4),
                    "historical_mismatch": round(historical_mismatch, 4),
                },
                "cost_error_components": {
                    "severity_weight": severity_weight,
                    "sub_cost_factor": sub_cost,
                    "building_modifier_norm": round(building_modifier_norm, 4),
                },
                "threshold": RISK_HITL_THRESHOLD,
                "reason": " | ".join(reason_parts) if reason_parts else "within normal parameters",
            },
            "needs_human_review": needs_human,
            "needs_clarification": needs_clarification,
        }

    return compute_risk_score
