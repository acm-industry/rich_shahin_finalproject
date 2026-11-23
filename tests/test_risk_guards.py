"""Unit tests for the deterministic risk layer in agent/risk.py.

These nodes run after the LLM and are pure functions of graph state, so they
are tested without any API calls: build a minimal state, run the node, check
the result.
"""
import pytest

from agent.config import GENUINE_EMERGENCY_SUBCATEGORIES, RISK_HITL_THRESHOLD
from agent.risk import (
    TRAP_PATTERNS,
    make_compute_risk_score,
    make_risk_policy_cap,
    over_escalation_guard,
)


def _state(transcript, **classification):
    return {"transcript": transcript, "classification": dict(classification)}


def _guard(transcript, **classification):
    return over_escalation_guard(_state(transcript, **classification))["classification"]


# ── over_escalation_guard: trap patterns ─────────────────────────────────────


def test_burnt_popcorn_is_not_a_fire():
    out = _guard(
        "[CALLER] Someone burnt popcorn in the breakroom and it smells awful.",
        category="LIFE_SAFETY",
        subcategory="fire_smoke",
        risk_level="EMERGENCY",
        dispatched_emergency_services=True,
    )
    assert out["subcategory"] == "air_quality"
    assert out["category"] == "HVAC"
    assert out["risk_level"] == "LOW"
    assert out["dispatched_emergency_services"] is False
    assert "fire_smoke" in out["guard_applied"]


def test_trap_deescalation_always_routes_to_a_human():
    """A rewrite away from a dangerous label must be verified by a person."""
    out = _guard(
        "[CALLER] The elevator stopped but the doors open fine and it's empty.",
        category="ELEVATOR",
        subcategory="entrapment",
        risk_level="HIGH",
        needs_human_review=False,
    )
    assert out["subcategory"] == "malfunction"
    assert out["risk_level"] == "MEDIUM"
    assert out["needs_human_review"] is True


def test_trap_caps_risk_but_never_raises_it():
    out = _guard(
        "[CALLER] The man was escorted out and is gone now.",
        category="SECURITY",
        subcategory="active_threat",
        risk_level="LOW",
    )
    assert out["subcategory"] == "suspicious_person"
    assert out["risk_level"] == "LOW"  # cap is MEDIUM; LOW must stay LOW


def test_genuine_fire_is_left_alone():
    out = _guard(
        "[CALLER] There are flames coming out of the electrical closet!",
        category="LIFE_SAFETY",
        subcategory="fire_smoke",
        risk_level="EMERGENCY",
        dispatched_emergency_services=True,
    )
    assert out["subcategory"] == "fire_smoke"
    assert out["risk_level"] == "EMERGENCY"
    assert out["dispatched_emergency_services"] is True
    assert "guard_applied" not in out


def test_exclude_keywords_block_the_spoiled_food_trap():
    """'rotten eggs' is the classic natural-gas description, not a food smell."""
    out = _guard(
        "[CALLER] There's a garbage smell, kind of like rotten eggs, by the boiler.",
        category="LIFE_SAFETY",
        subcategory="gas_chemical",
        risk_level="EMERGENCY",
        dispatched_emergency_services=True,
    )
    assert out["subcategory"] == "gas_chemical"
    assert out["dispatched_emergency_services"] is True


def test_spoiled_food_without_gas_language_is_a_janitorial_odor():
    out = _guard(
        "[CALLER] There's a rotten food odor coming from the fridge on 4.",
        category="LIFE_SAFETY",
        subcategory="gas_chemical",
        risk_level="HIGH",
    )
    assert out["subcategory"] == "waste_odor"
    assert out["category"] == "JANITORIAL"
    assert out["risk_level"] == "LOW"


def test_trap_keywords_match_whole_words_only():
    """'tile' must not fire on 'textile'; 'left' must not fire on 'cleft'."""
    out = _guard(
        "[CALLER] The wall of the textile showroom is cracking and bowing out.",
        category="LIFE_SAFETY",
        subcategory="structural",
        risk_level="EMERGENCY",
    )
    assert out["subcategory"] == "structural"


def test_traps_only_apply_to_their_own_dangerous_subcategory():
    out = _guard(
        "[CALLER] The microwave in the kitchen stopped heating food.",
        category="PEST_SPECIALTY",
        subcategory="appliance_kitchen",
        risk_level="LOW",
    )
    assert out["subcategory"] == "appliance_kitchen"
    assert "guard_applied" not in out


@pytest.mark.parametrize("trap", TRAP_PATTERNS, ids=lambda t: f"{t['dangerous']}->{t['correction']['subcategory']}")
def test_every_trap_corrects_to_a_non_emergency_subcategory(trap):
    """Invariant: a trap can never move a call *into* the 911-eligible set."""
    assert trap["dangerous"] in GENUINE_EMERGENCY_SUBCATEGORIES
    assert trap["correction"]["subcategory"] not in GENUINE_EMERGENCY_SUBCATEGORIES
    assert trap["correction"]["max_risk"] in ("LOW", "MEDIUM")


@pytest.mark.xfail(
    strict=True,
    reason="Known limitation: keyword traps cannot tell a burnt-food smell from "
    "a real fire that happens to involve a microwave. The call is still routed "
    "to a human, but the label and risk are wrongly demoted. See README > Limitations.",
)
def test_real_microwave_fire_should_not_be_deescalated():
    out = _guard(
        "[CALLER] The microwave is on fire, there are flames and the room is filling with smoke!",
        category="LIFE_SAFETY",
        subcategory="fire_smoke",
        risk_level="EMERGENCY",
        dispatched_emergency_services=True,
    )
    assert out["subcategory"] == "fire_smoke"


# ── over_escalation_guard: emergency-dispatch allowlist ──────────────────────


@pytest.mark.parametrize("subcategory", ["pipe_leak", "lighting", "waste_odor", "power_outage"])
def test_emergency_dispatch_is_cleared_outside_the_allowlist(subcategory):
    out = _guard(
        "[CALLER] Please send someone right away.",
        subcategory=subcategory,
        risk_level="EMERGENCY",
        dispatched_emergency_services=True,
    )
    assert out["dispatched_emergency_services"] is False


@pytest.mark.parametrize("risk", ["LOW", "MEDIUM"])
def test_emergency_dispatch_requires_high_or_emergency_band(risk):
    out = _guard(
        "[CALLER] I think I smell gas near the loading dock.",
        subcategory="gas_chemical",
        risk_level=risk,
        dispatched_emergency_services=True,
    )
    assert out["dispatched_emergency_services"] is False


def test_guard_never_turns_emergency_dispatch_on():
    out = _guard(
        "[CALLER] There is a fire in the lobby!",
        subcategory="fire_smoke",
        risk_level="EMERGENCY",
        dispatched_emergency_services=False,
    )
    assert out["dispatched_emergency_services"] is False


def test_danger_words_on_a_routine_label_force_human_review():
    out = _guard(
        "[CALLER] There's smoke coming from the vent, I think.",
        subcategory="air_quality",
        risk_level="LOW",
        needs_human_review=False,
    )
    assert out["needs_human_review"] is True


def test_no_danger_words_leaves_review_flag_untouched():
    out = _guard(
        "[CALLER] The restroom on Floor 3 is out of paper towels.",
        subcategory="restroom_supplies",
        risk_level="LOW",
        needs_human_review=False,
    )
    assert out["needs_human_review"] is False


# ── risk_policy_cap ──────────────────────────────────────────────────────────

POLICY = {
    "threshold_pct": 3,
    "caps": {"restroom_fixture": "LOW", "roof_leak": "MEDIUM", "fire_smoke": "EMERGENCY"},
    "intake_vs_final_mismatch_rate": {"roof_leak": 0.5},
    "subcategory_cost_factor": {"roof_leak": 0.5, "fire_smoke": 1.0},
}


def _cap(**classification):
    node = make_risk_policy_cap(POLICY)
    return node({"classification": dict(classification)})["classification"]


def test_cap_demotes_band_above_historical_ceiling():
    out = _cap(subcategory="restroom_fixture", risk_level="HIGH")
    assert out["risk_level"] == "LOW"
    assert out["original_risk_level"] == "HIGH"
    assert "capped HIGH→LOW" in out["risk_policy_applied"]


@pytest.mark.parametrize("llm_band", ["LOW", "MEDIUM"])
def test_cap_is_demote_only(llm_band):
    out = _cap(subcategory="roof_leak", risk_level=llm_band)
    assert out["risk_level"] == llm_band
    assert "original_risk_level" not in out


def test_cap_does_not_touch_unknown_subcategories():
    out = _cap(subcategory="not_in_policy", risk_level="EMERGENCY")
    assert out["risk_level"] == "EMERGENCY"


# ── compute_risk_score: Risk = P(error) × Cost(error) ────────────────────────


def _score(building_type="office", **classification):
    node = make_compute_risk_score(POLICY)
    return node({"classification": dict(classification), "_building_type": building_type})


def test_risk_is_product_of_p_error_and_cost():
    out = _score(subcategory="roof_leak", risk_level="MEDIUM", confidence=0.8)
    d = out["risk_details"]
    # P = 0.4*(1-0.8) + 0.3*0 + 0.3*0.5 ; C = 0.5*0.4 + 0.3*0.5 + 0.2*0.5
    assert d["p_error"] == pytest.approx(0.23)
    assert d["cost_error"] == pytest.approx(0.45)
    assert out["risk_score"] == pytest.approx(0.23 * 0.45, abs=1e-4)


def test_confident_routine_call_is_auto_resolved():
    out = _score(subcategory="restroom_fixture", risk_level="LOW", confidence=0.95)
    assert out["risk_score"] < RISK_HITL_THRESHOLD
    assert out["needs_human_review"] is False
    assert out["risk_details"]["reason"] == "within normal parameters"


@pytest.mark.parametrize("band", ["HIGH", "EMERGENCY"])
def test_high_bands_always_go_to_a_human(band):
    out = _score(subcategory="fire_smoke", risk_level=band, confidence=0.99)
    assert out["needs_human_review"] is True
    assert f"band={band}" in out["risk_details"]["reason"]


def test_low_confidence_always_goes_to_a_human():
    out = _score(subcategory="restroom_fixture", risk_level="LOW", confidence=0.55)
    assert out["needs_human_review"] is True


def test_score_never_clears_an_existing_review_flag():
    out = _score(
        subcategory="restroom_fixture", risk_level="LOW", confidence=0.95, needs_human_review=True
    )
    assert out["needs_human_review"] is True


def test_rag_demotion_raises_p_error():
    base = _score(subcategory="roof_leak", risk_level="LOW", confidence=0.8)
    demoted = _score(
        subcategory="roof_leak", risk_level="LOW", confidence=0.8, original_risk_level_rag="EMERGENCY"
    )
    assert demoted["risk_details"]["p_error"] > base["risk_details"]["p_error"]
    assert demoted["risk_details"]["p_error_components"]["rag_disagreement"] == pytest.approx(1.0)


def test_same_call_costs_more_in_a_hospital_than_a_parking_garage():
    hospital = _score("medical", subcategory="roof_leak", risk_level="MEDIUM", confidence=0.8)
    garage = _score("parking", subcategory="roof_leak", risk_level="MEDIUM", confidence=0.8)
    assert hospital["risk_details"]["cost_error"] > garage["risk_details"]["cost_error"]
    assert hospital["risk_score"] > garage["risk_score"]


def test_malformed_confidence_is_treated_as_uncertain():
    out = _score(subcategory="restroom_fixture", risk_level="LOW", confidence="very sure")
    assert out["risk_details"]["p_error_components"]["confidence_uncertainty"] == pytest.approx(0.5)
    assert out["needs_human_review"] is True
