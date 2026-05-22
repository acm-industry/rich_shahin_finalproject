"""
Human-in-the-loop trinity nodes.

Maps to the brief's three required HITL patterns:
  - **Pattern 1: Validator** — validator_gate evaluates `needs_human_review`
    and either auto-routes (eval mode / not flagged) or pauses for a human
    reviewer via interrupt() (interactive mode, flagged).
  - **Pattern 2: Override** — override_node applies human corrections (from
    the validator_gate interrupt payload) to the classification before it
    proceeds to logging.
  - **Pattern 3: Trainer** — make_trainer_log emits the per-call record with
    full_transcript + ai_prediction + human_override + final_decision so the
    log is replayable for prompt tuning / fine-tuning.
"""
from __future__ import annotations

from langgraph.types import interrupt

from .state import WorkflowState


def validator_gate(state: WorkflowState):
    """HITL validator gate.

    In eval mode (batch grading), auto-approves and just builds a
    routing_decision string for the trainer log.

    In interactive mode (live operator workflow), if `needs_human_review` is
    True, fires interrupt() with the classification + risk context so the
    reviewer can approve, override, or reject. Resume semantics are handled
    by LangGraph's Command(resume=...) downstream.
    """
    if state.get("eval_mode"):
        risk_score = state.get("risk_score", 0.0)
        classification = state["classification"]
        cat = classification.get("category", "")
        sub = classification.get("subcategory", "")
        if state.get("needs_human_review"):
            routing = f"Flagged for review → {cat}/{sub} (risk {risk_score:.2f})"
        else:
            routing = f"Auto-routed → {cat}/{sub} (risk {risk_score:.2f})"
        return {"routing_decision": routing}

    # Interactive mode — interrupt for human review
    if state.get("needs_human_review"):
        decision = interrupt({
            "type": "review_required",
            "reason": state.get("risk_details", {}).get("reason", "elevated risk"),
            "risk_score": state.get("risk_score"),
            "classification": state["classification"],
            "risk_details": state.get("risk_details"),
        })

        if decision.get("override_code"):
            return {
                "human_override": decision,
                "routing_decision": f"Human override → {decision.get('override_code')}",
            }
        else:
            return {"routing_decision": f"Human approved → {state['classification'].get('subcategory')}"}

    classification = state["classification"]
    return {
        "routing_decision": f"Auto-routed → {classification.get('category')}/{classification.get('subcategory')} (risk {state.get('risk_score', 0.0):.2f})",
    }


def override_node(state: WorkflowState):
    """Apply human corrections from validator_gate to the classification."""
    human_override = state.get("human_override")
    if not human_override:
        return {}

    classification = state["classification"].copy()
    if human_override.get("category"):
        classification["category"] = human_override["category"]
    if human_override.get("subcategory"):
        classification["subcategory"] = human_override["subcategory"]
    if human_override.get("risk_level"):
        classification["risk_level"] = human_override["risk_level"]
    classification["human_reviewed"] = True

    return {"classification": classification}


def make_trainer_log(state: WorkflowState):
    """Build the per-call trainer log + ensure call_summary is non-trivial.

    Surfaces the full ai_prediction (including every correction provenance
    field: original_risk_level from the cap, original_risk_level_rag from
    consensus, clarification_reasons from the gate, risk_score + risk_details
    from the P × C trigger) alongside any human_override and the
    final_decision (AI prediction with overrides applied).
    """
    classification = state["classification"]
    category = (classification.get("category") or "").upper()
    subcategory = (classification.get("subcategory") or "").lower()
    risk_level = (classification.get("risk_level") or "MEDIUM").upper()

    if risk_level not in ("LOW", "MEDIUM", "HIGH", "EMERGENCY"):
        risk_level = "MEDIUM"

    building_name = state.get("building_name") or classification.get("building_name")
    floor = state.get("floor")
    address = state.get("address")
    vendor_id = state.get("vendor_id")
    needs_human = state.get("needs_human_review", False)
    needs_clarification = state.get("needs_clarification", False)
    emergency = bool(classification.get("dispatched_emergency_services", False))

    # Call summary with fallback (the brief's 2-3 sentence operator-style narrative)
    call_summary = (classification.get("call_summary") or "").strip()
    if len(call_summary) < 30:
        call_summary = (
            f"Caller reported {subcategory.replace('_', ' ')} at "
            f"{building_name or 'unknown building'}, {floor or 'unknown floor'}. "
            f"Classified {category}/{subcategory}, risk {risk_level}."
        )

    ai_prediction = {
        "category": category,
        "subcategory": subcategory,
        "risk_level": risk_level,
        "building_name": building_name,
        "floor": floor,
        "vendor_id": vendor_id,
        "needs_human_review": needs_human,
        "needs_clarification": needs_clarification,
        "dispatched_emergency_services": emergency,
        "original_risk_level": classification.get("original_risk_level"),
        "risk_policy_applied": classification.get("risk_policy_applied"),
        "original_risk_level_rag": classification.get("original_risk_level_rag"),
        "consensus_band_applied": classification.get("consensus_band_applied"),
        "risk_score": state.get("risk_score"),
        "risk_details": state.get("risk_details"),
        "clarification_reasons": list(state.get("clarification_reasons") or ()),
    }

    human_override = None
    if state.get("human_override"):
        human_override = state["human_override"]

    final_decision = ai_prediction.copy()
    if human_override:
        for k, v in human_override.items():
            if k in final_decision and v is not None:
                final_decision[k] = v

    trainer_log = {
        "full_transcript": state["transcript"],
        "ai_prediction": ai_prediction,
        "human_override": human_override,
        "final_decision": final_decision,
    }

    return {
        "trainer_log": trainer_log,
        "call_summary": call_summary,
        "dispatched_emergency_services": emergency,
    }
