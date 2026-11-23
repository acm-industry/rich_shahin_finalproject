"""
CBRE Facilities Dispatch Agent — Human-in-the-Loop RAG Pipeline.

Public API (used by `evaluation/run_eval.py --agent agent:classify`):
    classify(turns, caller_phone) -> dict

The package is organized by design-doc requirement. See each module for its
specific deliverable mapping:

    config.py       - paths, taxonomy (10×36), policy constants, data loading
    state.py        - WorkflowState + Pydantic schemas
    utils.py        - shared helpers (transcript flatten, building/vendor
                      lookup, floor normalize, timing, regex compiler)
    prompts.py      - LLM prompt templates (EXTRACT, CLASSIFY) + blocks
    intake.py       - Inarticulate-caller + clarification deliverable:
                      extract_and_check, gather_followup, clarification_gate
    rag_classify.py - RAG + hierarchical classification deliverable:
                      build_enriched_query, retrieve, classify_issue,
                      consensus_check
    risk.py         - Risk-policy + over-escalation deliverable:
                      TRAP_PATTERNS, over_escalation_guard, risk_policy_cap,
                      compute_risk_score (Risk = P × C HITL trigger)
    dispatch.py     - Location-fields + vendor-match deliverable:
                      resolve_location, dispatch_vendor
    hitl.py         - HITL trinity (Validator + Override + Trainer):
                      validator_gate, override_node, make_trainer_log
    graph.py        - LangGraph wiring + the classify() entry point

For graph visualization, see `scripts/render_graph.py`.
"""
from .graph import _build_graph, _format_prediction, _get_eval_graph, classify
from .intake import make_extract_and_check
from .utils import _flatten, _lookup_profile

__all__ = [
    "classify",
    "_get_eval_graph",
    "_build_graph",
    "_flatten",
    "_format_prediction",
    "_lookup_profile",
    "make_extract_and_check",
]
