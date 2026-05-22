"""
LangGraph wiring + public entry point.

This module knows about every node and conditional edge; everything below
only knows about its own slice. The wiring order is:

  START
    → extract_and_check
    → clarification_gate                      (intake — clarification axis)
    → (gather_followup loop OR build_enriched_query)
        gather_followup → extract_and_check   (interactive mode loop)
    → retrieve                                (RAG — k=5 historical tickets)
    → classify_issue                          (LLM classification, 10×36 taxonomy)
    → consensus_check                         (RAG sub + band consensus)
    → over_escalation_guard                   (false-911 protection)
    → risk_policy_cap                         (historical band cap)
    → compute_risk_score                      (Risk = P × C HITL trigger)
    → resolve_location                        (canonical building + floor)
    → dispatch_vendor                         (specialty → city → type cascade)
    → validator_gate                          (HITL pattern 1)
    → (override → trainer_log) OR trainer_log (HITL pattern 2)
    → END                                     (HITL pattern 3: trainer log)
"""
from __future__ import annotations

import json
from typing import Literal, Optional

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph

from .config import CHROMA_DIR, HERE, MODELS, SUBCATEGORY_TO_CATEGORY
from .dispatch import dispatch_vendor, resolve_location
from .hitl import make_trainer_log, override_node, validator_gate
from .intake import clarification_gate, gather_followup, make_extract_and_check
from .rag_classify import build_enriched_query, consensus_check, make_classify_issue, make_retrieve
from .risk import make_compute_risk_score, make_risk_policy_cap, over_escalation_guard
from .state import WorkflowState
from .utils import _flatten, _lookup_profile, timed_node


# ── Conditional edge functions ───────────────────────────────────────────────


def needs_followup(state: WorkflowState) -> Literal["gather_followup", "build_enriched_query"]:
    """Fired AFTER clarification_gate. In interactive mode, route to gather_followup
    when any pending follow_ups exist (either from extract_and_check's LLM extractor
    or from the clarification_gate's appended question). In eval mode, skip directly
    to build_enriched_query — the gate's verdict still drives the needs_clarification
    flag via the LLM review in classify_issue.
    """
    if state.get("eval_mode"):
        return "build_enriched_query"
    if state.get("extraction_complete"):
        return "build_enriched_query"
    if state.get("follow_ups"):
        return "gather_followup"
    return "build_enriched_query"


def override_route(state: WorkflowState) -> Literal["override", "trainer_log"]:
    if state.get("human_override"):
        return "override"
    return "trainer_log"


# ── Graph builder ────────────────────────────────────────────────────────────


def _build_graph(eval_mode: bool = True):
    """Construct and compile the LangGraph pipeline.

    Loads the LLM clients, the embedding model, the Chroma vector store, and
    the historical risk-policy artifact once at build time. The compiled
    graph is then cached as a module-level singleton (see `_get_eval_graph`).
    """
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, max_retries=3, request_timeout=30)
    llm_mini = ChatOpenAI(model="gpt-4.1-mini", temperature=0, max_retries=3, request_timeout=20)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        collection_name="historical_records",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    risk_policy = json.loads((MODELS / "risk_policy.json").read_text())

    workflow = StateGraph(WorkflowState)

    # Add nodes (timed for the per-node latency report)
    workflow.add_node("extract_and_check", timed_node("extract_and_check", make_extract_and_check(llm_mini)))
    workflow.add_node("gather_followup", gather_followup)
    workflow.add_node("clarification_gate", timed_node("clarification_gate", clarification_gate))
    workflow.add_node("build_enriched_query", timed_node("build_enriched_query", build_enriched_query))
    workflow.add_node("retrieve", timed_node("retrieve", make_retrieve(vector_store)))
    workflow.add_node("classify_issue", timed_node("classify_issue", make_classify_issue(llm)))
    workflow.add_node("consensus_check", timed_node("consensus_check", consensus_check))
    workflow.add_node("over_escalation_guard", timed_node("over_escalation_guard", over_escalation_guard))
    workflow.add_node("risk_policy_cap", timed_node("risk_policy_cap", make_risk_policy_cap(risk_policy)))
    workflow.add_node("compute_risk_score", timed_node("compute_risk_score", make_compute_risk_score(risk_policy)))
    workflow.add_node("resolve_location", timed_node("resolve_location", resolve_location))
    workflow.add_node("dispatch_vendor", timed_node("dispatch_vendor", dispatch_vendor))
    workflow.add_node("validator_gate", timed_node("validator_gate", validator_gate))
    workflow.add_node("override", timed_node("override", override_node))
    workflow.add_node("trainer_log", timed_node("trainer_log", make_trainer_log))

    # Edges
    # Flow: extract_and_check → clarification_gate → (gather_followup loop OR build_enriched_query)
    # Running clarification_gate BEFORE the gather conditional means its appended
    # question is available to gather_followup in interactive mode.
    workflow.add_edge(START, "extract_and_check")
    workflow.add_edge("extract_and_check", "clarification_gate")
    workflow.add_conditional_edges("clarification_gate", needs_followup)
    workflow.add_edge("gather_followup", "extract_and_check")
    workflow.add_edge("build_enriched_query", "retrieve")
    workflow.add_edge("retrieve", "classify_issue")
    workflow.add_edge("classify_issue", "consensus_check")
    workflow.add_edge("consensus_check", "over_escalation_guard")
    workflow.add_edge("over_escalation_guard", "risk_policy_cap")
    workflow.add_edge("risk_policy_cap", "compute_risk_score")
    workflow.add_edge("compute_risk_score", "resolve_location")
    workflow.add_edge("resolve_location", "dispatch_vendor")
    workflow.add_edge("dispatch_vendor", "validator_gate")
    workflow.add_conditional_edges("validator_gate", override_route)
    workflow.add_edge("override", "trainer_log")
    workflow.add_edge("trainer_log", END)

    return workflow.compile()


# ── Entry points ─────────────────────────────────────────────────────────────


_eval_graph = None


def _get_eval_graph():
    """Lazy singleton for the eval-mode compiled graph. Built on first call."""
    global _eval_graph
    if _eval_graph is None:
        _eval_graph = _build_graph(eval_mode=True)
    return _eval_graph


def classify(turns: list[dict], caller_phone: Optional[str] = None) -> dict:
    """The agent's public callable (the harness imports this).

    Args:
        turns: structured dialogue from the eval transcripts — list of
            {"speaker": "agent"|"caller", "text": ...} dicts.
        caller_phone: phone number for caller_profile lookup, or None.

    Returns:
        A prediction dict matching the rubric schema from `prediction.py`
        (category, subcategory, risk_level, needs_human_review,
        needs_clarification, building_name, address, floor,
        dispatched_vendor_id, dispatched_emergency_services, call_summary,
        trainer_log).
    """
    graph = _get_eval_graph()
    transcript = _flatten(turns)
    profile = _lookup_profile(caller_phone)

    result = graph.invoke({
        "transcript": transcript,
        "turns_split": turns,
        "caller_phone": caller_phone,
        "caller_profile": profile,
        "eval_mode": True,
        "entity": None,
        "follow_ups": [],
        "extraction_complete": False,
        "needs_clarification": False,
        "gate_clarification_verdict": False,
        "clarification_question": None,
        "clarification_reasons": (),
        "enriched_query": "",
        "retrieved_docs": [],
        "retrieved_context": "",
        "retrieval_attempts": 0,
        "classification": None,
        "consensus_agreement": None,
        "risk_score": None,
        "risk_details": None,
        "building": None,
        "building_name": None,
        "floor": None,
        "address": None,
        "_city": None,
        "_building_type": None,
        "vendor_id": None,
        "needs_human_review": False,
        "dispatched_emergency_services": False,
        "call_summary": "",
        "routing_decision": None,
        "human_override": None,
        "trainer_log": None,
        "_grade": None,
    })

    return _format_prediction(result)


def _format_prediction(result: dict) -> dict:
    """Project the rich WorkflowState end state down to the grader's schema."""
    classification = result.get("classification") or {}
    category = (classification.get("category") or "").upper()
    subcategory = (classification.get("subcategory") or "").lower()

    # Ensure category↔subcategory consistency one more time at the boundary.
    if subcategory in SUBCATEGORY_TO_CATEGORY:
        category = SUBCATEGORY_TO_CATEGORY[subcategory]

    risk_level = (classification.get("risk_level") or "MEDIUM").upper()
    if risk_level not in ("LOW", "MEDIUM", "HIGH", "EMERGENCY"):
        risk_level = "MEDIUM"

    return {
        "category": category,
        "subcategory": subcategory,
        "risk_level": risk_level,
        "needs_human_review": result.get("needs_human_review", False),
        "needs_clarification": result.get("needs_clarification", False),
        "building_name": result.get("building_name"),
        "address": result.get("address"),
        "floor": result.get("floor"),
        "dispatched_vendor_id": result.get("vendor_id"),
        "dispatched_emergency_services": result.get("dispatched_emergency_services", False),
        "call_summary": result.get("call_summary", ""),
        "trainer_log": result.get("trainer_log"),
    }
