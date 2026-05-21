"""
RAG + LLM classification nodes.

Maps to the brief's deliverables on:
  - **RAG strategy** — build_enriched_query forms a structured query from the
    extractor's output; make_retrieve hits Chroma for the 5 most-similar
    historical tickets (k=5).
  - **Hierarchical classification (10×36)** — make_classify_issue is the
    main gpt-4.1 call, sees the full taxonomy + retrieved historical context +
    extracted entities + transcript + caller profile + clarification-gate
    verdict, and emits the full classification JSON.
  - **Conflict resolution between LLM and history** — consensus_check
    aggregates the 5 retrieved tickets two ways: (1) subcategory vote that
    weakens confidence on disagreement, (2) per-call RAG-band consensus
    (same-subcategory filter) that demote-only-corrects the LLM's emitted
    risk_level when historical majority lands lower.
"""
from __future__ import annotations

import json
from collections import Counter

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from .prompts import CLASSIFY_PROMPT, _BUILDINGS_BLOCK, _TAXONOMY_BLOCK
from .state import WorkflowState


def build_enriched_query(state: WorkflowState):
    """Compose the natural-language query passed to Chroma similarity search."""
    ent = state["entity"]
    enriched = (
        f"Problem: {ent['problem_category']}."
        + (f" System: {ent['affected_system']}." if ent.get("affected_system") else "")
        + f" Severity: {ent['severity']}."
        + f" {ent['summary']}"
    )
    return {
        "enriched_query": enriched,
        "retrieval_attempts": 0,
    }


def make_retrieve(vector_store: Chroma):
    """Chroma similarity search node: pulls the 5 most-similar historical
    tickets (page_content + metadata including final_subcategory and
    final_risk_level) so downstream LLM and consensus_check can use them.
    """

    def retrieve(state: WorkflowState):
        query = state["enriched_query"]
        docs = vector_store.similarity_search(query, k=5)
        retrieved_docs = []
        context_parts = []
        for doc in docs:
            meta = doc.metadata
            retrieved_docs.append({
                "content": doc.page_content,
                "metadata": meta,
            })
            context_parts.append(
                f"[Past ticket] {doc.page_content}"
            )
        return {
            "retrieved_docs": retrieved_docs,
            "retrieved_context": "\n\n".join(context_parts),
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }

    return retrieve


def make_classify_issue(llm: ChatOpenAI):
    """Main classification LLM call (gpt-4.1, JSON output)."""

    def classify_issue(state: WorkflowState):
        ent = state.get("entity") or {}
        profile = state.get("caller_profile")

        profile_context = ""
        if profile:
            profile_context = (
                f"\nCALLER PROFILE (last-known default — caller may be elsewhere today):\n"
                f"  Name: {profile['caller_name']}\n"
                f"  Building: {profile['primary_building_name']}\n"
                f"  Address: {profile['primary_address']}, {profile['primary_city']}\n"
                f"  Floor: {profile['primary_floor']}\n"
                f"  Suite: {profile.get('primary_suite') or 'N/A'}\n"
                f"  Active: {profile['active']}"
            )

        entities_str = (
            f"Problem: {ent.get('problem_category', 'unknown')}\n"
            f"System: {ent.get('affected_system', 'unknown')}\n"
            f"Building: {ent.get('building_name', 'unknown')}\n"
            f"Floor: {ent.get('floor', 'unknown')}\n"
            f"Severity: {ent.get('severity', 'unknown')}\n"
            f"Summary: {ent.get('summary', '')}"
        )

        reasons = list(state.get("clarification_reasons") or ())
        gate_verdict = state.get("gate_clarification_verdict", False)

        prompt = CLASSIFY_PROMPT.format(
            taxonomy=_TAXONOMY_BLOCK,
            buildings=_BUILDINGS_BLOCK,
            rag_context=state.get("retrieved_context", "None"),
            entities=entities_str,
            transcript=state["transcript"],
            profile_context=profile_context,
            gate_verdict=gate_verdict,
            clarification_reasons=reasons,
        )

        response = llm.invoke(
            [{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        raw = json.loads(response.content)
        return {"classification": raw}

    return classify_issue


def consensus_check(state: WorkflowState):
    """Two-axis sanity check against the 5 retrieved historical tickets:

    1. Subcategory vote (legacy): if RAG majority disagrees with LLM subcategory,
       weaken confidence by 30%. Does NOT change the label.
    2. Risk-level vote: demote-only band correction. If RAG majority (>=60%)
       of *same-subcategory* retrieved docs names a band BELOW the LLM's
       emission, demote to that band. Uses per-call historical evidence
       rather than the aggregate subcategory cap.

    Demote-only by construction — RAG can never promote past the LLM's emission.
    """
    docs = state.get("retrieved_docs") or []
    classification = state["classification"]
    if not docs:
        return {"classification": classification, "consensus_agreement": None}

    # ── Vote 1: subcategory consensus (existing behavior) ──
    sub_votes = Counter(d["metadata"].get("final_subcategory", "") for d in docs)
    majority_sub, majority_count = sub_votes.most_common(1)[0]
    sub_agreement = majority_count / len(docs)
    if sub_agreement >= 0.6 and majority_sub != classification.get("subcategory"):
        conf = classification.get("confidence", 0.5)
        classification["confidence"] = conf * 0.7
        classification["consensus_note"] = (
            f"RAG majority ({sub_agreement:.0%}) suggests {majority_sub}"
        )

    # ── Vote 2: risk-level consensus (demote-only, same-sub filter) ──
    band_ord = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "EMERGENCY": 3}
    sub = (classification.get("subcategory") or "").lower()
    same_sub_docs = [
        d for d in docs
        if (d["metadata"].get("final_subcategory") or "").lower() == sub
    ]
    if len(same_sub_docs) >= 3:
        band_votes = Counter()
        for d in same_sub_docs:
            band = (d["metadata"].get("final_risk_level") or "").upper()
            if band in band_ord:
                band_votes[band] += 1
        if band_votes:
            majority_band, band_count = band_votes.most_common(1)[0]
            band_agreement = band_count / sum(band_votes.values())
            llm_band = (classification.get("risk_level") or "MEDIUM").upper()
            if (band_agreement >= 0.6
                    and band_ord.get(majority_band, 3) < band_ord.get(llm_band, 1)):
                classification["original_risk_level_rag"] = llm_band
                classification["risk_level"] = majority_band
                classification["consensus_band_applied"] = (
                    f"RAG-band majority ({band_agreement:.0%}={majority_band}, "
                    f"n={len(same_sub_docs)} same-sub) demoted from {llm_band}"
                )

    return {"classification": classification, "consensus_agreement": sub_agreement}
