"""
RAG + LLM classification nodes.

Maps to the brief's deliverables on:
  - **RAG strategy** — build_enriched_query forms a structured query from the
    extractor's output; make_retrieve hits Chroma for k=10 candidates and
    runs them through make_rerank_docs (deterministic metadata-aware
    reranker) to produce the top-5 reranked set used downstream.
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
from typing import Optional

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from .config import _building_by_name
from .prompts import CLASSIFY_SYSTEM_PROMPT, CLASSIFY_USER_TEMPLATE
from .state import WorkflowState
from .utils import _log_cache_usage


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


# ── RAG reranking ────────────────────────────────────────────────────────────
#
# Chroma's similarity_search picks the k most-embedding-similar docs against
# the enriched_query. That signal is fine but ignores the structured metadata
# we have at retrieval time — the caller's profile, the extracted entity, the
# doc's building/city/risk_level metadata. The reranker layers a small
# composite boost on top of Chroma's rank to re-order the k=10 candidates,
# then returns the top 5 for downstream nodes.

# Extractor's Severity enum → final_risk_level band, for the severity-class
# match boost. The extractor emits Severity strings ("Critical"/...) while
# the historical docs store final_risk_level ("EMERGENCY"/...).
_SEVERITY_TO_BAND = {
    "Critical": "EMERGENCY",
    "High": "HIGH",
    "Medium": "MEDIUM",
    "Low": "LOW",
}

# Additive boost weights. Base score is rank-derived in [0, 1]; boosts stack.
# Weights tuned so the most-specific signal (exact building_name match)
# dominates, with progressively smaller boosts for coarser signals.
_DEFAULT_RERANK_WEIGHTS = {
    "building_name": 0.50,   # exact building match — strongest signal
    "building_type": 0.20,   # same building category (medical, industrial, ...)
    "severity_class": 0.10,  # extractor's severity matches doc's risk band
    "city": 0.05,            # same city — weakest, local-context only
}


def _building_type_for(building_name: Optional[str]) -> Optional[str]:
    """Look up a building's type via the registry. Returns None if no match."""
    if not building_name:
        return None
    record = _building_by_name.get(building_name.strip().lower())
    return record.get("building_type") if record else None


def _score_doc(
    doc: dict,
    rank: int,
    total: int,
    entity: dict,
    profile: Optional[dict],
    weights: dict,
) -> float:
    """Composite rerank score for one Chroma-retrieved doc.

    Base = (total - rank) / total — Chroma's best-ranked candidate starts at
    1.0, worst at 1/total. Boosts add on top when the doc's metadata matches
    the caller's emerging context. Final score is unbounded above but
    relative-ordering is what matters for sort.
    """
    meta = doc["metadata"]
    base = (total - rank) / total if total else 0.0
    score = base

    # 1. Building-name match — strongest signal (exact-same-building case).
    extracted_bname = (entity.get("building_name") or "").strip().lower()
    doc_bname = (meta.get("building_name") or "").strip().lower()
    if extracted_bname and doc_bname and extracted_bname == doc_bname:
        score += weights["building_name"]

    # 2. Building-type match — needs a registry lookup since Chroma metadata
    #    doesn't carry building_type directly. Caller's type comes from the
    #    profile when available, else looked up via the extracted name.
    caller_btype: Optional[str] = None
    if profile and profile.get("primary_building_type"):
        caller_btype = profile["primary_building_type"].lower()
    elif extracted_bname:
        caller_btype = _building_type_for(extracted_bname)
    doc_btype = _building_type_for(doc_bname) if doc_bname else None
    if caller_btype and doc_btype and caller_btype == doc_btype:
        score += weights["building_type"]

    # 3. Severity-class match — extractor's severity vs doc's final_risk_level.
    severity = entity.get("severity")
    severity_band = _SEVERITY_TO_BAND.get(severity) if severity else None
    doc_band = (meta.get("final_risk_level") or "").upper()
    if severity_band and doc_band and severity_band == doc_band:
        score += weights["severity_class"]

    # 4. City match — weak local-context signal from the profile.
    profile_city = (profile or {}).get("primary_city", "") if profile else ""
    doc_city = meta.get("city", "")
    if profile_city and doc_city and profile_city == doc_city:
        score += weights["city"]

    return score


def make_rerank_docs(
    *,
    top_k: int = 5,
    weights: Optional[dict] = None,
):
    """Factory for the metadata-aware reranker. Returns a callable taking
    (docs, entity, profile) and returning the top-k reranked docs.

    Mirrors `make_retrieve`'s factory pattern so a different rerank policy
    (different weights, different top_k) can be swapped in without changing
    the call sites in `make_retrieve` or downstream nodes.
    """
    final_weights = weights if weights is not None else _DEFAULT_RERANK_WEIGHTS

    def rerank_docs(
        docs: list[dict],
        entity: dict,
        profile: Optional[dict],
    ) -> list[dict]:
        total = len(docs)
        if total == 0:
            return []
        scored = [
            (_score_doc(d, i, total, entity, profile, final_weights), d)
            for i, d in enumerate(docs)
        ]
        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored[:top_k]]

    return rerank_docs


def make_retrieve(
    vector_store: Chroma,
    *,
    over_retrieve_k: int = 10,
    rerank_fn=None,
):
    """Chroma similarity search node with metadata-aware reranking.

    Pulls `over_retrieve_k` candidates (default 10) from Chroma against the
    enriched_query, runs them through `rerank_fn` (default: make_rerank_docs
    with the standard weights), and writes the top 5 to retrieved_docs +
    retrieved_context. Downstream nodes (classify_issue, consensus_check)
    see only the reranked set.
    """
    if rerank_fn is None:
        rerank_fn = make_rerank_docs()

    def retrieve(state: WorkflowState):
        query = state["enriched_query"]
        entity = state.get("entity") or {}
        profile = state.get("caller_profile")

        # Over-retrieve from Chroma against the enriched query
        raw_docs = vector_store.similarity_search(query, k=over_retrieve_k)
        candidates = [
            {"content": d.page_content, "metadata": d.metadata}
            for d in raw_docs
        ]

        # Rerank the k=10 candidates → top 5 for downstream
        reranked = rerank_fn(candidates, entity, profile)
        context_parts = [f"[Past ticket] {d['content']}" for d in reranked]

        return {
            "retrieved_docs": reranked,
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

        # System message is byte-stable across every call (taxonomy + buildings
        # + rules pre-interpolated at module load). User message holds 100% of
        # the per-call dynamic payload so it never interrupts the cached prefix.
        user_message = CLASSIFY_USER_TEMPLATE.format(
            gate_verdict=gate_verdict,
            clarification_reasons=reasons,
            rag_context=state.get("retrieved_context", "None"),
            entities=entities_str,
            transcript=state["transcript"],
            profile_context=profile_context,
        )

        response = llm.invoke(
            [
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
        )

        _log_cache_usage("classify_issue", response)
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
