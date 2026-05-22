"""
LangGraph state schema + Pydantic models that flow through the agent.

WorkflowState is the TypedDict that every node receives and partially
returns. It carries:
  - the original transcript + structured turns
  - the LLM-extracted entity (problem, severity, location, urgency)
  - clarification-gate verdict + reasons (for the brief's clarification axis)
  - RAG outputs (retrieved_docs, retrieved_context)
  - the LLM's classification + downstream corrections (risk_policy_cap,
    consensus_band, etc.)
  - resolved location + dispatched vendor
  - HITL gate outputs + trainer log
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, TypedDict

from pydantic import BaseModel, Field


# ── Pydantic schemas for LLM structured output ──────────────────────────────


class Severity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class EntityWithQuality(BaseModel):
    """Combined entity extraction + quality check in a single LLM call."""
    problem_category: str = Field(
        description="Specific description of the problem (e.g., 'water leaking from ceiling pipe', "
                    "'elevator stuck between floors') — not vague terms like 'issue' or 'problem'"
    )
    affected_system: Optional[str] = Field(
        default=None,
        description="The exact device, system, or component affected"
    )
    building_name: Optional[str] = Field(
        default=None,
        description="Name of the building mentioned in the transcript"
    )
    address: Optional[str] = Field(
        default=None,
        description="Street address mentioned"
    )
    floor: Optional[str] = Field(
        default=None,
        description="Floor number or level (e.g., '3rd floor', 'Floor 7', 'basement')"
    )
    location_detail: Optional[str] = Field(
        default=None,
        description="Specific location within the building (suite, room, wing, hallway)"
    )
    severity: Severity = Field(
        description="Critical / High / Medium / Low based on safety and operational impact"
    )
    urgency_indicators: list[str] = Field(
        default_factory=list,
        description="Specific phrases indicating urgency or timeline"
    )
    summary: str = Field(description="One-sentence summary of the issue")
    # Quality check fields
    description_specific_enough: bool = Field(
        description="True if the problem describes a specific symptom or failure, not vague"
    )
    location_ok: bool = Field(
        default=True,
        description="True if location info is present and not conflicting"
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if description is too vague or location is conflicting/missing"
    )
    followup_question: str = Field(
        default="",
        description="Question to get more specific info. Empty string if ok."
    )


# ── Deterministic clarification-gate decision ────────────────────────────────


@dataclass(frozen=True)
class ClarificationDecision:
    needs_clarification: bool
    question: Optional[str]
    reasons: Tuple[str, ...]


# ── LangGraph state ─────────────────────────────────────────────────────────


class WorkflowState(TypedDict):
    transcript: str
    turns_split: list[dict]
    caller_phone: Optional[str]
    caller_profile: Optional[dict]
    eval_mode: bool
    entity: Optional[dict]
    follow_ups: list
    extraction_complete: bool
    needs_clarification: bool
    gate_clarification_verdict: bool
    clarification_question: Optional[str]
    clarification_reasons: Tuple[str, ...]
    enriched_query: str
    retrieved_docs: list
    retrieved_context: str
    retrieval_attempts: int
    classification: Optional[dict]
    consensus_agreement: Optional[float]
    risk_score: Optional[float]
    risk_details: Optional[dict]
    building: Optional[dict]
    building_name: Optional[str]
    floor: Optional[str]
    address: Optional[str]
    vendor_id: Optional[str]
    _city: Optional[str]
    _building_type: Optional[str]
    needs_human_review: bool
    dispatched_emergency_services: bool
    call_summary: str
    routing_decision: Optional[str]
    human_override: Optional[dict]
    trainer_log: Optional[dict]
    _grade: Optional[str]
