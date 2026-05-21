"""
One-time script to embed historical_records.json into a Chroma vector store.

Usage:
    python build_chroma.py            # idempotent — skip if collection populated
    python build_chroma.py --force    # wipe and re-embed (use when metadata schema changes)

Creates a persistent Chroma DB at ./chroma_db/ with collection "historical_records".

Each historical record is enriched with QA audit findings from
evaluation/qa_audit_findings.json:
  - Floors logged wrong at intake are overwritten with audit.actual_floor.
  - Audit flags (was_over_escalated, was_reclassified, floor_was_wrong_at_intake,
    audit_clean) are stored as sentinel strings since Chroma metadata can't be None.
"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / "Week5" / ".env")

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

HERE = Path(__file__).resolve().parent
OPS = HERE / "operational"
AUDIT = HERE / "evaluation" / "qa_audit_findings.json"
CHROMA_DIR = str(HERE / "chroma_db")
COLLECTION = "historical_records"


def _load_audit_by_ticket() -> dict:
    if not AUDIT.exists():
        return {}
    raw = json.loads(AUDIT.read_text())
    return {a["ticket_id"]: a for a in raw}


def _corrected_floor(rec: dict, audit_entry: dict | None) -> str:
    """Use audit.actual_floor when the audit flagged the intake floor as wrong."""
    if audit_entry and audit_entry.get("floor_was_wrong_at_intake"):
        actual = audit_entry.get("actual_floor")
        if actual:
            return actual
    return rec.get("floor") or ""


def _audit_flags(audit_entry: dict | None) -> dict:
    """Sentinel-stringify audit booleans — Chroma metadata rejects None."""
    if audit_entry is None:
        return {
            "was_over_escalated": "unaudited",
            "was_reclassified": "unaudited",
            "floor_was_wrong_at_intake": "unaudited",
            "audit_clean": "unaudited",
        }
    o = bool(audit_entry.get("was_over_escalated"))
    r = bool(audit_entry.get("was_reclassified"))
    f = bool(audit_entry.get("floor_was_wrong_at_intake"))
    return {
        "was_over_escalated": str(o).lower(),
        "was_reclassified": str(r).lower(),
        "floor_was_wrong_at_intake": str(f).lower(),
        "audit_clean": str((not o) and (not r) and (not f)).lower(),
    }


def build(force: bool = False):
    records = json.loads((OPS / "historical_records.json").read_text())
    audit_by_tid = _load_audit_by_ticket()
    print(f"Loaded {len(records)} historical records, {len(audit_by_tid)} audit entries.")

    docs = []
    floor_overwrites = 0
    for r in records:
        tid = r["ticket_id"]
        audit_entry = audit_by_tid.get(tid)
        floor = _corrected_floor(r, audit_entry)
        if audit_entry and audit_entry.get("floor_was_wrong_at_intake") and floor != r.get("floor"):
            floor_overwrites += 1

        text = (
            f"Problem: {r['call_recording_transcript']}\n"
            f"Category: {r['final_category']} / {r['final_subcategory']}\n"
            f"Risk: {r['final_risk_level']}\n"
            f"Building: {r['building_name']} ({r['building_type']}), Floor: {floor}"
        )
        metadata = {
            "ticket_id": tid,
            "final_category": r["final_category"],
            "final_subcategory": r["final_subcategory"],
            "final_risk_level": r["final_risk_level"],
            "building_name": r["building_name"],
            "city": r["city"],
            "floor": floor,
            "assigned_vendor_id": r.get("assigned_vendor_id", ""),
            **_audit_flags(audit_entry),
        }
        docs.append(Document(page_content=text, metadata=metadata))

    print(f"Floor overwrites from audit: {floor_overwrites}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    if force and db._collection.count() > 0:
        print(f"--force: dropping existing collection ({db._collection.count()} docs)")
        db._client.delete_collection(name=COLLECTION)
        db = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )

    if db._collection.count() == 0:
        batch_size = 500
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            db.add_documents(batch)
            print(f"  Embedded batch {i // batch_size + 1}/{(len(docs) - 1) // batch_size + 1} ({len(batch)} docs)")
        print(f"Done. Embedded {len(docs)} documents into Chroma at {CHROMA_DIR}")
    else:
        print(f"Collection already has {db._collection.count()} documents — skipping. Pass --force to rebuild.")

    return db


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Drop existing collection and re-embed from scratch.")
    args = ap.parse_args()
    build(force=args.force)
