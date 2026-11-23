"""
One-time script to embed historical_records.json into a Chroma vector store.

Usage:
    python scripts/build_chroma.py

Creates a persistent Chroma DB at ./chroma_db/ with collection "historical_records".
Idempotent — skips embedding if collection is already populated.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent.parent  # repo root
load_dotenv(HERE / ".env")

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

OPS = HERE / "operational"
CHROMA_DIR = str(HERE / "chroma_db")
COLLECTION = "historical_records"


def build():
    records = json.loads((OPS / "historical_records.json").read_text())
    print(f"Loaded {len(records)} historical records.")

    docs = []
    for r in records:
        text = (
            f"Problem: {r['call_recording_transcript']}\n"
            f"Category: {r['final_category']} / {r['final_subcategory']}\n"
            f"Risk: {r['final_risk_level']}\n"
            f"Building: {r['building_name']} ({r['building_type']})"
        )
        metadata = {
            "ticket_id": r["ticket_id"],
            "final_category": r["final_category"],
            "final_subcategory": r["final_subcategory"],
            "final_risk_level": r["final_risk_level"],
            "building_name": r["building_name"],
            "city": r["city"],
            "assigned_vendor_id": r.get("assigned_vendor_id", ""),
        }
        docs.append(Document(page_content=text, metadata=metadata))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    if db._collection.count() == 0:
        # Batch in chunks of 500 to avoid hitting API limits
        batch_size = 500
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            db.add_documents(batch)
            print(f"  Embedded batch {i // batch_size + 1}/{(len(docs) - 1) // batch_size + 1} ({len(batch)} docs)")
        print(f"Done. Embedded {len(docs)} documents into Chroma at {CHROMA_DIR}")
    else:
        print(f"Collection already has {db._collection.count()} documents — skipping.")

    return db


if __name__ == "__main__":
    build()
