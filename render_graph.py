"""
Render the compiled agent graph to a Mermaid PNG + source file.

Usage:
    python render_graph.py

Outputs:
    graph_visual/agent_graph.png   — rendered topology (via mermaid.ink, needs internet)
    graph_visual/agent_graph.mmd   — Mermaid source (offline-editable)

The OPENAI_API_KEY env var is set to a dummy value if missing — graph compilation
instantiates ChatOpenAI() but does NOT call the API. No tokens are consumed.
Re-run any time the graph structure changes to refresh the artifacts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "sk-render-only-no-llm-call-made")

from langchain_core.runnables.graph import MermaidDrawMethod

import agent


def main() -> int:
    out_dir = Path(__file__).resolve().parent / "graph_visual"
    out_dir.mkdir(exist_ok=True)

    graph = agent._get_eval_graph().get_graph()

    mmd_path = out_dir / "agent_graph.mmd"
    mmd_path.write_text(graph.draw_mermaid())
    print(f"wrote {mmd_path}  ({mmd_path.stat().st_size} bytes)")

    png_path = out_dir / "agent_graph.png"
    try:
        png_bytes = graph.draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    except Exception as e:
        print(f"PNG render failed (mermaid.ink may be flaky — retry, or use the .mmd source): {e}",
              file=sys.stderr)
        return 1
    png_path.write_bytes(png_bytes)
    print(f"wrote {png_path}  ({png_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
