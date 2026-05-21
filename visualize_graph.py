"""Regenerate docs/agent_graph.mmd and docs/agent_graph.png from the live graph."""
from pathlib import Path

from agent import _get_eval_graph

HERE = Path(__file__).resolve().parent
DOCS = HERE / "docs"
DOCS.mkdir(exist_ok=True)

graph = _get_eval_graph()
compiled = graph.get_graph()

mmd_path = DOCS / "agent_graph.mmd"
mmd_path.write_text(compiled.draw_mermaid())
print(f"Wrote {mmd_path}")

png_path = DOCS / "agent_graph.png"
png_path.write_bytes(compiled.draw_mermaid_png())
print(f"Wrote {png_path}")
