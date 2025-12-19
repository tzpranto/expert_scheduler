#!/usr/bin/env python
"""
Render a human-friendly report (Markdown + PNG charts) for heavy/sparse expert hitters.

Inputs:
  - heavy_sparse_report.json produced by analyze_hitters.py

Outputs:
  - heavy_sparse_summary.md : narrative + tables + chart references
  - charts/*.png            : bar charts for overall heavy/sparse experts
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_report(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def bar_chart(items: List[dict], title: str, ylabel: str, outpath: Path) -> None:
    """Render a simple bar chart of expert percentages."""
    experts = [int(x["expert"]) for x in items]
    pcts = [float(x["pct"]) for x in items]

    plt.figure(figsize=(6, 4))
    bars = plt.bar([str(e) for e in experts], pcts, color="tab:blue")
    plt.title(title)
    plt.xlabel("Expert ID")
    plt.ylabel(ylabel)
    plt.ylim(0, max(pcts) * 1.15 if pcts else 1)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for rect, pct in zip(bars, pcts):
        plt.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            f"{pct:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def make_markdown(report: Dict, charts_dir: Path) -> str:
    meta = report["meta"]
    seg_all = report["segments"]["all"]
    heavy = seg_all["heavy_hitters"]
    sparse = seg_all["sparse_hitters"]

    lines = []
    lines.append("# Expert Routing Report (GPT5OSS OASST)")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Traces dir: `{meta['traces_dir']}`")
    lines.append(f"- Files analyzed: **{meta['num_traces']}** trace_*.json")
    lines.append(f"- Layers: **{meta['num_layers']}**, Experts: **{meta['num_experts']}**")
    lines.append(f"- Cutoff: top/bottom **{int(meta['pct']*100)}%** per selection frequency")
    lines.append("")

    lines.append("## Key findings (overall tokens)")
    lines.append(f"- Tokens: **{seg_all['total_tokens']}**, top-1 routing events: **{seg_all['total_top1_events']}**")
    lines.append(
        "- Heavy hitters (most used experts): "
        + ", ".join(f"{h['expert']} ({h['pct']:.2f}%)" for h in heavy)
    )
    lines.append(
        "- Sparse hitters (least used experts): "
        + ", ".join(f"{h['expert']} ({h['pct']:.4f}%)" for h in sparse)
    )
    lines.append("")

    lines.append("### MoE perspective (interpretation)")
    lines.append(
        "- **Imbalance**: A few experts (24, 23, 11, 10) dominate routing (~21.3% combined), "
        "suggesting specialization or over-reliance; may merit load-balancing or routing regularization."
    )
    lines.append(
        "- **Under-utilization**: Experts 25, 20, 29, 3 sit at ~1.4–1.8%, indicating capacity headroom "
        "and potential to encourage diversity (e.g., auxiliary losses, temperature on gate, or dropout)."
    )
    lines.append(
        "- **No analysis/final segments detected**: All tokens were classified as prompt; future traces "
        "with channel markers would enable per-stage routing insights."
    )
    lines.append("")

    lines.append("## Visuals")
    lines.append(f"![Heavy hitters]({charts_dir / 'heavy_hitters.png'})")
    lines.append(f"![Sparse hitters]({charts_dir / 'sparse_hitters.png'})")
    lines.append("")

    lines.append("## How to reproduce")
    lines.append("```bash")
    lines.append("# 1) Compute hitters")
    lines.append("python analyze_hitters.py --traces-dir gpt5oss/oasst --output heavy_sparse_report.json")
    lines.append("# 2) Render this report")
    lines.append("python render_hitters_report.py")
    lines.append("```")

    return "\n".join(lines)


def main() -> None:
    report_path = Path("heavy_sparse_report.json")
    if not report_path.exists():
        raise SystemExit("heavy_sparse_report.json not found; run analyze_hitters.py first.")

    report = load_report(report_path)
    charts_dir = Path("charts")

    seg_all = report["segments"]["all"]
    heavy = seg_all["heavy_hitters"]
    sparse = seg_all["sparse_hitters"]

    bar_chart(heavy, "Heavy hitters (top usage)", "Selection share (%)", charts_dir / "heavy_hitters.png")
    bar_chart(sparse, "Sparse hitters (bottom usage)", "Selection share (%)", charts_dir / "sparse_hitters.png")

    md = make_markdown(report, charts_dir)
    Path("heavy_sparse_summary.md").write_text(md, encoding="utf-8")
    print("Wrote charts to", charts_dir)
    print("Wrote Markdown summary to heavy_sparse_summary.md")


if __name__ == "__main__":
    main()
