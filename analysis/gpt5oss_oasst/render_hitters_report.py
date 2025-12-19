#!/usr/bin/env python
"""
Inputs:
  - heavy_sparse_report.json produced by analyze_hitters.py

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



if __name__ == "__main__":
    main()
