#!/usr/bin/env python
"""
Generate publication-style heatmaps for MoE routing.

Modes:
- per_token: Two stacked heatmaps (router confidence + top-1 expert ID) per layer × token.
- freq: Single heatmap of layer × expert frequency (%) like the provided example figure.

Examples:
    # Per-token view (default)
    python plot_heatmaps.py --trace gpt5oss/oasst/trace_0000.json \\
        --output charts/per_layer_token_heatmap.png --max_tokens 256

    # Layer × expert frequency view (example style)
    python plot_heatmaps.py --trace gpt5oss/oasst/trace_0000.json \\
        --mode freq --output charts/layer_expert_freq.png --max_tokens 2048
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_tokens(layer: Dict) -> List[Dict]:
    if "token_data" in layer:
        return layer["token_data"]
    if "topk_per_token" in layer:
        return layer["topk_per_token"]
    raise KeyError("Layer missing token payload (expected token_data or topk_per_token).")


def load_trace(trace_path: Path, max_tokens: int | None) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
    with trace_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    layers = data["layers"]
    token_lists = [_load_tokens(layer) for layer in layers]
    seq_len = min(len(toks) for toks in token_lists)
    if max_tokens:
        seq_len = min(seq_len, max_tokens)

    num_layers = len(layers)
    top1_probs = np.zeros((num_layers, seq_len), dtype=float)
    top1_ids = np.zeros((num_layers, seq_len), dtype=float)

    # Use layer 0 tokens as display labels when present.
    token_labels = []

    for li, tokens in enumerate(token_lists):
        for ti in range(seq_len):
            entry = tokens[ti]
            experts = entry.get("topk_experts") or entry.get("top_experts")
            probs = entry["topk_probs"]
            top1_probs[li, ti] = float(probs[0])
            top1_ids[li, ti] = float(experts[0])
            if li == 0:
                token_labels.append(entry.get("token", str(ti)))

    meta = {
        "num_layers": data.get("num_layers", num_layers),
        "num_experts": data.get("num_experts"),
        "k_per_token": data.get("k_per_token"),
    }
    return top1_probs, top1_ids, token_labels, meta


def layer_expert_freq(top1_ids: np.ndarray, num_experts: int) -> np.ndarray:
    """Convert top-1 expert IDs (layers x tokens) into per-layer frequency (%) matrix."""
    layers, _ = top1_ids.shape
    freq = np.zeros((layers, num_experts), dtype=float)
    for li in range(layers):
        ids = top1_ids[li].astype(int)
        counts = np.bincount(ids, minlength=num_experts)
        total = counts.sum()
        freq[li] = counts / total * 100.0 if total > 0 else 0.0
    return freq


def _set_xticks(ax: plt.Axes, tokens: List[str]) -> None:
    n = len(tokens)
    max_ticks = 24
    step = max(1, n // max_ticks)
    positions = list(range(0, n, step))
    labels = [tokens[i] if i < len(tokens) else str(i) for i in positions]
    labels = [label.replace("\n", " ")[:10] for label in labels]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)


def plot_heatmaps(
    top1_probs: np.ndarray,
    top1_ids: np.ndarray,
    tokens: List[str],
    meta: Dict,
    output: Path,
    dpi: int,
) -> None:
    layers, seq_len = top1_probs.shape
    fig_width = max(10, min(20, 0.12 * seq_len + 4))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(fig_width, 8),
        constrained_layout=True,
        height_ratios=[1, 1],
    )

    im0 = axes[0].imshow(top1_probs, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    axes[0].set_ylabel("Layer")
    axes[0].set_title("Router confidence (top-1 gate probability)", loc="left", fontsize=11, pad=8)
    _set_xticks(axes[0], tokens)
    c0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
    c0.set_label("Probability", fontsize=9)

    im1 = axes[1].imshow(top1_ids, aspect="auto", cmap="tab20", interpolation="nearest")
    axes[1].set_ylabel("Layer")
    axes[1].set_xlabel("Token position (decoded order)")
    axes[1].set_title("Top-1 expert ID", loc="left", fontsize=11, pad=8)
    _set_xticks(axes[1], tokens)
    c1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
    c1.set_label("Expert ID", fontsize=9)

    fig.suptitle(
        f"Per-layer, per-token routing • layers={meta.get('num_layers')} "
        f"experts={meta.get('num_experts')} k={meta.get('k_per_token')} • n_tokens={seq_len}",
        fontsize=12,
        y=1.02,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_freq_heatmap(freq_pct: np.ndarray, meta: Dict, output: Path, dpi: int) -> None:
    layers, experts = freq_pct.shape
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    im = ax.imshow(freq_pct, aspect="auto", cmap="viridis")
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(layers))
    ax.set_yticklabels([f"L{l}" for l in range(layers)])
    ax.set_xticks(range(experts))
    ax.set_xticklabels(range(experts))
    ax.set_title("Prompt tokens: layer × expert frequency (%)", fontsize=13, pad=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Frequency (%)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MoE routing heatmaps.")
    parser.add_argument("--trace", required=True, help="Path to a trace_*.json prefill file.")
    parser.add_argument(
        "--mode",
        choices=["per_token", "freq"],
        default="per_token",
        help="per_token: stacked token-wise heatmaps; freq: layer × expert frequency (percent).",
    )
    parser.add_argument(
        "--output",
        default="charts/per_layer_token_heatmap.png",
        help="Output image path (PNG recommended).",
    )
    parser.add_argument("--max_tokens", type=int, default=256, help="Cap number of tokens considered.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI for publication-quality export.")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    output_path = Path(args.output)

    top1_probs, top1_ids, tokens, meta = load_trace(trace_path, args.max_tokens)

    if args.mode == "per_token":
        plot_heatmaps(top1_probs, top1_ids, tokens, meta, output_path, args.dpi)
        print(f"Wrote per-token heatmap to {output_path} (layers={meta['num_layers']}, experts={meta['num_experts']}).")
        print(
            f"  Tokens visualized: {top1_probs.shape[1]} | "
            f"mean top-1 prob: {top1_probs.mean():.3f} | "
            f"min/max top-1 prob: {top1_probs.min():.3f}/{top1_probs.max():.3f}"
        )
    else:
        freq_pct = layer_expert_freq(top1_ids, meta["num_experts"])
        plot_freq_heatmap(freq_pct, meta, output_path, args.dpi)
        print(
            f"Wrote frequency heatmap to {output_path} "
            f"(layers={meta['num_layers']}, experts={meta['num_experts']}, max={freq_pct.max():.1f}% per cell)."
        )


if __name__ == "__main__":
    main()
