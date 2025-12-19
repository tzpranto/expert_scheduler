#!/usr/bin/env python
"""
Generate per-layer per-token heatmaps (top-1 expert IDs) from GPT5OSS OASST traces.
Segments are split into prompt, analysis, and final channels using GPT-5 markers.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Marker tokens used by GPT-5 output traces
ANALYSIS_MARK = "<|channel|>analysis"
FINAL_MARK = "<|channel|>final"
RETURN_MARK = "<|return|>"


def find_first(tokens: List[str], needle: str) -> Optional[int]:
    """Return first index of needle in tokens or None if absent."""
    try:
        return tokens.index(needle)
    except ValueError:
        return None


def slice_tokens(tokens: List[str]) -> Dict[str, Tuple[int, int]]:
    """Return start/end indices for prompt, analysis, final segments."""
    a_start = find_first(tokens, ANALYSIS_MARK)
    f_start = find_first(tokens, FINAL_MARK)
    r_idx = find_first(tokens, RETURN_MARK)
    end = r_idx if r_idx is not None else len(tokens)

    if a_start is None:
        return {"prompt": (0, end)}
    if f_start is None:
        f_start = end

    return {
        "prompt": (0, a_start),
        "analysis": (a_start, f_start),
        "final": (f_start, end),
    }


def build_matrix(layers: List[dict], start: int, end: int) -> np.ndarray:
    """Build layer x token matrix of top-1 expert IDs."""
    num_layers = len(layers)
    num_tokens = end - start
    mat = np.zeros((num_layers, num_tokens), dtype=int)
    for li, layer in enumerate(layers):
        td = layer["token_data"]
        for ti in range(num_tokens):
            mat[li, ti] = td[start + ti]["topk_experts"][0]  # top-1 expert id
    return mat


def plot_heatmap(
    mat: np.ndarray,
    tokens: List[str],
    title: str,
    outpath: str,
    max_tokens: int = 512,
) -> None:
    """Plot and save a heatmap."""
    if mat.shape[1] > max_tokens:
        mat = mat[:, :max_tokens]
        tokens = tokens[:max_tokens]

    plt.figure(figsize=(max(8, mat.shape[1] / 8), 6))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest", cmap="tab20")
    plt.colorbar(im, label="Top expert id")
    plt.yticks(range(mat.shape[0]), [f"L{l}" for l in range(mat.shape[0])])
    plt.xticks(
        range(mat.shape[1]),
        [t if len(t) < 6 else t[:5] + "…" for t in tokens[:mat.shape[1]]],
        rotation=90,
        fontsize=6,
    )
    plt.title(title)
    plt.xlabel("Token position")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def build_freq_matrix(
    layers: List[dict],
    start: int,
    end: int,
    num_experts: int,
) -> np.ndarray:
    """Layer x expert frequency (counts) matrix across tokens."""
    num_layers = len(layers)
    freq = np.zeros((num_layers, num_experts), dtype=float)
    for li, layer in enumerate(layers):
        td = layer["token_data"]
        for ti in range(start, end):
            top1 = td[ti]["topk_experts"][0]
            freq[li, top1] += 1
    # convert to percentage per layer
    token_count = max(1, end - start)
    freq = freq / token_count * 100.0
    return freq


def plot_freq_heatmap(
    freq: np.ndarray,
    title: str,
    outpath: str,
) -> None:
    """Plot layer x expert frequency (%) heatmap."""
    plt.figure(figsize=(10, 6))
    im = plt.imshow(freq, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(im, label="Frequency (%)")
    plt.yticks(range(freq.shape[0]), [f"L{l}" for l in range(freq.shape[0])])
    plt.xticks(range(freq.shape[1]), range(freq.shape[1]))
    plt.xlabel("Expert ID")
    plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def compute_entropy_stats(
    layers: List[dict],
    start: int,
    end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-layer average entropy (bits) and effective experts (2**H)
    using the provided top-k probabilities.
    """
    num_layers = len(layers)
    entropies = np.zeros(num_layers, dtype=float)
    effs = np.zeros(num_layers, dtype=float)
    token_count = max(1, end - start)
    for li, layer in enumerate(layers):
        td = layer["token_data"]
        layer_ents = []
        for ti in range(start, end):
            probs = td[ti]["topk_probs"]
            # ensure numeric and normalized
            s = sum(probs)
            if s <= 0:
                continue
            p = [x / s for x in probs]
            h = -sum(pi * np.log2(pi) for pi in p if pi > 0)
            layer_ents.append(h)
        if layer_ents:
            ent = float(np.mean(layer_ents))
        else:
            ent = 0.0
        entropies[li] = ent
        effs[li] = 2 ** ent
    return entropies, effs


def plot_entropy_effective(
    entropies: np.ndarray,
    effs: np.ndarray,
    outpath: str,
    segment_name: str,
) -> None:
    """Plot per-layer entropy and effective experts."""
    layers = np.arange(len(entropies))
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(layers, entropies, marker="o", label="Entropy (bits)", color="tab:blue")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Entropy (bits)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(layers)
    ax1.set_xticklabels([f"L{l}" for l in layers], rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(layers, effs, marker="s", label="Effective experts", color="tab:orange")
    ax2.set_ylabel("Effective experts", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle(f"{segment_name.capitalize()} tokens: entropy and effective experts")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-layer per-token activation heatmaps (top-1 expert IDs)."
    )
    parser.add_argument("--trace", required=True, help="Path to trace_XXXX.json")
    parser.add_argument("--outdir", default="heatmaps", help="Output directory for PNGs")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens per heatmap (truncates longer sequences).",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.trace, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers = data["layers"]
    tokens = [td["token"] for td in layers[0]["token_data"]]
    segments = slice_tokens(tokens)

    for name, (s, e) in segments.items():
        if e <= s:
            continue
        base = os.path.splitext(os.path.basename(args.trace))[0]

        # Top-1 expert heatmap (token-level)
        mat = build_matrix(layers, s, e)
        seg_tokens = tokens[s:e]
        heat_out = os.path.join(args.outdir, f"{base}_{name}.png")
        plot_heatmap(mat, seg_tokens, f"{name.capitalize()} tokens: top-1 expert per layer", heat_out, args.max_tokens)
        print(f"Wrote {heat_out}")

        # Layer x expert frequency heatmap
        num_experts = layers[0]["num_experts"] if "num_experts" in layers[0] else 32
        freq = build_freq_matrix(layers, s, e, num_experts)
        freq_out = os.path.join(args.outdir, f"{base}_{name}_freq.png")
        plot_freq_heatmap(freq, f"{name.capitalize()} tokens: layer x expert frequency (%)", freq_out)
        print(f"Wrote {freq_out}")

        # Per-layer entropy / effective experts
        entropies, effs = compute_entropy_stats(layers, s, e)
        ent_out = os.path.join(args.outdir, f"{base}_{name}_entropy.png")
        plot_entropy_effective(entropies, effs, ent_out, name)
        print(f"Wrote {ent_out}")


if __name__ == "__main__":
    main()
