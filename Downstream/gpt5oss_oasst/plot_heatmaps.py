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
        mat = build_matrix(layers, s, e)
        seg_tokens = tokens[s:e]
        outpath = os.path.join(
            args.outdir,
            f"{os.path.splitext(os.path.basename(args.trace))[0]}_{name}.png",
        )
        title = f"{name.capitalize()} tokens: top-1 expert per layer"
        plot_heatmap(mat, seg_tokens, title, outpath, args.max_tokens)
        print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()
