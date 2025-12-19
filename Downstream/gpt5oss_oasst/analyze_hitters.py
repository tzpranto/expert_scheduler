#!/usr/bin/env python
"""
Compute heavy and sparse expert hitters (top/bottom 10%) from GPT5OSS OASST traces.

The script aggregates top-1 expert selections across all layers and tokens in the
trace_*.json files, and reports the most/least used experts globally, per segment
(prompt/analysis/final), and per layer.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np

ANALYSIS_MARK = "<|channel|>analysis"
FINAL_MARK = "<|channel|>final"
RETURN_MARK = "<|return|>"


def find_first(tokens: List[str], needle: str) -> int | None:
    """Return first index of needle in tokens or None if absent."""
    try:
        return tokens.index(needle)
    except ValueError:
        return None


def slice_tokens(tokens: List[str]) -> Dict[str, Tuple[int, int]]:
    """
    Return start/end indices for prompt, analysis, final segments.
    If channel markers are missing, fall back to treating everything as prompt.
    """
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


def hitters_from_counts(
    counts: np.ndarray, pct: float
) -> Tuple[List[dict], List[dict], np.ndarray]:
    """
    Given 1D counts per expert, return heavy/sparse hitters lists and freqs.
    pct is the fraction of experts to select (e.g., 0.1 for 10%).
    """
    total = float(counts.sum())
    freqs = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
    n = max(1, int(math.ceil(len(counts) * pct)))
    desc = np.argsort(-freqs)
    asc = np.argsort(freqs)

    def pack(order: np.ndarray) -> List[dict]:
        return [
            {
                "expert": int(idx),
                "count": int(counts[idx]),
                "pct": float(freqs[idx] * 100.0),
            }
            for idx in order[:n]
        ]

    return pack(desc), pack(asc), freqs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find heavy/sparse expert hitters (top/bottom 10%) from GPT5OSS OASST traces."
    )
    parser.add_argument(
        "--traces-dir",
        default=os.path.join("gpt5oss", "oasst"),
        help="Directory containing trace_*.json files.",
    )
    parser.add_argument(
        "--pct",
        type=float,
        default=0.10,
        help="Top/bottom percentage to select (default 0.10 -> 10%%).",
    )
    parser.add_argument(
        "--output",
        default="heavy_sparse_report.json",
        help="Path to write the JSON report.",
    )
    args = parser.parse_args()

    trace_paths = sorted(glob.glob(os.path.join(args.traces_dir, "trace_*.json")))
    if not trace_paths:
        raise SystemExit(f"No trace_*.json files found in {args.traces_dir}")

    num_traces = len(trace_paths)
    num_layers = None
    num_experts = None
    segments = ["prompt", "analysis", "final", "all"]

    seg_counts: Dict[str, np.ndarray] = {}
    seg_token_counts: Dict[str, int] = {s: 0 for s in segments}

    for path in trace_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        layers = data["layers"]
        if num_layers is None:
            num_layers = len(layers)
        if num_experts is None:
            num_experts = layers[0].get("num_experts", 32)
            for seg in segments:
                seg_counts[seg] = np.zeros((num_layers, num_experts), dtype=np.int64)

        tokens = [td["token"] for td in layers[0]["token_data"]]
        spans = slice_tokens(tokens)
        spans["all"] = (0, len(tokens))

        for seg in segments:
            start, end = spans.get(seg, (0, 0))
            if end <= start:
                continue
            num_tokens = end - start
            seg_token_counts[seg] += num_tokens
            for li, layer in enumerate(layers):
                td = layer["token_data"]
                for ti in range(start, end):
                    top1 = td[ti]["topk_experts"][0]
                    seg_counts[seg][li, top1] += 1

    assert num_layers is not None and num_experts is not None

    report = {
        "meta": {
            "traces_dir": args.traces_dir,
            "num_traces": num_traces,
            "num_layers": num_layers,
            "num_experts": num_experts,
            "pct": args.pct,
            "segments": segments,
        },
        "segments": {},
        "layers": {},
    }

    for seg in segments:
        layer_counts = seg_counts[seg]
        global_counts = layer_counts.sum(axis=0)
        heavy, sparse, freqs = hitters_from_counts(global_counts, args.pct)
        report["segments"][seg] = {
            "total_tokens": seg_token_counts[seg],
            "total_top1_events": int(global_counts.sum()),
            "heavy_hitters": heavy,
            "sparse_hitters": sparse,
        }

        per_layer = []
        for li in range(num_layers):
            l_counts = layer_counts[li]
            l_heavy, l_sparse, l_freqs = hitters_from_counts(l_counts, args.pct)
            per_layer.append(
                {
                    "layer": li,
                    "total_top1_events": int(l_counts.sum()),
                    "heavy_hitters": l_heavy,
                    "sparse_hitters": l_sparse,
                }
            )
        report["layers"][seg] = per_layer

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote report to {args.output}")

    # Brief console summary for the 'all' segment
    overall = report["segments"]["all"]
    print(
        f"Traces: {num_traces}, layers: {num_layers}, experts: {num_experts}, pct: {args.pct*100:.1f}%"
    )
    print("Overall heavy hitters (top):")
    for item in overall["heavy_hitters"]:
        print(
            f"  expert {item['expert']:2d}: {item['count']} selections ({item['pct']:.2f}%)"
        )
    print("Overall sparse hitters (bottom):")
    for item in overall["sparse_hitters"]:
        print(
            f"  expert {item['expert']:2d}: {item['count']} selections ({item['pct']:.4f}%)"
        )


if __name__ == "__main__":
    main()
