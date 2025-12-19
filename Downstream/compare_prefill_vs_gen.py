#!/usr/bin/env python
"""
Compare expert heavy hitters between prefill (encoder-style, parallel token ingest)
and generation (auto-regressive decode) stages.

The script expects a directory containing both `trace_*.json` (prefill) files and
`gen_*.json` (generation) files produced by the tracing pipeline used in this repo.

Outputs a JSON report summarizing:
- Top/bottom experts (top-1 router choice) for prefill and generation separately.
- Overlap between heavy-hitter sets.
- Distribution distances (total variation + Jensen-Shannon) between stages.

Example:
    python compare_prefill_vs_gen.py \\
        --dataset gpt5oss_oasst \\
        --dir Downstream/gpt5oss_oasst/gpt5oss/oasst \\
        --pct 0.10 \\
        --output Downstream/gpt5oss_oasst/prefill_vs_gen_report.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_prefill_counts(paths: List[str]) -> Tuple[np.ndarray, int]:
    """
    Aggregate top-1 expert counts across prefill trace_*.json files.
    Supports both GPT5-style (token_data) and OLMoE-style (topk_per_token) layouts.
    Returns (counts_per_layer_expert, token_count).
    """
    counts = None
    token_count = 0

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        num_layers = data["num_layers"]
        num_experts = data["num_experts"]
        if counts is None:
            counts = np.zeros((num_layers, num_experts), dtype=np.int64)

        for li, layer in enumerate(data["layers"]):
            tokens = None
            if "token_data" in layer:
                tokens = layer["token_data"]
            elif "topk_per_token" in layer:
                tokens = layer["topk_per_token"]
            else:
                raise KeyError(f"Unrecognized layer schema in {path}")

            if li == 0:
                token_count += len(tokens)

            for entry in tokens:
                top1 = int(entry["topk_experts"][0])
                counts[li, top1] += 1

    if counts is None:
        raise RuntimeError("No prefill traces found.")

    return counts, token_count


def load_generation_counts(paths: List[str]) -> Tuple[np.ndarray, int]:
    """
    Aggregate top-1 expert counts across gen_*.json decode traces.
    """
    counts = None
    token_count = 0

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        num_layers = data["num_layers"]
        num_experts = data["num_experts"]
        if counts is None:
            counts = np.zeros((num_layers, num_experts), dtype=np.int64)

        for step in data["decode_steps"]:
            token_count += 1
            for li, layer in enumerate(step["layers"]):
                top1 = int(layer["topk_experts"][0])
                counts[li, top1] += 1

    if counts is None:
        raise RuntimeError("No generation traces found.")

    return counts, token_count


def hitters_from_counts(counts: np.ndarray, pct: float) -> Tuple[List[dict], List[dict], np.ndarray]:
    """
    Given 2D counts [layers, experts], return heavy/sparse hitters lists and freqs.
    """
    total_counts = counts.sum(axis=0)
    total = float(total_counts.sum())
    freqs = total_counts / total if total > 0 else np.zeros_like(total_counts, dtype=float)

    n = max(1, int(math.ceil(len(total_counts) * pct)))
    desc = np.argsort(-freqs)
    asc = np.argsort(freqs)

    def pack(order: np.ndarray) -> List[dict]:
        return [
            {"expert": int(idx), "count": int(total_counts[idx]), "pct": float(freqs[idx] * 100.0)}
            for idx in order[:n]
        ]

    return pack(desc), pack(asc), freqs


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base e) with small epsilon for stability."""
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return 0.5 * (kl_pm + kl_qm)


def summarize_stage(counts: np.ndarray, pct: float) -> Dict:
    heavy, sparse, freqs = hitters_from_counts(counts, pct)
    return {
        "total_top1_events": int(counts.sum()),
        "heavy_hitters": heavy,
        "sparse_hitters": sparse,
        "freqs": freqs.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare prefill vs generation heavy hitters.")
    parser.add_argument("--dataset", required=True, help="Label for the dataset (used in report).")
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing trace_*.json (prefill) and gen_*.json (generation) files.",
    )
    parser.add_argument("--pct", type=float, default=0.10, help="Fraction for heavy/sparse hitter selection.")
    parser.add_argument("--output", required=True, help="Path to write the JSON comparison report.")
    args = parser.parse_args()

    trace_paths = sorted(glob.glob(os.path.join(args.dir, "trace_*.json")))
    gen_paths = sorted(glob.glob(os.path.join(args.dir, "gen_*.json")))

    if not trace_paths:
        raise SystemExit(f"No trace_*.json files in {args.dir}")
    if not gen_paths:
        raise SystemExit(f"No gen_*.json files in {args.dir}")

    pre_counts, pre_tokens = load_prefill_counts(trace_paths)
    gen_counts, gen_tokens = load_generation_counts(gen_paths)

    if pre_counts.shape != gen_counts.shape:
        raise SystemExit(
            f"Shape mismatch between prefill {pre_counts.shape} and generation {gen_counts.shape} counts."
        )

    num_layers, num_experts = pre_counts.shape
    pre_summary = summarize_stage(pre_counts, args.pct)
    gen_summary = summarize_stage(gen_counts, args.pct)

    pre_freqs = np.array(pre_summary["freqs"])
    gen_freqs = np.array(gen_summary["freqs"])

    shared = {h["expert"] for h in pre_summary["heavy_hitters"]}.intersection(
        {h["expert"] for h in gen_summary["heavy_hitters"]}
    )
    overlap = {
        "shared_experts": sorted(int(x) for x in shared),
        "prefill_only": sorted(
            int(x["expert"]) for x in pre_summary["heavy_hitters"] if x["expert"] not in shared
        ),
        "generation_only": sorted(
            int(x["expert"]) for x in gen_summary["heavy_hitters"] if x["expert"] not in shared
        ),
    }
    overlap["jaccard"] = (
        len(shared)
        / float(
            len(overlap["shared_experts"])
            + len(overlap["prefill_only"])
            + len(overlap["generation_only"])
        )
        if (overlap["shared_experts"] or overlap["prefill_only"] or overlap["generation_only"])
        else 0.0
    )

    tvd = 0.5 * float(np.abs(pre_freqs - gen_freqs).sum())
    jsd = float(js_divergence(pre_freqs, gen_freqs))

    report = {
        "dataset": args.dataset,
        "dir": args.dir,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "pct": args.pct,
        "files": {"prefill": len(trace_paths), "generation": len(gen_paths)},
        "tokens": {"prefill": pre_tokens, "generation": gen_tokens},
        "prefill": {k: v for k, v in pre_summary.items() if k != "freqs"},
        "generation": {k: v for k, v in gen_summary.items() if k != "freqs"},
        "distance": {"tvd": tvd, "js_divergence": jsd},
        "overlap": overlap,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[{args.dataset}] wrote comparison report to {args.output}")
    print(
        f"  tokens prefill/generation: {pre_tokens}/{gen_tokens}; "
        f"heavy-overlap jaccard: {overlap['jaccard']:.3f}; "
        f"TVD: {tvd:.4f}; JSD: {jsd:.4f}"
    )


if __name__ == "__main__":
    main()
