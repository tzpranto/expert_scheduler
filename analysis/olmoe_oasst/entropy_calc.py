"""Compute expert usage entropy from trace JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from typing import Dict, List


TRACE_ROOT = pathlib.Path(__file__).parent / "olmoe_oasst" / "olmoe" / "oasst1"
DEFAULT_OUTDIR = pathlib.Path(__file__).parent


def compute_entropy() -> Dict[str, object]:
    trace_paths = sorted(TRACE_ROOT.glob("trace_*.json"))
    if not trace_paths:
        raise FileNotFoundError(f"No trace_*.json files found in {TRACE_ROOT}")

    num_layers = None
    num_experts = None
    accum: List[List[float]] = []
    token_counts: List[int] = []

    for path in trace_paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if num_layers is None:
            num_layers = int(data["num_layers"])
            num_experts = int(data["num_experts"])
            accum = [[0.0] * num_experts for _ in range(num_layers)]
            token_counts = [0] * num_layers

        for layer in data["layers"]:
            layer_idx = int(layer["layer"])
            for token_entry in layer["topk_per_token"]:
                token_counts[layer_idx] += 1
                for expert_id, prob in zip(
                    token_entry["topk_experts"], token_entry["topk_probs"]
                ):
                    accum[layer_idx][int(expert_id)] += float(prob)

    layer_results = []
    for layer_idx, dist in enumerate(accum):
        total = sum(dist)
        if total == 0:
            entropy = float("nan")
            effective = float("nan")
        else:
            normalized = [v / total for v in dist]
            entropy = -sum(p * math.log(p, 2) for p in normalized if p > 0)
            effective = math.pow(2.0, entropy)
        layer_results.append(
            {
                "layer": layer_idx,
                "entropy": entropy,
                "effective_experts": effective,
                "token_count": token_counts[layer_idx],
                "total_mass": total,
            }
        )

    return {
        "trace_count": len(trace_paths),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "total_tokens": sum(token_counts),
        "layer_results": layer_results,
    }


def format_stdout(stats: Dict[str, object]) -> str:
    lines = [
        f"Traces: {stats['trace_count']}  Layers: {stats['num_layers']}  "
        f"Experts: {stats['num_experts']}  Tokens: {stats['total_tokens']}",
        "layer\tentropy_bits\teff_experts\ttokens\ttotal_mass",
    ]
    for entry in stats["layer_results"]:
        lines.append(
            f"{entry['layer']}\t{entry['entropy']:.4f}\t\t"
            f"{entry['effective_experts']:.2f}\t\t"
            f"{entry['token_count']}\t{entry['total_mass']:.2f}"
        )
    return "\n".join(lines)


def write_csv(stats: Dict[str, object], path: pathlib.Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "entropy_bits", "effective_experts", "tokens", "total_mass"])
        for entry in stats["layer_results"]:
            writer.writerow(
                [
                    entry["layer"],
                    f"{entry['entropy']:.6f}",
                    f"{entry['effective_experts']:.6f}",
                    entry["token_count"],
                    f"{entry['total_mass']:.6f}",
                ]
            )


def write_json(stats: Dict[str, object], path: pathlib.Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def write_markdown(stats: Dict[str, object], path: pathlib.Path) -> None:
    lines = [
        f"Traces: {stats['trace_count']}  |  Layers: {stats['num_layers']}  |  "
        f"Experts: {stats['num_experts']}  |  Tokens: {stats['total_tokens']}",
        "",
        "| layer | entropy_bits | effective_experts | tokens | total_mass |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in stats["layer_results"]:
        lines.append(
            f"| {entry['layer']} | {entry['entropy']:.4f} | {entry['effective_experts']:.2f} | "
            f"{entry['token_count']} | {entry['total_mass']:.2f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute expert entropy across trace_*.json files.")
    parser.add_argument("--outdir", type=pathlib.Path, default=DEFAULT_OUTDIR, help="Directory to write outputs.")
    parser.add_argument("--basename", default="entropy_analysis", help="Base name for saved files.")
    parser.add_argument("--save-csv", action="store_true", help="Write per-layer results to CSV.")
    parser.add_argument("--save-json", action="store_true", help="Write full stats to JSON.")
    parser.add_argument("--save-md", action="store_true", help="Write Markdown table.")
    args = parser.parse_args()

    stats = compute_entropy()
    print(format_stdout(stats))

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.save_csv:
        write_csv(stats, args.outdir / f"{args.basename}.csv")
    if args.save_json:
        write_json(stats, args.outdir / f"{args.basename}.json")
    if args.save_md:
        write_markdown(stats, args.outdir / f"{args.basename}.md")


if __name__ == "__main__":
    main()
