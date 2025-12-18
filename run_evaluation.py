#!/usr/bin/env python3
"""
Quick evaluation script for expert prediction models
Usage: python3 run_evaluation.py [--trace-dir <path>]
"""

import argparse
from pathlib import Path
from prediction.evaluate import compare_models


def main():
    parser = argparse.ArgumentParser(description="Evaluate expert prediction models")
    parser.add_argument('--trace-dir', type=str, default='moe_test/olmoe/oasst',
                       help='Path to trace directory (default: moe_test/olmoe/oasst)')
    parser.add_argument('--pool-size', type=int, default=100,
                       help='Expert pool size (default: 100)')
    parser.add_argument('--context-size', type=int, default=10,
                       help='Context window size (default: 10)')
    parser.add_argument('--predict-window', type=int, default=10,
                       help='Prediction window (default: 10)')
    parser.add_argument('--experts-per-layer', type=int, default=2,
                       help='Top-K experts per layer (default: 2)')

    args = parser.parse_args()

    config = {
        'context_size': args.context_size,
        'predict_window': args.predict_window,
        'experts_per_layer': args.experts_per_layer,
        'num_layers': 16,
        'expert_pool_size': args.pool_size
    }

    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists():
        print(f"Error: Trace directory {trace_dir} does not exist")
        return

    compare_models(trace_dir=trace_dir, config=config)


if __name__ == "__main__":
    main()
