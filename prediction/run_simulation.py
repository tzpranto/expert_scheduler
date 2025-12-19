#!/usr/bin/env python3
"""
Simulation Runner - Compare LRU vs LSTM Prediction

Runs expert pool simulations for different pool sizes and compares:
- LRU baseline
- LSTM-based prediction
- Metrics: hit rate, miss rate, by stage (prefill, analysis, gen)
"""

import torch
import json
from pathlib import Path
import argparse
import sys
import time
from typing import Dict, List

# Support both module and direct execution
try:
    from .lru_simulator import LRUSimulator
    from .lstm_simulator import LSTMSimulator
    from .data_loader import OLMOE_CONFIG, GPT5OSS_CONFIG, ModelConfig
    from .bilstm_model import BiLSTMExpertPredictor
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from lru_simulator import LRUSimulator
    from lstm_simulator import LSTMSimulator
    from data_loader import OLMOE_CONFIG, GPT5OSS_CONFIG, ModelConfig
    from bilstm_model import BiLSTMExpertPredictor


class SimulationRunner:
    """Run and compare expert pool simulations"""

    def __init__(self, config: ModelConfig, model_path: Path, device: str = 'cpu'):
        self.config = config
        self.device = device

        # Load trained model
        self.model = None
        if model_path.exists():
            self.model = BiLSTMExpertPredictor(
                num_layers=config.num_layers,
                num_experts=config.num_experts,
                hidden_size=512,
            ).to(device)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            print(f"✓ Loaded model from {model_path}")
        else:
            print(f"⚠ Model not found at {model_path}, running LSTM simulator without model")

    def run_simulation_on_file(
        self, trace_file: Path, gen_file: Path, pool_percentages: List[int]
    ) -> Dict:
        """
        Run both LRU and LSTM simulations on a single file.

        Args:
            trace_file: Path to prefill trace
            gen_file: Path to generation trace
            pool_percentages: Pool sizes as percentages (e.g., [15, 25, 50])

        Returns:
            Dictionary with results for each pool size
        """
        total_experts = self.config.num_layers * self.config.num_experts

        results = {
            'file': trace_file.name,
            'model': self.config.name,
            'total_experts': total_experts,
            'simulations': {},
        }

        for pct in pool_percentages:
            pool_size = max(1, int(total_experts * pct / 100))

            print(f"\n  Running simulations for pool size {pool_size} ({pct}%)")

            # LRU simulation
            start_time = time.time()
            lru_sim = LRUSimulator(pool_size, self.config.num_layers, self.config.num_experts, self.config.top_k)
            lru_result = lru_sim.load_trace_and_simulate(trace_file, gen_file)
            lru_time = time.time() - start_time

            # LSTM simulation
            start_time = time.time()
            lstm_sim = LSTMSimulator(self.config, pool_size, self.model, self.device)
            lstm_result = lstm_sim.load_trace_and_simulate(trace_file, gen_file)
            lstm_time = time.time() - start_time

            results['simulations'][pct] = {
                'pool_size': pool_size,
                'lru': {
                    'results': lru_result,
                    'time': lru_time,
                    'hit_rate': lru_result['overall']['hit_rate'],
                },
                'lstm': {
                    'results': lstm_result,
                    'time': lstm_time,
                    'hit_rate': lstm_result['overall']['hit_rate'],
                    'improvement': lstm_result['overall']['hit_rate'] - lru_result['overall']['hit_rate'],
                },
            }

            # Print summary
            print(f"    LRU  - Hit Rate: {lru_result['overall']['hit_rate']:.4f}, Time: {lru_time:.3f}s")
            print(f"    LSTM - Hit Rate: {lstm_result['overall']['hit_rate']:.4f}, Time: {lstm_time:.3f}s")
            improvement = lstm_result['overall']['hit_rate'] - lru_result['overall']['hit_rate']
            print(f"    Improvement: {improvement:+.4f}")

        return results

    def print_detailed_results(self, results: Dict):
        """Print detailed results for all simulations"""
        print("\n" + "=" * 80)
        print("DETAILED SIMULATION RESULTS")
        print("=" * 80)

        for pct in sorted(results['simulations'].keys()):
            sim_result = results['simulations'][pct]

            print(f"\n{'='*80}")
            print(f"Pool Size: {sim_result['pool_size']} ({pct}% of {results['total_experts']} total experts)")
            print(f"{'='*80}")

            # LRU Results
            print(f"\nLRU Baseline:")
            lru_result = sim_result['lru']['results']
            print(f"  Overall Hit Rate: {lru_result['overall']['hit_rate']:.4f}")
            print(f"  Overall Miss Rate: {lru_result['overall']['miss_rate']:.4f}")
            print(f"  Total Hits: {lru_result['overall']['total_hits']}, Misses: {lru_result['overall']['total_misses']}")

            for stage in ['prefill', 'analysis', 'gen']:
                if stage in lru_result['by_stage']:
                    stage_data = lru_result['by_stage'][stage]
                    print(
                        f"    {stage.capitalize():12} - Hit Rate: {stage_data['hit_rate']:.4f}, "
                        f"Tokens: {stage_data['tokens']}, Hits: {stage_data['hits']}, Misses: {stage_data['misses']}"
                    )

            # LSTM Results
            print(f"\nLSTM Prediction:")
            lstm_result = sim_result['lstm']['results']
            print(f"  Overall Hit Rate: {lstm_result['overall']['hit_rate']:.4f}")
            print(f"  Overall Miss Rate: {lstm_result['overall']['miss_rate']:.4f}")
            print(f"  Total Hits: {lstm_result['overall']['total_hits']}, Misses: {lstm_result['overall']['total_misses']}")

            for stage in ['prefill', 'analysis', 'gen']:
                if stage in lstm_result['by_stage']:
                    stage_data = lstm_result['by_stage'][stage]
                    print(
                        f"    {stage.capitalize():12} - Hit Rate: {stage_data['hit_rate']:.4f}, "
                        f"Tokens: {stage_data['tokens']}, Hits: {stage_data['hits']}, Misses: {stage_data['misses']}"
                    )

            # Comparison
            improvement = sim_result['lstm']['improvement']
            print(f"\nComparison:")
            print(f"  LSTM Improvement: {improvement:+.4f} ({100*improvement/lru_result['overall']['hit_rate']:+.2f}%)")
            print(f"  LRU Time: {sim_result['lru']['time']:.3f}s")
            print(f"  LSTM Time: {sim_result['lstm']['time']:.3f}s")

    def save_results(self, results: Dict, output_file: Path):
        """Save results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run expert pool simulations")
    parser.add_argument(
        "--model", type=str, choices=["olmoe", "gpt5oss"], default="olmoe", help="Model type"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (trace and gen files)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained LSTM model (optional)",
    )
    parser.add_argument(
        "--pool-percentages",
        type=int,
        nargs='+',
        default=[15, 25, 50],
        help="Pool sizes as percentages (default: 15 25 50)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    # Select config
    if args.model == "olmoe":
        config = OLMOE_CONFIG
        data_dir = Path("moe_test/olmoe/oasst") if args.data_dir is None else Path(args.data_dir)
        model_path = Path("models/olmoe_lstm.pt") if args.model_path is None else Path(args.model_path)
    else:
        config = GPT5OSS_CONFIG
        data_dir = Path("moe_test/gpt5oss/oasst") if args.data_dir is None else Path(args.data_dir)
        model_path = Path("models/gpt5oss_lstm.pt") if args.model_path is None else Path(args.model_path)

    print("=" * 80)
    print(f"EXPERT POOL SIMULATION - {config.name.upper()}")
    print("=" * 80)
    print(f"Model: {config.name}")
    print(f"Total Experts: {config.num_layers} layers × {config.num_experts} experts = {config.num_layers * config.num_experts}")
    print(f"Top-K: {config.top_k}")
    print(f"Pool Percentages: {args.pool_percentages}")
    print(f"Data Directory: {data_dir}")
    print()

    # Find trace and gen files
    trace_file = data_dir / "trace_0000.json"
    gen_file = data_dir / "gen_0000.json"

    if not trace_file.exists() or not gen_file.exists():
        print(f"ERROR: Data files not found:")
        print(f"  Trace: {trace_file}")
        print(f"  Gen: {gen_file}")
        return

    print(f"✓ Found data files")
    print(f"  Trace: {trace_file}")
    print(f"  Gen: {gen_file}")
    print()

    # Create runner
    runner = SimulationRunner(config, model_path, args.device)

    # Run simulations
    print("Starting simulations...")
    results = runner.run_simulation_on_file(trace_file, gen_file, args.pool_percentages)

    # Print detailed results
    runner.print_detailed_results(results)

    # Save results
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = Path("results") / f"simulation_{config.name}_{int(time.time())}.json"
        output_file.parent.mkdir(exist_ok=True)

    runner.save_results(results, output_file)


if __name__ == "__main__":
    main()
