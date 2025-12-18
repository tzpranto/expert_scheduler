"""
Evaluation framework for expert prediction models
Compares LSTM vs LRU baseline with detailed metrics
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

from .common import ExpertSequenceAnalyzer, ExpertPool, EvaluationMetrics
from .lstm_model import LSTMExpertPredictor
from .lru_baseline import LRUBaseline


def evaluate_model(model, test_sequences: List[Tuple],
                   config: Dict, model_name: str = "Model") -> EvaluationMetrics:
    """
    Evaluate a model on test sequences

    Args:
        model: Prediction model with predict_next_experts() method
        test_sequences: List of (expert_sequence, stage_labels) tuples
        config: Configuration dict with context_size, predict_window, expert_pool_size
        model_name: Name of model for logging

    Returns:
        EvaluationMetrics object with detailed results
    """
    print(f"\n=== Evaluating {model_name} ===")

    stage_metrics = defaultdict(lambda: {'hits': 0, 'total': 0})
    overall_hits = 0
    overall_total = 0
    prediction_times = []
    pool_sizes = []

    context_size = config.get('context_size', 10)
    predict_window = config.get('predict_window', 10)
    expert_pool_size = config.get('expert_pool_size', 100)

    for seq_idx, (sequence, stages) in enumerate(test_sequences):
        pool = ExpertPool(expert_pool_size)

        if len(sequence) < context_size:
            continue

        # Initialize pool with first context_size tokens
        for experts in sequence[:context_size]:
            pool.add(experts)

        # Evaluate on remaining tokens
        for i in range(context_size, len(sequence)):
            context = sequence[max(0, i - context_size):i]
            target_experts = sequence[i]
            stage = stages[i] if i < len(stages) else 'gen'

            # Get predictions
            start_time = time.time()
            predictions = model.predict_next_experts(context)
            pred_time = (time.time() - start_time) * 1000  # milliseconds
            prediction_times.append(pred_time)

            # Add predicted experts to pool
            for pred_set in predictions:
                pool.add(pred_set)

            # Evaluate
            hits, misses = pool.check_and_add(target_experts)

            overall_hits += hits
            overall_total += hits + misses

            # Track stage metrics
            stage_metrics[stage]['hits'] += hits
            stage_metrics[stage]['total'] += hits + misses

            # Update pool utilization
            pool_sizes.append(len(pool.get_pool()))

        if (seq_idx + 1) % max(1, len(test_sequences) // 5) == 0:
            print(f"  Progress: {seq_idx + 1}/{len(test_sequences)} sequences processed")

    # Calculate metrics
    hit_rate_overall = overall_hits / overall_total if overall_total > 0 else 0.0

    hit_rates_by_stage = {}
    for stage in ['prefill', 'gen', 'gen_final', 'gen_analysis']:
        if stage_metrics[stage]['total'] > 0:
            hit_rates_by_stage[stage] = (stage_metrics[stage]['hits'] /
                                        stage_metrics[stage]['total'])
        else:
            hit_rates_by_stage[stage] = 0.0

    avg_prediction_time = np.mean(prediction_times) if prediction_times else 0.0
    avg_pool_utilization = np.mean(pool_sizes) / expert_pool_size if pool_sizes else 0.0

    metrics = EvaluationMetrics(
        hit_rate_overall=hit_rate_overall,
        hit_rate_prefill=hit_rates_by_stage.get('prefill', 0.0),
        hit_rate_analysis=hit_rates_by_stage.get('gen_analysis', 0.0),
        hit_rate_gen=hit_rates_by_stage.get('gen', hit_rates_by_stage.get('gen_final', 0.0)),
        avg_prediction_time=avg_prediction_time,
        pool_utilization=avg_pool_utilization,
        misses_per_token=overall_total - overall_hits,
        stage_details=hit_rates_by_stage
    )

    # Print results
    print(f"\n{model_name} Results:")
    print(f"  Overall Hit Rate: {metrics.hit_rate_overall * 100:.2f}%")
    print(f"  Prefill Hit Rate: {metrics.hit_rate_prefill * 100:.2f}%")
    print(f"  Analysis Hit Rate: {metrics.hit_rate_analysis * 100:.2f}%")
    print(f"  Gen Hit Rate: {metrics.hit_rate_gen * 100:.2f}%")
    print(f"  Avg Prediction Time: {metrics.avg_prediction_time:.4f} ms")
    print(f"  Pool Utilization: {metrics.pool_utilization * 100:.2f}%")
    print(f"  Total Misses: {int(metrics.misses_per_token)}")
    print(f"  Stage Breakdown: {metrics.stage_details}")

    return metrics


def load_test_sequences(trace_dir: Path, config: Dict) -> Tuple[List, int]:
    """
    Load test sequences from trace files

    Args:
        trace_dir: Path to directory containing trace_*.json and gen_*.json files
        config: Configuration dict with analyzer parameters

    Returns:
        Tuple of (test_sequences, num_experts)
    """
    trace_files = sorted(trace_dir.glob("trace_*.json"))
    gen_files = sorted(trace_dir.glob("gen_*.json"))

    if not trace_files or not gen_files:
        raise ValueError(f"No trace or gen files found in {trace_dir}")

    # Prepare test sequences
    analyzer = ExpertSequenceAnalyzer(
        context_size=config['context_size'],
        predict_window=config['predict_window'],
        experts_per_layer=config['experts_per_layer'],
        num_layers=config['num_layers']
    )
    test_sequences = []

    print(f"Loading traces from {trace_dir}...")
    for trace_file, gen_file in zip(trace_files, gen_files):
        try:
            with open(trace_file) as f:
                trace_data = json.load(f)
            with open(gen_file) as f:
                gen_data = json.load(f)

            sequence, stages = analyzer.extract_expert_sequence(trace_data, gen_data)
            if sequence:
                test_sequences.append((sequence, stages))
        except Exception as e:
            print(f"Error loading {trace_file}: {e}")

    print(f"Loaded {len(test_sequences)} test sequences")
    print(f"Total unique experts: {len(analyzer.all_experts)}")

    return test_sequences, len(analyzer.all_experts)


def compare_models(trace_dir: Path = None, config: Dict = None):
    """
    Main evaluation pipeline comparing LSTM vs LRU

    Args:
        trace_dir: Path to test traces (default: moe_test/olmoe/oasst)
        config: Configuration dict (default: standard config)
    """
    if trace_dir is None:
        trace_dir = Path("moe_test/olmoe/oasst")

    if config is None:
        config = {
            'context_size': 10,
            'predict_window': 10,
            'experts_per_layer': 2,
            'num_layers': 16,
            'expert_pool_size': 100
        }

    # Load test data
    test_sequences, num_experts = load_test_sequences(trace_dir, config)

    if not test_sequences:
        print("No test sequences loaded!")
        return

    # Create models
    lstm_model = LSTMExpertPredictor(
        context_size=config['context_size'],
        predict_window=config['predict_window'],
        experts_per_layer=config['experts_per_layer'],
        num_experts=num_experts
    )

    lru_model = LRUBaseline(
        context_size=config['context_size'],
        predict_window=config['predict_window'],
        experts_per_layer=config['experts_per_layer']
    )

    # Evaluate models
    lstm_metrics = evaluate_model(lstm_model, test_sequences, config, "LSTM")
    lru_metrics = evaluate_model(lru_model, test_sequences, config, "LRU")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON: LSTM vs LRU")
    print("=" * 70)
    improvement = (lstm_metrics.hit_rate_overall - lru_metrics.hit_rate_overall) * 100
    print(f"\nOverall Hit Rate Improvement: {improvement:+.2f}%")
    print(f"  LSTM: {lstm_metrics.hit_rate_overall * 100:.2f}%")
    print(f"  LRU:  {lru_metrics.hit_rate_overall * 100:.2f}%")

    print(f"\nPrefill Stage:")
    print(f"  LSTM: {lstm_metrics.hit_rate_prefill * 100:.2f}%")
    print(f"  LRU:  {lru_metrics.hit_rate_prefill * 100:.2f}%")

    print(f"\nGeneration Stage:")
    print(f"  LSTM: {lstm_metrics.hit_rate_gen * 100:.2f}%")
    print(f"  LRU:  {lru_metrics.hit_rate_gen * 100:.2f}%")

    print(f"\nAnalysis Stage:")
    print(f"  LSTM: {lstm_metrics.hit_rate_analysis * 100:.2f}%")
    print(f"  LRU:  {lru_metrics.hit_rate_analysis * 100:.2f}%")

    print(f"\nPrediction Time:")
    print(f"  LSTM: {lstm_metrics.avg_prediction_time:.6f} ms")
    print(f"  LRU:  {lru_metrics.avg_prediction_time:.6f} ms")

    print(f"\nPool Utilization:")
    print(f"  LSTM: {lstm_metrics.pool_utilization * 100:.2f}%")
    print(f"  LRU:  {lru_metrics.pool_utilization * 100:.2f}%")

    print("\n" + "=" * 70)

    return lstm_metrics, lru_metrics


if __name__ == "__main__":
    compare_models()
