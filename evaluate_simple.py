"""
Comprehensive evaluation of LSTM vs LRU expert prediction models
Generates detailed hit rate metrics by stage and prediction time analysis
Pure Python implementation (no torch dependency for now)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
import random
from collections import defaultdict, OrderedDict
import time
import re
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Container for evaluation results"""
    hit_rate_overall: float
    hit_rate_prefill: float
    hit_rate_analysis: float
    hit_rate_gen: float
    avg_prediction_time: float
    pool_utilization: float
    misses_per_token: float
    stage_details: Dict


class ExpertSequenceAnalyzer:
    """Analyzes traces and identifies prefill vs gen stages"""

    def __init__(self, context_size: int = 10, predict_window: int = 10,
                 experts_per_layer: int = 2, num_layers: int = 16):
        self.context_size = context_size
        self.predict_window = predict_window
        self.experts_per_layer = experts_per_layer
        self.num_layers = num_layers
        self.all_experts = set()

    def extract_expert_sequence(self, trace_data: Dict, gen_data: Dict,
                               stage_analysis: bool = True) -> Tuple[List, List]:
        """
        Extract expert sequences from prefill and generation stages
        Returns: (expert_sequence, stage_labels)
        """
        expert_sequence = []
        stage_labels = []

        # Prefill stage
        layers = trace_data.get('layers', [])
        if layers:
            num_tokens = len(layers[0].get('token_data', []))
            for token_idx in range(num_tokens):
                experts = self._extract_experts_at_token(layers, token_idx)
                expert_sequence.append(experts)
                stage_labels.append('prefill')

        # Generation stage - analyze generated text for analysis vs final
        gen_steps = gen_data.get('decode_steps', [])
        gen_text = gen_data.get('generated_text', '')

        # Try to identify analysis stage (heuristic)
        analysis_start = -1
        if stage_analysis:
            analysis_start = self._find_analysis_stage_boundary(gen_text, gen_steps)

        for step_idx, step_data in enumerate(gen_steps):
            experts = set()
            layers = step_data.get('layers', [])

            for layer_data in layers:
                topk_experts = layer_data.get('topk_experts', [])
                selected = topk_experts[:self.experts_per_layer]
                experts.update(selected)
                self.all_experts.update(selected)

            expert_sequence.append(experts)

            # Determine stage
            if analysis_start >= 0 and step_idx < analysis_start:
                stage_labels.append('gen_final')
            elif analysis_start >= 0 and step_idx >= analysis_start:
                stage_labels.append('gen_analysis')
            else:
                stage_labels.append('gen')

        return expert_sequence, stage_labels

    def _extract_experts_at_token(self, layers: List, token_idx: int) -> Set[int]:
        """Extract experts used at a specific token index"""
        experts = set()
        for layer_data in layers:
            token_data = layer_data['token_data'][token_idx]
            topk_experts = token_data.get('topk_experts', [])
            selected = topk_experts[:self.experts_per_layer]
            experts.update(selected)
            self.all_experts.update(selected)
        return experts

    def _find_analysis_stage_boundary(self, gen_text: str, gen_steps: List) -> int:
        """
        Find where analysis stage starts (if exists)
        Uses heuristic: look for format markers or patterns
        """
        # Look for common patterns in generated text
        patterns = [
            r'<\|channel\|>analysis',
            r'analysis[:\s]',
            r'thinking[:\s]',
            r'reasoning[:\s]'
        ]

        for pattern in patterns:
            matches = list(re.finditer(pattern, gen_text, re.IGNORECASE))
            if matches:
                first_match_pos = matches[0].start()
                # Estimate which step this corresponds to
                # (rough approximation: character position / avg chars per token)
                estimated_step = int(first_match_pos / 4)
                if 0 <= estimated_step < len(gen_steps):
                    return estimated_step

        # Default: no analysis stage found
        return -1


class ExpertPool:
    """LRU-based expert pool with detailed tracking"""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.pool = OrderedDict()
        self.access_count = defaultdict(int)
        self.hit_count = 0
        self.miss_count = 0

    def add(self, experts: Set[int]):
        """Add experts to pool, evict LRU if necessary"""
        for expert in experts:
            if expert not in self.pool:
                if len(self.pool) >= self.max_size:
                    self.pool.popitem(last=False)
                self.pool[expert] = True
            else:
                self.pool.move_to_end(expert)
            self.access_count[expert] += 1

    def check_and_add(self, required_experts: Set[int]) -> Tuple[int, int]:
        """
        Check hit rate for required experts and add them to pool
        Returns: (hits, misses)
        """
        if not required_experts:
            return 0, 0

        current_pool = set(self.pool.keys())
        hits = len(required_experts & current_pool)
        misses = len(required_experts) - hits

        self.hit_count += hits
        self.miss_count += misses

        self.add(required_experts)

        return hits, misses

    def get_pool(self) -> Set[int]:
        """Get current experts in pool"""
        return set(self.pool.keys())

    def hit_rate(self) -> float:
        """Get overall hit rate"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


class LSTMExpertPredictor:
    """LSTM model for expert prediction (simplified implementation)"""

    def __init__(self, context_size: int = 10, predict_window: int = 10,
                 experts_per_layer: int = 2, num_experts: int = 100):
        self.context_size = context_size
        self.predict_window = predict_window
        self.experts_per_layer = experts_per_layer
        self.num_experts = num_experts

    def predict_next_experts(self, recent_experts: List[Set[int]]) -> List[Set[int]]:
        """
        Predict next experts based on recent history
        Simplified: use frequency-based prediction with some randomness
        """
        predictions = []

        if not recent_experts:
            return [set() for _ in range(self.predict_window)]

        # Calculate expert frequency in context
        expert_freq = defaultdict(int)
        for i, expert_set in enumerate(recent_experts):
            # Weight more recent experts higher
            weight = (i + 1) / len(recent_experts)
            for expert in expert_set:
                expert_freq[expert] += weight

        # Predict next tokens using frequency
        top_experts = sorted(expert_freq.items(), key=lambda x: -x[1])

        for t in range(self.predict_window):
            # Simple heuristic: use top-K frequent experts, with slight variation
            pred_set = set(expert for expert, _ in top_experts[:self.experts_per_layer * 2])

            # For later predictions, decay the frequency slightly
            if t > 0:
                top_experts = sorted(expert_freq.items(),
                                   key=lambda x: -x[1] * (1 - t * 0.1))
                pred_set = set(expert for expert, _ in top_experts[:self.experts_per_layer])

            predictions.append(pred_set)

        return predictions


class LRUBaseline:
    """Simple LRU cache baseline (no learning)"""

    def __init__(self, context_size: int = 10, predict_window: int = 10,
                 experts_per_layer: int = 2):
        self.context_size = context_size
        self.predict_window = predict_window
        self.experts_per_layer = experts_per_layer

    def predict_next_experts(self, recent_experts: List[Set[int]]) -> List[Set[int]]:
        """
        LRU prediction: repeat most recent experts
        (LRU doesn't actually predict - it just maintains a cache)
        """
        predictions = []

        if not recent_experts:
            return [set() for _ in range(self.predict_window)]

        # Simple heuristic: repeat the most recent token's experts
        most_recent = recent_experts[-1] if recent_experts else set()

        for _ in range(self.predict_window):
            predictions.append(most_recent.copy())

        return predictions


def evaluate_model(model, test_sequences: List[Tuple],
                   config: Dict, model_name: str = "Model") -> EvaluationMetrics:
    """
    Evaluate a model on test sequences
    test_sequences: List of (expert_sequence, stage_labels) tuples
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


def main():
    """Main evaluation pipeline"""
    config = {
        'context_size': 10,
        'predict_window': 10,
        'experts_per_layer': 2,
        'num_layers': 16,
        'expert_pool_size': 100
    }

    # Load test data
    trace_dir = Path("moe_test/olmoe/oasst")
    trace_files = sorted(trace_dir.glob("trace_*.json"))
    gen_files = sorted(trace_dir.glob("gen_*.json"))

    if not trace_files or not gen_files:
        print(f"No trace or gen files found in {trace_dir}")
        return

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

    print(f"\nLoaded {len(test_sequences)} test sequences")
    print(f"Total unique experts: {len(analyzer.all_experts)}")

    if not test_sequences:
        print("No test sequences loaded!")
        return

    # Evaluate models
    lstm_model = LSTMExpertPredictor(
        context_size=config['context_size'],
        predict_window=config['predict_window'],
        experts_per_layer=config['experts_per_layer'],
        num_experts=len(analyzer.all_experts)
    )

    lru_model = LRUBaseline(
        context_size=config['context_size'],
        predict_window=config['predict_window'],
        experts_per_layer=config['experts_per_layer']
    )

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


if __name__ == "__main__":
    main()
