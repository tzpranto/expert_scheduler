"""
LSTM-based Expert Pool Predictor Simulator

Uses trained LSTM model to predict upcoming experts and prefetch them into the pool.
Compares with LRU baseline for expert pool management.
"""

import json
import torch
import numpy as np
import sys
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

# Support both module execution and direct execution
try:
    from .bilstm_model import BiLSTMExpertPredictor
    from .data_loader import ModelConfig, ExpertDataLoader
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from bilstm_model import BiLSTMExpertPredictor
    from data_loader import ModelConfig, ExpertDataLoader


class ExpertPoolWithPrediction:
    """Expert pool with LSTM prediction"""

    def __init__(self, pool_size: int, num_layers: int, num_experts: int):
        self.pool_size = pool_size
        self.num_layers = num_layers
        self.num_experts = num_experts

        # Pool state: maps (layer, expert) -> access_time
        self.pool = OrderedDict()
        self.timestamp = 0

        # Statistics
        self.hits = 0
        self.misses = 0
        self.total_accesses = 0
        self.predictions_used = 0
        self.predictions_accurate = 0

    def access(self, layer: int, expert: int) -> bool:
        """Access an expert. Returns True if hit, False if miss."""
        key = (layer, expert)
        self.total_accesses += 1
        self.timestamp += 1

        if key in self.pool:
            self.pool.move_to_end(key)
            self.pool[key] = self.timestamp
            self.hits += 1
            return True
        else:
            self.misses += 1
            self.pool[key] = self.timestamp

            if len(self.pool) > self.pool_size:
                self.pool.popitem(last=False)

            return False

    def prefetch(self, predicted_experts: List[Tuple[int, int]]):
        """Prefetch predicted experts into pool"""
        for layer, expert in predicted_experts:
            if (layer, expert) not in self.pool:
                self.pool[(layer, expert)] = self.timestamp
                self.predictions_used += 1

                if len(self.pool) > self.pool_size:
                    self.pool.popitem(last=False)

    def check_prediction_accuracy(self, predicted_experts: set, actual_experts: set):
        """Check if predicted experts match actual experts"""
        if len(predicted_experts) > 0:
            accuracy = len(predicted_experts & actual_experts) / len(predicted_experts)
            self.predictions_accurate += accuracy

    def access_batch(self, required_experts: List[Tuple[int, int]]) -> int:
        """Access a batch of experts, returns number of hits"""
        hits_in_batch = 0
        for layer, expert in required_experts:
            if self.access(layer, expert):
                hits_in_batch += 1
        return hits_in_batch

    def get_stats(self) -> Dict:
        """Get current statistics"""
        hit_rate = self.hits / max(self.total_accesses, 1)
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_accesses': self.total_accesses,
            'hit_rate': hit_rate,
            'predictions_used': self.predictions_used,
            'predictions_accurate': self.predictions_accurate,
        }


class LSTMSimulator:
    """LSTM-based expert pool simulator with prediction"""

    def __init__(
        self,
        config: ModelConfig,
        pool_size: int,
        model: Optional[BiLSTMExpertPredictor] = None,
        device: str = 'cpu',
    ):
        self.config = config
        self.pool_size = pool_size
        self.model = model
        self.device = device

        self.pool = ExpertPoolWithPrediction(
            pool_size,
            config.num_layers,
            config.num_experts,
        )

        # Context buffer for LSTM (stores last 10 probability distributions)
        self.context_buffer = []
        self.context_size = 10

        # Stage-specific statistics
        self.stage_stats = {
            'prefill': {'hits': 0, 'misses': 0, 'tokens': 0},
            'analysis': {'hits': 0, 'misses': 0, 'tokens': 0},
            'gen': {'hits': 0, 'misses': 0, 'tokens': 0},
        }

    def _get_probability_distribution(
        self, topk_experts_per_layer: List[List[int]], topk_probs_per_layer: Optional[List[List[float]]] = None
    ) -> np.ndarray:
        """
        Create probability distribution from top-K experts.
        Returns [num_layers, num_experts] array.
        """
        probs = np.zeros((self.config.num_layers, self.config.num_experts), dtype=np.float32)

        for layer_idx, experts in enumerate(topk_experts_per_layer):
            if layer_idx >= self.config.num_layers:
                continue

            # Get top-K experts
            topk = experts[:self.config.top_k]

            # Get probabilities if available
            if topk_probs_per_layer and layer_idx < len(topk_probs_per_layer):
                probs_list = topk_probs_per_layer[layer_idx][:self.config.top_k]
            else:
                probs_list = [1.0 / len(topk)] * len(topk)

            # Assign probabilities
            sum_topk = sum(probs_list) if probs_list else 0
            remaining_prob = (1.0 - sum_topk) / (self.config.num_experts - self.config.top_k)

            for expert_id, prob in zip(topk, probs_list):
                if 0 <= expert_id < self.config.num_experts:
                    probs[layer_idx, expert_id] = prob

            for expert_id in range(self.config.num_experts):
                if expert_id not in topk:
                    probs[layer_idx, expert_id] = remaining_prob

        return probs

    def _predict_next_experts(self) -> Optional[np.ndarray]:
        """
        Use LSTM to predict next experts.
        Returns [num_layers, num_experts] probability distribution or None if can't predict.
        """
        if self.model is None or len(self.context_buffer) < self.context_size:
            return None

        # Prepare input: last 10 probability distributions
        X = np.array(self.context_buffer[-self.context_size:], dtype=np.float32)  # [10, layers, experts]
        X = torch.from_numpy(X).unsqueeze(0).to(self.device)  # [1, 10, layers, experts]

        with torch.no_grad():
            pred = self.model(X)  # [1, layers, experts]
            pred = pred.squeeze(0).cpu().numpy()  # [layers, experts]

        return pred

    def _get_top_k_experts_from_prediction(self, pred: np.ndarray) -> List[Tuple[int, int]]:
        """Extract top-K experts per layer from prediction"""
        experts = []
        for layer_idx in range(pred.shape[0]):
            topk_indices = np.argsort(pred[layer_idx, :])[-self.config.top_k:][::-1]
            for expert_id in topk_indices:
                experts.append((layer_idx, int(expert_id)))
        return experts

    def process_token(
        self,
        topk_experts_per_layer: List[List[int]],
        topk_probs_per_layer: Optional[List[List[float]]] = None,
        stage: str = 'gen',
    ):
        """
        Process a single token and update expert pool with predictions.

        Args:
            topk_experts_per_layer: Top-K expert indices per layer
            topk_probs_per_layer: Top-K probabilities per layer (optional)
            stage: 'prefill', 'analysis', or 'gen'
        """
        if stage not in self.stage_stats:
            raise ValueError(f"Unknown stage: {stage}")

        # Get probability distribution for this token
        prob_dist = self._get_probability_distribution(topk_experts_per_layer, topk_probs_per_layer)

        # Add to context buffer
        self.context_buffer.append(prob_dist)
        if len(self.context_buffer) > self.context_size:
            self.context_buffer.pop(0)

        # Try to predict next experts and prefetch
        if len(self.context_buffer) >= self.context_size and self.model is not None:
            pred = self._predict_next_experts()
            if pred is not None:
                predicted_experts = self._get_top_k_experts_from_prediction(pred)
                self.pool.prefetch(predicted_experts)

        # Extract required experts for this token
        required_experts = []
        actual_experts = set()
        for layer_idx, experts in enumerate(topk_experts_per_layer):
            for expert_id in experts[:self.config.top_k]:
                required_experts.append((layer_idx, expert_id))
                actual_experts.add((layer_idx, expert_id))

        # Access experts in pool
        hits = self.pool.access_batch(required_experts)
        misses = len(required_experts) - hits

        # Update stage statistics
        self.stage_stats[stage]['hits'] += hits
        self.stage_stats[stage]['misses'] += misses
        self.stage_stats[stage]['tokens'] += 1

    def _get_analysis_split_point(self, gen_data: Dict) -> Optional[int]:
        """
        Determine where analysis phase ends and gen phase begins.
        For GPT5OSS, the generated_text contains markers like:
        <|channel|>analysis<|message|>...
        <|end|><|start|>assistant<|channel|>final<|message|>...

        Returns the approximate decode_step index where analysis ends.
        Returns None if no split found (assume all gen).
        """
        generated_text = gen_data.get('generated_text', '')

        # Look for markers indicating analysis/final split
        analysis_marker = '<|channel|>analysis'
        final_marker = '<|channel|>final<|message|>'

        if analysis_marker not in generated_text or final_marker not in generated_text:
            return None

        # Find position of final marker
        final_pos = generated_text.find(final_marker)
        if final_pos == -1:
            return None

        # Estimate which decode_step this corresponds to
        # This is approximate: assume ~average tokens per step
        # For GPT5OSS, typically each token is 4-10 characters
        avg_chars_per_token = 6
        estimated_tokens_in_analysis = final_pos / avg_chars_per_token

        # Clamp to reasonable range
        split_point = max(0, int(estimated_tokens_in_analysis * 0.8))  # Conservative estimate

        return split_point if split_point > 0 else None

    def load_trace_and_simulate(self, trace_file: Path, gen_file: Path) -> Dict:
        """
        Load trace files and simulate with LSTM prediction.

        Args:
            trace_file: Path to prefill trace JSON
            gen_file: Path to generation trace JSON

        Returns:
            Dictionary with simulation results
        """
        # Load prefill trace
        with open(trace_file) as f:
            trace_data = json.load(f)

        layers = trace_data.get('layers', [])
        if not layers:
            return None

        # Process prefill tokens
        num_tokens = len(layers[0].get('token_data', []))
        for token_idx in range(num_tokens):
            topk_experts = []
            topk_probs = []
            for layer_idx, layer_data in enumerate(layers):
                token_data = layer_data.get('token_data', [])
                if token_idx < len(token_data):
                    experts = token_data[token_idx].get('topk_experts', [])
                    probs = token_data[token_idx].get('topk_probs', [])
                    topk_experts.append(experts)
                    topk_probs.append(probs)

            self.process_token(topk_experts, topk_probs, 'prefill')

        # Load generation trace
        with open(gen_file) as f:
            gen_data = json.load(f)

        decode_steps = gen_data.get('decode_steps', [])

        # Try to find analysis/gen split for GPT5OSS
        analysis_split_point = self._get_analysis_split_point(gen_data)

        # Process generation tokens
        for step_idx, step_data in enumerate(decode_steps):
            step_layers = step_data.get('layers', [])
            topk_experts = [[] for _ in range(self.config.num_layers)]
            topk_probs = [[] for _ in range(self.config.num_layers)]

            for layer_data in step_layers:
                layer_idx = layer_data.get('layer', 0)
                if layer_idx < self.config.num_layers:
                    experts = layer_data.get('topk_experts', [])
                    probs = layer_data.get('topk_probs', [])
                    topk_experts[layer_idx] = experts
                    topk_probs[layer_idx] = probs

            # Determine stage for this step
            if analysis_split_point is not None and step_idx < analysis_split_point:
                stage = 'analysis'
            else:
                stage = 'gen'

            self.process_token(topk_experts, topk_probs, stage)

        return self._get_results()

    def _get_results(self) -> Dict:
        """Compute final results"""
        results = {
            'pool_size': self.pool_size,
            'total_experts': self.config.num_layers * self.config.num_experts,
            'pool_percentage': round(100 * self.pool_size / (self.config.num_layers * self.config.num_experts), 1),
        }

        # Overall statistics
        total_hits = sum(s['hits'] for s in self.stage_stats.values())
        total_misses = sum(s['misses'] for s in self.stage_stats.values())
        total_accesses = total_hits + total_misses

        results['overall'] = {
            'hit_rate': total_hits / total_accesses if total_accesses > 0 else 0,
            'miss_rate': total_misses / total_accesses if total_accesses > 0 else 0,
            'total_hits': total_hits,
            'total_misses': total_misses,
        }

        # Per-stage statistics
        results['by_stage'] = {}
        for stage, stats in self.stage_stats.items():
            if stats['tokens'] > 0:
                total = stats['hits'] + stats['misses']
                results['by_stage'][stage] = {
                    'tokens': stats['tokens'],
                    'hit_rate': stats['hits'] / total if total > 0 else 0,
                    'miss_rate': stats['misses'] / total if total > 0 else 0,
                    'hits': stats['hits'],
                    'misses': stats['misses'],
                }

        return results

    def get_summary(self) -> str:
        """Get human-readable summary"""
        results = self._get_results()

        lines = [
            f"\nLSTM Predictor Results",
            f"{'='*60}",
            f"Pool Size: {results['pool_size']} ({results['pool_percentage']}% of total)",
            f"Overall Hit Rate: {results['overall']['hit_rate']:.4f}",
            f"Overall Miss Rate: {results['overall']['miss_rate']:.4f}",
            f"Total Hits: {results['overall']['total_hits']}, Total Misses: {results['overall']['total_misses']}",
            f"\nBy Stage:",
        ]

        for stage, stats in results['by_stage'].items():
            lines.append(
                f"  {stage.capitalize():12} - Hit Rate: {stats['hit_rate']:.4f}, "
                f"Tokens: {stats['tokens']}, Hits: {stats['hits']}, Misses: {stats['misses']}"
            )

        return "\n".join(lines)
