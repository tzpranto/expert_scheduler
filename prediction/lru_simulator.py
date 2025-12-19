"""
LRU (Least Recently Used) Expert Pool Simulator

Simulates expert pool management using traditional LRU cache policy.
Tracks hit/miss rates and processing stages (prefill, analysis, gen).
"""

import json
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np


class ExpertPool:
    """LRU-based expert pool"""

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

    def access(self, layer: int, expert: int) -> bool:
        """
        Access an expert. Returns True if hit, False if miss.
        If miss, expert is brought into pool and LRU expert is evicted.
        """
        key = (layer, expert)
        self.total_accesses += 1
        self.timestamp += 1

        if key in self.pool:
            # Hit: move to end (most recently used)
            self.pool.move_to_end(key)
            self.pool[key] = self.timestamp
            self.hits += 1
            return True
        else:
            # Miss: bring into pool
            self.misses += 1

            # Add new expert
            self.pool[key] = self.timestamp

            # Evict LRU if pool is full
            if len(self.pool) > self.pool_size:
                self.pool.popitem(last=False)

            return False

    def access_batch(self, required_experts: List[Tuple[int, int]]) -> int:
        """
        Access a batch of experts in a token.
        Returns number of hits.
        """
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
            'pool_occupancy': len(self.pool),
        }


class LRUSimulator:
    """LRU cache simulator for expert pool"""

    def __init__(self, pool_size: int, num_layers: int, num_experts: int, top_k: int):
        self.pool_size = pool_size
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = top_k

        self.pool = ExpertPool(pool_size, num_layers, num_experts)

        # Stage-specific statistics
        self.stage_stats = {
            'prefill': {'hits': 0, 'misses': 0, 'tokens': 0},
            'analysis': {'hits': 0, 'misses': 0, 'tokens': 0},
            'gen': {'hits': 0, 'misses': 0, 'tokens': 0},
        }

    def process_token(self, topk_experts_per_layer: List[List[int]], stage: str):
        """
        Process a single token with its top-K experts.

        Args:
            topk_experts_per_layer: List of top-K expert indices for each layer
            stage: 'prefill', 'analysis', or 'gen'
        """
        if stage not in self.stage_stats:
            raise ValueError(f"Unknown stage: {stage}")

        # Extract required experts for this token
        required_experts = []
        for layer_idx, experts in enumerate(topk_experts_per_layer):
            for expert_id in experts[:self.top_k]:
                required_experts.append((layer_idx, expert_id))

        # Access experts in pool
        hits = self.pool.access_batch(required_experts)
        misses = len(required_experts) - hits

        # Update stage statistics
        self.stage_stats[stage]['hits'] += hits
        self.stage_stats[stage]['misses'] += misses
        self.stage_stats[stage]['tokens'] += 1

    def load_trace_and_simulate(self, trace_file: Path, gen_file: Path, include_analysis: bool = False) -> Dict:
        """
        Load trace files and simulate expert pool access.

        Args:
            trace_file: Path to prefill trace JSON
            gen_file: Path to generation trace JSON
            include_analysis: Include analysis stage (for GPT5OSS)

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
            for layer_idx, layer_data in enumerate(layers):
                token_data = layer_data.get('token_data', [])
                if token_idx < len(token_data):
                    experts = token_data[token_idx].get('topk_experts', [])
                    topk_experts.append(experts)

            self.process_token(topk_experts, 'prefill')

        # Load generation trace
        with open(gen_file) as f:
            gen_data = json.load(f)

        decode_steps = gen_data.get('decode_steps', [])

        # Process generation tokens
        for step_data in decode_steps:
            step_layers = step_data.get('layers', [])
            topk_experts = [[] for _ in range(self.num_layers)]

            for layer_data in step_layers:
                layer_idx = layer_data.get('layer', 0)
                if layer_idx < self.num_layers:
                    experts = layer_data.get('topk_experts', [])
                    topk_experts[layer_idx] = experts

            self.process_token(topk_experts, 'gen')

        return self._get_results()

    def _get_results(self) -> Dict:
        """Compute final results"""
        results = {
            'pool_size': self.pool_size,
            'total_experts': self.num_layers * self.num_experts,
            'pool_percentage': round(100 * self.pool_size / (self.num_layers * self.num_experts), 1),
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
            f"\nLRU Simulator Results",
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
