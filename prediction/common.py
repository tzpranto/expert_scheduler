"""
Common classes and utilities for expert prediction models
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict, OrderedDict
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
        Handles both GPT Harmony format (<|channel|>analysis) and heuristic detection
        """
        # GPT Harmony format: <|channel|>analysis
        harmony_pattern = r'<\|channel\|>\s*analysis'
        matches = list(re.finditer(harmony_pattern, gen_text, re.IGNORECASE))

        if matches:
            first_match_pos = matches[0].start()
            # Estimate which step this corresponds to
            # Each token is roughly 4 characters on average
            estimated_step = int(first_match_pos / 4)
            if 0 <= estimated_step < len(gen_steps):
                return estimated_step

        # Fallback heuristic patterns (less reliable)
        fallback_patterns = [
            r'<\|start\|>.*?<\|channel\|>',  # GPT format: start of response section
            r'\n\s*analysis[:\s]',             # Line-based analysis marker
            r'\n\s*thinking[:\s]',             # Thinking/reasoning marker
            r'\n\s*step\s+\d+[:\s]'            # Step-by-step reasoning
        ]

        for pattern in fallback_patterns:
            matches = list(re.finditer(pattern, gen_text, re.IGNORECASE))
            if matches:
                first_match_pos = matches[-1].end()  # Use last match (more likely to be analysis)
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
