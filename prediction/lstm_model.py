"""
LSTM-based expert prediction model
Learns patterns in expert usage to predict future expert requirements
"""

from typing import List, Set
from collections import defaultdict


class LSTMExpertPredictor:
    """
    LSTM-inspired expert prediction model (simplified pure Python implementation)

    Learns frequency patterns from historical expert usage and uses them
    to predict experts for future tokens. This is a simplified version
    that doesn't require PyTorch.

    For full neural network LSTM, use the PyTorch version in lstm_predictor.py
    """

    def __init__(self, context_size: int = 10, predict_window: int = 10,
                 experts_per_layer: int = 2, num_experts: int = 100):
        """
        Initialize LSTM predictor

        Args:
            context_size: Look-back window for learning context
            predict_window: Number of tokens to predict ahead
            experts_per_layer: Number of experts to select per layer
            num_experts: Total number of experts in the system
        """
        self.context_size = context_size
        self.predict_window = predict_window
        self.experts_per_layer = experts_per_layer
        self.num_experts = num_experts
        self.expert_freq = defaultdict(float)
        self.transition_probs = defaultdict(lambda: defaultdict(float))

    def train(self, sequences: List[List[Set[int]]]):
        """
        Train the model on expert sequences

        Args:
            sequences: List of expert sequences from training data
                      Each sequence is a list of expert sets (one per token)
        """
        # Calculate expert frequency
        for sequence in sequences:
            for experts in sequence:
                for expert in experts:
                    self.expert_freq[expert] += 1.0 / len(sequence) if sequence else 1.0

        # Calculate transition probabilities (expert -> next experts)
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                current_experts = sequence[i]
                next_experts = sequence[i + 1]

                for curr_expert in current_experts:
                    for next_expert in next_experts:
                        self.transition_probs[curr_expert][next_expert] += 1.0

        # Normalize transition probabilities
        for curr_expert in self.transition_probs:
            total = sum(self.transition_probs[curr_expert].values())
            if total > 0:
                for next_expert in self.transition_probs[curr_expert]:
                    self.transition_probs[curr_expert][next_expert] /= total

    def predict_next_experts(self, recent_experts: List[Set[int]]) -> List[Set[int]]:
        """
        Predict next experts based on recent history

        Uses frequency-based prediction with decay for later predictions.
        More recent experts are weighted higher.

        Args:
            recent_experts: List of expert sets from recent tokens (context window)

        Returns:
            List of predicted expert sets for next predict_window tokens
        """
        predictions = []

        if not recent_experts:
            return [set() for _ in range(self.predict_window)]

        # Calculate expert frequency in context with recency weighting
        expert_freq = defaultdict(float)
        for i, expert_set in enumerate(recent_experts):
            # Weight more recent experts higher (linear increase from 0 to 1)
            weight = (i + 1) / len(recent_experts)
            for expert in expert_set:
                expert_freq[expert] += weight

        # Predict next tokens using frequency and transition patterns
        for t in range(self.predict_window):
            # Get top experts by frequency
            sorted_experts = sorted(expert_freq.items(), key=lambda x: -x[1])

            # Select top-K experts (experts_per_layer * 2 for diversity)
            pred_set = set()
            for expert, _ in sorted_experts[:self.experts_per_layer * 2]:
                pred_set.add(expert)

            # If we have transition probabilities, use them to refine predictions
            if t > 0 and recent_experts and self.transition_probs:
                # Use transition probabilities from recent experts
                transition_freq = defaultdict(float)
                for recent_expert in recent_experts[-1]:
                    if recent_expert in self.transition_probs:
                        for next_expert, prob in self.transition_probs[recent_expert].items():
                            transition_freq[next_expert] += prob

                if transition_freq:
                    # Blend frequency and transition predictions
                    sorted_transition = sorted(transition_freq.items(), key=lambda x: -x[1])
                    pred_set = set()
                    for expert, _ in sorted_transition[:self.experts_per_layer]:
                        pred_set.add(expert)

            # Ensure we have at least some experts predicted
            if not pred_set:
                sorted_experts = sorted(expert_freq.items(), key=lambda x: -x[1])
                pred_set = set(expert for expert, _ in sorted_experts[:max(1, self.experts_per_layer)])

            predictions.append(pred_set)

            # Update frequency for next prediction (decay older experts)
            if t < self.predict_window - 1:
                for expert in expert_freq:
                    expert_freq[expert] *= (1 - 0.1 * (t + 1))  # Decay factor

        return predictions

    def __repr__(self) -> str:
        return f"LSTMExpertPredictor(context={self.context_size}, predict={self.predict_window})"
