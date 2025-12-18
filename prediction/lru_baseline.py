"""
LRU (Least Recently Used) baseline model for expert prediction
No training required - uses simple heuristic strategy
"""

from typing import List, Set


class LRUBaseline:
    """
    Simple LRU cache baseline without learning

    Strategy: Repeat the most recent token's experts for future predictions
    Assumes recent experts are likely to appear again in the near future
    """

    def __init__(self, context_size: int = 10, predict_window: int = 10,
                 experts_per_layer: int = 2):
        """
        Initialize LRU baseline

        Args:
            context_size: Number of historical tokens to consider (unused in simple LRU)
            predict_window: Number of tokens to predict ahead
            experts_per_layer: Number of experts to select per layer
        """
        self.context_size = context_size
        self.predict_window = predict_window
        self.experts_per_layer = experts_per_layer

    def predict_next_experts(self, recent_experts: List[Set[int]]) -> List[Set[int]]:
        """
        LRU prediction: repeat most recent experts

        Args:
            recent_experts: List of expert sets from recent tokens (context window)

        Returns:
            List of predicted expert sets for next predict_window tokens
        """
        predictions = []

        if not recent_experts:
            return [set() for _ in range(self.predict_window)]

        # Simple strategy: repeat the most recent token's experts
        most_recent = recent_experts[-1] if recent_experts else set()

        # Predict same experts for all future tokens
        for _ in range(self.predict_window):
            predictions.append(most_recent.copy())

        return predictions

    def __repr__(self) -> str:
        return f"LRUBaseline(context={self.context_size}, predict={self.predict_window})"
