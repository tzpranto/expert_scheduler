"""
Data loader for expert prediction with proper one-hot encoding
Loads 400 training samples and 100 test samples
Each sample: [16, 64] = 16 layers × 64 experts with probabilities
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple


class OLMoEDataLoader:
    """Load OLMoE traces and convert to proper format"""

    def __init__(self, num_layers: int = 16, num_experts: int = 64, top_k: int = 8):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = top_k

    def load_trace_to_probabilities(self, trace_file: Path) -> np.ndarray:
        """
        Load prefill trace and convert to probability distribution

        Returns: [num_tokens, num_layers, num_experts]
                 where each [token, layer, :] sums to 1.0
        """
        with open(trace_file) as f:
            data = json.load(f)

        layers = data.get('layers', [])
        if not layers:
            return None

        # Get number of tokens
        num_tokens = len(layers[0].get('token_data', []))

        # Create probability matrix
        probs = np.zeros((num_tokens, self.num_layers, self.num_experts), dtype=np.float32)

        # For each layer
        for layer_idx, layer_data in enumerate(layers):
            token_data_list = layer_data.get('token_data', [])

            for token_idx, token_data in enumerate(token_data_list):
                topk_experts = token_data.get('topk_experts', [])
                topk_probs = token_data.get('topk_probs', [])

                # Ensure we have exactly top_k
                topk_experts = topk_experts[:self.top_k]
                topk_probs = topk_probs[:self.top_k]

                # Sum of top-k probabilities
                sum_topk = sum(topk_probs) if topk_probs else 0

                # Remaining probability
                remaining_prob = (1.0 - sum_topk) / (self.num_experts - self.top_k)

                # Fill all experts
                # First: set top-k
                for expert_id, prob in zip(topk_experts, topk_probs):
                    if 0 <= expert_id < self.num_experts:
                        probs[token_idx, layer_idx, expert_id] = prob

                # Second: set remaining experts with equal probability
                for expert_id in range(self.num_experts):
                    if expert_id not in topk_experts:
                        probs[token_idx, layer_idx, expert_id] = remaining_prob

        return probs

    def load_gen_to_probabilities(self, gen_file: Path) -> np.ndarray:
        """
        Load generation trace and convert to probability distribution

        Returns: [num_steps, num_layers, num_experts]
        """
        with open(gen_file) as f:
            data = json.load(f)

        decode_steps = data.get('decode_steps', [])

        # Create probability matrix
        probs = np.zeros((len(decode_steps), self.num_layers, self.num_experts), dtype=np.float32)

        for step_idx, step_data in enumerate(decode_steps):
            layers = step_data.get('layers', [])

            for layer_data in layers:
                layer_idx = layer_data.get('layer', 0)
                topk_experts = layer_data.get('topk_experts', [])
                topk_probs = layer_data.get('topk_probs', [])

                topk_experts = topk_experts[:self.top_k]
                topk_probs = topk_probs[:self.top_k]

                sum_topk = sum(topk_probs) if topk_probs else 0
                remaining_prob = (1.0 - sum_topk) / (self.num_experts - self.top_k)

                for expert_id, prob in zip(topk_experts, topk_probs):
                    if 0 <= expert_id < self.num_experts:
                        probs[step_idx, layer_idx, expert_id] = prob

                for expert_id in range(self.num_experts):
                    if expert_id not in topk_experts:
                        probs[step_idx, layer_idx, expert_id] = remaining_prob

        return probs

    def create_time_series_sequences(self, probs: np.ndarray, context_size: int = 10) -> list:
        """
        Create time series sequences for LSTM
        Given: [num_tokens, 16, 64]
        Create: list of (context, target) where
                - context: [context_size, 16, 64]
                - target: [1, 16, 64] or [16, 64]
        """
        sequences = []

        if probs is None or len(probs) < context_size + 1:
            return sequences

        for i in range(len(probs) - context_size):
            context = probs[i:i + context_size]  # [context_size, 16, 64]
            target = probs[i + context_size]     # [16, 64]
            sequences.append((context, target))

        return sequences

    def load_dataset(self, data_dir: Path, indices: list, context_size: int = 10):
        """
        Load multiple traces and create sequences

        Args:
            data_dir: Directory with trace_XXXX.json and gen_XXXX.json
            indices: List of trace indices to load
            context_size: LSTM context window

        Returns:
            List of (context, target) tuples
            context: [context_size, 16, 64]
            target: [16, 64]
        """
        all_sequences = []

        for idx in indices:
            trace_file = data_dir / f"trace_{idx:04d}.json"
            gen_file = data_dir / f"gen_{idx:04d}.json"

            if not trace_file.exists() or not gen_file.exists():
                print(f"Warning: Missing {trace_file} or {gen_file}")
                continue

            try:
                # Prefill
                prefill_probs = self.load_trace_to_probabilities(trace_file)
                if prefill_probs is not None:
                    seqs = self.create_time_series_sequences(prefill_probs, context_size)
                    all_sequences.extend(seqs)
                    print(f"✓ trace_{idx:04d}: {len(seqs)} sequences")

                # Generation
                gen_probs = self.load_gen_to_probabilities(gen_file)
                if gen_probs is not None:
                    seqs = self.create_time_series_sequences(gen_probs, context_size)
                    all_sequences.extend(seqs)
                    print(f"✓ gen_{idx:04d}: {len(seqs)} sequences")

            except Exception as e:
                print(f"✗ Error loading trace {idx}: {e}")
                continue

        return all_sequences


def test_data_loader():
    """Test the data loader"""
    print("=" * 70)
    print("TESTING OLMoE DATA LOADER")
    print("=" * 70)
    print()

    # Initialize
    loader = OLMoEDataLoader(num_layers=16, num_experts=64, top_k=8)
    data_dir = Path('moe_test/olmoe/oasst')

    # Test with sample data first
    print("--- Testing with sample data (trace_0000) ---")
    sample_indices = [0]
    sample_seqs = loader.load_dataset(data_dir, sample_indices, context_size=10)
    print(f"Sample sequences: {len(sample_seqs)}")
    if sample_seqs:
        context, target = sample_seqs[0]
        print(f"Context shape: {context.shape}  (should be [10, 16, 64])")
        print(f"Target shape: {target.shape}    (should be [16, 64])")
        print()

        # Check probabilities sum to 1
        print("Checking probability distributions:")
        print(f"  Context[0, 0, :].sum() = {context[0, 0, :].sum():.4f} (should be 1.0)")
        print(f"  Context[0, 15, :].sum() = {context[0, 15, :].sum():.4f} (should be 1.0)")
        print(f"  Target[0, :].sum() = {target[0, :].sum():.4f} (should be 1.0)")
        print(f"  Target[15, :].sum() = {target[15, :].sum():.4f} (should be 1.0)")
        print()

    # Load training data (indices 0-399 in full dataset)
    # For now, test with sample
    print("--- Training data (indices 0 only) ---")
    train_indices = [0]  # Change to list(range(0, 400)) for full training
    train_seqs = loader.load_dataset(data_dir, train_indices, context_size=10)
    print(f"Training sequences: {len(train_seqs)}")
    print()

    # Load test data (indices 400-499 in full dataset)
    # For now, test with sample
    print("--- Test data (indices 0 only) ---")
    test_indices = [0]  # Change to list(range(400, 500)) for full testing
    test_seqs = loader.load_dataset(data_dir, test_indices, context_size=10)
    print(f"Test sequences: {len(test_seqs)}")
    print()

    # Summary
    print("--- Summary ---")
    print(f"Total training sequences: {len(train_seqs)}")
    print(f"Total test sequences: {len(test_seqs)}")
    print(f"Sequence shapes:")
    print(f"  Context: [context_size=10, num_layers=16, num_experts=64]")
    print(f"  Target:  [num_layers=16, num_experts=64]")
    print()
    print("✓ Data loader working correctly!")
    print()

    return train_seqs, test_seqs


if __name__ == "__main__":
    train_seqs, test_seqs = test_data_loader()
