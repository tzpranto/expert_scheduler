"""
Simple dataset loader for expert prediction
Each sample: 10 tokens of [16 layers x 8 experts] → predict next token [16 x 8]
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
import random


class ExpertDataLoader:
    """Load expert activation traces into simple numpy arrays"""

    def __init__(self, context_size: int = 10, experts_per_token: int = 8, num_layers: int = 16):
        """
        Args:
            context_size: Number of previous tokens to use as context
            experts_per_token: Number of experts activated per token (top-K)
            num_layers: Number of layers in MoE model
        """
        self.context_size = context_size
        self.experts_per_token = experts_per_token
        self.num_layers = num_layers

    def load_trace(self, trace_file: Path) -> np.ndarray:
        """
        Load prefill trace and convert to expert activations

        Returns: array of shape [num_tokens, num_layers, experts_per_token]
                 with expert IDs (0-63 for OLMoE)
        """
        with open(trace_file) as f:
            data = json.load(f)

        layers = data.get('layers', [])
        if not layers:
            return None

        # Get number of tokens from first layer
        num_tokens = len(layers[0].get('token_data', []))

        # Create array: [num_tokens, num_layers, experts_per_token]
        activations = np.zeros((num_tokens, self.num_layers, self.experts_per_token), dtype=np.int32)

        # Fill with expert IDs
        for layer_idx, layer_data in enumerate(layers):
            token_data_list = layer_data.get('token_data', [])
            for token_idx, token_data in enumerate(token_data_list):
                topk_experts = token_data.get('topk_experts', [])
                # Take top K experts (should be 8 for OLMoE)
                selected = topk_experts[:self.experts_per_token]
                activations[token_idx, layer_idx, :len(selected)] = selected

        return activations

    def load_gen(self, gen_file: Path) -> np.ndarray:
        """
        Load generation trace and convert to expert activations

        Returns: array of shape [num_steps, num_layers, experts_per_token]
        """
        with open(gen_file) as f:
            data = json.load(f)

        decode_steps = data.get('decode_steps', [])

        # Create array: [num_steps, num_layers, experts_per_token]
        activations = np.zeros((len(decode_steps), self.num_layers, self.experts_per_token), dtype=np.int32)

        # Fill with expert IDs
        for step_idx, step_data in enumerate(decode_steps):
            layers = step_data.get('layers', [])
            for layer_data in layers:
                layer_idx = layer_data.get('layer', 0)
                topk_experts = layer_data.get('topk_experts', [])
                # Take top K experts
                selected = topk_experts[:self.experts_per_token]
                activations[step_idx, layer_idx, :len(selected)] = selected

        return activations

    def create_sequences(self, activations: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create sliding window sequences from activations

        Input: [num_tokens, num_layers, experts_per_token]
        Output: List of (context, target) where:
                - context: [context_size, num_layers, experts_per_token]
                - target: [num_layers, experts_per_token]
        """
        sequences = []

        if activations is None or len(activations) < self.context_size + 1:
            return sequences

        for i in range(len(activations) - self.context_size):
            context = activations[i:i + self.context_size]  # [context_size, 16, 8]
            target = activations[i + self.context_size]      # [16, 8]
            sequences.append((context, target))

        return sequences

    def load_dataset(self, data_dir: Path, trace_indices: List[int]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Load multiple traces and create sequences

        Args:
            data_dir: Directory containing trace_*.json and gen_*.json files
            trace_indices: List of trace indices to load (e.g., [0, 1, ..., 399] for training)

        Returns:
            List of (context, target) tuples
        """
        all_sequences = []

        for idx in trace_indices:
            trace_file = data_dir / f"trace_{idx:04d}.json"
            gen_file = data_dir / f"gen_{idx:04d}.json"

            if not trace_file.exists() or not gen_file.exists():
                print(f"Warning: {trace_file} or {gen_file} not found")
                continue

            try:
                # Load prefill trace
                prefill_acts = self.load_trace(trace_file)
                if prefill_acts is not None:
                    sequences = self.create_sequences(prefill_acts)
                    all_sequences.extend(sequences)

                # Load generation trace
                gen_acts = self.load_gen(gen_file)
                if gen_acts is not None:
                    sequences = self.create_sequences(gen_acts)
                    all_sequences.extend(sequences)

            except Exception as e:
                print(f"Error loading trace {idx}: {e}")
                continue

        print(f"Loaded {len(all_sequences)} sequences from {len(trace_indices)} traces")
        return all_sequences


def main():
    """Test the data loader"""

    # Configuration
    data_dir = Path('moe_test/olmoe/oasst')  # For testing
    context_size = 10
    experts_per_token = 8
    num_layers = 16

    # Check if we have full dataset or just test data
    full_data_dir = Path('moe_traces/olmoe/oasst')
    if full_data_dir.exists():
        data_dir = full_data_dir
        print(f"Using full dataset from {data_dir}")
    else:
        print(f"Using test dataset from {data_dir}")

    # Initialize loader
    loader = ExpertDataLoader(context_size, experts_per_token, num_layers)

    # Load training data (indices 0-399)
    print("\n=== Loading Training Data ===")
    train_indices = list(range(0, 400))
    train_sequences = loader.load_dataset(data_dir, train_indices)

    # Load test data (indices 400-499)
    print("\n=== Loading Test Data ===")
    test_indices = list(range(400, 500))
    test_sequences = loader.load_dataset(data_dir, test_indices)

    # Print statistics
    print(f"\n=== Dataset Statistics ===")
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")
    print(f"Context size: {context_size}")
    print(f"Number of layers: {num_layers}")
    print(f"Experts per token: {experts_per_token}")

    if train_sequences:
        context, target = train_sequences[0]
        print(f"\nSample context shape: {context.shape}")  # Should be [10, 16, 8]
        print(f"Sample target shape: {target.shape}")      # Should be [16, 8]
        print(f"\nFirst token, layer 0 experts: {context[0, 0, :]}")
        print(f"Next token (target), layer 0 experts: {target[0, :]}")

    return train_sequences, test_sequences


if __name__ == "__main__":
    train_data, test_data = main()
