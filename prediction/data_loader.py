"""
Expert Prediction Data Loader - Configurable for OLMoE and GPT5OSS

Supports:
  - OLMoE: 16 layers, 64 experts, top-K=8
  - GPT5OSS: 24 layers, 32 experts, top-K=4

Each file has multiple tokens:
  - Prefill: variable tokens
  - Gen: 256 tokens (OLMoE), 512 tokens (GPT5OSS)

Creates sliding windows: [10, 16, 64] → [16, 64]
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    num_layers: int
    num_experts: int
    top_k: int
    gen_tokens: int

#Hardcoded for easy use
OLMOE_CONFIG = ModelConfig(
    name="olmoe",
    num_layers=16,
    num_experts=64,
    top_k=8,
    gen_tokens=256
)

GPT5OSS_CONFIG = ModelConfig(
    name="gpt5oss",
    num_layers=24,
    num_experts=32,
    top_k=4,
    gen_tokens=512
)


class ExpertDataLoader:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.num_layers = config.num_layers
        self.num_experts = config.num_experts
        self.top_k = config.top_k

    def load_trace_to_probabilities(self, trace_file: Path) -> np.ndarray:
        with open(trace_file) as f:
            data = json.load(f)

        layers = data.get('layers', [])
        if not layers:
            return None

        num_tokens = len(layers[0].get('token_data', []))
        probs = np.zeros((num_tokens, self.num_layers, self.num_experts), dtype=np.float32)

        for layer_idx, layer_data in enumerate(layers):
            token_data_list = layer_data.get('token_data', [])

            for token_idx, token_data in enumerate(token_data_list):
                topk_experts = token_data.get('topk_experts', [])
                topk_probs = token_data.get('topk_probs', [])

                topk_experts = topk_experts[:self.top_k]
                topk_probs = topk_probs[:self.top_k]

                sum_topk = sum(topk_probs) if topk_probs else 0
                remaining_prob = (1.0 - sum_topk) / (self.num_experts - self.top_k) if self.num_experts > self.top_k else 0

                for expert_id, prob in zip(topk_experts, topk_probs):
                    if 0 <= expert_id < self.num_experts:
                        probs[token_idx, layer_idx, expert_id] = prob

                for expert_id in range(self.num_experts):
                    if expert_id not in topk_experts:
                        probs[token_idx, layer_idx, expert_id] = remaining_prob

        return probs

    def load_gen_to_probabilities(self, gen_file: Path) -> np.ndarray:
        with open(gen_file) as f:
            data = json.load(f)

        decode_steps = data.get('decode_steps', [])

        probs = np.zeros((len(decode_steps), self.num_layers, self.num_experts), dtype=np.float32)

        for step_idx, step_data in enumerate(decode_steps):
            layers = step_data.get('layers', [])

            for layer_data in layers:
                layer_idx = layer_data.get('layer', 0)
                if layer_idx >= self.num_layers:
                    continue

                topk_experts = layer_data.get('topk_experts', [])
                topk_probs = layer_data.get('topk_probs', [])

                topk_experts = topk_experts[:self.top_k]
                topk_probs = topk_probs[:self.top_k]

                sum_topk = sum(topk_probs) if topk_probs else 0
                remaining_prob = (1.0 - sum_topk) / (self.num_experts - self.top_k) if self.num_experts > self.top_k else 0

                for expert_id, prob in zip(topk_experts, topk_probs):
                    if 0 <= expert_id < self.num_experts:
                        probs[step_idx, layer_idx, expert_id] = prob

                for expert_id in range(self.num_experts):
                    if expert_id not in topk_experts:
                        probs[step_idx, layer_idx, expert_id] = remaining_prob

        return probs

    def create_sequences(self, probs: np.ndarray, context_size: int = 10) -> List[Tuple]:
        sequences = []

        if probs is None or len(probs) < context_size + 1:
            return sequences

        for i in range(len(probs) - context_size):
            context = probs[i:i + context_size]  #[context_size, num_layers, num_experts]
            target = probs[i + context_size]     #[num_layers, num_experts]
            sequences.append((context, target))

        return sequences

    def load_dataset(self, data_dir: Path, indices: List[int], context_size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        all_sequences = []
        processed_files = 0

        for idx in indices:
            trace_file = data_dir / f"trace_{idx:04d}.json"
            gen_file = data_dir / f"gen_{idx:04d}.json"

            if not trace_file.exists() or not gen_file.exists():
                print(f"  ✗ Missing trace_{idx:04d}.json or gen_{idx:04d}.json")
                continue

            try:
                # Prefill
                prefill_probs = self.load_trace_to_probabilities(trace_file)
                if prefill_probs is not None:
                    seqs = self.create_sequences(prefill_probs, context_size)
                    all_sequences.extend(seqs)
                    processed_files += 1

                gen_probs = self.load_gen_to_probabilities(gen_file)
                if gen_probs is not None:
                    seqs = self.create_sequences(gen_probs, context_size)
                    all_sequences.extend(seqs)

            except Exception as e:
                print(f"  ✗ Error loading trace {idx}: {e}")
                continue

        if all_sequences:
            X = np.array([context for context, _ in all_sequences], dtype=np.float32)
            Y = np.array([target for _, target in all_sequences], dtype=np.float32)
        else:
            X = np.array([], dtype=np.float32).reshape(0, context_size, self.num_layers, self.num_experts)
            Y = np.array([], dtype=np.float32).reshape(0, self.num_layers, self.num_experts)

        return X, Y, processed_files


def test_data_loader(config: ModelConfig, data_dir: Path):
    print(f"\n{'='*70}")
    print(f"DATA LOADER TEST - {config.name.upper()}")
    print(f"{'='*70}\n")

    print(f"Configuration:")
    print(f"  Model: {config.name}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Experts per layer: {config.num_experts}")
    print(f"  Top-K: {config.top_k}")
    print(f"  Gen tokens: {config.gen_tokens}\n")

    loader = ExpertDataLoader(config)

    print(f"Loading training data (indices 0-399)...")
    train_indices = list(range(0, 400))
    X_train, Y_train, train_files = loader.load_dataset(data_dir, train_indices, context_size=10)
    print(f"  ✓ Processed {train_files} files")
    print(f"  ✓ X_train shape: {X_train.shape}")
    print(f"  ✓ Y_train shape: {Y_train.shape}\n")

    print(f"Loading test data (indices 400-499)...")
    test_indices = list(range(400, 500))
    X_test, Y_test, test_files = loader.load_dataset(data_dir, test_indices, context_size=10)
    print(f"  ✓ Processed {test_files} files")
    print(f"  ✓ X_test shape: {X_test.shape}")
    print(f"  ✓ Y_test shape: {Y_test.shape}\n")

    print(f"Verification:")
    if len(X_train) > 0:
        print(f"  X_train[0, 0, 0, :].sum() = {X_train[0, 0, 0, :].sum():.4f} (should be 1.0)")
        print(f"  Y_train[0, 0, :].sum() = {Y_train[0, 0, :].sum():.4f} (should be 1.0)")
    print()

    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Training:   {len(X_train)} sequences from {train_files} files")
    print(f"Testing:    {len(X_test)} sequences from {test_files} files")
    print(f"Total:      {len(X_train) + len(X_test)} sequences")
    print(f"\nInput shape:  [N, 10, {config.num_layers}, {config.num_experts}]")
    print(f"Output shape: [N, {config.num_layers}, {config.num_experts}]")
    print()


if __name__ == "__main__":
    olmoe_dir = Path('../moe_traces/olmoe/oasst')
    if olmoe_dir.exists():
        test_data_loader(OLMOE_CONFIG, olmoe_dir)
    else:
        print(f"OLMoE data directory not found: {olmoe_dir}")
        print("Using sample data instead...")
        test_data_loader(OLMOE_CONFIG, Path('../moe_test/olmoe/oasst'))

    gpt5_dir = Path('../moe_traces/gpt5oss/oasst')
    if gpt5_dir.exists():
        test_data_loader(GPT5OSS_CONFIG, gpt5_dir)
    else:
        print(f"\nGPT5OSS data directory not found: {gpt5_dir}")
        print("Skipping GPT5OSS test...")
