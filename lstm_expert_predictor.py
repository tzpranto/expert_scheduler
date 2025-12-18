"""
LSTM-based Expert Prediction Model for MoE Systems
Predicts which experts will be needed for upcoming tokens to enable expert prefetching
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import time


@dataclass
class ExpertPredictionConfig:
    """Configuration for LSTM expert prediction"""
    context_size: int = 10  # Look-back window for LSTM
    predict_window: int = 10  # Number of future tokens to predict
    expert_pool_size: int = 100  # Size of expert prefetch pool
    num_layers: int = 16  # Number of MoE layers
    experts_per_layer: int = 2  # Top-K experts to select per layer
    hidden_size: int = 128
    num_lstm_layers: int = 2
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ExpertSequenceDataset:
    """Converts trace data into sequences for LSTM training"""

    def __init__(self, config: ExpertPredictionConfig):
        self.config = config
        self.sequences = []  # List of (context, targets) tuples
        self.all_experts = set()

    def load_traces(self, trace_dir: Path, split_ratio: float = 0.8):
        """Load and preprocess traces from directory"""
        trace_files = sorted(Path(trace_dir).glob("trace_*.json"))
        gen_files = sorted(Path(trace_dir).glob("gen_*.json"))

        if not trace_files or not gen_files:
            raise ValueError(f"No trace or gen files found in {trace_dir}")

        # Pair trace and gen files
        samples = []
        for trace_file, gen_file in zip(trace_files, gen_files):
            try:
                with open(trace_file) as f:
                    trace_data = json.load(f)
                with open(gen_file) as f:
                    gen_data = json.load(f)
                samples.append((trace_data, gen_data))
            except Exception as e:
                print(f"Error loading {trace_file}: {e}")
                continue

        # Split into train/test
        random.seed(42)
        random.shuffle(samples)
        split_idx = int(len(samples) * split_ratio)
        train_samples = samples[:split_idx]
        test_samples = samples[split_idx:]

        print(f"Loaded {len(samples)} samples ({len(train_samples)} train, {len(test_samples)} test)")

        # Process samples
        for samples_list, is_train in [(train_samples, True), (test_samples, False)]:
            for trace_data, gen_data in samples_list:
                self._process_sample(trace_data, gen_data, is_train=is_train)

        return train_samples, test_samples

    def _process_sample(self, trace_data: Dict, gen_data: Dict, is_train: bool = True):
        """Convert a single trace/gen pair into training sequences"""
        # Extract expert sequences from prefill
        prefill_sequence = self._extract_expert_sequence(trace_data)

        # Extract expert sequences from generation
        gen_sequence = self._extract_generation_sequence(gen_data)

        # Combine: prefill + generation
        full_sequence = prefill_sequence + gen_sequence

        if len(full_sequence) < self.config.context_size + self.config.predict_window:
            return

        # Create training examples
        for i in range(len(full_sequence) - self.config.context_size - self.config.predict_window + 1):
            context = full_sequence[i:i + self.config.context_size]
            targets = full_sequence[i + self.config.context_size:
                                   i + self.config.context_size + self.config.predict_window]

            # Store as tuple (context, targets, is_train)
            self.sequences.append({
                'context': context,
                'targets': targets,
                'is_train': is_train
            })

    def _extract_expert_sequence(self, trace_data: Dict) -> List[Set[int]]:
        """Extract expert IDs from prefill trace"""
        sequence = []
        layers = trace_data.get('layers', [])

        if not layers:
            return sequence

        num_tokens = len(layers[0].get('token_data', []))

        for token_idx in range(num_tokens):
            experts_for_token = set()

            # Select top experts_per_layer from each layer
            for layer_idx, layer_data in enumerate(layers):
                token_data = layer_data['token_data'][token_idx]
                topk_experts = token_data.get('topk_experts', [])

                # Take top K experts for this layer
                selected = topk_experts[:self.config.experts_per_layer]
                experts_for_token.update(selected)
                self.all_experts.update(selected)

            sequence.append(experts_for_token)

        return sequence

    def _extract_generation_sequence(self, gen_data: Dict) -> List[Set[int]]:
        """Extract expert IDs from generation trace"""
        sequence = []
        decode_steps = gen_data.get('decode_steps', [])

        for step_data in decode_steps:
            experts_for_token = set()
            layers = step_data.get('layers', [])

            for layer_data in layers:
                topk_experts = layer_data.get('topk_experts', [])
                selected = topk_experts[:self.config.experts_per_layer]
                experts_for_token.update(selected)
                self.all_experts.update(selected)

            sequence.append(experts_for_token)

        return sequence

    def get_train_test_split(self):
        """Split into train and test sets"""
        train = [s for s in self.sequences if s['is_train']]
        test = [s for s in self.sequences if not s['is_train']]
        return train, test


class LSTMExpertPredictor(nn.Module):
    """LSTM model for predicting expert IDs for future tokens"""

    def __init__(self, config: ExpertPredictionConfig, num_experts: int):
        super().__init__()
        self.config = config
        self.num_experts = num_experts

        # Embedding layer: each expert ID gets embedded
        self.expert_embedding = nn.Embedding(num_experts + 1, config.hidden_size, padding_idx=num_experts)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_lstm_layers,
            batch_first=True,
            dropout=0.2 if config.num_lstm_layers > 1 else 0
        )

        # Output projection: predict logits for each expert at each time step
        self.output_proj = nn.Linear(config.hidden_size, num_experts)

    def forward(self, expert_sequences, lengths=None):
        """
        Forward pass
        expert_sequences: (batch, context_size) with expert IDs
        returns: (batch, predict_window, num_experts) predictions
        """
        # Embed expert sequences
        embedded = self.expert_embedding(expert_sequences)  # (batch, context, hidden)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # (batch, context, hidden)

        # Use final hidden state to predict future experts
        # Repeat hidden state for prediction window
        final_hidden = h_n[-1]  # (batch, hidden)

        predictions = []
        hidden = (h_n, c_n)

        for _ in range(self.config.predict_window):
            # Project to logits
            logits = self.output_proj(final_hidden)  # (batch, num_experts)
            predictions.append(logits)

            # For autoregressive prediction, we could sample or use greedy selection
            # For now, we'll use the logits directly

        predictions = torch.stack(predictions, dim=1)  # (batch, predict_window, num_experts)
        return predictions


class ExpertPool:
    """LRU-based expert pool for managing expert cache"""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.pool = OrderedDict()

    def add(self, experts: Set[int]):
        """Add experts to pool, evict LRU if necessary"""
        for expert in experts:
            if expert not in self.pool:
                if len(self.pool) >= self.max_size:
                    # Remove least recently used
                    self.pool.popitem(last=False)
                self.pool[expert] = True
            else:
                # Move to end (most recently used)
                self.pool.move_to_end(expert)

    def get_pool(self) -> Set[int]:
        """Get current experts in pool"""
        return set(self.pool.keys())

    def hit_rate(self, required_experts: Set[int]) -> float:
        """Calculate hit rate for required experts"""
        if not required_experts:
            return 1.0
        hits = len(required_experts & self.get_pool())
        return hits / len(required_experts)


class LRUBaseline:
    """Simple LRU cache baseline (no training needed)"""

    def __init__(self, config: ExpertPredictionConfig):
        self.config = config
        self.pool = ExpertPool(config.expert_pool_size)
        self.prediction_time = 0

    def predict_next_experts(self, recent_experts: List[Set[int]]) -> List[Set[int]]:
        """LRU prediction: just return next experts based on frequency"""
        predictions = []
        for _ in range(self.config.predict_window):
            # LRU doesn't actually "predict" - it assumes recent experts are likely
            # For simplicity, we return the most common experts from recent history
            if recent_experts:
                most_recent = recent_experts[-1]
                predictions.append(most_recent.copy())
            else:
                predictions.append(set())
        return predictions


class ExpertPrefetcher:
    """Manages expert prefetching and cache efficiency calculation"""

    def __init__(self, config: ExpertPredictionConfig, model: nn.Module = None):
        self.config = config
        self.model = model
        self.pool = ExpertPool(config.expert_pool_size)
        self.device = config.device

    def predict_experts_lstm(self, recent_experts: List[Set[int]]) -> List[Set[int]]:
        """Predict next experts using LSTM model"""
        if not self.model:
            raise ValueError("Model not provided")

        if len(recent_experts) < self.config.context_size:
            # Pad with empty sets
            recent_experts = [set()] * (self.config.context_size - len(recent_experts)) + recent_experts
        else:
            recent_experts = recent_experts[-self.config.context_size:]

        # Convert to tensor
        expert_ids = self._experts_to_tensor(recent_experts)

        with torch.no_grad():
            predictions = self.model(expert_ids)  # (1, predict_window, num_experts)

        # Convert predictions to expert sets
        pred_experts = []
        for t in range(self.config.predict_window):
            logits = predictions[0, t, :]
            top_experts = torch.topk(logits, k=min(self.config.experts_per_layer, logits.shape[0])).indices
            pred_experts.append(set(top_experts.cpu().numpy()))

        return pred_experts

    def _experts_to_tensor(self, expert_sets: List[Set[int]], pad_idx: int = -1) -> torch.Tensor:
        """Convert list of expert sets to tensor"""
        # Flatten: for each token, we have multiple experts
        # Create a matrix where each row is padded expert IDs
        max_experts = self.config.experts_per_layer
        batch_size = 1

        tensor = []
        for expert_set in expert_sets:
            experts_list = sorted(list(expert_set))
            # Pad or truncate to max_experts
            if len(experts_list) < max_experts:
                experts_list = experts_list + [pad_idx] * (max_experts - len(experts_list))
            else:
                experts_list = experts_list[:max_experts]
            tensor.append(experts_list)

        tensor = torch.tensor(tensor, dtype=torch.long, device=self.device)
        # Reshape to (batch, context_size)
        tensor = tensor.view(batch_size, -1)
        return tensor

    def evaluate_single_sequence(self, context: List[Set[int]],
                                targets: List[Set[int]],
                                use_lru: bool = False) -> Dict:
        """Evaluate model on a single sequence"""
        results = {
            'hits': 0,
            'misses': 0,
            'total': 0,
            'pool_used': 0,
            'stages': {
                'prefill': {'hits': 0, 'total': 0},
                'gen': {'hits': 0, 'total': 0}
            }
        }

        pool = ExpertPool(self.config.expert_pool_size)

        # Initialize pool with context
        for experts in context:
            pool.add(experts)

        # Get predictions
        if use_lru:
            predictions = LRUBaseline(self.config).predict_next_experts(context)
        else:
            predictions = self.predict_experts_lstm(context)

        # Prefetch predicted experts
        for pred_experts in predictions:
            pool.add(pred_experts)

        # Evaluate against targets
        for i, target_experts in enumerate(targets):
            if not target_experts:
                continue

            results['total'] += len(target_experts)

            # Check which experts are in pool
            hits = len(target_experts & pool.get_pool())
            results['hits'] += hits
            results['misses'] += len(target_experts) - hits

            # Determine stage (arbitrary: first 30% is prefill)
            stage = 'prefill' if i < len(targets) * 0.3 else 'gen'
            results['stages'][stage]['hits'] += hits
            results['stages'][stage]['total'] += len(target_experts)

            # Readjust: add any missing experts to pool
            missing = target_experts - pool.get_pool()
            if missing:
                pool.add(missing)

        results['pool_used'] = len(pool.get_pool())

        return results


def train_lstm_model(model: nn.Module, train_data: List[Dict],
                    config: ExpertPredictionConfig, num_experts: int):
    """Train LSTM model"""
    device = config.device
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()  # For multi-label prediction

    for epoch in range(config.epochs):
        total_loss = 0
        num_batches = 0

        # Shuffle training data
        random.shuffle(train_data)

        for i in range(0, len(train_data), config.batch_size):
            batch = train_data[i:i + config.batch_size]

            # Create tensors
            contexts = []
            targets_tensor = []

            for sample in batch:
                context_experts = sample['context']
                target_experts = sample['targets']

                # Flatten context to tensor (simplified: first expert per token)
                context_ids = torch.tensor(
                    [list(experts)[0] if experts else 0 for experts in context_experts],
                    dtype=torch.long
                )
                contexts.append(context_ids)

                # Create target tensor: binary labels for each expert at each time step
                target_tensor = torch.zeros(config.predict_window, num_experts)
                for t, experts in enumerate(target_experts):
                    for expert_id in experts:
                        if expert_id < num_experts:
                            target_tensor[t, expert_id] = 1
                targets_tensor.append(target_tensor)

            # Batch and move to device
            contexts = torch.stack([c for c in contexts]).to(device)
            targets = torch.stack(targets_tensor).to(device)

            # Forward pass
            predictions = model(contexts)  # (batch, predict_window, num_experts)

            # Calculate loss
            loss = criterion(predictions, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{config.epochs}: Loss = {avg_loss:.4f}")

    return model


def main():
    """Main training and evaluation pipeline"""
    config = ExpertPredictionConfig()

    # Load traces
    trace_dir = Path("moe_test/olmoe/oasst")  # Use test data for now
    dataset = ExpertSequenceDataset(config)

    train_samples, test_samples = dataset.load_traces(trace_dir, split_ratio=0.8)
    train_data, test_data = dataset.get_train_test_split()

    num_experts = len(dataset.all_experts)
    print(f"Total unique experts: {num_experts}")
    print(f"Training sequences: {len(train_data)}")
    print(f"Test sequences: {len(test_data)}")

    # Create and train LSTM model
    print("\n=== Training LSTM Model ===")
    model = LSTMExpertPredictor(config, num_experts)
    model = train_lstm_model(model, train_data, config, num_experts)

    # Evaluate
    print("\n=== Evaluation ===")
    prefetcher = ExpertPrefetcher(config, model)

    lstm_results = {'hits': 0, 'misses': 0, 'total': 0, 'stages': defaultdict(lambda: {'hits': 0, 'total': 0})}
    lru_results = {'hits': 0, 'misses': 0, 'total': 0, 'stages': defaultdict(lambda: {'hits': 0, 'total': 0})}

    for sample in test_data:
        lstm_eval = prefetcher.evaluate_single_sequence(sample['context'], sample['targets'], use_lru=False)
        lru_eval = prefetcher.evaluate_single_sequence(sample['context'], sample['targets'], use_lru=True)

        # Aggregate results
        for key in ['hits', 'misses', 'total']:
            lstm_results[key] += lstm_eval[key]
            lru_results[key] += lru_eval[key]

    # Print results
    print("\n=== Results ===")
    print(f"LSTM Hit Rate: {lstm_results['hits'] / max(lstm_results['total'], 1) * 100:.2f}%")
    print(f"LRU Hit Rate: {lru_results['hits'] / max(lru_results['total'], 1) * 100:.2f}%")


if __name__ == "__main__":
    main()
