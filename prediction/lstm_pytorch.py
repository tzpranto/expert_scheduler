"""
Full PyTorch LSTM implementation for expert prediction
Includes model architecture, training, and inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Set, Dict, Tuple
from collections import defaultdict
import time
import json
from pathlib import Path


class ExpertSequenceDataset(Dataset):
    """PyTorch Dataset for expert sequences"""

    def __init__(self, sequences: List[List[Set[int]]],
                 context_size: int = 10,
                 predict_window: int = 10,
                 experts_per_layer: int = 2,
                 max_experts: int = 128):
        """
        Initialize dataset

        Args:
            sequences: List of expert sequences
            context_size: Look-back window
            predict_window: Prediction window
            experts_per_layer: Top-K experts per layer
            max_experts: Maximum expert ID (for vocab size)
        """
        self.sequences = sequences
        self.context_size = context_size
        self.predict_window = predict_window
        self.experts_per_layer = experts_per_layer
        self.max_experts = max_experts
        self.examples = []

        self._prepare_examples()

    def _prepare_examples(self):
        """Prepare training examples from sequences"""
        for sequence in self.sequences:
            if len(sequence) < self.context_size + self.predict_window:
                continue

            # Create sliding windows
            for i in range(len(sequence) - self.context_size - self.predict_window + 1):
                context = sequence[i:i + self.context_size]
                targets = sequence[i + self.context_size:i + self.context_size + self.predict_window]
                self.examples.append((context, targets))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        context, targets = self.examples[idx]

        # Convert expert sets to tensor
        # Each token has multiple experts, we'll use a fixed-size representation
        context_tensor = self._experts_to_tensor(context)
        targets_tensor = self._targets_to_tensor(targets)

        return context_tensor, targets_tensor

    def _experts_to_tensor(self, expert_sets: List[Set[int]]) -> torch.Tensor:
        """Convert expert sets to input tensor (batch_size, context_size, max_experts)"""
        batch_size = len(expert_sets)
        tensor = torch.zeros(batch_size, self.max_experts, dtype=torch.float32)

        for i, experts in enumerate(expert_sets):
            for expert_id in experts:
                if expert_id < self.max_experts:
                    tensor[i, expert_id] = 1.0

        return tensor

    def _targets_to_tensor(self, expert_sets: List[Set[int]]) -> torch.Tensor:
        """Convert target expert sets to tensor (predict_window, max_experts)"""
        tensor = torch.zeros(self.predict_window, self.max_experts, dtype=torch.float32)

        for t, experts in enumerate(expert_sets):
            for expert_id in experts:
                if expert_id < self.max_experts:
                    tensor[t, expert_id] = 1.0

        return tensor


class LSTMExpertPredictorPyTorch(nn.Module):
    """
    Full PyTorch LSTM model for expert prediction

    Architecture:
    - Embedding layer for expert representations
    - Bi-directional LSTM encoder
    - Attention-based decoder for multi-step prediction
    """

    def __init__(self,
                 max_experts: int = 128,
                 embedding_dim: int = 64,
                 hidden_dim: int = 128,
                 num_lstm_layers: int = 2,
                 predict_window: int = 10,
                 dropout: float = 0.2,
                 device: str = 'cpu'):
        """
        Initialize LSTM model

        Args:
            max_experts: Total number of experts (vocabulary size)
            embedding_dim: Dimension of expert embeddings
            hidden_dim: LSTM hidden dimension
            num_lstm_layers: Number of LSTM layers
            predict_window: Number of time steps to predict
            dropout: Dropout rate
            device: Device to run on (cpu/cuda)
        """
        super().__init__()
        self.max_experts = max_experts
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.predict_window = predict_window
        self.device = device

        # Encoder: Input projection (max_experts -> embedding_dim)
        self.input_projection = nn.Linear(max_experts, embedding_dim)

        # Bi-directional LSTM encoder
        self.lstm_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Context vector projection
        self.context_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        # LSTM decoder
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Output projection to expert probabilities
        self.output_projection = nn.Linear(hidden_dim, max_experts)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_len, max_experts)

        Returns:
            Output tensor of shape (batch_size, predict_window, max_experts)
        """
        batch_size = x.size(0)

        # Project input
        x_proj = self.dropout(self.input_projection(x))  # (batch, seq_len, embedding_dim)

        # Encode with bi-directional LSTM
        encoder_out, (h_n, c_n) = self.lstm_encoder(x_proj)  # (batch, seq_len, hidden*2)

        # Apply attention to get context
        attn_weights = torch.softmax(self.attention(encoder_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(encoder_out * attn_weights, dim=1)  # (batch, hidden*2)

        # Project context
        context_proj = self.context_projection(context)  # (batch, hidden)

        # Initialize decoder hidden state from context
        decoder_h = context_proj.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1)  # (layers, batch, hidden)
        decoder_c = torch.zeros_like(decoder_h)

        # Decode multiple time steps
        outputs = []
        decoder_input = context_proj.unsqueeze(1)  # (batch, 1, hidden)

        for t in range(self.predict_window):
            decoder_out, (decoder_h, decoder_c) = self.lstm_decoder(
                decoder_input, (decoder_h, decoder_c)
            )  # decoder_out: (batch, 1, hidden)

            # Project to expert probabilities
            logits = self.output_projection(decoder_out[:, -1, :])  # (batch, max_experts)
            outputs.append(logits)

            # Use output as next input (teacher forcing could be applied here)
            decoder_input = decoder_out

        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, predict_window, max_experts)
        return output


class LSTMTrainer:
    """Trainer class for LSTM model"""

    def __init__(self,
                 model: LSTMExpertPredictorPyTorch,
                 device: str = 'cpu',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        """
        Initialize trainer

        Args:
            model: LSTM model to train
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch

        Args:
            train_loader: DataLoader for training

        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass
            logits = self.model(x)

            # Calculate loss
            loss = self.criterion(logits, y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}: Loss = {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model

        Args:
            val_loader: DataLoader for validation

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None,
              epochs: int = 20) -> Dict[str, List[float]]:
        """
        Train model for multiple epochs

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs

        Returns:
            Dictionary with training history
        """
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)

            print(f"  Train Loss: {train_loss:.4f}")

            # Validate if validation loader provided
            if val_loader is not None and len(val_loader) > 0:
                val_loss = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                print(f"  Val Loss: {val_loss:.4f}")

                # Update learning rate
                self.scheduler.step(val_loss)
            else:
                # Use training loss for scheduling if no validation
                self.scheduler.step(train_loss)

        return history

    def save_model(self, path: Path):
        """Save model checkpoint"""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"Model loaded from {path}")


class PyTorchLSTMPredictor:
    """Wrapper for using trained PyTorch LSTM model for inference"""

    def __init__(self, model: LSTMExpertPredictorPyTorch,
                 max_experts: int = 128,
                 context_size: int = 10,
                 predict_window: int = 10,
                 experts_per_layer: int = 2,
                 device: str = 'cpu'):
        """
        Initialize predictor

        Args:
            model: Trained LSTM model
            max_experts: Total number of experts
            context_size: Context window size
            predict_window: Prediction window
            experts_per_layer: Top-K experts per layer
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.max_experts = max_experts
        self.context_size = context_size
        self.predict_window = predict_window
        self.experts_per_layer = experts_per_layer

    def predict_next_experts(self, recent_experts: List[Set[int]]) -> List[Set[int]]:
        """
        Predict next experts using trained LSTM

        Args:
            recent_experts: List of expert sets from recent tokens

        Returns:
            List of predicted expert sets for next predict_window tokens
        """
        # Pad context if needed
        if len(recent_experts) < self.context_size:
            recent_experts = [set()] * (self.context_size - len(recent_experts)) + recent_experts
        else:
            recent_experts = recent_experts[-self.context_size:]

        # Convert to tensor
        context_tensor = torch.zeros(1, self.context_size, self.max_experts, dtype=torch.float32)
        for i, experts in enumerate(recent_experts):
            for expert_id in experts:
                if expert_id < self.max_experts:
                    context_tensor[0, i, expert_id] = 1.0

        context_tensor = context_tensor.to(self.device)

        # Get predictions
        with torch.no_grad():
            logits = self.model(context_tensor)  # (1, predict_window, max_experts)

        # Convert logits to expert sets
        predictions = []
        for t in range(self.predict_window):
            # Get top-K experts for this timestep
            probs = torch.sigmoid(logits[0, t, :])
            top_k_indices = torch.topk(probs, k=min(self.experts_per_layer * 2, self.max_experts)).indices
            pred_set = set(top_k_indices.cpu().numpy().tolist())
            predictions.append(pred_set)

        return predictions


def load_sequences_from_traces(trace_dir: Path,
                                context_size: int = 10,
                                experts_per_layer: int = 2) -> Tuple[List, List, int]:
    """Load training sequences from trace files"""
    from prediction.common import ExpertSequenceAnalyzer

    trace_files = sorted(trace_dir.glob("trace_*.json"))
    gen_files = sorted(trace_dir.glob("gen_*.json"))

    analyzer = ExpertSequenceAnalyzer(
        context_size=context_size,
        experts_per_layer=experts_per_layer
    )

    train_sequences = []
    test_sequences = []

    for trace_file, gen_file in zip(trace_files, gen_files):
        try:
            with open(trace_file) as f:
                trace_data = json.load(f)
            with open(gen_file) as f:
                gen_data = json.load(f)

            sequence, stages = analyzer.extract_expert_sequence(trace_data, gen_data)
            if sequence:
                # Simple 80/20 split (in practice, use proper train/test split)
                if hash(str(trace_file)) % 100 < 80:
                    train_sequences.append(sequence)
                else:
                    test_sequences.append(sequence)
        except Exception as e:
            print(f"Error loading {trace_file}: {e}")

    num_experts = len(analyzer.all_experts)
    print(f"Loaded {len(train_sequences)} train and {len(test_sequences)} test sequences")
    print(f"Total unique experts: {num_experts}")

    return train_sequences, test_sequences, num_experts


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Train LSTM expert predictor")
    parser.add_argument('--trace-dir', type=str, default='moe_traces/olmoe/oasst',
                       help='Path to trace directory')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--save-path', type=str, default='models/lstm_expert_predictor.pt',
                       help='Path to save model')
    parser.add_argument('--context-size', type=int, default=10,
                       help='Context window size')
    parser.add_argument('--predict-window', type=int, default=10,
                       help='Prediction window')

    args = parser.parse_args()

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load sequences
    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists():
        print(f"Trace directory {trace_dir} does not exist")
        print("Using test data from moe_test/ instead...")
        trace_dir = Path('moe_test/olmoe/oasst')

    train_sequences, test_sequences, num_experts = load_sequences_from_traces(
        trace_dir,
        context_size=args.context_size
    )

    if not train_sequences:
        print("No training sequences loaded!")
        exit(1)

    # Create datasets
    print("\nPreparing datasets...")
    train_dataset = ExpertSequenceDataset(
        train_sequences,
        context_size=args.context_size,
        predict_window=args.predict_window,
        max_experts=num_experts + 1
    )
    test_dataset = ExpertSequenceDataset(
        test_sequences,
        context_size=args.context_size,
        predict_window=args.predict_window,
        max_experts=num_experts + 1
    )

    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Test dataset: {len(test_dataset)} examples")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    print("\nCreating model...")
    model = LSTMExpertPredictorPyTorch(
        max_experts=num_experts + 1,
        embedding_dim=64,
        hidden_dim=128,
        num_lstm_layers=2,
        predict_window=args.predict_window,
        dropout=0.2,
        device=device
    )

    print(model)

    # Train
    print("\nTraining...")
    trainer = LSTMTrainer(model, device=device, learning_rate=args.learning_rate)
    # Use test_loader if it has data, otherwise None
    val_loader = test_loader if len(test_dataset) > 0 else None
    history = trainer.train(train_loader, val_loader, epochs=args.epochs)

    # Save model
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(Path(args.save_path))

    print("\nTraining complete!")
