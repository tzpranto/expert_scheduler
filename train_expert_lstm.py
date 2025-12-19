#!/usr/bin/env python3
"""
LSTM Expert Prediction - Final Clean Implementation
====================================================

Task: Predict which 8 experts (out of 64) are active in each of 16 layers
Input: Last 10 tokens of expert activations → Predict next token

Data format:
  - Context: [10, 16, 8] = 10 timesteps × 16 layers × 8 active experts per layer
  - Target: [16, 8] = 16 layers × 8 active experts (with expert IDs 0-63)

Metrics: Accuracy, F1, Precision, Recall (per expert per layer)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from data_loader import ExpertDataLoader
import json


class ExpertLSTM(nn.Module):
    """LSTM for predicting expert activations"""

    def __init__(self, hidden_size=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=128,  # 16 layers * 8 experts flattened
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 128)

    def forward(self, x):
        # x: [batch, 10, 16, 8]
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, 10, -1)  # [batch, 10, 128]

        out, _ = self.lstm(x_flat)  # [batch, 10, hidden]
        out = out[:, -1, :]  # [batch, hidden]
        out = self.fc(out)  # [batch, 128]
        out = out.reshape(batch_size, 16, 8)  # [batch, 16, 8]

        return out


def train(model, train_loader, epochs=20, lr=0.001, device='cpu'):
    """Train the model"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        total_loss = 0
        count = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward
            pred = model(batch_X)
            loss = criterion(pred, batch_y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1:2d}/{epochs} - Loss: {avg_loss:.4f}")

    return losses


def evaluate(model, test_loader, device='cpu'):
    """Evaluate the model"""
    model.eval()

    correct_matches = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Predict
            pred = model(batch_X)

            # Round to nearest integer (expert IDs are 0-63)
            pred_clipped = torch.clamp(torch.round(pred), 0, 63)

            # Compare
            matches = (pred_clipped == batch_y).float().mean()
            correct_matches += matches.item()
            total_samples += 1

            all_preds.append(pred_clipped.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    accuracy = correct_matches / max(total_samples, 1)

    return {
        'accuracy': accuracy,
        'predictions': np.concatenate(all_preds),
        'targets': np.concatenate(all_targets)
    }


def create_dataloader(sequences, batch_size=8, shuffle=True):
    """Create a simple dataloader"""
    X = torch.stack([torch.from_numpy(ctx).float() for ctx, _ in sequences])
    y = torch.stack([torch.from_numpy(tgt).float() for _, tgt in sequences])

    class TensorDataset:
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    dataset = TensorDataset(X, y)

    # Simple dataloader
    def dataloader(dataset, batch_size, shuffle=False):
        indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(dataset), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_X = torch.stack([dataset[j][0] for j in batch_idx])
            batch_y = torch.stack([dataset[j][1] for j in batch_idx])
            yield batch_X, batch_y

    return dataloader(dataset, batch_size, shuffle)


def main():
    print("=" * 80)
    print("LSTM EXPERT PREDICTION MODEL")
    print("=" * 80)
    print()
    print("Task: Predict which 8 experts (out of 64) activate in each of 16 layers")
    print("Input:  [10, 16, 8] = context of 10 tokens")
    print("Output: [16, 8]     = next token expert activations")
    print()

    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    # Load data
    print("--- Loading Data ---")
    loader = ExpertDataLoader(context_size=10, experts_per_token=8, num_layers=16)
    data_dir = Path('moe_test/olmoe/oasst')

    train_indices = [0]  # Can expand to [0-399] for full training
    test_indices = [0]   # Can expand to [400-499] for full testing

    train_seqs = loader.load_dataset(data_dir, train_indices)
    test_seqs = loader.load_dataset(data_dir, test_indices)

    print(f"Train sequences: {len(train_seqs)}")
    print(f"Test sequences:  {len(test_seqs)}")
    print()

    # Create model
    print("--- Creating Model ---")
    model = ExpertLSTM(hidden_size=256, num_layers=2)
    print(model)
    print()

    # Train
    print("--- Training ---")
    train_loader = create_dataloader(train_seqs, batch_size=8, shuffle=True)
    losses = train(model, train_loader, epochs=20, lr=0.001, device=device)
    print()

    # Evaluate
    print("--- Evaluation ---")
    test_loader = create_dataloader(test_seqs, batch_size=8, shuffle=False)
    results = evaluate(model, test_loader, device=device)

    print()
    print("Results:")
    print(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"    (Fraction of experts predicted exactly correctly)")
    print()

    # Save
    Path('models').mkdir(exist_ok=True)
    torch.save(model.state_dict(), 'models/expert_lstm.pt')
    print("✓ Model saved to models/expert_lstm.pt")


if __name__ == "__main__":
    main()
