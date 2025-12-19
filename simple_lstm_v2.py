"""
LSTM for expert prediction - SIMPLE AND CLEAN
Input: [context_size, 16, 8] → Output: [16, 8]
Predicts which 8 experts are active in each of 16 layers
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from data_loader import ExpertDataLoader
from sklearn.metrics import accuracy_score, f1_score
import json


class SimpleLSTMPredictor(nn.Module):
    """
    LSTM that predicts expert activations

    Input: [batch, context_size, 16, 8] - expert activations over time
    Output: [batch, 16, 8] - predicted next expert activations
    """

    def __init__(self, hidden_size=256, num_layers=2):
        super().__init__()
        # Flatten 16*8=128 experts per timestep
        self.input_size = 128
        self.output_size = 128

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Two fully connected layers
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, self.output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: [batch, context_size, 16, 8]
        return: [batch, 16, 8]
        """
        batch_size, context_size, num_layers, experts_per_token = x.shape

        # Flatten: [batch, context_size, 16*8]
        x_flat = x.reshape(batch_size, context_size, -1)

        # LSTM: [batch, context_size, hidden_size]
        lstm_out, (h_n, c_n) = self.lstm(x_flat)

        # Take last timestep: [batch, hidden_size]
        last_out = lstm_out[:, -1, :]

        # FC layers
        out = self.relu(self.fc1(last_out))
        out = self.fc2(out)  # [batch, 128]

        # Reshape back: [batch, 16, 8]
        out = out.reshape(batch_size, 16, 8)

        return out


def create_binary_targets(expert_tensor):
    """
    Convert expert IDs (0-63) to binary matrix (which experts are active)
    Input: [16, 8] with expert IDs
    Output: [16, 64] with 1 where expert is active, 0 otherwise
    """
    batch_size = expert_tensor.shape[0]
    num_layers = expert_tensor.shape[1]
    experts_per_token = expert_tensor.shape[2]

    # Create binary matrix
    binary = np.zeros((batch_size, num_layers, 64), dtype=np.float32)

    for b in range(batch_size):
        for l in range(num_layers):
            for e in range(experts_per_token):
                expert_id = expert_tensor[b, l, e].item()
                if 0 <= expert_id < 64:
                    binary[b, l, expert_id] = 1

    return binary


def train_epoch(model, optimizer, criterion, X_train, y_train, batch_size=8, device='cpu'):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for i in range(0, len(X_train), batch_size):
        batch_X = torch.stack(X_train[i:i+batch_size]).to(device)  # [batch, 10, 16, 8]
        batch_y = torch.stack(y_train[i:i+batch_size]).to(device)  # [batch, 16, 8]

        # Forward
        logits = model(batch_X)  # [batch, 16, 8]

        # For now, just use simple regression loss (predict the expert values)
        # We'll use MSE loss to predict the expert IDs
        loss = criterion(logits.float(), batch_y.float())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(model, X_test, y_test, device='cpu'):
    """Evaluate model"""
    model.eval()

    total_samples = len(X_test)
    correct_experts = 0
    total_experts = 0

    with torch.no_grad():
        for i in range(total_samples):
            x = X_test[i].unsqueeze(0).to(device)  # [1, 10, 16, 8]
            logits = model(x)  # [1, 16, 8]

            # Get predicted experts (round to nearest integer 0-63)
            preds = torch.round(logits).clamp(0, 63)
            preds = preds.long().cpu().numpy()[0]  # [16, 8]

            target = y_test[i].numpy()  # [16, 8]

            # Count exact matches
            matches = (preds == target).sum()
            correct_experts += matches
            total_experts += preds.size

    accuracy = correct_experts / total_experts if total_experts > 0 else 0

    return {'accuracy': accuracy}


def main():
    print("=" * 70)
    print("SIMPLE LSTM FOR EXPERT PREDICTION")
    print("=" * 70)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load data
    print("\n--- Loading Data ---")
    loader = ExpertDataLoader(context_size=10, experts_per_token=8, num_layers=16)
    data_dir = Path('moe_test/olmoe/oasst')

    # Load sample data
    train_sequences = loader.load_dataset(data_dir, [0])
    test_sequences = loader.load_dataset(data_dir, [0])

    print(f"Training sequences: {len(train_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")

    # Convert to tensors
    X_train = [torch.from_numpy(context).float() for context, _ in train_sequences]
    y_train = [torch.from_numpy(target).float() for _, target in train_sequences]
    X_test = [torch.from_numpy(context).float() for context, _ in test_sequences]
    y_test = [torch.from_numpy(target).float() for _, target in test_sequences]

    print(f"\nX_train shape per sample: {X_train[0].shape}")  # [10, 16, 8]
    print(f"y_train shape per sample: {y_train[0].shape}")    # [16, 8]

    # Create model
    print("\n--- Creating Model ---")
    model = SimpleLSTMPredictor(hidden_size=256, num_layers=2)
    model = model.to(device)
    print(model)

    # Training
    print("\n--- Training ---")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # Simple MSE for regression

    epochs = 20
    batch_size = 8

    for epoch in range(epochs):
        loss = train_epoch(model, optimizer, criterion, X_train, y_train, batch_size, device)
        print(f"Epoch {epoch+1:2d}/{epochs} - Loss: {loss:.4f}")

    # Evaluate
    print("\n--- Evaluation ---")
    test_metrics = evaluate(model, X_test, y_test, device)

    print(f"\nTest Metrics:")
    print(f"  Expert Prediction Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"    (Fraction of experts predicted exactly correct)")

    # Save model
    torch.save(model.state_dict(), 'models/simple_lstm_v2.pt')
    print("\n✓ Model saved to models/simple_lstm_v2.pt")

    # Save metrics
    with open('models/metrics_v2.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print("✓ Metrics saved to models/metrics_v2.json")


if __name__ == "__main__":
    main()
