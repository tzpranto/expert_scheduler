"""
Simple LSTM for expert prediction
Input: [context_size, 16, 8] → Output: [16, 8]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from data_loader import ExpertDataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
import json


class SimpleLSTM(nn.Module):
    """Simple LSTM model for expert prediction"""

    def __init__(self, input_size=128, hidden_size=256, num_layers=2, output_size=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch, context_size, input_size]
        # Flatten: [context_size, 16, 8] -> [context_size, 128]
        batch_size = x.shape[0]
        context_size = x.shape[1]

        # Reshape: [batch, context_size, 16*8]
        x_flat = x.reshape(batch_size, context_size, -1)

        # LSTM
        lstm_out, _ = self.lstm(x_flat)  # [batch, context_size, hidden_size]

        # Take last time step
        last_output = lstm_out[:, -1, :]  # [batch, hidden_size]

        # Project to output
        output = self.fc(last_output)  # [batch, 128]

        return output


def prepare_data(train_sequences, test_sequences):
    """Convert sequences to tensors"""
    X_train, y_train = [], []
    X_test, y_test = [], []

    for context, target in train_sequences:
        X_train.append(torch.from_numpy(context).float())
        y_train.append(torch.from_numpy(target).float())

    for context, target in test_sequences:
        X_test.append(torch.from_numpy(context).float())
        y_test.append(torch.from_numpy(target).float())

    return X_train, y_train, X_test, y_test


def compute_metrics(predictions, targets):
    """
    Compute classification metrics
    predictions: [batch, 16, 8] (probabilities)
    targets: [batch, 16, 8] (binary labels)
    """
    # Convert to binary predictions
    pred_binary = (predictions > 0.5).astype(np.int32)
    target_binary = targets.astype(np.int32)

    # Flatten for metrics
    pred_flat = pred_binary.reshape(-1)
    target_flat = target_binary.reshape(-1)

    accuracy = accuracy_score(target_flat, pred_flat)

    # Handle edge case where target might be all 0s or all 1s
    if len(np.unique(target_flat)) == 1:
        # If target is all same class, just return simple metrics
        precision = 0.0 if target_flat[0] == 0 else 1.0
        recall = 0.0 if target_flat[0] == 0 else 1.0
        f1 = 0.0 if target_flat[0] == 0 else 1.0
    else:
        precision = precision_score(target_flat, pred_flat, zero_division=0, average='micro')
        recall = recall_score(target_flat, pred_flat, zero_division=0, average='micro')
        f1 = f1_score(target_flat, pred_flat, zero_division=0, average='micro')

    hamming = hamming_loss(target_flat, pred_flat)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hamming_loss': hamming
    }


def train_epoch(model, optimizer, criterion, X_train, y_train, batch_size=4, device='cpu'):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    # Create batches
    for i in range(0, len(X_train), batch_size):
        batch_X = torch.stack(X_train[i:i+batch_size]).to(device)
        batch_y = torch.stack(y_train[i:i+batch_size]).to(device)

        # Forward
        logits = model(batch_X)  # [batch, 128]
        logits = logits.reshape(batch_X.shape[0], 16, 8)  # [batch, 16, 8]

        # Loss
        loss = criterion(logits, batch_y)

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
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(len(X_test)):
            x = X_test[i].unsqueeze(0).to(device)  # Add batch dim
            logits = model(x)  # [1, 128]
            logits = logits.reshape(1, 16, 8)  # [1, 16, 8]

            preds = torch.sigmoid(logits).cpu().numpy()
            target = y_test[i].numpy()

            all_preds.append(preds[0])
            all_targets.append(target)

    all_preds = np.array(all_preds)  # [num_samples, 16, 8]
    all_targets = np.array(all_targets)  # [num_samples, 16, 8]

    metrics = compute_metrics(all_preds, all_targets)
    return metrics, all_preds, all_targets


def main():
    print("=" * 70)
    print("LSTM EXPERT PREDICTION - SIMPLE IMPLEMENTATION")
    print("=" * 70)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load data
    print("\n--- Loading Data ---")
    loader = ExpertDataLoader(context_size=10, experts_per_token=8, num_layers=16)
    data_dir = Path('moe_test/olmoe/oasst')

    # For now, use only sample data (train on indices 0, test on indices 0)
    # In production, use indices 0-399 for training, 400-499 for testing
    train_indices = [0]  # Change to list(range(400)) for full training
    test_indices = [0]   # Change to list(range(400, 500)) for full testing

    train_sequences = loader.load_dataset(data_dir, train_indices)
    test_sequences = loader.load_dataset(data_dir, test_indices)

    print(f"Training sequences: {len(train_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")

    if not train_sequences or not test_sequences:
        print("ERROR: No data loaded. Check data directory.")
        return

    # Prepare tensors
    X_train, y_train, X_test, y_test = prepare_data(train_sequences, test_sequences)
    print(f"\nX_train shape: ({len(X_train)}, {X_train[0].shape})")
    print(f"y_train shape: ({len(y_train)}, {y_train[0].shape})")

    # Create model
    print("\n--- Creating Model ---")
    model = SimpleLSTM(input_size=128, hidden_size=256, num_layers=2, output_size=128)
    model = model.to(device)
    print(model)

    # Training
    print("\n--- Training ---")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 10
    batch_size = 4

    train_losses = []
    for epoch in range(epochs):
        loss = train_epoch(model, optimizer, criterion, X_train, y_train, batch_size, device)
        train_losses.append(loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

    # Evaluate
    print("\n--- Evaluation ---")
    test_metrics, test_preds, test_targets = evaluate(model, X_test, y_test, device)

    print("\nTest Metrics:")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  Precision:   {test_metrics['precision']:.4f}")
    print(f"  Recall:      {test_metrics['recall']:.4f}")
    print(f"  F1 Score:    {test_metrics['f1']:.4f}")
    print(f"  Hamming Loss: {test_metrics['hamming_loss']:.4f}")

    # Save model
    torch.save(model.state_dict(), 'models/simple_lstm.pt')
    print("\n✓ Model saved to models/simple_lstm.pt")

    # Save metrics
    results = {
        'train_losses': train_losses,
        'test_metrics': test_metrics
    }
    with open('models/metrics.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("✓ Metrics saved to models/metrics.json")


if __name__ == "__main__":
    main()
