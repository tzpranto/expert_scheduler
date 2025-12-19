"""
BiLSTM model for expert prediction
Input: [batch, 10, 16, 64] one-hot encoded probabilities
Output: [batch, 16, 64] probability distribution per layer
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from olmoe_data_loader import OLMoEDataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json


class BiLSTMExpertPredictor(nn.Module):
    """
    BiLSTM for expert prediction
    Input: [batch, context_size, 16, 64]
    Output: [batch, 16, 64]
    """

    def __init__(self, num_layers: int = 16, num_experts: int = 64, hidden_size: int = 512):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.hidden_size = hidden_size

        # Input: one-hot encoded [batch, context_size, 16*64=1024]
        self.input_size = num_layers * num_experts

        # Bi-directional LSTM
        self.bilstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Output projection
        self.fc = nn.Linear(hidden_size * 2, self.input_size)

        # Softmax for probability distribution
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: [batch, context_size, 16, 64]
        return: [batch, 16, 64]
        """
        batch_size = x.shape[0]

        # Flatten: [batch, context_size, 16*64]
        x_flat = x.reshape(batch_size, x.shape[1], -1)

        # BiLSTM: [batch, context_size, hidden_size*2]
        lstm_out, _ = self.bilstm(x_flat)

        # Attention: weight each timestep
        attn_scores = self.attention(lstm_out)  # [batch, context_size, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch, context_size, 1]

        # Context vector: weighted sum
        context = (lstm_out * attn_weights).sum(dim=1)  # [batch, hidden_size*2]

        # Output projection
        output_flat = self.fc(context)  # [batch, 16*64]

        # Reshape and apply softmax per layer
        output = output_flat.reshape(batch_size, self.num_layers, self.num_experts)
        output = self.softmax(output)  # [batch, 16, 64]

        return output


def prepare_tensors(sequences):
    """Convert sequences to tensors"""
    X = torch.stack([torch.from_numpy(context).float() for context, _ in sequences])
    y = torch.stack([torch.from_numpy(target).float() for _, target in sequences])
    return X, y


def train_epoch(model, X_train, y_train, optimizer, criterion, batch_size=32, device='cpu'):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    indices = np.arange(len(X_train))
    np.random.shuffle(indices)

    for i in range(0, len(X_train), batch_size):
        batch_idx = indices[i:i + batch_size]
        batch_X = X_train[batch_idx].to(device)
        batch_y = y_train[batch_idx].to(device)

        # Forward
        pred = model(batch_X)

        # Loss
        loss = criterion(pred, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, X_test, y_test, device='cpu'):
    """Evaluate model and compute metrics"""
    model.eval()

    with torch.no_grad():
        pred = model(X_test.to(device))  # [batch, 16, 64]
        pred = pred.cpu().numpy()

    y_true = y_test.numpy()

    # For each layer, get top-K experts
    top_k = 8

    # Compute accuracy: how many experts match top-K
    correct = 0
    total = 0

    for b in range(len(pred)):
        for layer in range(16):
            # Get top-K indices
            true_topk_idx = np.argsort(y_true[b, layer, :])[-top_k:]
            pred_topk_idx = np.argsort(pred[b, layer, :])[-top_k:]

            # Count matches
            matches = len(set(true_topk_idx) & set(pred_topk_idx))
            correct += matches
            total += top_k

    accuracy = correct / total if total > 0 else 0

    # Flatten for other metrics
    pred_topk = pred.copy()
    for b in range(len(pred)):
        for layer in range(16):
            # Zero out non-top-K
            topk_idx = np.argsort(pred[b, layer, :])[-top_k:]
            mask = np.zeros(64)
            mask[topk_idx] = 1
            pred_topk[b, layer, :] = mask

    y_true_topk = y_true.copy()
    for b in range(len(y_true)):
        for layer in range(16):
            topk_idx = np.argsort(y_true[b, layer, :])[-top_k:]
            mask = np.zeros(64)
            mask[topk_idx] = 1
            y_true_topk[b, layer, :] = mask

    # Flatten
    pred_flat = pred_topk.reshape(-1)
    true_flat = y_true_topk.reshape(-1)

    # Metrics
    precision = precision_score(true_flat, pred_flat, zero_division=0)
    recall = recall_score(true_flat, pred_flat, zero_division=0)
    f1 = f1_score(true_flat, pred_flat, zero_division=0)
    acc_exact = accuracy_score(true_flat, pred_flat)

    return {
        'accuracy': accuracy,
        'accuracy_exact': acc_exact,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    print("=" * 70)
    print("BILSTM EXPERT PREDICTOR FOR OLMoE")
    print("=" * 70)
    print()

    # Config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    # Load data
    print("--- Loading Data ---")
    loader = OLMoEDataLoader(num_layers=16, num_experts=64, top_k=8)
    data_dir = Path('moe_test/olmoe/oasst')

    # For testing: use only sample data
    # For production: use list(range(0, 400)) and list(range(400, 500))
    train_indices = [0]
    test_indices = [0]

    print(f"\nLoading training data (indices: {train_indices})...")
    train_seqs = loader.load_dataset(data_dir, train_indices, context_size=10)

    print(f"Loading test data (indices: {test_indices})...")
    test_seqs = loader.load_dataset(data_dir, test_indices, context_size=10)

    print()
    print(f"Training sequences: {len(train_seqs)}")
    print(f"Test sequences: {len(test_seqs)}")
    print()

    # Prepare tensors
    X_train, y_train = prepare_tensors(train_seqs)
    X_test, y_test = prepare_tensors(test_seqs)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print()

    # Create model
    print("--- Creating Model ---")
    model = BiLSTMExpertPredictor(num_layers=16, num_experts=64, hidden_size=512)
    model = model.to(device)
    print(model)
    print()

    # Train
    print("--- Training ---")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.KLDivLoss()  # KL divergence for probability distributions

    epochs = 20
    batch_size = 32

    train_losses = []
    for epoch in range(epochs):
        loss = train_epoch(model, X_train, y_train, optimizer, criterion, batch_size, device)
        train_losses.append(loss)
        print(f"Epoch {epoch+1:2d}/{epochs} - Loss: {loss:.6f}")

    print()

    # Evaluate
    print("--- Evaluation ---")
    metrics = evaluate(model, X_test, y_test, device)

    print()
    print("Test Results:")
    print(f"  Accuracy (top-K match):    {metrics['accuracy']:.4f}")
    print(f"  Accuracy (exact match):    {metrics['accuracy_exact']:.4f}")
    print(f"  Precision:                 {metrics['precision']:.4f}")
    print(f"  Recall:                    {metrics['recall']:.4f}")
    print(f"  F1 Score:                  {metrics['f1']:.4f}")
    print()

    # Save model
    Path('models').mkdir(exist_ok=True)
    model_path = 'models/olmoe_lstm.pt'
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to {model_path}")

    # Save results
    results_path = 'models/olmoe_lstm_result.txt'
    with open(results_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BILSTM EXPERT PREDICTOR - RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  Layers: 16\n")
        f.write(f"  Experts per layer: 64\n")
        f.write(f"  Top-K: 8\n")
        f.write(f"  Context size: 10\n")
        f.write(f"  Hidden size: 512\n")
        f.write(f"  Epochs: {epochs}\n")
        f.write(f"  Batch size: {batch_size}\n\n")
        f.write("Test Metrics:\n")
        f.write(f"  Accuracy (top-K match):    {metrics['accuracy']:.4f}\n")
        f.write(f"  Accuracy (exact match):    {metrics['accuracy_exact']:.4f}\n")
        f.write(f"  Precision:                 {metrics['precision']:.4f}\n")
        f.write(f"  Recall:                    {metrics['recall']:.4f}\n")
        f.write(f"  F1 Score:                  {metrics['f1']:.4f}\n\n")
        f.write("Training Loss:\n")
        for epoch, loss in enumerate(train_losses):
            f.write(f"  Epoch {epoch+1:2d}: {loss:.6f}\n")

    print(f"✓ Results saved to {results_path}")
    print()


if __name__ == "__main__":
    main()
