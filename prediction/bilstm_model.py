"""
BiLSTM Expert Predictor - Configurable for OLMoE and GPT5OSS

Input: [batch, 10, num_layers, num_experts] probability distributions
Output: [batch, num_layers, num_experts] predicted probability distribution
"""

import torch
import torch.nn as nn
import numpy as np


class BiLSTMExpertPredictor(nn.Module):
    """
    BiLSTM for expert prediction
    Works with any configuration (OLMoE, GPT5OSS, etc.)
    """

    def __init__(self, num_layers: int, num_experts: int, hidden_size: int = 512):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.hidden_size = hidden_size

        # Input: flattened [batch, context_size, num_layers * num_experts]
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

        # Softmax for probability distribution per layer
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: [batch, context_size, num_layers, num_experts]
        return: [batch, num_layers, num_experts]
        """
        batch_size = x.shape[0]

        # Flatten: [batch, context_size, num_layers * num_experts]
        x_flat = x.reshape(batch_size, x.shape[1], -1)

        # BiLSTM: [batch, context_size, hidden_size * 2]
        lstm_out, _ = self.bilstm(x_flat)

        # Attention: weight each timestep
        attn_scores = self.attention(lstm_out)  # [batch, context_size, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch, context_size, 1]

        # Context vector: weighted sum
        context = (lstm_out * attn_weights).sum(dim=1)  # [batch, hidden_size * 2]

        # Output projection
        output_flat = self.fc(context)  # [batch, num_layers * num_experts]

        # Reshape and apply softmax per layer
        output = output_flat.reshape(batch_size, self.num_layers, self.num_experts)

        # Apply softmax per layer to ensure probability distribution
        output = self.softmax(output)  # [batch, num_layers, num_experts]

        return output


def train_epoch(model, train_loader, optimizer, criterion, device='cpu'):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

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

    return total_loss / max(num_batches, 1)


def evaluate(model, test_loader, num_experts: int, top_k: int = None, device='cpu'):
    """
    Evaluate model and compute metrics

    Args:
        model: BiLSTM model
        test_loader: test dataloader
        num_experts: number of experts per layer
        top_k: if provided, compute top-K accuracy (e.g., 8 for OLMoE, 4 for GPT5OSS)
        device: 'cpu' or 'cuda'

    Returns:
        Dictionary with metrics
    """
    model.eval()

    with torch.no_grad():
        all_pred = []
        all_target = []

        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_X)  # [batch, num_layers, num_experts]

            all_pred.append(pred.cpu().numpy())
            all_target.append(batch_y.cpu().numpy())

    pred = np.concatenate(all_pred)  # [N, num_layers, num_experts]
    true = np.concatenate(all_target)  # [N, num_layers, num_experts]

    # Compute top-K accuracy if provided
    metrics = {}

    if top_k is not None:
        # For each layer, check if top-K experts match
        correct_topk = 0
        total_topk = 0

        for batch_idx in range(len(pred)):
            for layer_idx in range(pred.shape[1]):
                # Get top-K indices
                true_topk_idx = np.argsort(true[batch_idx, layer_idx, :])[-top_k:]
                pred_topk_idx = np.argsort(pred[batch_idx, layer_idx, :])[-top_k:]

                # Count matches
                matches = len(set(true_topk_idx) & set(pred_topk_idx))
                correct_topk += matches
                total_topk += top_k

        metrics['top_k_accuracy'] = correct_topk / total_topk if total_topk > 0 else 0
        metrics['top_k_recall'] = metrics['top_k_accuracy']

    # MSE loss
    mse = np.mean((pred - true) ** 2)
    metrics['mse'] = mse

    # L1 loss
    l1 = np.mean(np.abs(pred - true))
    metrics['l1'] = l1

    # KL divergence (averaged across all positions)
    epsilon = 1e-7
    pred_safe = np.clip(pred, epsilon, 1.0)
    true_safe = np.clip(true, epsilon, 1.0)
    kl_div = np.mean(true_safe * (np.log(true_safe) - np.log(pred_safe)))
    metrics['kl_divergence'] = kl_div

    return metrics
