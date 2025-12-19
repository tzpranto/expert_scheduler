"""
BiLSTM Expert Predictor - Configurable for OLMoE and GPT5OSS

Input: [batch, 10, num_layers, num_experts] probability distributions
Output: [batch, num_layers, num_experts] predicted probability distribution
"""

import torch
import torch.nn as nn
import numpy as np


class BiLSTMExpertPredictor(nn.Module):
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
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, x.shape[1], -1)
        lstm_out, _ = self.bilstm(x_flat)
        attn_scores = self.attention(lstm_out) 
        attn_weights = torch.softmax(attn_scores, dim=1)  
        context = (lstm_out * attn_weights).sum(dim=1)  
        output_flat = self.fc(context) 
        output = output_flat.reshape(batch_size, self.num_layers, self.num_experts)
        output = self.softmax(output)

        return output


def train_epoch(model, train_loader, optimizer, criterion, device='cpu'):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        pred = model(batch_X)
        loss = criterion(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(model, test_loader, num_experts: int, top_k: int = None, device='cpu'):
    model.eval()

    with torch.no_grad():
        all_pred = []
        all_target = []

        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_X)

            all_pred.append(pred.cpu().numpy())
            all_target.append(batch_y.cpu().numpy())

    pred = np.concatenate(all_pred)  
    true = np.concatenate(all_target) 
    metrics = {}

    if top_k is not None:
        correct_topk = 0
        total_topk = 0

        for batch_idx in range(len(pred)):
            for layer_idx in range(pred.shape[1]):
                true_topk_idx = np.argsort(true[batch_idx, layer_idx, :])[-top_k:]
                pred_topk_idx = np.argsort(pred[batch_idx, layer_idx, :])[-top_k:]

                matches = len(set(true_topk_idx) & set(pred_topk_idx))
                correct_topk += matches
                total_topk += top_k

        metrics['top_k_accuracy'] = correct_topk / total_topk if total_topk > 0 else 0
        metrics['top_k_recall'] = metrics['top_k_accuracy']

    mse = np.mean((pred - true) ** 2)
    metrics['mse'] = mse

    l1 = np.mean(np.abs(pred - true))
    metrics['l1'] = l1

    epsilon = 1e-7
    pred_safe = np.clip(pred, epsilon, 1.0)
    true_safe = np.clip(true, epsilon, 1.0)
    kl_div = np.mean(true_safe * (np.log(true_safe) - np.log(pred_safe)))
    metrics['kl_divergence'] = kl_div

    return metrics
