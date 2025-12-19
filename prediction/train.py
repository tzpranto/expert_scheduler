"""
Expert Prediction LSTM Training - Configurable for OLMoE and GPT5OSS

Usage:
    python3 train.py --model olmoe
    python3 train.py --model gpt5oss
    python3 train.py --model olmoe --epochs 50 --batch-size 32
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import argparse
import json
import sys

try:
    from .data_loader import ExpertDataLoader, OLMOE_CONFIG, GPT5OSS_CONFIG
    from .bilstm_model import BiLSTMExpertPredictor, train_epoch, evaluate
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import ExpertDataLoader, OLMOE_CONFIG, GPT5OSS_CONFIG
    from bilstm_model import BiLSTMExpertPredictor, train_epoch, evaluate


def create_dataloader(X, Y, batch_size=32, shuffle=True):
    if len(X) == 0:
        return []

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()

    class SimpleDataset:
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]

    dataset = SimpleDataset(X, Y)

    def batch_loader(dataset, batch_size, shuffle=False):
        indices = np.arange(len(dataset))
        if shuffle:
            np.random.seed(42)
            np.random.shuffle(indices)

        for i in range(0, len(dataset), batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_X = torch.stack([dataset[j][0] for j in batch_idx])
            batch_Y = torch.stack([dataset[j][1] for j in batch_idx])
            yield batch_X, batch_Y

    return batch_loader(dataset, batch_size, shuffle)


def main():
    parser = argparse.ArgumentParser(description="Train expert prediction model")
    parser.add_argument(
        "--model",
        type=str,
        choices=["olmoe", "gpt5oss"],
        default="olmoe",
        help="Model type (olmoe or gpt5oss)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (uses default if not specified)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden-size", type=int, default=512, help="LSTM hidden size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    args = parser.parse_args()

    # Select configuration
    if args.model == "olmoe":
        config = OLMOE_CONFIG
        data_dir = Path("../moe_trace/olmoe/oasst") if args.data_dir is None else Path(args.data_dir)
    else:
        config = GPT5OSS_CONFIG
        data_dir = Path("../moe_trace/gpt5oss/oasst") if args.data_dir is None else Path(args.data_dir)

    print("=" * 80)
    print(f"EXPERT PREDICTION LSTM - {config.name.upper()}")
    print("=" * 80)
    print()

    print(f"Configuration:")
    print(f"  Model: {config.name}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Experts per layer: {config.num_experts}")
    print(f"  Top-K: {config.top_k}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Data dir: {data_dir}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    print("--- Loading Data ---")
    loader = ExpertDataLoader(config)

    print("Loading training data (indices 0-399)...")
    train_indices = list(range(0, 400))
    X_train, Y_train, train_files = loader.load_dataset(
        data_dir, train_indices, context_size=10
    )
    print(f"  ✓ Processed {train_files} files")
    print(f"  ✓ Training sequences: {len(X_train)}")
    print(f"  ✓ X_train shape: {X_train.shape}")
    print(f"  ✓ Y_train shape: {Y_train.shape}")
    print()

    print("Loading test data (indices 400-499)...")
    test_indices = list(range(400, 500))
    X_test, Y_test, test_files = loader.load_dataset(
        data_dir, test_indices, context_size=10
    )
    print(f"  ✓ Processed {test_files} files")
    print(f"  ✓ Test sequences: {len(X_test)}")
    print(f"  ✓ X_test shape: {X_test.shape}")
    print(f"  ✓ Y_test shape: {Y_test.shape}")
    print()

    if len(X_train) == 0:
        print("ERROR: No training data found. Check data directory path.")
        return

    print("--- Creating Model ---")
    model = BiLSTMExpertPredictor(
        num_layers=config.num_layers,
        num_experts=config.num_experts,
        hidden_size=args.hidden_size,
    )
    model = model.to(device)
    print(model)
    print()

    print("--- Training ---")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.KLDivLoss(reduction="batchmean")

    train_losses = []

    train_loader = list(create_dataloader(X_train, Y_train, args.batch_size, shuffle=True))
    test_loader = list(create_dataloader(X_test, Y_test, args.batch_size, shuffle=False)) if len(X_test) > 0 else []

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(loss)
        print(f"Epoch {epoch + 1:3d}/{args.epochs} - Loss: {loss:.6f}")

    print()

    print("--- Evaluation ---")
    if len(test_loader) > 0:
        metrics = evaluate(
            model, test_loader, config.num_experts, top_k=config.top_k, device=device
        )

        print()
        print("Test Results:")
        print(f"  Top-K accuracy:  {metrics.get('top_k_accuracy', 0):.4f}")
        print(f"  MSE loss:        {metrics.get('mse', 0):.6f}")
        print(f"  L1 loss:         {metrics.get('l1', 0):.6f}")
        print(f"  KL divergence:   {metrics.get('kl_divergence', 0):.6f}")
        print()
    else:
        print("No test data available for evaluation")
        metrics = {}

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f"{config.name}_lstm.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to {model_path}")
    print()

    results_path = models_dir / f"{config.name}_lstm_result.txt"
    with open(results_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"LSTM EXPERT PREDICTOR - {config.name.upper()}\n")
        f.write("=" * 70 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Model: {config.name}\n")
        f.write(f"  Layers: {config.num_layers}\n")
        f.write(f"  Experts per layer: {config.num_experts}\n")
        f.write(f"  Top-K: {config.top_k}\n")
        f.write(f"  Context size: 10\n")
        f.write(f"  Hidden size: {args.hidden_size}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Learning rate: {args.lr}\n\n")

        f.write("Dataset:\n")
        f.write(f"  Training sequences: {len(X_train)} from {train_files} files\n")
        f.write(f"  Test sequences: {len(X_test)} from {test_files} files\n")
        f.write(f"  Total: {len(X_train) + len(X_test)} sequences\n\n")

        f.write("Test Metrics:\n")
        if metrics:
            f.write(f"  Top-K accuracy: {metrics.get('top_k_accuracy', 0):.4f}\n")
            f.write(f"  MSE loss: {metrics.get('mse', 0):.6f}\n")
            f.write(f"  L1 loss: {metrics.get('l1', 0):.6f}\n")
            f.write(f"  KL divergence: {metrics.get('kl_divergence', 0):.6f}\n\n")
        else:
            f.write("  No test data available\n\n")

        f.write("Training Loss:\n")
        for epoch, loss in enumerate(train_losses):
            f.write(f"  Epoch {epoch + 1:3d}: {loss:.6f}\n")

    print(f"✓ Results saved to {results_path}")
    print()


if __name__ == "__main__":
    main()
