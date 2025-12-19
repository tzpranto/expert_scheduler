#!/usr/bin/env python3
"""
Evaluate PyTorch-trained LSTM model against LRU baseline
"""

import torch
import argparse
from pathlib import Path
from prediction.lstm_pytorch import PyTorchLSTMPredictor, LSTMExpertPredictorPyTorch
from prediction.evaluate import load_test_sequences, evaluate_model
from prediction.lru_baseline import LRUBaseline


def main():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch LSTM model")
    parser.add_argument('--model-path', type=str, default='models/lstm_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--trace-dir', type=str, default='moe_test/olmoe/oasst',
                       help='Path to test traces')
    parser.add_argument('--context-size', type=int, default=10,
                       help='Context window size')
    parser.add_argument('--predict-window', type=int, default=10,
                       help='Prediction window')
    parser.add_argument('--experts-per-layer', type=int, default=2,
                       help='Top-K experts per layer')
    parser.add_argument('--pool-size', type=int, default=100,
                       help='Expert pool size')

    args = parser.parse_args()

    # Configuration
    config = {
        'context_size': args.context_size,
        'predict_window': args.predict_window,
        'experts_per_layer': args.experts_per_layer,
        'num_layers': 16,
        'expert_pool_size': args.pool_size
    }

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load test sequences
    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists():
        print(f"Trace directory {trace_dir} does not exist")
        exit(1)

    test_sequences, num_experts = load_test_sequences(trace_dir, config)

    if not test_sequences:
        print("No test sequences loaded!")
        exit(1)

    # Load trained PyTorch model
    print(f"\nLoading trained model from {args.model_path}...")
    model_path = Path(args.model_path)

    if not model_path.exists():
        print(f"Model file {model_path} not found!")
        print("Please train the model first using:")
        print("  python3 -m prediction.lstm_pytorch --trace-dir moe_test/olmoe/oasst")
        exit(1)

    # Recreate model architecture
    pytorch_model = LSTMExpertPredictorPyTorch(
        max_experts=num_experts + 1,
        embedding_dim=64,
        hidden_dim=128,
        num_lstm_layers=2,
        predict_window=args.predict_window,
        dropout=0.2,
        device=device
    )

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    pytorch_model.load_state_dict(checkpoint['model_state'])
    print(f"✓ Model loaded from {model_path}")

    # Create predictor wrapper
    lstm_predictor = PyTorchLSTMPredictor(
        model=pytorch_model,
        max_experts=num_experts + 1,
        context_size=args.context_size,
        predict_window=args.predict_window,
        experts_per_layer=args.experts_per_layer,
        device=device
    )

    # Create LRU baseline
    lru_baseline = LRUBaseline(
        context_size=args.context_size,
        predict_window=args.predict_window,
        experts_per_layer=args.experts_per_layer
    )

    # Evaluate
    print(f"\nEvaluating on {len(test_sequences)} test sequences...")

    lstm_metrics = evaluate_model(lstm_predictor, test_sequences, config, "PyTorch LSTM")
    lru_metrics = evaluate_model(lru_baseline, test_sequences, config, "LRU Baseline")

    # Comparison
    print("\n" + "=" * 75)
    print("COMPARISON: PyTorch LSTM vs LRU Baseline")
    print("=" * 75)

    improvement = (lstm_metrics.hit_rate_overall - lru_metrics.hit_rate_overall) * 100
    print(f"\n📊 Overall Hit Rate Improvement: {improvement:+.2f}%")
    print(f"   PyTorch LSTM: {lstm_metrics.hit_rate_overall * 100:.2f}%")
    print(f"   LRU Baseline: {lru_metrics.hit_rate_overall * 100:.2f}%")

    print(f"\n📍 Prefill Stage:")
    print(f"   PyTorch LSTM: {lstm_metrics.hit_rate_prefill * 100:.2f}%")
    print(f"   LRU Baseline: {lru_metrics.hit_rate_prefill * 100:.2f}%")

    print(f"\n📍 Generation Stage:")
    print(f"   PyTorch LSTM: {lstm_metrics.hit_rate_gen * 100:.2f}%")
    print(f"   LRU Baseline: {lru_metrics.hit_rate_gen * 100:.2f}%")

    print(f"\n📍 Analysis Stage:")
    print(f"   PyTorch LSTM: {lstm_metrics.hit_rate_analysis * 100:.2f}%")
    print(f"   LRU Baseline: {lru_metrics.hit_rate_analysis * 100:.2f}%")

    print(f"\n⏱️  Prediction Time:")
    print(f"   PyTorch LSTM: {lstm_metrics.avg_prediction_time:.6f} ms")
    print(f"   LRU Baseline: {lru_metrics.avg_prediction_time:.6f} ms")
    time_overhead = ((lstm_metrics.avg_prediction_time - lru_metrics.avg_prediction_time)
                     / lru_metrics.avg_prediction_time * 100 if lru_metrics.avg_prediction_time > 0 else 0)
    print(f"   Overhead: {time_overhead:+.1f}%")

    print(f"\n💾 Pool Utilization:")
    print(f"   PyTorch LSTM: {lstm_metrics.pool_utilization * 100:.2f}%")
    print(f"   LRU Baseline: {lru_metrics.pool_utilization * 100:.2f}%")

    print(f"\n📉 Total Misses:")
    print(f"   PyTorch LSTM: {int(lstm_metrics.misses_per_token)}")
    print(f"   LRU Baseline: {int(lru_metrics.misses_per_token)}")
    miss_reduction = ((lru_metrics.misses_per_token - lstm_metrics.misses_per_token)
                      / lru_metrics.misses_per_token * 100
                      if lru_metrics.misses_per_token > 0 else 0)
    print(f"   Miss Reduction: {miss_reduction:+.1f}%")

    print("\n" + "=" * 75)

    # Summary
    print("\n📋 Summary:")
    if improvement > 0:
        print(f"✓ PyTorch LSTM outperforms LRU by {improvement:.2f}% in overall hit rate")
    elif improvement < 0:
        print(f"✗ LRU outperforms PyTorch LSTM by {-improvement:.2f}% in overall hit rate")
    else:
        print("• PyTorch LSTM and LRU have equivalent hit rates")

    print(f"✓ Model successfully loaded and evaluated")


if __name__ == "__main__":
    main()
