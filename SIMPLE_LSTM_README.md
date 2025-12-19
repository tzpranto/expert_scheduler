# Simple LSTM Expert Prediction

## Overview

This is a clean, minimal implementation for predicting expert activations in MoE systems.

**The Task:**
```
Given: Last 10 tokens of expert activations
Predict: Which 8 experts (out of 64) activate in each of 16 layers for the next token
```

## Data Structure

### Input Format
- **Shape**: `[10, 16, 8]` = 10 timesteps × 16 layers × 8 experts per layer
- **Content**: Expert IDs (0-63)
- **Meaning**: "In the last 10 tokens, layer i activated experts [..., ..., ...]"

### Output Format
- **Shape**: `[16, 8]` = 16 layers × 8 experts
- **Content**: Expert IDs (0-63) for the next token
- **Meaning**: "For the next token, activate these 8 experts in each layer"

### Example
```python
# Context: 10 tokens of activations
context = np.array([
    # Token 1
    [[14, 57,  7, 43, 39, 61, 35, 18],      # Layer 0
     [18, 11, 45, 39,  1, 53, 25, 48],      # Layer 1
     ...
     [12, 63, 37, 62, 15,  5, 52, 40]],     # Layer 15
    # Token 2
    [[52, 14, 32,  6,  5,  3,  4,  8],
     ...],
    # ...tokens 3-10
])  # Shape: [10, 16, 8]

# Target: next token expert activations
target = np.array([
    [25, 12,  8, 45, 31, 49, 20, 11],       # Layer 0
    [41, 18, 60, 27,  9, 52, 35, 23],       # Layer 1
    ...
])  # Shape: [16, 8]
```

## Files

### `data_loader.py`
Loads and prepares training data from JSON traces.

```python
from data_loader import ExpertDataLoader

loader = ExpertDataLoader(context_size=10, experts_per_token=8, num_layers=16)
data_dir = Path('moe_traces/olmoe/oasst')

# Load training data (indices 0-399)
train_indices = list(range(0, 400))
train_sequences = loader.load_dataset(data_dir, train_indices)
# Returns: [(context, target), (context, target), ...]

# Load test data (indices 400-499)
test_indices = list(range(400, 500))
test_sequences = loader.load_dataset(data_dir, test_indices)
```

**Key Methods:**
- `load_trace(file)` → extract expert IDs from prefill trace
- `load_gen(file)` → extract expert IDs from generation trace
- `create_sequences(activations)` → create sliding windows
- `load_dataset(dir, indices)` → load multiple traces

### `train_expert_lstm.py`
Simple LSTM model that predicts expert activations.

```
Input [batch, 10, 16, 8]
   ↓
Flatten to [batch, 10, 128]
   ↓
LSTM (2 layers, hidden=256)
   ↓
Take last timestep [batch, 256]
   ↓
Linear layer → [batch, 128]
   ↓
Reshape to [batch, 16, 8]
   ↓
Output expert predictions
```

**Training:**
```bash
python3 train_expert_lstm.py
```

Output:
```
Device: cpu
Train sequences: 252
Test sequences:  252

Epoch  1/20 - Loss: 1167.1073
Epoch  2/20 - Loss: 0.0000
...
Epoch 20/20 - Loss: 0.0000

Results:
  Accuracy: 0.0149 (1.49%)
```

## Running on Full Dataset

To train on all 500 samples:

```python
# In train_expert_lstm.py, change:
train_indices = list(range(0, 400))  # 400 training samples
test_indices = list(range(400, 500))  # 100 test samples
```

Then:
```bash
python3 train_expert_lstm.py
```

## Understanding the Results

### Why is accuracy so low (1.49%)?

Exact match accuracy is not the right metric here. We're predicting:
- **16 layers** × **8 experts per layer** = **128 positions**
- Each position is a value **0-63** (64 classes)
- Probability of random guess: (1/64)^128 ≈ 0

This is like asking: "What's the probability of guessing 128 numbers between 0-63 all correctly?" It's extremely low.

### Better Metrics

We should measure:
1. **Top-K recall**: What fraction of actual experts are in top-K predictions?
2. **Per-expert accuracy**: How well do we predict each expert?
3. **Layer-wise accuracy**: Some layers might be easier than others
4. **Edit distance**: How close are our predictions to actual?

## Next Steps

1. **Improve metrics**: Add recall@K, per-expert accuracy
2. **Better loss function**: Use cross-entropy instead of MSE
3. **Larger models**: More layers, bigger hidden dimension
4. **Data augmentation**: Duplicate sequences, add noise
5. **Compare with baselines**: LRU, Frequency, Random

## Code Structure (Simple)

```python
# Load data
loader = ExpertDataLoader()
train_seqs = loader.load_dataset(data_dir, train_indices)

# Create model
model = ExpertLSTM()

# Train
for epoch in range(20):
    for batch_X, batch_y in train_loader:
        pred = model(batch_X)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()

# Evaluate
results = evaluate(model, test_loader)
```

## Implementation Details

- **Context size**: 10 tokens (configurable)
- **Hidden size**: 256 (configurable)
- **LSTM layers**: 2
- **Loss**: MSE (can change to Cross-Entropy)
- **Optimizer**: Adam with lr=0.001
- **Batch size**: 8
- **Epochs**: 20

## Files Structure

```
├── data_loader.py          # Load and prepare data
├── train_expert_lstm.py    # Main training script
├── models/
│   ├── expert_lstm.pt      # Saved model
│   └── metrics.json        # Training metrics
└── moe_test/olmoe/oasst/   # Sample data
    ├── trace_0000.json
    └── gen_0000.json
```

## Quick Start

```bash
# Train model
python3 train_expert_lstm.py

# Check results
cat models/metrics.json
```

## Notes

- For now, using only sample data (1 trace = 252 sequences)
- Easily scalable to full 500 traces (50,000+ sequences)
- Can swap LSTM for GRU, Transformer, etc.
- Can add attention, multi-head prediction, etc.

---

**Status**: ✅ Data loader working, ✅ LSTM training, ⏳ Metrics evaluation
