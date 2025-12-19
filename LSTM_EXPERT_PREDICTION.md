# LSTM Expert Prediction Model for MoE Systems

## Overview

This implementation designs and evaluates an LSTM-based model for predicting which experts will be needed in upcoming tokens for Mixture-of-Experts (MoE) language models. The goal is to enable expert prefetching to improve performance.

## Architecture

### Key Components

1. **ExpertSequenceAnalyzer**: Extracts expert sequences from prefill and generation traces
   - Identifies different stages: prefill, generation, analysis
   - Extracts top-K experts per layer per token
   - Handles both OLMoE and GPT-5 OSS models

2. **ExpertPool**: LRU-based expert cache management
   - Manages expert prefetch buffer (configurable size, default 100)
   - Tracks hits, misses, and access patterns
   - Supports dynamic eviction when pool is full

3. **LSTMExpertPredictor**: LSTM-based prediction model
   - Context window: 10 tokens (configurable)
   - Prediction window: 10 tokens (configurable)
   - Predicts top-2 experts per layer (configurable)
   - Uses historical expert sequences to predict future experts

4. **LRUBaseline**: Simple LRU cache without learning
   - No training required
   - Serves as baseline for comparison
   - Assumes recent experts are likely to appear again

### Data Structure

The model operates on expert sequences extracted from MoE traces:
- **Prefill Stage**: Expert selections during prompt encoding
- **Generation Stage**: Expert selections during token generation
- **Analysis Stage**: Expert selections during response analysis (if applicable)

Each token requires experts from multiple layers:
- OLMoE: 16 layers × 8 experts per layer (uses top-K)
- GPT-5 OSS: 16 layers × 4 experts per layer

## Configuration

```python
@dataclass
class ExpertPredictionConfig:
    context_size: int = 10           # Look-back window for LSTM
    predict_window: int = 10         # Number of future tokens to predict
    expert_pool_size: int = 100      # Size of expert prefetch pool
    num_layers: int = 16             # Number of MoE layers
    experts_per_layer: int = 2       # Top-K experts to select per layer
    hidden_size: int = 128           # LSTM hidden dimension
    num_lstm_layers: int = 2         # Number of LSTM layers
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 0.001
```

## Evaluation Metrics

The model is evaluated on the following metrics:

### Hit Rate
- **Overall**: % of required experts found in pool
- **Prefill**: Hit rate during prompt encoding
- **Generation**: Hit rate during token generation
- **Analysis**: Hit rate during response analysis phase

### Performance Metrics
- **Avg Prediction Time**: Average time to generate predictions (ms)
- **Pool Utilization**: % of pool size used (avg across sequence)
- **Total Misses**: Number of expert cache misses

## Dataset Structure

Training data consists of 500 samples with 80% (400) for training and 20% (100) for testing.

Each sample contains:
- **Trace file** (`trace_XXXX.json`): Prefill stage router logits
- **Gen file** (`gen_XXXX.json`): Generation stage router logits and text

### Trace File Format
```json
{
  "prompt": "...",
  "num_layers": 16,
  "num_experts": 64,
  "k_per_token": 8,
  "layers": [
    {
      "layer": 0,
      "token_data": [
        {
          "token": "word",
          "topk_experts": [14, 57, 7, ...],
          "topk_probs": [0.134, 0.125, ...]
        },
        ...
      ]
    }
  ]
}
```

### Gen File Format
```json
{
  "prompt": "...",
  "generated_text": "...",
  "generated_ids": [...],
  "decode_steps": [
    {
      "step": 0,
      "token_id": 123,
      "token": "word",
      "layers": [
        {
          "layer": 0,
          "topk_experts": [52, 14, 32, ...],
          "topk_probs": [0.093, 0.093, ...]
        }
      ]
    }
  ]
}
```

## Prefetching Strategy

### Initial Prefetching (Context Phase)
1. Initialize expert pool with context_size tokens
2. For each token, pool is populated with its required experts

### Prediction & Readjustment
1. LSTM predicts experts for next predict_window tokens
2. Add predicted experts to pool
3. When actual token arrives:
   - Check for hits in pool
   - Track misses for missed experts
   - Add any new required experts (readjustment)

### Pool Management
- Pool size limited to expert_pool_size (e.g., 100 experts)
- LRU eviction when pool exceeds capacity
- Most recently used experts stay in pool

## Running the Evaluation

### Simple Evaluation (Pure Python)
```bash
python3 evaluate_simple.py
```

Outputs:
- Hit rates by stage (prefill, generation, analysis)
- Average prediction time
- Pool utilization
- Comparison between LSTM and LRU baseline

### Full Training Pipeline (Requires PyTorch)
```bash
python3 lstm_expert_predictor.py
```

Requirements:
- PyTorch >= 1.9
- NumPy
- Transformers (for models)

## Expected Results

Based on the single test sample:
- **LSTM Hit Rate**: ~99.97% overall
  - Prefill: ~98.73%
  - Generation: ~100%
- **LRU Baseline**: ~99.97% overall (similar to LSTM for this sample)
- **Prediction Time**: ~0.11ms (LSTM), ~0.004ms (LRU)

The expert pool with size 100 can contain most of the 64 unique experts in the test sample, resulting in very high hit rates.

## Future Improvements

1. **Larger Dataset**: Evaluate on full 500 samples to see model differentiation
2. **Model Variants**: Compare with GRU, Transformer attention-based models
3. **Advanced Scheduling**: Implement expert prefetching with actual execution time tracking
4. **Stage-Specific Models**: Train separate models for prefill vs generation phases
5. **Context Adaptation**: Dynamic context size based on sequence properties

## Key Insights

1. **Small Expert Pool**: With 64 total experts and pool size 100, most experts are always available
   - Explains high hit rates for both models
   - Realistic scenarios may have larger expert pools

2. **LSTM Potential**: LSTM can learn patterns in expert usage over time
   - Especially useful for longer sequences with expert specialization
   - May show benefits with larger/more diverse datasets

3. **Prefetch Window**: 10-token lookahead is reasonable for latency-critical scenarios
   - Can be adjusted based on generation latency constraints

## Files

- `evaluate_simple.py`: Main evaluation script (no PyTorch dependency)
- `lstm_expert_predictor.py`: Full training pipeline with PyTorch
- `trace_generator.py`: Trace collection from models (existing)
- `LSTM_EXPERT_PREDICTION.md`: This documentation

## Usage Notes

### For Testing
- Use `moe_test/` directory which has 1 sample per model/dataset combo
- Run `evaluate_simple.py` for quick evaluation

### For Production
- Collect traces using `trace_generator.py` with larger sample size
- Train model on 400 samples
- Evaluate on held-out 100 samples
- Monitor hit rates and prediction latency
