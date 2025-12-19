"""
Expert Prediction Module
Provides configurable LSTM model for predicting expert usage in MoE systems.

Supports:
  - OLMoE: 16 layers, 64 experts, top-K=8
  - GPT5OSS: 24 layers, 32 experts, top-K=4
  - Custom configurations via ModelConfig

Usage:
  from prediction.data_loader import ExpertDataLoader, OLMOE_CONFIG
  from prediction.bilstm_model import BiLSTMExpertPredictor
  from prediction.train import main as train_model
"""

from .data_loader import ExpertDataLoader, OLMOE_CONFIG, GPT5OSS_CONFIG, ModelConfig
from .bilstm_model import BiLSTMExpertPredictor, train_epoch, evaluate

__all__ = [
    'ExpertDataLoader',
    'OLMOE_CONFIG',
    'GPT5OSS_CONFIG',
    'ModelConfig',
    'BiLSTMExpertPredictor',
    'train_epoch',
    'evaluate',
]
