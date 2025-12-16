#!/usr/bin/env python3
"""
Debug script to test GPT5 OSS router hooks on CPU (bypasses GPU MXFP4 limitation).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_hook_registration_cpu():
    """Test hooks by running on CPU to avoid GPU MXFP4 incompatibility."""

    print(f"\n{'='*80}")
    print("Testing Hook Registration on CPU")
    print("(Using CPU to avoid GPU MXFP4 compatibility issues)")
    print(f"{'='*80}\n")

    print("Loading GPT5-OSS model on CPU (bf16 precision)...")
    try:
        tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            torch_dtype=torch.bfloat16,
            device_map="cpu"  # Force CPU
        )
        print("✓ Model loaded on CPU\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Find layers
    layers = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers

    if layers is None:
        print("✗ Could not find layers!")
        return

    print(f"Found {len(layers)} layers\n")

    # Set up hooks
    hooked_count = 0
    hooks = []
    captured_outputs = {}

    def make_hook(layer_idx, module_name):
        def hook(mod, inputs, outputs):
            # Handle multiple output formats
            t = None

            if isinstance(outputs, torch.Tensor):
                t = outputs
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                t = outputs[0]
            elif isinstance(outputs, dict):
                for key in ['logits', 'router_logits', 'routing_logits', 'output']:
                    if key in outputs and isinstance(outputs[key], torch.Tensor):
                        t = outputs[key]
                        break

            if t is not None and isinstance(t, torch.Tensor):
                captured_outputs[layer_idx] = {
                    'module': module_name,
                    'shape': tuple(t.shape),
                    'dtype': str(t.dtype),
                    'device': str(t.device)
                }
                print(f"  ✓ Hook captured: Layer {layer_idx} ({module_name}): shape={t.shape}, dtype={t.dtype}")
        return hook

    # Hook router modules in first 3 layers
    print("Registering hooks on router modules...")
    for i, layer in enumerate(layers[:3]):
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            for name, module in mlp.named_modules():
                name_lower = name.lower()
                if any(x in name_lower for x in ['gate', 'router', 'select', 'route']):
                    print(f"  Hooking: Layer {i} -> mlp.{name} (type: {type(module).__name__})")
                    h = module.register_forward_hook(make_hook(i, f"mlp.{name}"))
                    hooks.append(h)
                    hooked_count += 1
                    break

    if hooked_count == 0:
        print("✗ No router modules found!")
        return

    print(f"\n✓ Registered {hooked_count} hooks\n")

    # Test forward pass with small input
    print("Running test forward pass with small input (seq_len=5)...")
    print("-" * 80)

    try:
        # Small test input
        test_text = "Hello world"
        test_input = tok(test_text, return_tensors="pt")

        with torch.no_grad():
            out = model(**test_input)

        print("-" * 80)
        if captured_outputs:
            print(f"\n✓ SUCCESS! Hooks captured {len(captured_outputs)} router outputs:\n")
            for layer_idx in sorted(captured_outputs.keys()):
                info = captured_outputs[layer_idx]
                print(f"  Layer {layer_idx}:")
                print(f"    Module: {info['module']}")
                print(f"    Shape:  {info['shape']}")
                print(f"    Dtype:  {info['dtype']}")
                print(f"    Device: {info['device']}")
                print()

            # Infer number of experts
            first_output_shape = captured_outputs[0]['shape']
            if len(first_output_shape) >= 1:
                num_experts = first_output_shape[-1]
                print(f"  Inferred num_experts: {num_experts}")
        else:
            print("\n✗ Hooks registered but no outputs captured!")

    except Exception as e:
        print(f"\n✗ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

    # Clean up
    for h in hooks:
        h.remove()

if __name__ == "__main__":
    test_hook_registration_cpu()
