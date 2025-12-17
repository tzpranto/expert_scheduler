#!/usr/bin/env python3
"""
Workaround to extract GPT5-OSS router logits by preventing expert execution.

The issue: GptOssTopKRouter works fine, but Mxfp4GptOssExperts fails on older GPUs.
Solution: Temporarily replace expert forward with a dummy to skip MXFP4 execution.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_router_logits_with_dummy_experts():
    """
    Extract router logits by:
    1. Monkey-patching expert forward to return zeros (skip MXFP4)
    2. Running forward pass (router hooks capture logits)
    3. Restoring expert forward
    """

    print(f"\n{'='*80}")
    print("Extracting Router Logits (Expert Execution Bypass)")
    print(f"{'='*80}\n")

    print("Loading GPT5-OSS model...")
    try:
        tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✓ Model loaded\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Find layers and experts
    layers = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers

    if layers is None:
        print("✗ Could not find layers!")
        return

    print(f"Found {len(layers)} layers\n")

    # Set up hooks to capture router outputs
    hooked_count = 0
    hooks = []
    captured_logits = {}

    def make_hook(layer_idx):
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
                captured_logits[layer_idx] = {
                    'shape': tuple(t.shape),
                    'dtype': str(t.dtype),
                    'tensor': t.detach().cpu()
                }
                print(f"  ✓ Captured router logits Layer {layer_idx}: shape={t.shape}")

        return hook

    # Register hooks on all routers
    print("Registering hooks on router modules...")
    for i, layer in enumerate(layers):
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            for name, module in mlp.named_modules():
                name_lower = name.lower()
                if 'router' in name_lower:
                    h = module.register_forward_hook(make_hook(i))
                    hooks.append(h)
                    hooked_count += 1
                    if i < 3:  # Only print first few
                        print(f"  Hooked: Layer {i} -> router")
                    break

    print(f"✓ Registered {hooked_count} hooks\n")

    # Now the critical part: Monkey-patch expert forward to skip execution
    print("Monkey-patching expert forward to skip MXFP4 execution...")
    original_forwards = {}

    for i, layer in enumerate(layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            experts = layer.mlp.experts
            original_forwards[i] = experts.forward

            # Create a dummy forward that returns zeros with the right shape
            def make_dummy_forward(layer_idx, original_forward):
                def dummy_forward(hidden_states, expert_indices):
                    """
                    Return dummy output with correct shape to avoid MXFP4 execution.
                    hidden_states: (batch, seq_len, hidden_dim)
                    expert_indices: indices of experts to "run"
                    Returns: (batch, seq_len, hidden_dim) dummy output
                    """
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    # Return zeros with same shape as expected output
                    return torch.zeros_like(hidden_states)

                return dummy_forward

            experts.forward = make_dummy_forward(i, original_forwards[i])

    print("✓ Experts patched\n")

    # Run forward pass with dummy experts
    print("Running forward pass (experts skipped via dummy forward)...")
    print("-" * 80)

    try:
        test_text = "Hello world, this is a test"
        test_input = tok(test_text, return_tensors="pt")

        with torch.no_grad():
            out = model(**test_input)

        print("-" * 80)
        print()

        if captured_logits:
            print(f"✓ SUCCESS! Captured router logits from {len(captured_logits)} layers:\n")

            for layer_idx in sorted(captured_logits.keys()):
                info = captured_logits[layer_idx]
                shape = info['shape']
                num_experts = shape[-1] if len(shape) >= 1 else -1
                print(f"  Layer {layer_idx}:")
                print(f"    Shape: {shape}")
                print(f"    Num Experts: {num_experts}")
                print(f"    Dtype: {info['dtype']}")

            if captured_logits:
                first_shape = captured_logits[0]['shape']
                num_experts = first_shape[-1] if len(first_shape) >= 1 else -1
                print(f"\n  Inferred num_experts: {num_experts}")
        else:
            print("✗ Hooks registered but no outputs captured!")

    except Exception as e:
        print(f"\n✗ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Restore original expert forwards
        print("\nRestoring original expert forward methods...")
        for i, layer in enumerate(layers):
            if i in original_forwards:
                layer.mlp.experts.forward = original_forwards[i]
        print("✓ Restoration complete")

        # Clean up hooks
        for h in hooks:
            h.remove()

if __name__ == "__main__":
    extract_router_logits_with_dummy_experts()
