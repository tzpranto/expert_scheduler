#!/usr/bin/env python3
"""
Debug script to inspect GPT5 OSS model structure and find router modules.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def inspect_model_structure(model, model_name="gpt5oss"):
    """Inspect model layer structure to find router/gate modules."""

    print(f"\n{'='*80}")
    print(f"Inspecting {model_name} Model Structure")
    print(f"{'='*80}\n")

    # Find layers
    layers = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        print("✓ Found layers at: model.model.layers")
    elif hasattr(model, 'layers'):
        layers = model.layers
        print("✓ Found layers at: model.layers")
    else:
        print("✗ Could not find layers!")
        return

    print(f"  Total layers: {len(layers)}\n")

    # Inspect first few layers
    num_to_inspect = min(3, len(layers))
    print(f"Inspecting first {num_to_inspect} layers for MoE modules:\n")

    for layer_idx in range(num_to_inspect):
        layer = layers[layer_idx]
        print(f"[Layer {layer_idx}]")
        print(f"  Layer type: {type(layer).__name__}")
        print(f"  Layer attributes: {[x for x in dir(layer) if not x.startswith('_')]}")

        # Check for MLP/FFN modules
        if hasattr(layer, 'mlp'):
            print(f"\n  ✓ Found mlp attribute")
            mlp = layer.mlp
            print(f"    MLP type: {type(mlp).__name__}")
            print(f"    MLP structure:")
            for name, module in mlp.named_modules():
                if isinstance(module, torch.nn.Linear):
                    print(f"      {name:<50} Linear(in={module.in_features}, out={module.out_features})")
                else:
                    print(f"      {name:<50} {type(module).__name__}")

        # List all named modules in this layer
        print(f"\n  All Linear modules in layer:")
        linear_modules = []
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_modules.append((name, module.in_features, module.out_features))

        if linear_modules:
            print(f"    Found: {len(linear_modules)}")
            for name, in_feat, out_feat in linear_modules:
                # Highlight potential router modules
                is_router = any(x in name.lower() for x in ['gate', 'router', 'select', 'route', 'moe', 'expert'])
                marker = "👈 ROUTER?" if is_router else "  "
                print(f"      {marker} {name:<50} in={in_feat:>5}, out={out_feat:>5}")
        else:
            print(f"    No Linear modules found")

        print()

def test_hook_registration(model):
    """Test if we can register hooks on potential router modules."""

    print(f"\n{'='*80}")
    print("Testing Hook Registration")
    print(f"{'='*80}\n")

    layers = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers

    if layers is None:
        print("✗ Could not find layers!")
        return

    hooked_count = 0
    hooks = []
    captured_outputs = {}

    def make_hook(layer_idx, module_name):
        def hook(mod, inputs, outputs):
            if isinstance(outputs, tuple):
                t = outputs[0]
            else:
                t = outputs

            if isinstance(t, torch.Tensor):
                captured_outputs[layer_idx] = {
                    'module': module_name,
                    'shape': tuple(t.shape),
                    'dtype': str(t.dtype),
                    'device': str(t.device)
                }
                print(f"  ✓ Hook captured output from layer {layer_idx} ({module_name}): shape={t.shape}, dtype={t.dtype}")
        return hook

    # Try to hook potential router modules with better detection
    for i, layer in enumerate(layers[:2]):  # Just first 2 layers for testing
        # First try to find MoE/FFN modules
        found_moe = False

        # Check if layer has mlp attribute
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            for name, module in mlp.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if any(x in name.lower() for x in ['gate', 'router', 'select', 'route']):
                        print(f"Hooking: Layer {i} -> mlp.{name}")
                        h = module.register_forward_hook(make_hook(i, f"mlp.{name}"))
                        hooks.append(h)
                        hooked_count += 1
                        found_moe = True
                        break

        # Fallback: search all named modules
        if not found_moe:
            for name, module in layer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if any(x in name.lower() for x in ['gate', 'router', 'select', 'route', 'moe']):
                        print(f"Hooking: Layer {i} -> {name}")
                        h = module.register_forward_hook(make_hook(i, name))
                        hooks.append(h)
                        hooked_count += 1
                        break

    if hooked_count == 0:
        print("✗ No potential router modules found to hook!")
        return

    print(f"\n✓ Registered {hooked_count} hooks\n")

    # Run a test forward pass
    print("Running test forward pass...")
    test_input = torch.randint(0, 50000, (1, 10)).to(model.device)

    try:
        with torch.no_grad():
            out = model(input_ids=test_input)

        if captured_outputs:
            print(f"\n✓ Hooks captured {len(captured_outputs)} outputs:")
            for layer_idx, info in sorted(captured_outputs.items()):
                print(f"  Layer {layer_idx}: {info}")
        else:
            print("\n✗ Hooks were registered but no outputs captured!")
    except Exception as e:
        print(f"\n✗ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

    # Clean up
    for h in hooks:
        h.remove()

if __name__ == "__main__":
    print("Loading GPT5-OSS model...")
    try:
        tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✓ Model loaded successfully\n")

        # Inspect structure
        inspect_model_structure(model, "GPT5-OSS")

        # Test hooks
        test_hook_registration(model)

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()

