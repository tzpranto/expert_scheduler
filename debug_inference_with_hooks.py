#!/usr/bin/env python3
"""
Debug script to test GPT5-OSS inference WITH and WITHOUT hooks.
This helps identify if:
1. The model can run inference at all
2. Hooks interfere with normal operation
3. MXFP4 is the real blocker
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

class SimpleRouterHook:
    """Minimal hook that just captures router outputs without patching"""
    def __init__(self, model, verbose=False):
        self.model = model
        self.verbose = verbose
        self.hooks = []
        self.captured = {}

    def register(self):
        """Register hooks on all router modules"""
        layers = None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers

        if layers is None:
            print("❌ Could not find layers")
            return 0

        hooked = 0
        for i, layer in enumerate(layers):
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                for name, module in mlp.named_modules():
                    if 'router' in name.lower():
                        h = module.register_forward_hook(self._make_hook(i))
                        self.hooks.append(h)
                        hooked += 1
                        if self.verbose:
                            print(f"  Hooked layer {i}: {name}")
                        break

        if self.verbose:
            print(f"✓ Registered {hooked} hooks")
        return hooked

    def _make_hook(self, layer_idx):
        def hook(mod, inputs, outputs):
            # Capture only metadata as strings - no tensor references
            if isinstance(outputs, tuple) and len(outputs) > 0:
                t = outputs[0]
            else:
                t = outputs

            if isinstance(t, torch.Tensor):
                # Store only string representations to avoid device issues
                self.captured[layer_idx] = {
                    'shape': str(t.shape),
                    'dtype': str(t.dtype),
                    'device': str(t.device)
                }
        return hook

    def cleanup(self):
        """Remove all hooks"""
        for h in self.hooks:
            h.remove()
        self.hooks = []

def test_inference_without_hooks():
    """Test inference WITHOUT any hooks"""
    print("\n" + "="*80)
    print("TEST 1: Inference WITHOUT Hooks")
    print("="*80 + "\n")

    print("Loading model...")
    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    print("✓ Model loaded\n")

    # Simple inference test
    prompt = "Hello world, this is a test of"
    print(f"Prompt: {prompt}\n")

    try:
        print("Running inference (no hooks)...")
        inputs = tok(prompt, return_tensors="pt")
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=10,
                do_sample=False
            )

        elapsed = time.time() - start
        generated_text = tok.decode(outputs[0], skip_special_tokens=True)

        print(f"✓ Inference succeeded in {elapsed:.2f}s\n")
        print(f"Output:\n{generated_text}\n")
        return True, generated_text

    except RuntimeError as e:
        error_msg = str(e)
        print(f"✗ Inference failed: {error_msg}\n")

        if "cvt with .e4m3x2/.e5m2x2" in error_msg or "sm_89" in error_msg:
            print("  → MXFP4 incompatibility detected (expected on older GPUs)")
            return False, None
        else:
            print("  → Different error (unexpected)")
            return False, None

    except Exception as e:
        print(f"✗ Unexpected error: {e}\n")
        return False, None

    finally:
        # Clean up
        del model
        del tok
        torch.cuda.empty_cache()


def test_inference_with_hooks():
    """Test inference WITH hooks (non-interfering)"""
    print("\n" + "="*80)
    print("TEST 2: Inference WITH Hooks (capturing only, no patching)")
    print("="*80 + "\n")

    print("Loading model...")
    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    print("✓ Model loaded\n")

    # Register hooks
    hook_manager = SimpleRouterHook(model, verbose=True)
    hooked = hook_manager.register()

    if hooked == 0:
        print("❌ No router modules found to hook")
        return False, None

    print()

    # Simple inference test
    prompt = "Hello world, this is a test of"
    print(f"Prompt: {prompt}\n")

    try:
        print("Running inference (with hooks)...")
        inputs = tok(prompt, return_tensors="pt")
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=10,
                do_sample=False
            )

        elapsed = time.time() - start
        generated_text = tok.decode(outputs[0], skip_special_tokens=True)

        print(f"✓ Inference succeeded in {elapsed:.2f}s\n")
        print(f"Output:\n{generated_text}\n")

        # Show captured router outputs
        if hook_manager.captured:
            print(f"✓ Captured router outputs from {len(hook_manager.captured)} layers:")
            for layer_idx in sorted(hook_manager.captured.keys()):
                info = hook_manager.captured[layer_idx]
                print(f"  Layer {layer_idx}: shape={info['shape']}, dtype={info['dtype']}")
            print()

        return True, generated_text

    except RuntimeError as e:
        error_msg = str(e)
        print(f"✗ Inference failed: {error_msg}\n")

        if "cvt with .e4m3x2/.e5m2x2" in error_msg or "sm_89" in error_msg:
            print("  → MXFP4 incompatibility detected (expected on older GPUs)")
            return False, None
        else:
            print("  → Different error (unexpected)")
            return False, None

    except Exception as e:
        print(f"✗ Unexpected error: {e}\n")
        return False, None

    finally:
        hook_manager.cleanup()
        del model
        del tok
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GPT5-OSS Inference Debug: WITH vs WITHOUT Hooks")
    print("="*80)

    # Test 1: Without hooks
    success1, output1 = test_inference_without_hooks()

    # Test 2: With hooks
    success2, output2 = test_inference_with_hooks()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")

    print(f"Inference WITHOUT hooks: {'✓ Success' if success1 else '✗ Failed'}")
    print(f"Inference WITH hooks:    {'✓ Success' if success2 else '✗ Failed'}")

    if success1 and success2:
        print("\n✓ Both tests passed! Hooks don't interfere with inference.")
        print("  → You can use hooks for trace collection without affecting output")
        if output1 == output2:
            print("  → Outputs are identical (hooks are truly non-interfering)")
        else:
            print("  ⚠ Outputs differ (hooks may have side effects)")
    elif success1 and not success2:
        print("\n❌ Hooks break inference! Need to debug hook implementation")
    elif not success1 and not success2:
        print("\n❌ Inference fails with or without hooks")
        print("   MXFP4 is likely the root cause on older GPUs")
    else:
        print("\n⚠ Unexpected result pattern")
