import os, json, time, torch
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import argparse

# --- Global variable for top-k setting (can be overridden by argument) ---
TOPK_PER_TOKEN = None

class RouterTraceContext:
    """
    Context manager to capture router logits via forward hooks.
    Useful when model.config.output_router_logits is not supported or not working.
    """
    def __init__(self, model, verbose=False):
        self.model = model
        self.hooks = []
        self.captured_logits = {} # layer_idx -> list of logits (one per call)
        self.num_experts = None   # Stores the inferred total number of experts (E)
        self.verbose = verbose
        self.hooked_modules = []  # Track which modules we hooked

    def __enter__(self):
        self._register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()
        
    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.captured_logits = {}
        # Keep self.num_experts unless we know we are reloading a new model

    def _register_hooks(self):
        # Heuristic: Find router modules in MoE layers.
        layers = None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers

        if layers is None:
            print("Warning: Could not find layers to attach hooks.")
            return

        if self.verbose:
            print(f"[Hooks] Scanning {len(layers)} layers for router modules...")

        for i, layer in enumerate(layers):
            target_modules = []  # Collect all candidates

            # Strategy 1: Check for MLP/FFN attribute (common in MoE architectures)
            # This hooks ANY router-like module, not just Linear layers
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                for name, module in mlp.named_modules():
                    name_lower = name.lower()
                    # Hook ANY module with router/gate/select/route in name
                    if 'gate' in name_lower or 'router' in name_lower:
                        target_modules.append((f"mlp.{name}", module, 'mlp_primary'))
                    elif any(x in name_lower for x in ['select', 'route']):
                        target_modules.append((f"mlp.{name}", module, 'mlp_secondary'))

            # Strategy 2: Search all named modules (fallback)
            if not target_modules:
                for name, module in layer.named_modules():
                    # Skip self references
                    if module is layer:
                        continue

                    name_lower = name.lower()

                    # Primary targets: Standard router/gate names (any module type)
                    if 'gate' in name_lower or 'router' in name_lower:
                        target_modules.append((name, module, 'primary'))
                    elif 'moe' in name_lower and any(x in name_lower for x in ['gate', 'router', 'select', 'route']):
                        target_modules.append((name, module, 'moe_keyword'))
                    elif 'moe' in name_lower:
                        target_modules.append((name, module, 'moe_fallback'))

            # Select the best candidate based on priority
            target_module = None
            target_name = None
            priority_order = ['mlp_primary', 'mlp_secondary', 'primary', 'moe_keyword', 'moe_fallback']

            for priority in priority_order:
                candidates = [m for m in target_modules if m[2] == priority]
                if candidates:
                    # If multiple candidates, prefer the one with fewest submodules (more direct)
                    target_name, target_module, _ = min(candidates, key=lambda x: len(list(x[1].modules())))
                    if self.verbose:
                        print(f"  [Layer {i}] Found router module ({priority}): {target_name} (type: {type(target_module).__name__})")
                    break

            if target_module:
                self._hook_module(i, target_module, target_name)
                self.hooked_modules.append((i, target_name))
            elif self.verbose:
                print(f"  [Layer {i}] No router module found")

        if self.verbose:
            print(f"[Hooks] Successfully hooked {len(self.hooked_modules)} layers")

    def _hook_module(self, layer_idx, module, target_name=None):
        def hook(mod, inputs, outputs):
            # Handle various output formats from different router implementations
            t = None

            # Case 1: Direct tensor output
            if isinstance(outputs, torch.Tensor):
                t = outputs
            # Case 2: Tuple of tensors (common: logits, routing_indices, etc.)
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                t = outputs[0]
            # Case 3: Dict output (some MoE routers return dicts)
            elif isinstance(outputs, dict):
                # Try common keys
                for key in ['logits', 'router_logits', 'routing_logits', 'output']:
                    if key in outputs and isinstance(outputs[key], torch.Tensor):
                        t = outputs[key]
                        break

            # If we have a tensor, process it
            if t is not None and isinstance(t, torch.Tensor):
                if layer_idx not in self.captured_logits:
                    self.captured_logits[layer_idx] = []

                # Detach and move to CPU immediately to save VRAM
                t_detached = t.detach().cpu()
                self.captured_logits[layer_idx].append(t_detached)

                # Infer and store num_experts (E) from the last dimension
                if self.num_experts is None and t_detached.dim() >= 1:
                    self.num_experts = t_detached.shape[-1]

                if self.verbose:
                    print(f"    [Hook captured] Layer {layer_idx}: shape={t_detached.shape}, dtype={t_detached.dtype}")

        self.hooks.append(module.register_forward_hook(hook))
    
    def get_logits(self, layer_idx):
        return self.captured_logits.get(layer_idx, [])


def format_prompt_harmony(tok, prompt, reasoning_effort="medium"):
    """
    Format prompt with harmony chat template for GPT-5 OSS.
    reasoning_effort: 'low', 'medium', or 'high'
    """
    messages = [
        {"role": "developer", "content": "Try to avoid hallucination"},
        {"role": "user", "content": prompt},
    ]
    
    # Apply chat template with reasoning effort (assumes custom tokenizer supports this)
    formatted = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,  # Return string, not tokens
        reasoning_effort=reasoning_effort,
    )
    
    return formatted

def get_oasst1_prompts(sample_n=500, split="train", lang="en", min_chars=30, seed=42):
    """
    Build conversational, context-free prompts from OpenAssistant/oasst1.
    """
    ds = load_dataset("OpenAssistant/oasst1")[split]

    def is_root_user_msg(row):
        return (row.get("role") == "prompter") and (row.get("parent_id") is None)

    rows = [r for r in ds if is_root_user_msg(r)]
    if lang:
        rows = [r for r in rows if r.get("lang") == lang]

    def clean(s): return " ".join(str(s).split())
    rows = [r for r in rows if r.get("text")]
    rows = [{"text": clean(r["text"])} for r in rows if len(clean(r["text"])) >= min_chars]

    seen = set()
    unique = []
    for r in rows:
        t = r["text"]
        if t not in seen:
            seen.add(t)
            unique.append(t)

    random.seed(seed)
    random.shuffle(unique)
    prompts = unique[:sample_n]

    prompts = [f"{p}\nA:" for p in prompts]
    ds = Dataset.from_dict({
        "prompt": prompts,
        "category": ["conversation"] * len(prompts)
    })
    return ds

def get_dataset(dataset_id, subset=None, split=None):
    if dataset_id == 'orbench':
        return load_dataset("bench-llm/or-bench", "or-bench-hard-1k")['train']
    elif dataset_id == 'xstest':
        ds = load_dataset("walledai/XSTest")['test']
        ds = ds.rename_column("label", "category")
        return ds
    elif dataset_id == 'oasst':
        return get_oasst1_prompts()
    else:
        print('Dataset not supported. Make changes in get_data() function to add support.')
        exit(0)

def load_model_and_tok(model_id, dtype, device_map):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map
    )
    # Attempt to enable built-in router logits output (for models that support it)
    try:
        model.config.output_router_logits = True
    except:
        pass
    return tok, model

def get_model(model_id):
    if model_id == 'olmoe':
        return load_model_and_tok("allenai/OLMoE-1B-7B-0125", torch.float16, "auto")
    elif model_id == 'gpt5oss':
        return load_model_and_tok("openai/gpt-oss-20b", "auto", "auto")
    else:
        print("Model not supported!")
        exit(0)


def _norm_router_tensor(t):
    """
    Standardize router tensor shape to (B, T, E) where:
    - B = batch size (usually 1 during inference)
    - T = sequence length
    - E = number of experts

    Handles various shapes from different MoE implementations.
    """
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"Expected tensor, got {type(t)}")

    if t.dim() == 3:
        # Already in correct format (B, T, E)
        return t
    elif t.dim() == 2:
        # Could be (T, E) or (B*T, E), assume (T, E) and add batch dim
        return t.unsqueeze(0)
    elif t.dim() == 1:
        # Single value, expand to (1, 1, E)
        return t.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected router tensor dim {t.dim()} with shape {tuple(t.shape)}")

@torch.no_grad()
def collect_prompt_router_trace(model, tok, text, use_hooks=False, verbose=False):
    global TOPK_PER_TOKEN
    enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    ctx = RouterTraceContext(model, verbose=verbose) if use_hooks else None

    if use_hooks:
        ctx.__enter__()
        # Check if hooks were registered
        if not ctx.hooked_modules:
            raise RuntimeError(
                "No router modules found! Could not register hooks. "
                "Try running: python debug_gpt5_structure.py to inspect model structure."
            )
        out = model(**enc)
    else:
        out = model(**enc, output_router_logits=True)

    # Retrieve logits
    if use_hooks:
        captured = ctx.captured_logits
        sorted_layers = sorted(captured.keys())

        if not captured:
            raise RuntimeError(
                f"Hooks registered on {len(ctx.hooked_modules)} layers but no logits captured! "
                "The forward pass may not have triggered the hooked modules."
            )

        router_logits = []
        for i in sorted_layers:
            # We take the first one (should be only one for prefill)
            if captured[i]:
                router_logits.append(captured[i][0])

        if not router_logits:
            raise RuntimeError(
                f"Hooked {len(ctx.hooked_modules)} layers and captured {len(captured)} layers, "
                "but captured tensors are empty!"
            )

        ctx.__exit__(None, None, None)
        router = tuple(_norm_router_tensor(x) for x in router_logits)
        E = ctx.num_experts # Get E from the context manager
    else:
        if not out.router_logits:
            raise RuntimeError(
                "Built-in output_router_logits=True returned empty! "
                "This is a known issue with GPT5-OSS. Try --force_hooks flag."
            )
        router = tuple(_norm_router_tensor(x) for x in out.router_logits)
        E = model.config.num_experts if hasattr(model.config, "num_experts") else -1
        if E == -1:
            E = router[0].shape[-1] if router else -1

    if not router:
         raise RuntimeError("Could not collect router logits. Hook failed or built-in output is empty.")

    B, T, _ = router[0].shape
    
    # Determine k: experts per token
    if hasattr(model.config, "num_experts_per_tok"):
        k = TOPK_PER_TOKEN or model.config.num_experts_per_tok
    else:
        k = TOPK_PER_TOKEN or 2 # Default fallback
    
    current_k = min(k, E) if E > 0 else k
        
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    layers = []
    for L, r in enumerate(router):
        probs = torch.softmax(r[0], dim=-1)
        vals, idxs = torch.topk(probs, k=current_k, dim=-1)
        token_data = [
            {"token": tokens[t], "topk_experts": idxs[t].cpu().tolist(),
             "topk_probs": vals[t].cpu().tolist()}
            for t in range(len(tokens))
        ]
        layers.append({"layer": L, "num_experts": E, "topk_per_token": current_k, "token_data": token_data})
    
    return {
        "prompt": text,
        "num_layers": len(router),
        "num_experts": E,
        "k_per_token": current_k,
        "layers": layers,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

@torch.no_grad()
def collect_generate_router_trace(model, tok, prompt, max_new_tokens=64, use_hooks=False, verbose=False):
    global TOPK_PER_TOKEN
    enc = tok(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]

    ctx = RouterTraceContext(model, verbose=verbose) if use_hooks else None
    
    # Determine K (experts per tok)
    if hasattr(model.config, "num_experts_per_tok"):
        k = TOPK_PER_TOKEN or model.config.num_experts_per_tok
    else:
        k = TOPK_PER_TOKEN or 2

    # First forward pass (prefill)
    if use_hooks:
        ctx.__enter__()
        # Check if hooks were registered
        if not ctx.hooked_modules:
            raise RuntimeError(
                "No router modules found! Could not register hooks. "
                "Try running: python debug_gpt5_structure.py to inspect model structure."
            )
        out = model(input_ids=input_ids, use_cache=True)
        # Determine E from hooks after prefill
        if not ctx.captured_logits:
            raise RuntimeError(
                f"Hooks registered on {len(ctx.hooked_modules)} layers but no logits captured! "
                "The forward pass may not have triggered the hooked modules."
            )
        E = ctx.num_experts if ctx.num_experts is not None else -1
    else:
        out = model(input_ids=input_ids, use_cache=True, output_router_logits=True)
        # Determine E from model config or output tensor
        if not out.router_logits:
            raise RuntimeError(
                "Built-in output_router_logits=True returned empty! "
                "This is a known issue with GPT5-OSS. Try --force_hooks flag."
            )
        if hasattr(model.config, "num_experts"):
            E = model.config.num_experts
        elif out.router_logits:
            E = out.router_logits[0].shape[-1]
        else:
            E = -1
    
    past = out.past_key_values
    
    # Safety check for k and E
    current_k = min(k, E) if E > 0 else k

    generated = []
    steps = []
    start = time.time()
    curr_out = out
    
    # Initialize layers info for the loop structure.
    step_layers_init = []
    num_layers = 0
    if E > 0 and use_hooks:
        num_layers = len(ctx.captured_logits)
        sorted_layers = sorted(ctx.captured_logits.keys())
    elif E > 0 and not use_hooks and out.router_logits:
        num_layers = len(out.router_logits)
    
    
    for step in range(max_new_tokens):
        # 1. Get next token
        logits = curr_out.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated.append(next_token.item())

        # 2. Capture router logits for this token
        step_layers = []
        
        if use_hooks and E > 0:
            # We rely on the hook mechanism capturing the latest tensor
            for L in sorted_layers:
                # Get the last captured tensor
                r = _norm_router_tensor(ctx.captured_logits[L][-1])
                probs = torch.softmax(r[0, -1], dim=-1)
                vals, idxs = torch.topk(probs, k=current_k)
                step_layers.append({
                    "layer": L,
                    "topk_experts": idxs.cpu().tolist(),
                    "topk_probs": vals.cpu().tolist()
                })
        elif not use_hooks and E > 0 and curr_out.router_logits:
            for L, r_raw in enumerate(curr_out.router_logits):
                r = _norm_router_tensor(r_raw)
                probs = torch.softmax(r[0, -1], dim=-1)
                vals, idxs = torch.topk(probs, k=current_k)
                step_layers.append({
                    "layer": L,
                    "topk_experts": idxs.cpu().tolist(),
                    "topk_probs": vals.cpu().tolist()
                })
        
        # If the number of layers is zero, this loop will still run, 
        # but step_layers will be empty. We check E>0 to ensure we save data.
        steps.append({"step": step, "token_id": next_token.item(), "layers": step_layers})

        if (step + 1) % 64 == 0 or step == max_new_tokens - 1:
            elapsed = time.time() - start
            print(f"    └── Generation step {step+1}/{max_new_tokens} "
                  f"({elapsed:.1f}s elapsed)")

        # Stop if end of sentence is reached
        if next_token.item() == tok.eos_token_id:
            break

        # 3. Run next step
        curr_out = model(input_ids=next_token, past_key_values=past,
                    use_cache=True, output_router_logits=not use_hooks)
        past = curr_out.past_key_values

    if use_hooks:
        ctx.__exit__(None, None, None)

    gen_text = tok.decode(generated, skip_special_tokens=True)
    
    return {
        "prompt": prompt,
        "generated_text": gen_text,
        "generated_ids": generated,
        "decode_steps": steps,
        "num_layers": num_layers,
        "num_experts": E, 
        "k_per_token": current_k,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect router traces for MoE models."
    )

    parser.add_argument(
        "--model_id",
        type=str,
        choices=["olmoe", "gpt5oss"],
        default="olmoe",
        help="Model type to run (default: olmoe)"
    )

    parser.add_argument(
        "--dataset_id",
        type=str,
        choices=["orbench", "xstest", "oasst"],
        default="oasst",
        help="Dataset to use (default: oasst)"
    )

    parser.add_argument(
        "--force_hooks",
        action="store_true",
        help="Force usage of hooks for router logits collection (recommended for gpt5oss)"
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting index of samples to process (default: 0)"
    )

    parser.add_argument(
        "--total",
        type=int,
        default=None,
        help="Total number of samples to process (default: None)"
    )

    parser.add_argument(
        "--max_token",
        type=int,
        default=256,
        help="Number of new tokens (default: 256)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["exec", "test"],
        default="test",
        help="Model of execution. Select 'test' mode to generate one sample. (default: test)"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Override for top-K experts to save (default: use model config)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging hook registration"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # Set global K
    TOPK_PER_TOKEN = args.top_k
    
    ds = get_dataset(args.dataset_id)
    start_idx = args.start
    total_samples = args.total if args.total is not None else len(ds)
    
    # Adjust total based on start
    total_samples = min(total_samples, len(ds) - start_idx)
    
    print(f"Processing {total_samples} samples from {args.dataset_id}, starting at index {start_idx}...\n")
    
    OUT_DIR = Path(f"moe_traces/{args.model_id}/{args.dataset_id}")
    MAX_NEW_TOKENS = args.max_token
    tok, model = get_model(args.model_id)
    
    # Force hooks for GPT-5 OSS or if requested
    # The fix ensures this is robust.
    USE_HOOKS = (args.model_id == 'gpt5oss') or args.force_hooks
    
    if args.model_id == 'gpt5oss':
        print("--- GPT-5 OSS Harmony Prompt Format Sample ---")
        print(format_prompt_harmony(tok, ds[0]["prompt"]))
        print("-------------------------------------------\n")

    if args.mode == 'test':
        print("--- Running in TEST mode (1 sample only) ---")
        start_idx = 0
        total_samples = 1
        OUT_DIR = Path(f"moe_test/{args.model_id}/{args.dataset_id}")
        
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i in range(start_idx, start_idx + total_samples, 1):
        row = ds[i]
        prompt = row["prompt"]
        category = row["category"]
    
        print(f"[{i+1-start_idx}/{total_samples}] ⏳ Collecting router trace for sample index {i}, prompt "
                f"{repr(prompt[:60])}...")

        # 1. Prefill Trace
        try:
            prefill = collect_prompt_router_trace(model, tok, prompt, use_hooks=USE_HOOKS, verbose=args.verbose)
            prefill["category"] = category
            json.dump(prefill, open(OUT_DIR / f"trace_{i:04d}.json","w"), indent=2)
            print(f"    ✅ Prefill trace saved ({prefill.get('num_layers', 0)} layers, {prefill.get('num_experts', -1)} experts).")
        except Exception as e:
            print(f"    ❌ Error collecting prefill trace for index {i}: {e}")
            continue # Skip generation if prefill fails

        # 2. Generation Trace
        try:
            print(f"    ⚙️  Starting generation for prompt index {i}...")
            gen = collect_generate_router_trace(model, tok, prompt, MAX_NEW_TOKENS, use_hooks=USE_HOOKS, verbose=args.verbose)
            gen["category"] = category
            json.dump(gen, open(OUT_DIR / f"gen_{i:04d}.json","w"), indent=2)
            print(f"    ✅ Generation trace saved "
                    f"({len(gen['decode_steps'])} steps, "
                    f"total records: {len(gen['decode_steps'])*gen.get('num_layers', 0)}).")
        except Exception as e:
            print(f"    ❌ Error collecting generation trace for index {i}: {e}")
            
        elapsed = time.time() - t0
        print(f"✔️  Done prompt {i} in {elapsed:.1f}s total (Average: {elapsed/(i+1-start_idx):.1f}s/sample)\n")

    print(f"🏁 All {total_samples} prompts processed in {(time.time()-t0)/60:.1f} min.")
    print(f"Traces saved to: {OUT_DIR.resolve()}")