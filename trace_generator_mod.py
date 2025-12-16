import os, json, time, torch
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import argparse

class RouterTraceContext:
    """
    Context manager to capture router logits via forward hooks.
    Useful when model.config.output_router_logits is not supported or not working.
    """
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.captured_logits = {} # layer_idx -> list of logits (one per call)
        self.num_experts = None # Store the inferred number of experts E

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

    def _register_hooks(self):
        # Heuristic: Find router modules.
        
        layers = None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        
        if layers is None:
            print("Warning: Could not find layers to attach hooks.")
            return

        for i, layer in enumerate(layers):
            # Find the gate module in this layer.
            target_module = None
            for name, module in layer.named_modules():
                # We are looking for the linear layer that is the gate.
                if ('gate' in name or 'router' in name) and not name.endswith('.fn'): 
                     target_module = module
                     break
            
            if target_module:
                self._hook_module(i, target_module)
            else:
                # print(f"Warning: No gate found in layer {i}")
                pass

    def _hook_module(self, layer_idx, module):
        def hook(mod, inputs, outputs):
            # outputs might be tensor or tuple
            if isinstance(outputs, tuple):
                t = outputs[0]
            else:
                t = outputs
            
            if layer_idx not in self.captured_logits:
                self.captured_logits[layer_idx] = []
            
            # Detach and move to CPU immediately to save VRAM
            t_detached = t.detach().cpu()
            self.captured_logits[layer_idx].append(t_detached)

            # Infer and store num_experts (E) from the last dimension
            # We only need to do this once.
            if self.num_experts is None and t_detached.dim() >= 1:
                self.num_experts = t_detached.shape[-1]
            
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
    
    # Apply chat template with reasoning effort
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
    We take only the FIRST user turn (root 'prompter'), English, length>=min_chars,
    deduplicate, and randomly sample 'sample_n'. We append 'A:' to cue generation.
    """
    ds = load_dataset("OpenAssistant/oasst1")[split]

    def is_root_user_msg(row):
        return (row.get("role") == "prompter") and (row.get("parent_id") is None)

    # filter: first user turns
    rows = [r for r in ds if is_root_user_msg(r)]

    # language
    if lang:
        rows = [r for r in rows if r.get("lang") == lang]

    # clean + min length
    def clean(s): return " ".join(str(s).split())
    rows = [r for r in rows if r.get("text")]
    rows = [{"text": clean(r["text"])} for r in rows if len(clean(r["text"])) >= min_chars]

    # dedupe
    seen = set()
    unique = []
    for r in rows:
        t = r["text"]
        if t not in seen:
            seen.add(t)
            unique.append(t)

    # sample
    random.seed(seed)
    random.shuffle(unique)
    prompts = unique[:sample_n]

    # conversational QA style
    prompts = [f"{p}\nA:" for p in prompts]
    ds = Dataset.from_dict({
        "prompt": prompts,
        "category": ["conversation"] * len(prompts)
    })
    return ds

def get_dataset(dataset_id, subset=None, split=None):
    # Returned Dataset must have two cloumns
    # 1. prompt
    # 2. category
    # Add support for new dataset by following this format
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
    # Ask the model to return router logits
    # We set it, but we also rely on hooks for models that don't support it fully
    try:
        model.config.output_router_logits = True
    except:
        pass
    return tok, model

def get_model(model_id):
    if model_id == 'olmoe':
        # Default dtype for OLMoE
        return load_model_and_tok("allenai/OLMoE-1B-7B-0125", torch.float16, "auto")
    elif model_id == 'gpt5oss':
        # Use 'auto' for dtype to let HF handle it
        return load_model_and_tok("openai/gpt-oss-20b", "auto", "auto")
    else:
        print("Model not supported!")
        exit(0)


def _norm_router_tensor(t):
    # Expected: (B, T, E). Sometimes comes as (T, E) or (E,)
    if t.dim() == 3:
        return t  # (B, T, E)
    if t.dim() == 2:
        return t.unsqueeze(0)  # -> (1, T, E)
    if t.dim() == 1:
        return t.unsqueeze(0).unsqueeze(0)  # -> (1, 1, E)
    raise ValueError(f"Unexpected router tensor dim {t.dim()} with shape {tuple(t.shape)}")

# Global for k-per-token if not in config
TOPK_PER_TOKEN = None

@torch.no_grad()
def collect_prompt_router_trace(model, tok, text, use_hooks=False):
    global TOPK_PER_TOKEN
    enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    ctx = RouterTraceContext(model) if use_hooks else None
    
    if use_hooks:
        ctx.__enter__()
        out = model(**enc)
    else:
        out = model(**enc, output_router_logits=True)

    # Retrieve logits
    if use_hooks:
        captured = ctx.captured_logits
        sorted_layers = sorted(captured.keys())
        router_logits = []
        for i in sorted_layers:
            # We take the first one (should be only one for prefill)
            router_logits.append(captured[i][0])
        ctx.__exit__(None, None, None)
        router = tuple(_norm_router_tensor(x) for x in router_logits)
        E = ctx.num_experts # Get E from the context manager
    else:
        router = tuple(_norm_router_tensor(x) for x in out.router_logits)
        if not router:
            raise RuntimeError("Model returned no router logits.")
        E = router[0].shape[-1] # Get E from the tensor shape
        
    B, T, _ = router[0].shape
    
    # Handle num_experts_per_tok safely
    if hasattr(model.config, "num_experts_per_tok"):
        k = TOPK_PER_TOKEN or model.config.num_experts_per_tok
    else:
        # Fallback default (common value is 2)
        k = TOPK_PER_TOKEN or 2
        
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    layers = []
    for L, r in enumerate(router):
        # r is (1, T, E)
        probs = torch.softmax(r[0], dim=-1)
        # Use min(k, E) to avoid error if k > E (e.g., if E is small)
        current_k = min(k, E) 
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
def collect_generate_router_trace(model, tok, prompt, max_new_tokens=64, use_hooks=False):
    global TOPK_PER_TOKEN
    enc = tok(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    
    ctx = RouterTraceContext(model) if use_hooks else None
    
    # Determine K (num_experts_per_tok)
    if hasattr(model.config, "num_experts_per_tok"):
        k = TOPK_PER_TOKEN or model.config.num_experts_per_tok
    else:
        k = TOPK_PER_TOKEN or 2

    # First forward pass (prefill)
    if use_hooks:
        ctx.__enter__()
        out = model(input_ids=input_ids, use_cache=True)
        # Determine E from hooks after prefill
        E = ctx.num_experts if ctx.num_experts is not None else -1
    else:
        out = model(input_ids=input_ids, use_cache=True, output_router_logits=True)
        # Determine E from model config or output tensor
        if hasattr(model.config, "num_experts"):
            E = model.config.num_experts
        elif out.router_logits:
            E = out.router_logits[0].shape[-1]
        else:
            E = -1 # Fallback, should not happen if output_router_logits=True works
    
    past = out.past_key_values
    
    # Safety check for k and E
    current_k = min(k, E) if E > 0 else k

    generated = []
    steps = []
    start = time.time()
    
    curr_out = out
    
    # We ignore router logits from the prefill pass, only capture for generated tokens
    
    for step in range(max_new_tokens):
        # 1. Get next token from curr_out.logits
        logits = curr_out.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated.append(next_token.item())
        
        # 2. Capture router logits for this token (from the forward pass that just happened)
        step_layers = []
        
        if use_hooks:
            # Retrieve from ctx. The LAST entry in the list is the most recent forward pass.
            sorted_layers = sorted(ctx.captured_logits.keys())
            
            for L in sorted_layers:
                # Get the last captured tensor (should be (1, 1, E) for generation steps)
                r = _norm_router_tensor(ctx.captured_logits[L][-1])
                probs = torch.softmax(r[0, -1], dim=-1) # softmax on the last token of the batch/sequence
                vals, idxs = torch.topk(probs, k=current_k)
                step_layers.append({
                    "layer": L,
                    "topk_experts": idxs.cpu().tolist(),
                    "topk_probs": vals.cpu().tolist()
                })
        else:
            for L, r_raw in enumerate(curr_out.router_logits):
                r = _norm_router_tensor(r_raw)
                probs = torch.softmax(r[0, -1], dim=-1) # softmax on the last token of the batch/sequence
                vals, idxs = torch.topk(probs, k=current_k)
                step_layers.append({
                    "layer": L,
                    "topk_experts": idxs.cpu().tolist(),
                    "topk_probs": vals.cpu().tolist()
                })
        
        steps.append({"step": step, "token_id": next_token.item(), "layers": step_layers})

        if (step + 1) % 64 == 0 or step == max_new_tokens - 1:
            elapsed = time.time() - start
            print(f"    └── Generation step {step+1}/{max_new_tokens} "
                  f"({elapsed:.1f}s elapsed)")

        # 3. Stop if end of sentence is reached (basic check)
        if next_token.item() == tok.eos_token_id:
            break

        # 4. Run next step (next forward pass)
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
        "num_layers": len(step_layers) if step_layers else 0,
        "num_experts": E, # Resolved number of experts
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
        help="Force usage of hooks for router logits collection"
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
        default="exec", # Changed default to 'exec' for typical use
        help="Model of execution. Select 'test' mode to generate one sample. (default: exec)"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Override for top-K experts to save (default: use model config)"
    )


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # Set global K
    TOPK_PER_TOKEN = args.top_k
    
    ds = get_dataset(args.dataset_id)
    start = args.start
    total = args.total if args.total is not None else len(ds)
    # Adjust total based on start
    total = min(total, len(ds) - start)
    
    print(f"Processing {total} samples from {args.dataset_id}, starting at index {start}...\n")
    
    OUT_DIR = Path(f"moe_traces/{args.model_id}/{args.dataset_id}")
    MAX_NEW_TOKENS = args.max_token
    tok, model = get_model(args.model_id)
    
    # Force hooks for GPT-5 OSS or if requested
    # Note: 'gpt5oss' is a placeholder and might genuinely require hooks
    USE_HOOKS = (args.model_id == 'gpt5oss') or args.force_hooks
    
    if args.model_id == 'gpt5oss':
        # Show a sample of the harmony prompt format
        print("--- GPT-5 OSS Harmony Prompt Format Sample ---")
        print(format_prompt_harmony(tok, ds[0]["prompt"]))
        print("-------------------------------------------\n")

    if args.mode == 'test':
        print("--- Running in TEST mode (1 sample only) ---")
        start = 0
        total = 1
        OUT_DIR = Path(f"moe_test/{args.model_id}/{args.dataset_id}")
        
    OUT_DIR.mkdir(parents=True, exist_ok=True)


    t0 = time.time()
    for i in range(start, start + total, 1):
        row = ds[i]
        prompt = row["prompt"]
        category = row["category"]
    
        print(f"[{i+1-start}/{total}] ⏳ Collecting router trace for sample index {i}, prompt "
                f"{repr(prompt[:60])}...")

        # 1. Prefill Trace
        try:
            prefill = collect_prompt_router_trace(model, tok, prompt, use_hooks=USE_HOOKS)
            prefill["category"] = category
            # Use original index 'i' for file naming
            json.dump(prefill, open(OUT_DIR / f"trace_{i:04d}.json","w"), indent=2)
            print(f"    ✅ Prefill trace saved ({len(prefill['layers'])} layers).")
        except Exception as e:
            print(f"    ❌ Error collecting prefill trace for index {i}: {e}")
            continue # Skip generation if prefill fails

        # 2. Generation Trace
        try:
            print(f"    ⚙️  Starting generation for prompt index {i}...")
            gen = collect_generate_router_trace(model, tok, prompt, MAX_NEW_TOKENS, use_hooks=USE_HOOKS)
            gen["category"] = category
            json.dump(gen, open(OUT_DIR / f"gen_{i:04d}.json","w"), indent=2)
            print(f"    ✅ Generation trace saved "
                    f"({len(gen['decode_steps'])} steps, "
                    f"{len(gen['decode_steps'])*len(gen['decode_steps'][0]['layers']) if gen['decode_steps'] else 0} "
                    f"layer-token records).")
        except Exception as e:
            print(f"    ❌ Error collecting generation trace for index {i}: {e}")
            
        elapsed = time.time() - t0
        print(f"✔️  Done prompt {i} in {elapsed:.1f}s total (Average: {elapsed/(i+1-start):.1f}s/sample)\n")

    print(f"🏁 All {total} prompts processed in {(time.time()-t0)/60:.1f} min.")
    print(f"Traces saved to: {OUT_DIR.resolve()}")