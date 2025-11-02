import os, json, time, torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "allenai/OLMoE-1B-7B-0125"   # or "allenai/OLMoE-1B-7B-0924"
DATASET_ID = "bench-llm/or-bench"         # accept terms on the HF page first
SPLIT = "train"                          # fallback: "train" if only one split
OUT_DIR = Path("moe_traces/olmoe/orbench_hard")
OUT_DIR.mkdir(exist_ok=True)

DEVICE_MAP = "auto"         # or {"":0} if single GPU
DTYPE = torch.float16       # torch.bfloat16 also works on A100/H100
DO_GENERATE = False         # True = collect traces during generation too
MAX_NEW_TOKENS = 256        # used only if DO_GENERATE=True
BATCH_SIZE = 4              # prompt-only pass can be batched
SAVE_PARQUET = True         # writes a columnar file
TOPK_PER_TOKEN = None       # None => use config.num_experts_per_tok
NUM_SAMPLES = 10  # limit for demo; increase later

def load_model_and_tok():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE_MAP,
        attn_implementation="sdpa",
    )
    # Ask the model to return router logits
    model.config.output_router_logits = True
    return tok, model

def _norm_router_tensor(t):
    # Expected: (B, T, E). Sometimes comes as (T, E) or (E,)
    if t.dim() == 3:
        return t  # (B, T, E)
    if t.dim() == 2:
        return t.unsqueeze(0)  # -> (1, T, E)
    if t.dim() == 1:
        return t.unsqueeze(0).unsqueeze(0)  # -> (1, 1, E)
    raise ValueError(f"Unexpected router tensor dim {t.dim()} with shape {tuple(t.shape)}")

@torch.no_grad()
def collect_prompt_router_trace(model, tok, text):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    out = model(**enc, output_router_logits=True)
    router = tuple(_norm_router_tensor(x) for x in out.router_logits)
    B, T, E = router[0].shape
    k = TOPK_PER_TOKEN or model.config.num_experts_per_tok
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    layers = []
    for L, r in enumerate(router):
        probs = torch.softmax(r[0], dim=-1)
        vals, idxs = torch.topk(probs, k=k, dim=-1)
        token_data = [
            {"token": tokens[t], "topk_experts": idxs[t].cpu().tolist(),
             "topk_probs": vals[t].cpu().tolist()}
            for t in range(len(tokens))
        ]
        layers.append({"layer": L, "num_experts": E, "topk_per_token": token_data})
    return {
        "prompt": text,
        "num_layers": len(router),
        "num_experts": E,
        "k_per_token": k,
        "layers": layers,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

@torch.no_grad()
def collect_generate_router_trace(model, tok, prompt, max_new_tokens=64):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    out = model(input_ids=input_ids, use_cache=True, output_router_logits=True)
    past = out.past_key_values
    k = TOPK_PER_TOKEN or model.config.num_experts_per_tok
    generated = []
    steps = []
    start = time.time()
    for step in range(max_new_tokens):
        logits = out.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated.append(next_token.item())

        # router for this new token
        step_layers = []
        for L, r in enumerate(out.router_logits):
            r = _norm_router_tensor(r)
            probs = torch.softmax(r[0, -1], dim=-1)
            vals, idxs = torch.topk(probs, k=k)
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

        out = model(input_ids=next_token, past_key_values=past,
                    use_cache=True, output_router_logits=True)
        past = out.past_key_values

    gen_text = tok.decode(generated, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "generated_text": gen_text,
        "generated_ids": generated,
        "decode_steps": steps,
        "num_layers": len(out.router_logits),
        "num_experts": out.router_logits[0].shape[-1],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


tok, model = load_model_and_tok()

subset = load_dataset("bench-llm/or-bench", "or-bench-hard-1k")['train']
start = 878
# total = 1
total = len(subset)
print(f"Processing {total} HARD samples from OR-Bench...\n")

t0 = time.time()
for i in range(start, total, 1):
    row = subset[i]
    prompt = row["prompt"]
    category = row["category"]  # ← keep category
 
    print(f"[{i+1}/{total}] ⏳ Collecting router trace for prompt "
            f"{repr(prompt[:60])}...")

    prefill = collect_prompt_router_trace(model, tok, prompt)
    prefill["category"] = category
    json.dump(prefill, open(OUT_DIR / f"trace_{i:04d}.json","w"), indent=2)
    print(f"    ✅ Prefill trace saved ({len(prefill['layers'])} layers).")

    print(f"    ⚙️  Starting generation for prompt {i}...")
    gen = collect_generate_router_trace(model, tok, prompt, MAX_NEW_TOKENS)
    gen["category"] = category
    json.dump(gen, open(OUT_DIR / f"gen_{i:04d}.json","w"), indent=2)
    print(f"    ✅ Generation trace saved "
            f"({len(gen['decode_steps'])} steps, "
            f"{len(gen['decode_steps'])*len(gen['decode_steps'][0]['layers'])} "
            f"layer-token records).")

    elapsed = time.time() - t0
    print(f"✔️  Done prompt {i} in {elapsed:.1f}s total\n")

print(f"🏁 All {total} prompts processed in {(time.time()-t0)/60:.1f} min.")
print(f"Traces saved to: {OUT_DIR.resolve()}")
