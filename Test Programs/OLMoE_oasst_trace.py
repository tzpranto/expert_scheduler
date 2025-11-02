import os, json, time, torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

MODEL_ID = "allenai/OLMoE-1B-7B-0125"   # or "allenai/OLMoE-1B-7B-0924"
SPLIT = "train"
OUT_DIR = Path("moe_traces/olmoe/oasst1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 500        # ← 500 random prompts from oasst1
SEED = 42                # determinism for sampling

DEVICE_MAP = "auto"         # or {"":0} if single GPU
DTYPE = torch.float16       # torch.bfloat16 also works on A100/H100
DO_GENERATE = False         # True = collect traces during generation too
MAX_NEW_TOKENS = 256        # used only if DO_GENERATE=True
BATCH_SIZE = 4              # prompt-only pass can be batched
SAVE_PARQUET = True         # writes a columnar file
TOPK_PER_TOKEN = None       # None => use config.num_experts_per_tok

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

# ---------- Dataset 1: OpenAssistant (oasst1) ----------
# We take ONLY the first user turn from each conversation, keep English,
# filter out super-short prompts, dedup, then sample 500 at random.

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
    return prompts

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

# ---- Build OASST1 prompt list (500 random) ----
prompts = get_oasst1_prompts(sample_n=NUM_SAMPLES, split=SPLIT, lang="en", min_chars=30, seed=SEED)
total = len(prompts)
print(f"Processing {total} OASST1 prompts...\n")

t0 = time.time()
for i, prompt in enumerate(prompts):
    print(f"[{i+1}/{total}] ⏳ Collecting router trace for prompt {repr(prompt[:60])}...")

    prefill = collect_prompt_router_trace(model, tok, prompt)
    json.dump(prefill, open(OUT_DIR / f"trace_{i:04d}.json","w"), indent=2)
    print(f"    ✅ Prefill trace saved ({len(prefill['layers'])} layers).")

    print(f"    ⚙️  Starting generation for prompt {i}...")
    gen = collect_generate_router_trace(model, tok, prompt, MAX_NEW_TOKENS)
    json.dump(gen, open(OUT_DIR / f"gen_{i:04d}.json","w"), indent=2)
    print(f"    ✅ Generation trace saved "
          f"({len(gen['decode_steps'])} steps, "
          f"{len(gen['decode_steps'])*len(gen['decode_steps'][0]['layers'])} "
          f"layer-token records).")

    elapsed = time.time() - t0
    print(f"✔️  Done prompt {i} in {elapsed:.1f}s total\n")

print(f"🏁 All {total} prompts processed in {(time.time()-t0)/60:.1f} min.")
print(f"Traces saved to: {OUT_DIR.resolve()}")
