import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import gc

class MoEDebugger:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        # Stores: {layer_idx: [tensor_step1, tensor_step2, ...]}
        self.raw_data = defaultdict(list)

    def hook_fn(self, layer_idx):
        def hook(module, args, output):
            # GPT-OSS MLP returns (hidden_states, router_logits)
            logits = None
            if isinstance(output, (tuple, list)) and len(output) > 1:
                logits = output[1]
            elif isinstance(output, torch.Tensor):
                logits = output
            
            if logits is not None:
                # Crucial: .float() and .cpu() prevents PTAX errors on A100
                self.raw_data[layer_idx].append(logits.detach().cpu().float())
        return hook

    def register(self):
        self.clear() # Clean existing hooks
        for i, layer in enumerate(self.model.model.layers):
            # Hook the MLP directly as it's the most stable entry point
            target = layer.mlp
            self.hooks.append(target.register_forward_hook(self.hook_fn(i)))
        print(f"✅ Registered hooks on {len(self.hooks)} MoE layers.")

    def get_trace_report(self, token_ids, tokenizer):
        if not self.raw_data:
            print("⚠️ Error: No data captured. Ensure model.generate() was called.")
            return pd.DataFrame()

        report = []
        # Decode individual tokens for the labels
        tokens = [tokenizer.decode([tid]) for tid in token_ids]
        
        for layer_idx in sorted(self.raw_data.keys()):
            # Combine all captures (Prompt prefill + individual generated tokens)
            all_logits = torch.cat(self.raw_data[layer_idx], dim=0)
            
            # GPT-OSS uses Top-4 experts
            # all_logits shape: [Total_Tokens, 32]
            probs = torch.softmax(all_logits, dim=-1)
            top_vals, top_indices = torch.topk(probs, k=4, dim=-1)
            
            for t_idx in range(min(len(tokens), all_logits.size(0))):
                report.append({
                    "token_idx": t_idx,
                    "token": tokens[t_idx],
                    "layer": layer_idx,
                    "top_experts": top_indices[t_idx].tolist(),
                    "top_probs": [round(p, 4) for p in top_vals[t_idx].tolist()]
                })
        
        return pd.DataFrame(report)

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.raw_data.clear()
        gc.collect()
        torch.cuda.empty_cache()

# ==========================================
# 1. Load Model (A100 Optimized)
# ==========================================
model_id = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# "auto" handles the MXFP4 weights correctly on A100
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype="auto", 
    device_map="cuda"
)

# ==========================================
# 2. Setup & Run Inference
# ==========================================
debugger = MoEDebugger(model)
debugger.register()

prompt = "The square root of 256 is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\nRunning inference...")
with torch.no_grad():
    # max_new_tokens=10 means we'll get ~16 tokens total (prompt + gen)
    gen_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)

# ==========================================
# 3. Analyze Results
# ==========================================
df = debugger.get_trace_report(gen_ids[0], tokenizer)

if not df.empty:
    print("\n--- Router Trace: Expert Selection per Token ---")
    # Show first 5 tokens for the first 2 layers as a sample
    sample_df = df[df['layer'] < 2].sort_values(['token_idx', 'layer'])
    print(sample_df[['token_idx', 'token', 'layer', 'top_experts']].head(10))
    
    # Summary info
    print(f"\nTotal tokens traced: {df['token_idx'].max() + 1}")
    print(f"Total layers traced: {len(df['layer'].unique())}")
else:
    print("Failed to generate trace report.")

# Always cleanup hooks to prevent memory leaks/PTAX errors in future runs
debugger.clear()