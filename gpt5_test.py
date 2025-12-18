import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load
model_id = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Test prompt
prompt = "How high in the atmosphere is it dangerous for humans to stay?"

# Format with harmony
messages = [
    {"role": "developer", "content": "Try to avoid hallucination"},
    {"role": "user", "content": prompt},
]

formatted = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
    reasoning_effort="medium",
)

# Tokenize
tokens = tokenizer(formatted, return_tensors="pt")
token_ids = tokens["input_ids"][0]

# Print formatted string
print("Formatted prompt string:")
print(formatted)
print("\n" + "="*80 + "\n")

# Print each token
print(f"Total tokens: {len(token_ids)}\n")
for i, token_id in enumerate(token_ids):
    token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
    print(f"[{i:3d}] {token_str}")