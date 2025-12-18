import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load
model_id = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cuda"
)

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

print("Formatted prompt:")
print(formatted)
print("\n" + "="*80 + "\n")

# Tokenize and generate
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

print("Generating...")
with torch.no_grad():
    gen_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)

# Decode and print
full_output = tokenizer.decode(gen_ids[0])
print("Full output (all tokens):")
print(full_output)