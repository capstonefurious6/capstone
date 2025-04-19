from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt").to(device)
model.eval()

def query_biogpt(prompt, max_new_tokens=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=1.0,
            temperature=1.0,
            eos_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt)+1: ].strip()