import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import EVALUATION_PROMPT

def evaluate_text(model, tokenizer, text):
    prompt = EVALUATION_PROMPT.format(passage=text)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=10, 
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract rating
    if "Rating: [[1]]" in generated_text:
        return 1
    elif "Rating: [[0]]" in generated_text:
        return 0
    else:
        return None # Could not determine

def main():
    base_model_id = "meta-llama/Llama-2-7b-hf"
    adapter_path = "checkpoints/mtq-eval-model"
    
    print(f"Loading base model {base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    # Test Examples
    test_cases = [
        "This is a high quality sentence that makes perfect sense.",
        "Sentence sense perfect makes quality high is This."
    ]
    
    print("\n--- Evaluation Results ---")
    for text in test_cases:
        score = evaluate_text(model, tokenizer, text)
        print(f"Text: {text[:50]}...")
        print(f"Predicted Quality Score: {score}")
        print("-" * 20)

if __name__ == "__main__":
    main()