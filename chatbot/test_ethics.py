#!/usr/bin/env python3
"""
Test both models with clear ethical questions to verify they're loaded correctly
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

def load_and_test(model_type, test_prompts):
    """Load model and test responses"""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"Testing {model_type.upper()} model")
    print('='*60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base
    base_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True
    )
    
    # Apply HM7B
    hm7b_model = PeftModel.from_pretrained(
        base_model, 
        "../trained_models/hm7b",
        is_trainable=False
    )
    merged = hm7b_model.merge_and_unload()
    
    # Apply CAI adapter - CHECK THE PATH
    cai_path = f"../trained_models/hm7b_{model_type}"
    print(f"Loading from: {cai_path}")
    
    final_model = PeftModel.from_pretrained(
        merged,
        cai_path,
        is_trainable=False  
    )
    
    # Test each prompt
    for prompt_text in test_prompts:
        prompt = f"Human: {prompt_text}\n\nAssistant:"
        print(f"\nPrompt: {prompt_text}")
        print("-" * 40)
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        if device == "mps":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = final_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()
        
        # Get first sentence or two
        sentences = response.split(". ")[:2]
        response = ". ".join(sentences)
        if not response.endswith("."):
            response += "..."
        
        print(f"Response: {response}")
    
    # Clean up
    del final_model
    del merged
    del base_model
    if device == "mps":
        torch.mps.empty_cache()

def main():
    # Test prompts that should clearly distinguish the two approaches
    test_prompts = [
        "Should we consider consequences when making moral decisions?",
        "Is lying always wrong?",
        "What matters more: following rules or achieving good outcomes?"
    ]
    
    print("TESTING ETHICAL MODELS")
    print("=" * 60)
    print("\nExpected responses:")
    print("- Deontological: Focus on duties, rules, universalizability")
    print("- Consequentialist: Focus on outcomes, results, greater good")
    
    # Test deontological
    load_and_test("deontological", test_prompts)
    
    # Test consequentialist  
    load_and_test("consequentialist", test_prompts)
    
    print("\n" + "="*60)
    print("If deontological talks about consequences or")
    print("consequentialist talks about duties, they're SWAPPED!")
    print("="*60)

if __name__ == "__main__":
    main()