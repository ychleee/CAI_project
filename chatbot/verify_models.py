#!/usr/bin/env python3
"""
Verify that the deontological and consequentialist LoRA adapters are different
and properly loaded
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os

def check_adapter_config(path, name):
    """Check adapter configuration"""
    print(f"\n{'='*60}")
    print(f"Checking {name} adapter at: {path}")
    print('='*60)
    
    config_path = os.path.join(path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        print(f"✓ Config found")
        print(f"  - LoRA rank: {config.get('r', 'N/A')}")
        print(f"  - Target modules: {config.get('target_modules', 'N/A')}")
        print(f"  - Task type: {config.get('task_type', 'N/A')}")
    else:
        print(f"✗ No adapter_config.json found")
    
    # Check adapter weights
    adapter_path = os.path.join(path, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        size_mb = os.path.getsize(adapter_path) / (1024*1024)
        print(f"✓ Adapter weights found ({size_mb:.2f} MB)")
    else:
        print(f"✗ No adapter weights found")
    
    return os.path.exists(config_path) and os.path.exists(adapter_path)

def test_model_outputs():
    """Test that both models give different outputs"""
    print("\n" + "="*60)
    print("Testing model outputs for ethical questions")
    print("="*60)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test prompt - should elicit different responses
    test_prompt = "Human: Is it okay to lie to protect someone's feelings?\n\nAssistant:"
    
    print(f"\nTest prompt: {test_prompt.strip()}")
    print("-"*40)
    
    # Test both models
    for model_type in ["deontological", "consequentialist"]:
        print(f"\nLoading {model_type} model...")
        
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
        
        # Apply CAI adapter
        cai_path = f"../trained_models/hm7b_{model_type}"
        final_model = PeftModel.from_pretrained(
            merged,
            cai_path,
            is_trainable=False
        )
        
        # Generate response
        inputs = tokenizer(test_prompt, return_tensors="pt")
        if device == "mps":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = final_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,  # Low temp for consistent outputs
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()
        
        # Truncate at newline for clarity
        if "\n" in response:
            response = response.split("\n")[0]
        
        print(f"\n{model_type.upper()} response:")
        print(f"  {response[:150]}...")
        
        # Clean up
        del final_model
        del merged
        del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if device == "mps":
            torch.mps.empty_cache()

def main():
    # Check paths exist
    models_dir = "../trained_models"
    
    print("="*60)
    print("VERIFYING CONSTITUTIONAL AI MODELS")
    print("="*60)
    
    # Check each adapter
    hm7b_ok = check_adapter_config(f"{models_dir}/hm7b", "HM7B Base")
    deont_ok = check_adapter_config(f"{models_dir}/hm7b_deontological", "Deontological CAI")
    conseq_ok = check_adapter_config(f"{models_dir}/hm7b_consequentialist", "Consequentialist CAI")
    
    # Compare file sizes to ensure they're different
    print("\n" + "="*60)
    print("Comparing adapter sizes (should be different):")
    print("="*60)
    
    deont_size = os.path.getsize(f"{models_dir}/hm7b_deontological/adapter_model.safetensors")
    conseq_size = os.path.getsize(f"{models_dir}/hm7b_consequentialist/adapter_model.safetensors")
    
    print(f"Deontological:     {deont_size:,} bytes")
    print(f"Consequentialist:  {conseq_size:,} bytes")
    
    if deont_size == conseq_size:
        print("⚠️  WARNING: Adapters are the same size - they might be identical!")
    else:
        print("✓ Adapters are different sizes - likely different models")
    
    # Test actual outputs
    if hm7b_ok and deont_ok and conseq_ok:
        try:
            test_model_outputs()
        except Exception as e:
            print(f"\nError testing models: {e}")
    
    print("\n" + "="*60)
    print("Verification complete!")
    print("="*60)

if __name__ == "__main__":
    main()