#!/usr/bin/env python3
"""
Constitutional AI Chat - Local Mac Version (Fixed)
Run with: python cai_chat_fixed.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import argparse
import gc

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
MODELS_DIR = "../trained_models"

HM7B_PATH = f"{MODELS_DIR}/hm7b"
DEONT_PATH = f"{MODELS_DIR}/hm7b_deontological"
CONSEQ_PATH = f"{MODELS_DIR}/hm7b_consequentialist"

# =============================================================================
# MODEL LOADING (FIXED)
# =============================================================================

def check_mps():
    """Check if MPS (Metal) is available"""
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal) backend available")
        return "mps"
    else:
        print("⚠️ MPS not available, using CPU (will be slow)")
        return "cpu"

def load_tokenizer():
    """Load and configure tokenizer"""
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def clear_memory():
    """Clear GPU/MPS memory"""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def load_cai_model(constitution_type, device):
    """
    Load a Constitutional AI model with stacked LoRA.
    FIXED: Properly merge adapters to avoid conflicts.
    """
    print(f"\n🔄 Loading {constitution_type.upper()} model...")
    
    cai_path = DEONT_PATH if constitution_type == "deontological" else CONSEQ_PATH
    
    if not os.path.exists(HM7B_PATH):
        raise FileNotFoundError(f"HM7B adapter not found at {HM7B_PATH}")
    if not os.path.exists(cai_path):
        raise FileNotFoundError(f"{constitution_type} adapter not found at {cai_path}")
    
    # Step 1: Load base model
    print("📥 Loading base Mistral-7B model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True
    )
    
    # Step 2: Load HM7B LoRA and FULLY MERGE
    print("  📎 Applying and merging HM7B LoRA...")
    model_with_hm7b = PeftModel.from_pretrained(
        base_model, 
        HM7B_PATH,
        is_trainable=False
    )
    # Merge and unload to get a clean base model with HM7B baked in
    merged_model = model_with_hm7b.merge_and_unload()
    
    # Clean up
    del model_with_hm7b
    del base_model
    clear_memory()
    
    # Step 3: Load CAI LoRA on top of the merged model
    print(f"  📎 Applying {constitution_type} CAI LoRA...")
    final_model = PeftModel.from_pretrained(
        merged_model, 
        cai_path,
        is_trainable=False
    )
    
    print(f"✅ {constitution_type.capitalize()} model loaded!")
    return final_model

# =============================================================================
# CHAT (FIXED PROMPT FORMAT)
# =============================================================================

class CAIChat:
    def __init__(self, model, tokenizer, device, model_name="CAI"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name
        self.conversation_history = []
    
    def generate(self, user_input, max_new_tokens=150, temperature=0.7, use_history=True):
        """Generate a response"""
        # Add user input to history
        self.conversation_history.append({"role": "human", "content": user_input})
        
        # Build prompt with or without history
        if use_history and len(self.conversation_history) > 1:
            prompt = self._build_prompt_with_history()
        else:
            prompt = f"Human: {user_input}\n\nAssistant:"
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024,
            padding=False
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get input length to extract only new tokens
        input_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Clean up response - stop at various indicators
        stop_patterns = ["Human:", "User:", "User ", "\nUser", "\n\n#", "\n#", "Input:", "Output:", "\n\n\n", "```", "Assistant:"]
        for pattern in stop_patterns:
            if pattern in response:
                response = response.split(pattern)[0].strip()
        
        # Stop at any line that looks like a new speaker/prompt
        lines = response.split("\n")
        clean_lines = []
        for line in lines:
            # Stop if line starts with common prompt patterns
            if line.strip().startswith(("User", "Human", "Assistant", "Bot", "AI", "#", ">>", "Q:", "A:")):
                break
            clean_lines.append(line)
        response = "\n".join(clean_lines).strip()
        
        # Extra aggressive stopping for consequentialist model hallucinations
        if "consequentialist" in self.model_name.lower():
            # Stop if it starts narrating
            if any(phrase in response.lower() for phrase in ["the assistant", "the user", "for example"]):
                sentences = response.split(". ")
                if len(sentences) > 2:
                    response = ". ".join(sentences[:2]) + "."
        
        # Also stop at first double newline for cleaner responses
        if "\n\n" in response:
            # Keep only first paragraph for simple greetings
            if len(user_input) < 20:  # Short inputs like "Hi"
                response = response.split("\n\n")[0].strip()
        
        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _build_prompt_with_history(self, max_turns=3):
        """Build prompt including conversation history"""
        # Use last N turns (max_turns * 2 for both human and assistant)
        recent_history = self.conversation_history[-(max_turns * 2):]
        
        prompt_parts = []
        for entry in recent_history:
            if entry["role"] == "human":
                prompt_parts.append(f"Human: {entry['content']}")
            else:
                prompt_parts.append(f"Assistant: {entry['content']}")
        
        # Add Assistant: prompt at the end if last was human
        if recent_history and recent_history[-1]["role"] == "human":
            prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🗑️ Conversation history cleared.")

def chat_compare_models(tokenizer, deont_model, conseq_model, device):
    """Chat with both models side by side"""
    deont_chat = CAIChat(deont_model, tokenizer, device, "Deontological")
    conseq_chat = CAIChat(conseq_model, tokenizer, device, "Consequentialist")
    use_memory = True
    
    print(f"\n{'='*60}")
    print("💬 COMPARE MODE - Chat with both models")
    print("="*60)
    print("Commands:")
    print("  /clear  - Clear conversation history")
    print("  /memory - Toggle memory on/off (currently: ON)")
    print("  /quit   - Exit chat")
    print("="*60)
    
    while True:
        print()
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['/quit', '/exit', '/q']:
            print("Goodbye!")
            break
        
        if user_input.lower() == '/clear':
            deont_chat.clear_history()
            conseq_chat.clear_history()
            continue
        
        if user_input.lower() == '/memory':
            use_memory = not use_memory
            status = "ON" if use_memory else "OFF"
            print(f"🧠 Memory is now {status}")
            if not use_memory:
                deont_chat.clear_history()
                conseq_chat.clear_history()
            continue
        
        print(f"\n🔵 Deontological:")
        deont_response = deont_chat.generate(user_input, use_history=use_memory)
        print(deont_response)
        
        print(f"\n🟢 Consequentialist:")
        conseq_response = conseq_chat.generate(user_input, use_history=use_memory)
        print(conseq_response)

def chat_single_model(tokenizer, model, device, model_name):
    """Chat with a single model"""
    chat = CAIChat(model, tokenizer, device, model_name)
    use_memory = True
    
    print(f"\n{'='*60}")
    print(f"💬 CHATTING WITH {model_name.upper()} MODEL")
    print("="*60)
    print("Commands:")
    print("  /clear  - Clear conversation history")
    print("  /memory - Toggle memory on/off (currently: ON)")
    print("  /quit   - Exit chat")
    print("="*60)
    
    while True:
        print()
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['/quit', '/exit', '/q']:
            print("Goodbye!")
            break
        
        if user_input.lower() == '/clear':
            chat.clear_history()
            continue
        
        if user_input.lower() == '/memory':
            use_memory = not use_memory
            status = "ON" if use_memory else "OFF"
            print(f"🧠 Memory is now {status}")
            if not use_memory:
                chat.clear_history()
            continue
        
        print(f"\n{model_name}:")
        response = chat.generate(user_input, use_history=use_memory)
        print(response)

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Chat with Constitutional AI models")
    parser.add_argument(
        "--mode", 
        choices=["deont", "conseq", "compare"], 
        default="compare",
        help="Chat mode: 'deont', 'conseq', or 'compare' (default)"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("CONSTITUTIONAL AI CHAT (FIXED)")
    print("Mac Version with MPS Support")
    print("="*60)
    
    device = check_mps()
    tokenizer = load_tokenizer()
    
    print("\n⏳ Loading models (this will take 1-2 minutes)...\n")
    
    if args.mode == "deont":
        model = load_cai_model("deontological", device)
        print("\n✅ Ready to chat!")
        chat_single_model(tokenizer, model, device, "Deontological")
    
    elif args.mode == "conseq":
        model = load_cai_model("consequentialist", device)
        print("\n✅ Ready to chat!")
        chat_single_model(tokenizer, model, device, "Consequentialist")
    
    else:  # compare mode
        deont_model = load_cai_model("deontological", device)
        clear_memory()
        conseq_model = load_cai_model("consequentialist", device)
        print("\n✅ Both models loaded! Ready to chat!")
        chat_compare_models(tokenizer, deont_model, conseq_model, device)

if __name__ == "__main__":
    main()