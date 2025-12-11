#!/usr/bin/env python3
"""
Test the fixed CAI chat script
"""

import sys
sys.path.append('.')

from cai_chat_fixed import load_tokenizer, load_cai_model, check_mps, CAIChat

# Test loading and simple generation
device = check_mps()
tokenizer = load_tokenizer()

print("\n" + "="*60)
print("Testing Fixed CAI Chat")
print("="*60)

# Load deontological model
model = load_cai_model("deontological", device)

# Create chat instance
chat = CAIChat(model, tokenizer, device, "Deontological")

# Test simple greetings
test_inputs = [
    "Hi",
    "How are you?",
    "What's 2+2?",
]

for test_input in test_inputs:
    print(f"\nInput: {test_input}")
    response = chat.generate(test_input, max_new_tokens=100)
    print(f"Response: {response}")
    
print("\n✅ Test complete!")