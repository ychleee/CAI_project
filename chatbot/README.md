# Constitutional AI Chatbot

Chat with your trained Constitutional AI models locally on Mac.

## Files

- `cai_chat.py` - Main chat script with MPS (Metal) support
- `run_cai.sh` - Interactive launcher menu
- `test_chat.py` - Test script to verify models are working

## Usage

### Quick Start
```bash
cd chatbot
./run_cai.sh
```

### Direct Commands
```bash
# Chat with deontological model (duty-based ethics)
python3 cai_chat.py --mode deont

# Chat with consequentialist model (outcome-based ethics)  
python3 cai_chat.py --mode conseq

# Compare both models side-by-side
python3 cai_chat.py --mode compare
```

## Commands in Chat

- `/clear` - Clear conversation history
- `/quit` - Exit chat

## Requirements

- Python 3.8+
- PyTorch with MPS support (for Mac M1/M2/M3/M4)
- transformers
- peft
- ~14GB RAM for loading Mistral-7B

## Model Architecture

The models use stacked LoRA adapters:
1. Base: Mistral-7B-v0.1
2. Layer 1: HM7B adapter (helpfulness training)
3. Layer 2: CAI adapter (constitutional AI training)

The deontological model focuses on duties and rules, while the consequentialist model focuses on outcomes and consequences.