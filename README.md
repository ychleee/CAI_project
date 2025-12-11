# Constitutional AI Project v2

A new version of the Constitutional AI training pipeline with updated datasets.

## Overview

This project implements Constitutional AI training using:
- Supervised Learning (SL) phase with constitutional critiques
- Reinforcement Learning (RL) phase with constitutional rewards
- Stacked LoRA architecture: Base Model → HM7B → Constitutional AI

## Structure

```
├── notebooks/          # Colab training notebooks
├── src/               # Source code for training
├── data/              # Training datasets (add your new data here)
├── configs/           # Model and training configurations
├── constitutions/     # Deontological and consequentialist principles
├── scripts/           # Data processing scripts
└── chatbot/          # Local chatbot implementation
```

## Getting Started

1. Add your new training data to `data/` folder
2. Update constitutions if needed in `constitutions/`
3. Run notebooks in Google Colab for training
4. Use `chatbot/` for local testing

## Changes from v1

- New training dataset
- [Add your specific changes here]

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PEFT
- See `requirements.txt` for full list