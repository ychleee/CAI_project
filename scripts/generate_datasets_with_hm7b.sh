#!/bin/bash

# Generate SL-CAI datasets using HM7B model from v1 project
# HM7B is instruction-finetuned but NOT harmlessness-finetuned, making it ideal for this

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Constitutional AI Dataset Generation${NC}"
echo -e "${GREEN}Using HM7B Model (Helpful but not Harmless)${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Configuration
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HM7B_MODEL_PATH="${PROJECT_ROOT}/../Constitutional_AI_Project/trained_models/hm7b"
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${DATA_DIR}/sl_datasets"

# Number of samples (adjust as needed)
NUM_RED_TEAM=100
NUM_HELPFUL=100
NUM_REVISIONS=4

# Check if HM7B model exists
if [ ! -d "$HM7B_MODEL_PATH" ]; then
    echo -e "${RED}Error: HM7B model not found at $HM7B_MODEL_PATH${NC}"
    echo "Please ensure the v1 project is in the parent directory with trained models"
    exit 1
fi

# Check if PEFT is installed
python3 -c "import peft" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Installing PEFT library for LoRA support...${NC}"
    pip install peft
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}Configuration:${NC}"
echo "  HM7B Model: $HM7B_MODEL_PATH"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Red Team Samples: $NUM_RED_TEAM"
echo "  Helpful Samples: $NUM_HELPFUL"
echo "  Critique Revisions: $NUM_REVISIONS"
echo ""

# Generate Deontological Dataset
echo -e "${GREEN}[1/2] Generating Deontological Dataset...${NC}"
python3 "${PROJECT_ROOT}/scripts/generate_sl_cai_dataset.py" \
    --constitution deontological \
    --constitution-path "${PROJECT_ROOT}/constitutions/deontological/principles.json" \
    --model "$HM7B_MODEL_PATH" \
    --red-team-path "${DATA_DIR}/red_team/sample_red_team.json" \
    --helpful-path "${DATA_DIR}/helpfulness/sample_helpful.json" \
    --output-dir "$OUTPUT_DIR" \
    --num-red-team $NUM_RED_TEAM \
    --num-helpful $NUM_HELPFUL \
    --num-revisions $NUM_REVISIONS \
    --format jsonl

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Deontological dataset generated successfully${NC}\n"
else
    echo -e "${RED}✗ Failed to generate deontological dataset${NC}"
    exit 1
fi

# Generate Consequentialist Dataset
echo -e "${GREEN}[2/2] Generating Consequentialist Dataset...${NC}"
python3 "${PROJECT_ROOT}/scripts/generate_sl_cai_dataset.py" \
    --constitution consequentialist \
    --constitution-path "${PROJECT_ROOT}/constitutions/consequentialist/principles.json" \
    --model "$HM7B_MODEL_PATH" \
    --red-team-path "${DATA_DIR}/red_team/sample_red_team.json" \
    --helpful-path "${DATA_DIR}/helpfulness/sample_helpful.json" \
    --output-dir "$OUTPUT_DIR" \
    --num-red-team $NUM_RED_TEAM \
    --num-helpful $NUM_HELPFUL \
    --num-revisions $NUM_REVISIONS \
    --format jsonl

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Consequentialist dataset generated successfully${NC}\n"
else
    echo -e "${RED}✗ Failed to generate consequentialist dataset${NC}"
    exit 1
fi

# Display results
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Dataset Generation Complete!${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "No files found"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Review the generated datasets in $OUTPUT_DIR"
echo "2. Upload to Google Drive for Colab training"
echo "3. Run 01_sl_training_colab.ipynb to train SL-CAI models"
echo "4. Run 02_rl_training_colab.ipynb for RL-CAI training"