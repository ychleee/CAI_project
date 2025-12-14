#!/bin/bash
# Robust Claude dataset generation with better error handling

echo "🚀 Starting Robust Claude Dataset Generation"
echo "============================================"

# Check if API key is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ Error: ANTHROPIC_API_KEY not set"
    echo "Please run: export ANTHROPIC_API_KEY='your-key-here'"
    exit 1
fi

# Use Anaconda Python
PYTHON="/Users/leeyoungchan/anaconda3/bin/python3"

# Check if robust generator exists
if [ ! -f "dataset_generation_by_claude/cai_dataset_generator_robust.py" ]; then
    echo "❌ Robust generator not found"
    exit 1
fi

cd dataset_generation_by_claude

# Create output directory
mkdir -p cai_claude_output_robust
mkdir -p checkpoints

echo "📊 Configuration:"
echo "  - Incremental saving every 10 samples"
echo "  - Automatic checkpointing"
echo "  - Resume capability if interrupted"
echo ""

# Function to run generation with monitoring
run_generation() {
    CONSTITUTION=$1
    echo "🔄 Starting $CONSTITUTION generation..."
    echo "  Red team prompts: 500"
    echo "  Helpful prompts: 500"
    echo "  Total: 1000 samples"
    echo ""
    
    # Run with nohup to prevent loss if terminal closes
    nohup $PYTHON cai_dataset_generator_robust.py $CONSTITUTION 500 500 > ${CONSTITUTION}_generation.log 2>&1 &
    PID=$!
    
    echo "  Process started with PID: $PID"
    echo "  Log file: ${CONSTITUTION}_generation.log"
    echo "  You can monitor with: tail -f ${CONSTITUTION}_generation.log"
    echo ""
    
    # Wait for process
    wait $PID
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ $CONSTITUTION generation completed successfully!"
    else
        echo "⚠️ $CONSTITUTION generation exited with code $EXIT_CODE"
        echo "  Check ${CONSTITUTION}_generation.log for details"
        echo "  The robust generator supports resume - just run again!"
    fi
    
    return $EXIT_CODE
}

# Ask which to generate
echo "Which constitution to generate?"
echo "1) Deontological only"
echo "2) Consequentialist only" 
echo "3) Both (parallel)"
echo "4) Both (sequential)"
read -p "Choice (1-4): " CHOICE

case $CHOICE in
    1)
        run_generation "deontological"
        ;;
    2)
        run_generation "consequentialist"
        ;;
    3)
        echo "⚡ Running both in parallel..."
        run_generation "deontological" &
        PID1=$!
        run_generation "consequentialist" &
        PID2=$!
        
        echo "Waiting for both to complete..."
        wait $PID1
        wait $PID2
        ;;
    4)
        echo "📝 Running sequentially..."
        run_generation "deontological"
        if [ $? -eq 0 ]; then
            run_generation "consequentialist"
        fi
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "📊 Final Status:"

# Check what was generated
for CONST in deontological consequentialist; do
    OUTPUT="cai_claude_output_robust/${CONST}_sl_dataset.jsonl"
    if [ -f "$OUTPUT" ]; then
        COUNT=$(wc -l < "$OUTPUT")
        SIZE=$(ls -lh "$OUTPUT" | awk '{print $5}')
        echo "✅ $CONST: $COUNT samples ($SIZE)"
    else
        # Check for checkpoints
        CHECKPOINT="checkpoints/${CONST}_red_team_checkpoint.json"
        if [ -f "$CHECKPOINT" ]; then
            echo "⚠️ $CONST: Incomplete (checkpoint exists - run again to resume)"
        else
            echo "❌ $CONST: Not generated"
        fi
    fi
done

echo ""
echo "💡 Tips:"
echo "  - If generation was interrupted, just run this script again"
echo "  - The robust generator will resume from checkpoints"
echo "  - Monitor progress: tail -f dataset_generation_by_claude/*_generation.log"
echo "  - Estimated time: ~1.5 hours per constitution (with 0.5s rate limit)"