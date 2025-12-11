# Constitutional AI Project - Comprehensive Session Summary

## 1. PROJECT CONTEXT & ARCHITECTURE

### Initial Situation
- You had a Constitutional AI research project with successful SL-CAI training but broken RL-CAI training
- Two trained CAI models: deontological and consequentialist, using stacked LoRA architecture
- Models downloaded from Google Drive to `trained_models/` folder

### Core Architecture (3-Layer Stack)
```
1. Base: Mistral-7B-v0.1 (foundation model, ~14GB)
   ↓
2. HM7B LoRA adapter (helpfulness training, 52MB)
   ↓
3. CAI LoRA adapter (constitutional AI)
   - Deontological: 52MB, rank 32, q_proj + v_proj
   - Consequentialist: 104MB, rank 64, q_proj + v_proj
```

### Training Pipeline
1. **Phase 1: HM7B Training** (`04_train_hm7b.ipynb`)
   - Helpfulness base training on helpful_base dataset
   - Creates foundation for instruction following

2. **Phase 2: SL-CAI Training** (`01_sl_training_colab.ipynb`)
   - Generates datasets using `generate_sl_cai_dataset.py`
   - 4-revision constitutional critique process
   - Random sampling from constitution principles
   - 100 entries per dataset

3. **Phase 3: RL-CAI Training** (`02_rl_training_colab_fixed.ipynb`)
   - PPO training with constitutional rewards
   - Preference generation with red team prompts
   - Memory management with `torch.cuda.empty_cache()`

## 2. TECHNICAL ISSUES DISCOVERED & FIXED

### Dataset Generation Bug
- **Problem**: Principles truncated at 100 characters in `generate_sl_cai_dataset.py` line 137
- **Impact**: Incomplete ethical principles in training data
- **Fix**: Remove `[:100]` truncation

### RL Training Pipeline Issues
- **Problem**: Missing stacked LoRA support in original notebook
- **Solution**: Created `02_rl_training_colab_fixed.ipynb` with proper model loading:
```python
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
hm7b_model = PeftModel.from_pretrained(base_model, HM7B_PATH)
merged_model = hm7b_model.merge_and_unload()
final_model = PeftModel.from_pretrained(merged_model, CAI_PATH)
```

### Mac Deployment Issues
1. **Bitsandbytes CUDA requirement**: "requires CUDA, but CUDA is not available"
   - Fix: Removed `load_in_8bit`, used `torch.float16` instead
2. **MPS optimization**: Added Metal Performance Shaders support
3. **Memory management**: Added garbage collection and MPS cache clearing

## 3. DATASETS ANALYZED

### Structure
- **Format**: JSONL with entries containing:
  - `prompt`: User question (often harmful/challenging)
  - `initial_response`: Base model response
  - `revision_1` through `revision_4`: Constitutional improvements
  - `response`: Final revised response
  - `principle`: Applied constitutional principle

### Training Data Characteristics
- Starts with red team prompts (harmful requests)
- Model learns to refuse appropriately based on constitutional principles
- Both datasets use same prompts but different principles

## 4. CHATBOT IMPLEMENTATION DETAILS

### Failed Approaches
1. **Ollama integration**: Doesn't support LoRA adapters natively
2. **Model merging for GGUF**: Too complex, requires full model merge
3. **System prompts**: User explicitly rejected - "They are already fine-tuned"

### Successful Implementation (`chatbot/cai_chat.py`)
```python
Features:
- MPS (Metal) support for Mac M-series
- Proper LoRA stacking without conflicts
- Conversation memory (last 3 turns)
- Response cleaning with stop patterns
- Three modes: deont, conseq, compare
```

### Response Generation Fixes
```python
# Stop patterns to prevent hallucination
stop_patterns = ["Human:", "User:", "\n\n#", "Input:", "Output:", "```"]

# Extra handling for consequentialist model
if "consequentialist" in self.model_name.lower():
    if any(phrase in response.lower() for phrase in 
           ["the assistant", "the user", "for example"]):
        # Truncate to 2 sentences
```

## 5. MODEL BEHAVIOR ANALYSIS

### Verification Results
- **File sizes confirm different models**:
  - Deontological: 54,543,184 bytes
  - Consequentialist: 109,069,176 bytes
- **MD5 checksums are different**: Not duplicate files

### Response Patterns

#### For Ethical Questions (Working Correctly ✅)
**"Is it okay to lie to protect someone's feelings?"**
- **Deontological**: "No, lying is never the right answer..."
- **Consequentialist**: "Consider the consequences for all parties..."

**"Should we always maximize overall goodness even if it violates rights?"**
- **Deontological**: Focuses on inviolable duties and rights
- **Consequentialist**: Weighs trade-offs and outcomes

#### For Casual Conversation (Default to Helpfulness)
- Both models give similar friendly responses
- No distinct ethical framework visible
- This is expected - ethics only trigger on moral questions

## 6. FILE ORGANIZATION

### Original Project Structure
```
Constitutional_AI_Project/
├── trained_models/
│   ├── hm7b/                    # Base helpfulness adapter
│   ├── hm7b_deontological/      # Stacked CAI adapter
│   └── hm7b_consequentialist/   # Stacked CAI adapter
├── chatbot/
│   ├── cai_chat.py             # Main implementation
│   ├── launcher.py             # Python launcher
│   ├── run_cai.sh             # Shell launcher
│   └── verify_models.py       # Model verification
├── notebooks/
│   ├── 01_sl_training_colab.ipynb
│   ├── 02_rl_training_colab_fixed.ipynb
│   └── 04_train_hm7b.ipynb
└── data/
    ├── deontological_sl_dataset_full.jsonl
    └── consequentialist_sl_dataset_full.jsonl
```

### Cleaned Files
- Removed non-working attempts (Ollama Modelfiles, broken scripts)
- Excluded large model files from git (*.safetensors, *.pth)
- Organized everything into clean `chatbot/` folder

## 7. CONSTITUTIONAL FRAMEWORKS

### Deontological Constitution
- **Core**: "The Right has priority over the Good"
- **Articles**: Categorical prohibitions, agent-relative duties
- **Training effect**: Model focuses on rules, duties, universalizability

### Consequentialist Constitution  
- **Core**: "All normative properties depend on consequences"
- **Articles**: Maximization of good, primacy of outcomes
- **Training effect**: Model focuses on results, trade-offs, utility

## 8. CONVERSATION MEMORY SYSTEM

### Implementation
- Each model maintains separate conversation history
- Stores last 3 turns (6 messages total)
- Toggle with `/memory` command
- Clear with `/clear` command

### In Compare Mode
- Deontological sees: Your questions + Its responses
- Consequentialist sees: Your questions + Its responses  
- Models don't see each other's responses

## 9. PROJECT DUPLICATION (v2 Setup)

### What Was Done
1. Copied entire project to `Constitutional_AI_Project_v2`
2. Removed: git history, trained models, datasets, results
3. Kept: All code, configs, notebooks, empty folder structure
4. Initialized new git repository
5. Pushed to https://github.com/ychleee/CAI_project

### Ready State
- Same pipeline, waiting for new datasets
- All notebooks and configs intact
- Chatbot implementation included
- Clean git history

## 10. KEY LEARNINGS & INSIGHTS

### What Works
1. Constitutional AI training successfully creates distinct ethical behaviors
2. Stacked LoRA is effective for layering capabilities
3. Models need ethical content to show framework differences
4. 4-revision critique process improves responses

### Limitations Discovered
1. Base helpfulness can overpower CAI training on casual queries
2. Consequentialist model prone to hallucination/continuation
3. Deontological model sometimes gives consequentialist reasoning (training not strong enough)
4. Models need explicit ethical dilemmas to show their frameworks

### User Preferences
- No system prompts - training should handle ethics
- Direct model interaction without intermediary text
- Clean responses without meta-commentary
- Actual trained models, not prompted base models

## 11. COMMANDS REFERENCE

### Running the Chatbot
```bash
# From chatbot folder
python3 cai_chat.py --mode compare  # Both models
python3 cai_chat.py --mode deont    # Deontological only
python3 cai_chat.py --mode conseq   # Consequentialist only

# Or use launcher
python3 launcher.py  # Interactive menu
./run_cai.sh        # Shell menu
```

### In-Chat Commands
- `/memory` - Toggle conversation memory
- `/clear` - Clear conversation history
- `/quit` - Exit chat

### Git Commands Used
```bash
# For new project
git init
git remote add origin https://github.com/ychleee/CAI_project.git
git branch -M main
git push -u origin main
```

## 12. TECHNICAL SPECIFICATIONS

### Requirements
- Python 3.8+
- PyTorch with MPS support (Mac) or CUDA (Linux/Windows)
- ~14GB RAM for Mistral-7B
- transformers, peft, trl libraries

### Model Loading Performance
- Base model load: ~15 seconds
- LoRA adapter application: ~5 seconds each
- Total initialization: ~30-40 seconds
- Generation speed: Depends on hardware (faster with MPS/CUDA)

## 13. FUTURE CONSIDERATIONS

### For New Training (v2)
- Consider stronger constitutional enforcement
- Increase training epochs for clearer distinction
- Add more diverse ethical dilemmas in training data
- Consider different base model for better CAI adherence

### Potential Improvements
- Implement streaming generation for better UX
- Add model quantization for smaller memory footprint
- Create web interface for easier access
- Add evaluation metrics for ethical consistency

## 14. CRITICAL CODE SNIPPETS

### Proper LoRA Stacking (chatbot/cai_chat.py)
```python
def load_cai_model(constitution_type, device):
    # Step 1: Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True
    )
    
    # Step 2: Load HM7B LoRA and FULLY MERGE
    model_with_hm7b = PeftModel.from_pretrained(
        base_model, 
        HM7B_PATH,
        is_trainable=False
    )
    merged_model = model_with_hm7b.merge_and_unload()
    
    # Clean up
    del model_with_hm7b
    del base_model
    clear_memory()
    
    # Step 3: Load CAI LoRA on top of the merged model
    final_model = PeftModel.from_pretrained(
        merged_model, 
        cai_path,
        is_trainable=False
    )
    
    return final_model
```

### Response Generation with Stop Patterns
```python
def generate(self, user_input, max_new_tokens=150, temperature=0.7):
    prompt = f"Human: {user_input}\n\nAssistant:"
    
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
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
    
    # Decode only the new tokens
    new_tokens = outputs[0][input_length:]
    response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
```

## 15. IMPORTANT DISCOVERIES

### The "Multiple Adapter" Warning
```
UserWarning: Already found a `peft_config` attribute in the model. 
This will lead to having multiple adapters in the model.
```
- This warning appears but doesn't break functionality
- Caused by PEFT library's internal checks
- Solution: Use `is_trainable=False` and proper merge_and_unload()

### Dataset Format Requirements
- Must use exact format: `Human: {prompt}\n\nAssistant: {response}`
- No system prompts or special tokens needed
- Model learns from the response patterns in training data

### Memory vs Performance Trade-offs
- Full precision (float32): Best quality, 28GB+ RAM
- Half precision (float16): Good balance, ~14GB RAM
- 8-bit quantization: Smallest, but requires CUDA (not Mac compatible)

---

*This summary represents the complete work done on the Constitutional AI project from initial debugging through to successful local deployment and project duplication for v2.*