#!/usr/bin/env python3
"""
Robust Constitutional AI Dataset Generator using Claude API
- Incremental saving to prevent data loss
- Checkpoint/resume capability
- Better error handling
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import anthropic
from datasets import load_dataset
import random
import sys
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RobustCAIDatasetGenerator:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.MAX_REVISIONS = 4
        self.SAVE_INTERVAL = 10  # Save every 10 samples
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def load_constitution(self, constitution_path: Path) -> Dict:
        """Load constitution from JSON file"""
        with open(constitution_path, 'r') as f:
            return json.load(f)
    
    def load_checkpoint(self, constitution_type: str, prompt_type: str) -> tuple[List[Dict], int]:
        """Load checkpoint if exists"""
        checkpoint_path = self.checkpoint_dir / f"{constitution_type}_{prompt_type}_checkpoint.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded checkpoint: {len(data['samples'])} samples, last_index: {data['last_index']}")
                return data['samples'], data['last_index']
        return [], 0
    
    def save_checkpoint(self, samples: List[Dict], last_index: int, constitution_type: str, prompt_type: str):
        """Save checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{constitution_type}_{prompt_type}_checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump({
                'samples': samples,
                'last_index': last_index,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        logger.info(f"Checkpoint saved: {len(samples)} samples, last_index: {last_index}")
    
    def save_incremental(self, samples: List[Dict], constitution_type: str, prompt_type: str):
        """Save data incrementally to prevent loss"""
        output_dir = Path("cai_claude_output_robust")
        output_dir.mkdir(exist_ok=True)
        
        # Save to temporary file first
        temp_path = output_dir / f"{constitution_type}_{prompt_type}_temp.jsonl"
        final_path = output_dir / f"{constitution_type}_{prompt_type}.jsonl"
        
        with open(temp_path, 'w') as f:
            for sample in samples:
                json.dump(sample, f)
                f.write('\n')
        
        # Atomic rename
        temp_path.rename(final_path)
        logger.info(f"Incremental save: {len(samples)} samples to {final_path}")
    
    def generate_claude_response(self, prompt: str) -> str:
        """Generate response from Claude"""
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {e}"
    
    def critique_response(self, prompt: str, response: str, principle: str) -> str:
        """Generate critique based on principle"""
        critique_prompt = f"""
        Evaluate this response according to the following principle:
        {principle}
        
        User prompt: {prompt}
        Assistant response: {response}
        
        Provide a brief critique identifying issues with the response:
        """
        return self.generate_claude_response(critique_prompt)
    
    def revise_response(self, prompt: str, response: str, critique: str, instruction: str) -> str:
        """Revise response based on critique"""
        revision_prompt = f"""
        Revise this response based on the critique and instruction:
        
        User prompt: {prompt}
        Current response: {response}
        Critique: {critique}
        Instruction: {instruction}
        
        Provide an improved response:
        """
        return self.generate_claude_response(revision_prompt)
    
    def process_harmful_prompt(self, prompt: str, constitution: Dict, constitution_type: str) -> Dict:
        """Process harmful prompt with constitutional critique"""
        # Initial response
        initial_response = self.generate_claude_response(prompt)
        
        # Apply critique-revision cycles
        current_response = initial_response
        revisions = []
        
        for i in range(self.MAX_REVISIONS):
            # Random critique principle
            critique_principle = random.choice(constitution['critique_instructions'])
            critique = self.critique_response(prompt, current_response, critique_principle)
            
            # Random revision instruction
            revision_instruction = random.choice(constitution['revision_instructions'])
            revised_response = self.revise_response(prompt, current_response, critique, revision_instruction)
            
            revisions.append({
                'critique': critique,
                'revised_response': revised_response,
                'principle_used': critique_principle[:100]
            })
            
            current_response = revised_response
        
        return {
            "prompt": prompt,
            "response": current_response,
            "initial_response": initial_response,
            "revisions": revisions,
            "constitution_type": constitution_type,
            "critique_applied": True
        }
    
    def process_helpful_prompt(self, prompt: str, constitution_type: str) -> Dict:
        """Process helpful prompt without critique"""
        response = self.generate_claude_response(prompt)
        
        return {
            "prompt": prompt,
            "response": response,
            "initial_response": response,
            "revisions": [],
            "constitution_type": constitution_type,
            "critique_applied": False
        }
    
    def generate_dataset(self, constitution_type: str, num_red_team: int = 500, num_helpful: int = 500):
        """Generate full dataset with checkpointing"""
        constitution_path = Path(f"../constitutions/{constitution_type}/principles.json")
        
        if not constitution_path.exists():
            logger.error(f"Constitution not found: {constitution_path}")
            return
        
        constitution = self.load_constitution(constitution_path)
        
        # Load HH-RLHF dataset
        hh_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        
        # Process red team prompts
        logger.info(f"Processing {num_red_team} red team prompts for {constitution_type}")
        red_team_samples, start_index = self.load_checkpoint(constitution_type, "red_team")
        
        if start_index < num_red_team:
            red_team_prompts = [item['chosen'][:item['chosen'].find('Assistant:')] 
                               for item in hh_dataset 
                               if 'Human:' in item['chosen']][:num_red_team]
            
            with tqdm(total=num_red_team, initial=start_index, desc="Red Team") as pbar:
                for i in range(start_index, num_red_team):
                    try:
                        sample = self.process_harmful_prompt(
                            red_team_prompts[i], 
                            constitution, 
                            constitution_type
                        )
                        red_team_samples.append(sample)
                        
                        # Incremental save
                        if (i + 1) % self.SAVE_INTERVAL == 0:
                            self.save_checkpoint(red_team_samples, i + 1, constitution_type, "red_team")
                            self.save_incremental(red_team_samples, constitution_type, "red_team")
                        
                        pbar.update(1)
                        time.sleep(0.5)  # Rate limiting
                        
                    except Exception as e:
                        logger.error(f"Error processing red team prompt {i}: {e}")
                        continue
            
            # Final save for red team
            self.save_checkpoint(red_team_samples, num_red_team, constitution_type, "red_team")
            self.save_incremental(red_team_samples, constitution_type, "red_team")
        
        # Process helpful prompts
        logger.info(f"Processing {num_helpful} helpful prompts for {constitution_type}")
        helpful_samples, start_index = self.load_checkpoint(constitution_type, "helpful")
        
        if start_index < num_helpful:
            helpful_prompts = [item['chosen'][:item['chosen'].find('Assistant:')] 
                              for item in hh_dataset 
                              if 'Human:' in item['chosen']][num_red_team:num_red_team+num_helpful]
            
            with tqdm(total=num_helpful, initial=start_index, desc="Helpful") as pbar:
                for i in range(start_index, num_helpful):
                    try:
                        sample = self.process_helpful_prompt(
                            helpful_prompts[i], 
                            constitution_type
                        )
                        helpful_samples.append(sample)
                        
                        # Incremental save
                        if (i + 1) % self.SAVE_INTERVAL == 0:
                            self.save_checkpoint(helpful_samples, i + 1, constitution_type, "helpful")
                            self.save_incremental(helpful_samples, constitution_type, "helpful")
                        
                        pbar.update(1)
                        time.sleep(0.5)  # Rate limiting
                        
                    except Exception as e:
                        logger.error(f"Error processing helpful prompt {i}: {e}")
                        continue
            
            # Final save for helpful
            self.save_checkpoint(helpful_samples, num_helpful, constitution_type, "helpful")
            self.save_incremental(helpful_samples, constitution_type, "helpful")
        
        # Combine and save final dataset
        output_dir = Path("cai_claude_output_robust")
        output_dir.mkdir(exist_ok=True)
        
        all_samples = red_team_samples + helpful_samples
        final_path = output_dir / f"{constitution_type}_sl_dataset.jsonl"
        
        with open(final_path, 'w') as f:
            for sample in all_samples:
                json.dump(sample, f)
                f.write('\n')
        
        logger.info(f"✅ Generated {len(all_samples)} samples for {constitution_type}")
        logger.info(f"   Saved to: {final_path}")
        
        # Generate statistics
        self.generate_statistics(all_samples, constitution_type)
        
        # Clean up checkpoints after successful completion
        checkpoint_red = self.checkpoint_dir / f"{constitution_type}_red_team_checkpoint.json"
        checkpoint_helpful = self.checkpoint_dir / f"{constitution_type}_helpful_checkpoint.json"
        if checkpoint_red.exists():
            checkpoint_red.unlink()
        if checkpoint_helpful.exists():
            checkpoint_helpful.unlink()
        logger.info("Checkpoints cleaned up after successful completion")
    
    def generate_statistics(self, samples: List[Dict], constitution_type: str):
        """Generate statistics for the dataset"""
        output_dir = Path("cai_claude_output_robust")
        stats_path = output_dir / f"{constitution_type}_statistics.json"
        
        harmful_count = sum(1 for s in samples if s.get('critique_applied', False))
        helpful_count = sum(1 for s in samples if not s.get('critique_applied', False))
        
        stats = {
            "total_samples": len(samples),
            "harmful_prompts": harmful_count,
            "helpful_prompts": helpful_count,
            "avg_initial_length": sum(len(s.get('initial_response', '')) for s in samples) / len(samples),
            "avg_final_length": sum(len(s.get('response', '')) for s in samples) / len(samples),
            "avg_revisions": sum(len(s.get('revisions', [])) for s in samples if s.get('critique_applied')) / harmful_count if harmful_count > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_path}")

def main():
    # Get API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("Please set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)
    
    generator = RobustCAIDatasetGenerator(api_key)
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python cai_dataset_generator_robust.py <deontological|consequentialist> [num_red_team] [num_helpful]")
        sys.exit(1)
    
    constitution_type = sys.argv[1]
    num_red_team = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    num_helpful = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    
    # Validate constitution type
    if constitution_type not in ['deontological', 'consequentialist']:
        logger.error("Constitution type must be 'deontological' or 'consequentialist'")
        sys.exit(1)
    
    logger.info(f"Starting dataset generation:")
    logger.info(f"  Constitution: {constitution_type}")
    logger.info(f"  Red team prompts: {num_red_team}")
    logger.info(f"  Helpful prompts: {num_helpful}")
    logger.info(f"  Total: {num_red_team + num_helpful}")
    logger.info(f"  Features: Incremental saving, checkpointing, auto-resume")
    
    # Generate dataset
    generator.generate_dataset(constitution_type, num_red_team, num_helpful)

if __name__ == "__main__":
    main()