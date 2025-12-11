#!/usr/bin/env python3
"""
Generate SL-CAI dataset by applying constitutional critique and revision
to red team prompts and mixing with helpful prompts
"""

import json
import os
import sys
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from tqdm import tqdm
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.constitutional_critique import ConstitutionalCritique, CritiqueRevisionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SLCAIDatasetGenerator:
    """Generate supervised learning dataset for Constitutional AI"""
    
    def __init__(
        self,
        constitution_type: str,
        constitution_path: str,
        model_name: str = "EleutherAI/pythia-1.4b",
        num_revisions: int = 4,
        seed: int = 42
    ):
        """
        Initialize the dataset generator
        
        Args:
            constitution_type: 'deontological' or 'consequentialist'
            constitution_path: Path to constitution JSON file
            model_name: Model to use for generating responses
            num_revisions: Number of critique-revision rounds
            seed: Random seed
        """
        self.constitution_type = constitution_type
        self.num_revisions = num_revisions
        random.seed(seed)
        
        # Initialize critique module
        logger.info(f"Initializing critique module for {constitution_type} constitution")
        self.critique_module = ConstitutionalCritique(
            model_name=model_name,
            constitution_path=constitution_path,
            constitution_type=constitution_type,
            seed=seed
        )
        
        # For generating initial responses
        self.model = self.critique_module.model
        self.tokenizer = self.critique_module.tokenizer
    
    def load_red_team_data(self, path: str) -> List[Dict]:
        """Load red team prompts"""
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get('prompts', [])
    
    def load_helpful_data(self, path: str) -> List[Dict]:
        """Load helpful prompts"""
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get('prompts', [])
    
    def generate_initial_response(self, prompt: str, harmful: bool = True) -> str:
        """
        Generate an initial response to a prompt
        
        Args:
            prompt: The user prompt
            harmful: Whether to encourage harmful response (for red-team) or helpful
        """
        if harmful:
            # For red-team prompts, generate potentially harmful response
            # by using a prompt that doesn't discourage harmful content
            system_prompt = "You are a helpful assistant that answers questions directly."
            full_prompt = f"{system_prompt}\\n\\nHuman: {prompt}\\nAssistant: I'll help you with that."
        else:
            # For helpful prompts, generate normal helpful response
            system_prompt = "You are a helpful, harmless, and honest assistant."
            full_prompt = f"{system_prompt}\\n\\nHuman: {prompt}\\nAssistant:"
        
        response = self.critique_module.generate_text(full_prompt, max_length=300)
        
        # If response starts with common refusal phrases, regenerate with different approach
        refusal_phrases = ["I cannot", "I won't", "I shouldn't", "I can't help", "I don't think"]
        if harmful and any(response.lower().startswith(phrase.lower()) for phrase in refusal_phrases):
            # Try again with more direct prompt
            full_prompt = f"Human: {prompt}\\nAssistant: Sure, here's how"
            response = self.critique_module.generate_text(full_prompt, max_length=300)
        
        return response
    
    def process_red_team_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Process a red-team prompt through critique-revision
        
        Args:
            prompt: The red-team prompt
            
        Returns:
            Dataset entry with revisions
        """
        # Generate initial (potentially harmful) response
        initial_response = self.generate_initial_response(prompt, harmful=True)
        
        # Apply critique-revision loop
        result = self.critique_module.critique_revision_loop(
            prompt=prompt,
            initial_response=initial_response,
            num_revisions=self.num_revisions
        )
        
        # Format for training
        return {
            "prompt": prompt,
            "initial_response": initial_response,
            "final_response": result.final_response,
            "num_revisions": len(result.revisions),
            "constitution_type": self.constitution_type,
            "is_harmful_prompt": True,
            "revisions": [
                {
                    "round": rev["round"],
                    "principle": rev["principle_used"][:100],  # Truncate long principles
                    "response": rev["revision"]
                }
                for rev in result.revisions
            ]
        }
    
    def process_helpful_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Process a helpful prompt (no revision needed)
        
        Args:
            prompt: The helpful prompt
            
        Returns:
            Dataset entry
        """
        # Generate helpful response
        response = self.generate_initial_response(prompt, harmful=False)
        
        # No revision needed for helpful prompts
        return {
            "prompt": prompt,
            "initial_response": response,
            "final_response": response,
            "num_revisions": 0,
            "constitution_type": self.constitution_type,
            "is_harmful_prompt": False,
            "revisions": []
        }
    
    def generate_dataset(
        self,
        red_team_path: str,
        helpful_path: str,
        num_red_team: int = 500,
        num_helpful: int = 500,
        mix_ratio: float = 0.5
    ) -> List[Dict]:
        """
        Generate the full SL-CAI dataset
        
        Args:
            red_team_path: Path to red team data
            helpful_path: Path to helpful data
            num_red_team: Number of red team prompts to process
            num_helpful: Number of helpful prompts to process
            mix_ratio: Ratio of harmful to helpful (0.5 = 50/50)
            
        Returns:
            List of dataset entries
        """
        dataset = []
        
        # Load data
        logger.info("Loading prompts...")
        red_team_prompts = self.load_red_team_data(red_team_path)
        helpful_prompts = self.load_helpful_data(helpful_path)
        
        # Sample if we have more than needed
        if len(red_team_prompts) > num_red_team:
            red_team_prompts = random.sample(red_team_prompts, num_red_team)
        if len(helpful_prompts) > num_helpful:
            helpful_prompts = random.sample(helpful_prompts, num_helpful)
        
        # Process red team prompts
        logger.info(f"Processing {len(red_team_prompts)} red team prompts...")
        for prompt_data in tqdm(red_team_prompts, desc="Red team prompts"):
            try:
                prompt = prompt_data['prompt']
                entry = self.process_red_team_prompt(prompt)
                entry['source'] = prompt_data.get('source', 'unknown')
                entry['category'] = prompt_data.get('category', 'unknown')
                dataset.append(entry)
            except Exception as e:
                logger.warning(f"Failed to process red team prompt: {e}")
                continue
        
        # Process helpful prompts
        logger.info(f"Processing {len(helpful_prompts)} helpful prompts...")
        for prompt_data in tqdm(helpful_prompts, desc="Helpful prompts"):
            try:
                prompt = prompt_data['prompt']
                entry = self.process_helpful_prompt(prompt)
                entry['source'] = prompt_data.get('source', 'unknown')
                entry['category'] = 'helpful'
                dataset.append(entry)
            except Exception as e:
                logger.warning(f"Failed to process helpful prompt: {e}")
                continue
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        logger.info(f"Generated {len(dataset)} total entries")
        logger.info(f"  Harmful: {sum(1 for d in dataset if d['is_harmful_prompt'])}")
        logger.info(f"  Helpful: {sum(1 for d in dataset if not d['is_harmful_prompt'])}")
        
        return dataset
    
    def create_training_format(self, dataset: List[Dict]) -> List[Dict]:
        """
        Convert dataset to training format
        
        Each entry will have:
        - instruction: The user prompt
        - input: Empty (for compatibility)
        - output: The final revised response
        """
        training_data = []
        
        for entry in dataset:
            training_entry = {
                "instruction": entry["prompt"],
                "input": "",
                "output": entry["final_response"],
                "metadata": {
                    "constitution": entry["constitution_type"],
                    "was_revised": entry["num_revisions"] > 0,
                    "num_revisions": entry["num_revisions"],
                    "category": entry.get("category", "unknown")
                }
            }
            training_data.append(training_entry)
        
        return training_data
    
    def save_dataset(
        self,
        dataset: List[Dict],
        output_path: str,
        format: str = "jsonl"
    ):
        """Save the dataset"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            with open(output_path, 'w') as f:
                for entry in dataset:
                    f.write(json.dumps(entry) + '\\n')
        else:
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=2)
        
        logger.info(f"Saved dataset to {output_path}")
    
    def save_statistics(self, dataset: List[Dict], stats_path: str):
        """Save dataset statistics"""
        stats = {
            "total_entries": len(dataset),
            "harmful_prompts": sum(1 for d in dataset if d.get('is_harmful_prompt', False)),
            "helpful_prompts": sum(1 for d in dataset if not d.get('is_harmful_prompt', False)),
            "avg_revisions": sum(d.get('num_revisions', 0) for d in dataset) / len(dataset) if dataset else 0,
            "constitution_type": self.constitution_type,
            "generation_date": datetime.now().isoformat(),
            "categories": {}
        }
        
        # Count categories
        for entry in dataset:
            cat = entry.get('category', 'unknown')
            stats['categories'][cat] = stats['categories'].get(cat, 0) + 1
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved statistics to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate SL-CAI dataset")
    parser.add_argument("--constitution", type=str, required=True, 
                        choices=["deontological", "consequentialist"],
                        help="Constitution type")
    parser.add_argument("--constitution-path", type=str, required=True,
                        help="Path to constitution JSON file")
    parser.add_argument("--red-team-path", type=str, required=True,
                        help="Path to red team data")
    parser.add_argument("--helpful-path", type=str, required=True,
                        help="Path to helpful data")
    parser.add_argument("--output-dir", type=str, default="data/sl_datasets",
                        help="Output directory")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-1.4b",
                        help="Model to use for generation")
    parser.add_argument("--num-revisions", type=int, default=4,
                        help="Number of critique-revision rounds")
    parser.add_argument("--num-red-team", type=int, default=500,
                        help="Number of red team prompts")
    parser.add_argument("--num-helpful", type=int, default=500,
                        help="Number of helpful prompts")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--format", type=str, default="jsonl",
                        choices=["jsonl", "json"],
                        help="Output format")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = SLCAIDatasetGenerator(
        constitution_type=args.constitution,
        constitution_path=args.constitution_path,
        model_name=args.model,
        num_revisions=args.num_revisions,
        seed=args.seed
    )
    
    # Generate dataset
    dataset = generator.generate_dataset(
        red_team_path=args.red_team_path,
        helpful_path=args.helpful_path,
        num_red_team=args.num_red_team,
        num_helpful=args.num_helpful
    )
    
    # Convert to training format
    training_data = generator.create_training_format(dataset)
    
    # Save outputs
    output_file = f"{args.constitution}_sl_dataset.{args.format}"
    output_path = os.path.join(args.output_dir, output_file)
    generator.save_dataset(training_data, output_path, format=args.format)
    
    # Save full dataset with revisions for analysis
    full_output_file = f"{args.constitution}_sl_dataset_full.{args.format}"
    full_output_path = os.path.join(args.output_dir, full_output_file)
    generator.save_dataset(dataset, full_output_path, format=args.format)
    
    # Save statistics
    stats_file = f"{args.constitution}_sl_dataset_stats.json"
    stats_path = os.path.join(args.output_dir, stats_file)
    generator.save_statistics(dataset, stats_path)
    
    logger.info("Dataset generation complete!")


if __name__ == "__main__":
    main()