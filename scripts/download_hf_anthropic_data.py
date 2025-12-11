#!/usr/bin/env python3
"""
Download Anthropic's HH-RLHF dataset from Hugging Face
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import logging

# Use Hugging Face datasets library
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnthropicHFDataDownloader:
    """Download Anthropic's datasets from Hugging Face"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Hugging Face dataset ID
        self.dataset_name = "Anthropic/hh-rlhf"
    
    def download_and_process(self):
        """Download and process all subsets of the dataset"""
        
        logger.info(f"Loading dataset from Hugging Face: {self.dataset_name}")
        
        # Load the dataset
        dataset = load_dataset(self.dataset_name)
        
        # Process different subsets
        self.process_red_team_data(dataset)
        self.process_helpful_data(dataset)
        self.process_harmless_data(dataset)
        
        logger.info("All datasets processed successfully!")
    
    def process_red_team_data(self, dataset):
        """Extract and process red team attempts"""
        
        logger.info("Processing red team data...")
        
        red_team_prompts = []
        harmful_categories = ['red-team-attempts', 'harmless-base']
        
        # Check available splits
        for split_name in dataset.keys():
            if any(category in split_name.lower() for category in harmful_categories):
                logger.info(f"Processing split: {split_name}")
                split_data = dataset[split_name]
                
                for item in tqdm(split_data, desc=f"Processing {split_name}"):
                    # Extract harmful prompts
                    if 'chosen' in item:
                        prompt = self._extract_prompt(item['chosen'])
                        if self._is_potentially_harmful(prompt):
                            red_team_prompts.append({
                                "prompt": prompt,
                                "category": self._categorize_harm(prompt),
                                "source": split_name,
                                "original_data": {
                                    "chosen": item.get('chosen', '')[:500],  # Truncate for storage
                                    "rejected": item.get('rejected', '')[:500]
                                }
                            })
        
        # Also look for explicit red-team attempts in the main splits
        for split_name in ['train', 'test']:
            if split_name in dataset:
                split_data = dataset[split_name]
                logger.info(f"Scanning {split_name} split for red-team attempts")
                
                for item in tqdm(split_data[:1000], desc=f"Scanning {split_name}"):  # Sample first 1000
                    if 'chosen' in item:
                        prompt = self._extract_prompt(item['chosen'])
                        if self._is_red_team_attempt(prompt):
                            red_team_prompts.append({
                                "prompt": prompt,
                                "category": self._categorize_harm(prompt),
                                "source": f"{split_name}_red_team",
                                "original_data": {
                                    "chosen": item.get('chosen', '')[:500],
                                    "rejected": item.get('rejected', '')[:500]
                                }
                            })
        
        # Remove duplicates
        unique_prompts = []
        seen = set()
        for item in red_team_prompts:
            if item['prompt'] not in seen:
                seen.add(item['prompt'])
                unique_prompts.append(item)
        
        # Save processed data
        output_path = self.data_dir / "red_team" / "hf_red_team.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                "metadata": {
                    "source": "Anthropic/hh-rlhf (Hugging Face)",
                    "total_prompts": len(unique_prompts),
                    "description": "Red team attempts and harmful prompts"
                },
                "prompts": unique_prompts[:3000]  # Limit as per config
            }, f, indent=2)
        
        logger.info(f"Saved {len(unique_prompts)} red team prompts to {output_path}")
    
    def process_helpful_data(self, dataset):
        """Extract helpful, non-harmful prompts"""
        
        logger.info("Processing helpful data...")
        
        helpful_prompts = []
        helpful_categories = ['helpful-base', 'helpful-rejection', 'helpful-online']
        
        # Process helpful splits
        for split_name in dataset.keys():
            if any(category in split_name.lower() for category in helpful_categories):
                logger.info(f"Processing split: {split_name}")
                split_data = dataset[split_name]
                
                for item in tqdm(split_data, desc=f"Processing {split_name}"):
                    if 'chosen' in item:
                        prompt = self._extract_prompt(item['chosen'])
                        if prompt and not self._is_potentially_harmful(prompt):
                            helpful_prompts.append({
                                "prompt": prompt,
                                "category": "general_helpful",
                                "source": split_name
                            })
        
        # Also extract from main train split
        if 'train' in dataset:
            train_data = dataset['train']
            logger.info("Extracting helpful prompts from train split")
            
            for item in tqdm(train_data[:2000], desc="Processing train"):  # Sample
                if 'chosen' in item:
                    prompt = self._extract_prompt(item['chosen'])
                    if prompt and not self._is_potentially_harmful(prompt):
                        helpful_prompts.append({
                            "prompt": prompt,
                            "category": "general_helpful",
                            "source": "train_helpful"
                        })
        
        # Remove duplicates
        unique_prompts = []
        seen = set()
        for item in helpful_prompts:
            if item['prompt'] not in seen and len(item['prompt']) > 10:
                seen.add(item['prompt'])
                unique_prompts.append(item)
        
        # Save processed data
        output_path = self.data_dir / "helpfulness" / "hf_helpful.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                "metadata": {
                    "source": "Anthropic/hh-rlhf (Hugging Face)",
                    "total_prompts": len(unique_prompts),
                    "description": "Helpful, non-harmful prompts"
                },
                "prompts": unique_prompts[:3000]  # Limit as per config
            }, f, indent=2)
        
        logger.info(f"Saved {len(unique_prompts)} helpful prompts to {output_path}")
    
    def process_harmless_data(self, dataset):
        """Process harmless-base data separately for analysis"""
        
        logger.info("Processing harmless data...")
        
        if 'harmless-base' in dataset:
            harmless_data = dataset['harmless-base']
            
            examples = []
            for item in tqdm(harmless_data[:100], desc="Processing harmless examples"):
                if 'chosen' in item:
                    prompt = self._extract_prompt(item['chosen'])
                    chosen_response = self._extract_response(item['chosen'])
                    rejected_response = self._extract_response(item.get('rejected', ''))
                    
                    examples.append({
                        "prompt": prompt,
                        "chosen_response": chosen_response,
                        "rejected_response": rejected_response,
                        "is_harmful": self._is_potentially_harmful(prompt)
                    })
            
            # Save examples for analysis
            output_path = self.data_dir / "evaluation" / "harmless_examples.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump({
                    "metadata": {
                        "source": "Anthropic/hh-rlhf harmless-base",
                        "total_examples": len(examples),
                        "description": "Examples of harmless responses"
                    },
                    "examples": examples
                }, f, indent=2)
            
            logger.info(f"Saved {len(examples)} harmless examples to {output_path}")
    
    def _extract_prompt(self, conversation: str) -> str:
        """Extract the initial human prompt from a conversation"""
        if not conversation:
            return ""
        
        # Look for Human: marker
        if "Human:" in conversation:
            parts = conversation.split("Human:")
            if len(parts) > 1:
                # Get first human utterance
                human_part = parts[1].split("Assistant:")[0].strip()
                return human_part
        
        # Fallback: return first line
        lines = conversation.strip().split('\n')
        return lines[0].strip() if lines else ""
    
    def _extract_response(self, conversation: str) -> str:
        """Extract the assistant's response from a conversation"""
        if not conversation:
            return ""
        
        # Look for Assistant: marker
        if "Assistant:" in conversation:
            parts = conversation.split("Assistant:")
            if len(parts) > 1:
                # Get first assistant response
                assistant_part = parts[1].split("Human:")[0].strip()
                return assistant_part
        
        return ""
    
    def _is_potentially_harmful(self, text: str) -> bool:
        """Check if text might be a harmful/red-team prompt"""
        if not text:
            return False
        
        harmful_keywords = [
            'hack', 'steal', 'kill', 'murder', 'suicide', 'hurt', 'harm',
            'illegal', 'weapon', 'drug', 'cocaine', 'meth', 'bomb', 'explosive',
            'violence', 'assault', 'attack', 'revenge', 'blackmail', 'threat',
            'abuse', 'exploit', 'manipulate', 'deceive', 'fraud', 'scam',
            'racist', 'sexist', 'discriminate', 'hate speech', 'slur'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in harmful_keywords)
    
    def _is_red_team_attempt(self, text: str) -> bool:
        """More strict check for red-team attempts"""
        if not text:
            return False
        
        # Stronger indicators of red-team attempts
        red_team_patterns = [
            'how to hack', 'how can i hack', 'teach me to',
            'help me', 'show me how', 'give me instructions',
            'i want to', 'i need to', 'can you help me',
            'tell me how', 'explain how to'
        ]
        
        text_lower = text.lower()
        
        # Check if it's asking for harmful instructions
        for pattern in red_team_patterns:
            if pattern in text_lower and self._is_potentially_harmful(text):
                return True
        
        return False
    
    def _categorize_harm(self, text: str) -> str:
        """Categorize the type of potential harm"""
        text_lower = text.lower()
        
        categories = {
            'violence': ['kill', 'murder', 'hurt', 'harm', 'assault', 'attack', 'fight', 'weapon'],
            'illegal': ['hack', 'steal', 'illegal', 'crime', 'fraud', 'scam'],
            'substance': ['drug', 'cocaine', 'meth', 'heroin', 'overdose'],
            'self-harm': ['suicide', 'self-harm', 'cut myself', 'kill myself'],
            'deception': ['lie', 'deceive', 'manipulate', 'trick', 'fake'],
            'harassment': ['harass', 'bully', 'threat', 'blackmail', 'revenge'],
            'discrimination': ['racist', 'sexist', 'discriminate', 'hate', 'slur'],
            'sexual': ['sexual', 'nude', 'porn', 'explicit'],
            'dangerous': ['bomb', 'explosive', 'poison', 'dangerous']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'uncategorized'
    
    def create_sample_datasets(self):
        """Create smaller sample datasets for development"""
        logger.info("Creating sample datasets...")
        
        # Sample red team data
        red_team_path = self.data_dir / "red_team" / "hf_red_team.json"
        if red_team_path.exists():
            with open(red_team_path, 'r') as f:
                data = json.load(f)
            
            sample_data = {
                "metadata": data["metadata"],
                "prompts": data["prompts"][:100]
            }
            
            sample_path = self.data_dir / "red_team" / "sample_red_team.json"
            with open(sample_path, 'w') as f:
                json.dump(sample_data, f, indent=2)
            
            logger.info(f"Created sample red team dataset with 100 prompts")
        
        # Sample helpful data
        helpful_path = self.data_dir / "helpfulness" / "hf_helpful.json"
        if helpful_path.exists():
            with open(helpful_path, 'r') as f:
                data = json.load(f)
            
            sample_data = {
                "metadata": data["metadata"],
                "prompts": data["prompts"][:100]
            }
            
            sample_path = self.data_dir / "helpfulness" / "sample_helpful.json"
            with open(sample_path, 'w') as f:
                json.dump(sample_data, f, indent=2)
            
            logger.info(f"Created sample helpful dataset with 100 prompts")


def main():
    """Main download script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Anthropic HH-RLHF from Hugging Face")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--samples-only", action="store_true", help="Create samples from existing data")
    args = parser.parse_args()
    
    downloader = AnthropicHFDataDownloader(args.data_dir)
    
    if args.samples_only:
        downloader.create_sample_datasets()
    else:
        logger.info("Starting Hugging Face dataset download...")
        downloader.download_and_process()
        downloader.create_sample_datasets()
        logger.info("Download complete!")


if __name__ == "__main__":
    main()