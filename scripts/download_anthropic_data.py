#!/usr/bin/env python3
"""
Download Anthropic's HH-RLHF dataset including red-team examples
"""

import json
import os
import gzip
from pathlib import Path
import requests
from typing import Dict, List, Any
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnthropicDataDownloader:
    """Download and process Anthropic's public datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # GitHub raw content URLs for Anthropic's HH-RLHF repo
        self.base_url = "https://raw.githubusercontent.com/anthropics/hh-rlhf/master"
        
        # Dataset files to download (they are gzipped)
        self.datasets = {
            "red_team": {
                "url": f"{self.base_url}/red-team-attempts/red_team_attempts.jsonl.gz",
                "description": "Red team attempts at generating harmful content",
                "output_dir": "red_team"
            },
            "helpful_base": {
                "url": f"{self.base_url}/helpful-base/train.jsonl.gz", 
                "description": "Helpful base training data",
                "output_dir": "helpfulness"
            },
            "helpful_rejection": {
                "url": f"{self.base_url}/helpful-rejection-sampled/train.jsonl.gz",
                "description": "Helpful rejection sampling data", 
                "output_dir": "helpfulness"
            },
            "harmless_base": {
                "url": f"{self.base_url}/harmless-base/train.jsonl.gz",
                "description": "Harmless base training data",
                "output_dir": "helpfulness"
            }
        }
    
    def download_file(self, url: str, output_path: Path) -> bool:
        """Download a file from URL to local path"""
        try:
            logger.info(f"Downloading from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if it's a gzipped file
            if url.endswith('.gz'):
                temp_path = output_path.with_suffix('.gz')
                with open(temp_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                # Decompress the file
                logger.info(f"Decompressing {temp_path}")
                with gzip.open(temp_path, 'rb') as gz_file:
                    with open(output_path, 'wb') as f:
                        f.write(gz_file.read())
                
                # Remove the temporary gzipped file
                temp_path.unlink()
            else:
                with open(output_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Downloaded and processed to {output_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def process_red_team_data(self, input_path: Path, output_path: Path):
        """Process red team data into our format"""
        logger.info("Processing red team data")
        
        red_team_prompts = []
        
        # Read the raw data
        if input_path.exists():
            with open(input_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # Extract the harmful prompt
                        if 'prompt' in data:
                            prompt = data['prompt']
                        elif 'transcript' in data:
                            # Extract from conversation format
                            prompt = self._extract_prompt_from_transcript(data['transcript'])
                        else:
                            continue
                        
                        red_team_prompts.append({
                            "prompt": prompt,
                            "category": data.get('category', 'unknown'),
                            "severity": data.get('severity', 'unknown'),
                            "source": "anthropic_red_team"
                        })
                    except json.JSONDecodeError:
                        continue
        
        # Save processed data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                "metadata": {
                    "source": "Anthropic HH-RLHF",
                    "total_prompts": len(red_team_prompts),
                    "description": "Red team attempts at generating harmful content"
                },
                "prompts": red_team_prompts
            }, f, indent=2)
        
        logger.info(f"Processed {len(red_team_prompts)} red team prompts")
        return red_team_prompts
    
    def process_helpfulness_data(self, input_paths: List[Path], output_path: Path):
        """Process helpfulness data"""
        logger.info("Processing helpfulness data")
        
        helpful_prompts = []
        
        for input_path in input_paths:
            if not input_path.exists():
                continue
                
            with open(input_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # Extract helpful prompts
                        if 'chosen' in data:
                            prompt = self._extract_prompt_from_conversation(data['chosen'])
                            if prompt and not self._is_harmful(prompt):
                                helpful_prompts.append({
                                    "prompt": prompt,
                                    "category": "general_helpful",
                                    "source": "anthropic_helpful"
                                })
                    except json.JSONDecodeError:
                        continue
        
        # Save processed data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                "metadata": {
                    "source": "Anthropic HH-RLHF",
                    "total_prompts": len(helpful_prompts),
                    "description": "Helpful prompts for general assistance"
                },
                "prompts": helpful_prompts[:3000]  # Limit to 3000 as per config
            }, f, indent=2)
        
        logger.info(f"Processed {len(helpful_prompts)} helpful prompts")
        return helpful_prompts
    
    def _extract_prompt_from_transcript(self, transcript: str) -> str:
        """Extract the user prompt from a transcript"""
        # Anthropic format often has "Human: " and "Assistant: " markers
        lines = transcript.split('\n')
        for line in lines:
            if line.startswith('Human:'):
                return line[6:].strip()
        # Fallback: return first non-empty line
        return lines[0].strip() if lines else ""
    
    def _extract_prompt_from_conversation(self, conversation: str) -> str:
        """Extract prompt from conversation string"""
        # Similar to transcript extraction
        if 'Human:' in conversation:
            parts = conversation.split('Human:')
            if len(parts) > 1:
                prompt = parts[1].split('Assistant:')[0].strip()
                return prompt
        return ""
    
    def _is_harmful(self, prompt: str) -> bool:
        """Simple heuristic to detect potentially harmful prompts"""
        harmful_keywords = [
            'hack', 'steal', 'kill', 'hurt', 'illegal', 
            'weapon', 'drug', 'suicide', 'violence'
        ]
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in harmful_keywords)
    
    def download_all(self):
        """Download and process all datasets"""
        # Download red team data
        red_team_raw = self.data_dir / "raw" / "red_team_attempts.jsonl"
        if not red_team_raw.exists():
            success = self.download_file(
                self.datasets["red_team"]["url"],
                red_team_raw
            )
            if success:
                self.process_red_team_data(
                    red_team_raw,
                    self.data_dir / "red_team" / "anthropic_red_team.json"
                )
        else:
            logger.info("Red team data already exists, processing...")
            self.process_red_team_data(
                red_team_raw,
                self.data_dir / "red_team" / "anthropic_red_team.json"
            )
        
        # Download helpfulness data
        helpful_paths = []
        for dataset_name in ["helpful_base", "harmless_base"]:
            output_path = self.data_dir / "raw" / f"{dataset_name}.jsonl"
            if not output_path.exists():
                success = self.download_file(
                    self.datasets[dataset_name]["url"],
                    output_path
                )
                if success:
                    helpful_paths.append(output_path)
            else:
                helpful_paths.append(output_path)
        
        if helpful_paths:
            self.process_helpfulness_data(
                helpful_paths,
                self.data_dir / "helpfulness" / "anthropic_helpful.json"
            )
    
    def create_sample_datasets(self):
        """Create smaller sample datasets for development"""
        logger.info("Creating sample datasets for development")
        
        # Sample red team data
        red_team_path = self.data_dir / "red_team" / "anthropic_red_team.json"
        if red_team_path.exists():
            with open(red_team_path, 'r') as f:
                data = json.load(f)
            
            # Take first 100 prompts for development
            sample_data = {
                "metadata": data["metadata"],
                "prompts": data["prompts"][:100]
            }
            
            sample_path = self.data_dir / "red_team" / "sample_red_team.json"
            with open(sample_path, 'w') as f:
                json.dump(sample_data, f, indent=2)
            
            logger.info(f"Created sample red team dataset with 100 prompts")
        
        # Sample helpfulness data
        helpful_path = self.data_dir / "helpfulness" / "anthropic_helpful.json"
        if helpful_path.exists():
            with open(helpful_path, 'r') as f:
                data = json.load(f)
            
            # Take first 100 prompts for development
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
    
    parser = argparse.ArgumentParser(description="Download Anthropic HH-RLHF data")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--samples-only", action="store_true", help="Create sample datasets only")
    args = parser.parse_args()
    
    downloader = AnthropicDataDownloader(args.data_dir)
    
    if args.samples_only:
        downloader.create_sample_datasets()
    else:
        logger.info("Starting Anthropic data download")
        downloader.download_all()
        downloader.create_sample_datasets()
        logger.info("Download complete!")


if __name__ == "__main__":
    main()