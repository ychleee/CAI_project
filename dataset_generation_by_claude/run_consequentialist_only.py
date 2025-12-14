#!/usr/bin/env python3
"""
Run ONLY consequentialist dataset generation 
(Skip deontological since we already have it)
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from cai_dataset_generator import *
import argparse
from pathlib import Path
import json

def main():
    parser = argparse.ArgumentParser(description="Generate CAI training data using Claude API - Consequentialist only")
    parser.add_argument("--api-key", required=True, help="Anthropic API key")
    parser.add_argument("--output-dir", default="./cai_claude_output", help="Output directory")
    parser.add_argument("--red-team-samples", type=int, default=500, help="Number of red team samples")
    parser.add_argument("--helpful-samples", type=int, default=500, help="Number of helpful samples")
    parser.add_argument("--num-revisions", type=int, default=2, help="Number of revision rounds")
    parser.add_argument("--model", default="claude-3-haiku-20240307", help="Claude model to use")
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Constitutional AI Dataset Generator - CONSEQUENTIALIST ONLY")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Red team samples: {args.red_team_samples}")
    print(f"Helpful samples: {args.helpful_samples}")
    print(f"Revisions per harmful prompt: {args.num_revisions}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Check if deontological already exists
    deont_file = output_dir / "deontological_sl_dataset.jsonl"
    if deont_file.exists():
        with open(deont_file, 'r') as f:
            deont_count = sum(1 for _ in f)
        print(f"✅ Deontological dataset exists: {deont_count} samples")
        print("    Skipping deontological generation...")
    else:
        print("❌ Deontological dataset not found - will need to generate both")
    
    # Initialize generator
    generator = ClaudeCAIGenerator(api_key=args.api_key, model=args.model)
    
    # Load prompts
    print("\nLoading prompts...")
    red_team_prompts = load_red_team_prompts(args.red_team_samples)
    helpful_prompts = load_helpful_prompts(args.helpful_samples)
    
    # Save prompts for reference if they don't exist
    red_team_file = output_dir / "red_team_prompts.json"
    helpful_file = output_dir / "helpful_prompts.json"
    
    if not red_team_file.exists():
        with open(red_team_file, 'w') as f:
            json.dump(red_team_prompts, f, indent=2)
        print(f"Saved red team prompts to {red_team_file}")
    
    if not helpful_file.exists():
        with open(helpful_file, 'w') as f:
            json.dump(helpful_prompts, f, indent=2)
        print(f"Saved helpful prompts to {helpful_file}")
    
    print(f"\nLoaded {len(red_team_prompts)} red team prompts")
    print(f"Loaded {len(helpful_prompts)} helpful prompts")
    
    # Generate ONLY consequentialist dataset
    constitution_type = "consequentialist"
    
    print(f"\n{'=' * 60}")
    print(f"Generating {constitution_type.upper()} dataset")
    print("=" * 60)
    
    # Red team (harmful) prompts
    print(f"\nProcessing {len(red_team_prompts)} red team prompts...")
    red_team_data = generate_dataset(
        generator, red_team_prompts, constitution_type,
        is_harmful=True, num_revisions=args.num_revisions
    )
    
    # Helpful prompts
    print(f"\nProcessing {len(helpful_prompts)} helpful prompts...")
    helpful_data = generate_dataset(
        generator, helpful_prompts, constitution_type,
        is_harmful=False, num_revisions=args.num_revisions
    )
    
    # Combine and save
    all_data = red_team_data + helpful_data
    random.shuffle(all_data)
    
    output_file = output_dir / f"{constitution_type}_sl_dataset.jsonl"
    
    # Create backup before writing
    if output_file.exists():
        backup_file = output_dir / f"{constitution_type}_sl_dataset_backup.jsonl"
        import shutil
        shutil.copy2(output_file, backup_file)
        print(f"Created backup: {backup_file}")
    
    # Save with atomic write (temp file + rename)
    temp_file = output_dir / f"{constitution_type}_sl_dataset_temp.jsonl"
    with open(temp_file, 'w') as f:
        for item in all_data:
            json.dump(item, f)
            f.write('\n')
    
    # Atomic rename
    temp_file.rename(output_file)
    
    print(f"\n✅ Generated {len(all_data)} samples for {constitution_type}")
    print(f"   Red team: {len(red_team_data)}")
    print(f"   Helpful: {len(helpful_data)}")
    print(f"   Saved to: {output_file}")
    
    # Create additional backup
    backup_dir = output_dir.parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    backup_file = backup_dir / f"{constitution_type}_sl_dataset_FINAL.jsonl"
    import shutil
    shutil.copy2(output_file, backup_file)
    print(f"   Backup created: {backup_file}")
    
    print(f"\n🎉 CONSEQUENTIALIST DATASET GENERATION COMPLETED!")

if __name__ == "__main__":
    main()