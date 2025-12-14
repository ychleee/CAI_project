#!/usr/bin/env python3
"""
Parse ALL samples from JSONL datasets into readable markdown format
"""

import json
from pathlib import Path

def parse_jsonl_to_markdown(jsonl_path, output_path, max_samples=None):
    """Parse JSONL dataset into readable markdown format"""
    
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples:
                break
    
    markdown = []
    markdown.append(f"# {jsonl_path.stem.replace('_', ' ').title()}")
    markdown.append(f"\nTotal samples: {len(samples)}\n")
    markdown.append("---\n")
    
    for i, sample in enumerate(samples, 1):
        markdown.append(f"## Sample {i}")
        
        # Prompt
        markdown.append(f"\n### Prompt:")
        markdown.append(f"```")
        prompt = sample.get('prompt', 'N/A')
        # Show full prompt if under 500 chars, otherwise truncate
        if len(prompt) <= 500:
            markdown.append(prompt)
        else:
            markdown.append(prompt[:500] + '...')
        markdown.append(f"```")
        
        # Initial Response (if different from final)
        if 'initial_response' in sample and sample['initial_response'] != sample.get('response'):
            markdown.append(f"\n### Initial Response:")
            markdown.append(f"```")
            initial = sample['initial_response']
            if len(initial) <= 400:
                markdown.append(initial)
            else:
                markdown.append(initial[:400] + '...')
            markdown.append(f"```")
        
        # Final Response
        markdown.append(f"\n### Final Response:")
        markdown.append(f"```")
        response = sample.get('response', sample.get('final_response', 'N/A'))
        if len(response) <= 600:
            markdown.append(response)
        else:
            markdown.append(response[:600] + '...')
        markdown.append(f"```")
        
        # Metadata
        markdown.append(f"\n### Metadata:")
        markdown.append(f"- **Constitution Type**: {sample.get('constitution_type', 'N/A')}")
        markdown.append(f"- **Critique Applied**: {sample.get('critique_applied', 'N/A')}")
        markdown.append(f"- **Is Harmful Prompt**: {sample.get('is_harmful_prompt', 'Unknown')}")
        markdown.append(f"- **Number of Revisions**: {len(sample.get('revisions', []))}")
        
        # Show revision summary if present
        if sample.get('revisions'):
            markdown.append(f"\n### Revision Summary:")
            markdown.append(f"- **Total Rounds**: {len(sample['revisions'])}")
            principles_used = [rev.get('principle_used', 'N/A') for rev in sample['revisions']]
            markdown.append(f"- **Principles Applied**: {', '.join(set(principles_used))}")
        
        markdown.append("\n---\n")
    
    # Save markdown
    with open(output_path, 'w') as f:
        f.write('\n'.join(markdown))
    
    print(f"✅ Parsed ALL {len(samples)} samples from {jsonl_path.name} -> {output_path.name}")
    return samples

def generate_summary_stats(samples, dataset_name):
    """Generate summary statistics for a dataset"""
    stats = []
    stats.append(f"\n## Summary Statistics for {dataset_name}")
    stats.append(f"- **Total Samples**: {len(samples)}")
    
    harmful_count = sum(1 for s in samples if s.get('is_harmful_prompt', False))
    helpful_count = len(samples) - harmful_count
    critiqued_count = sum(1 for s in samples if s.get('critique_applied', False))
    
    stats.append(f"- **Harmful Prompts**: {harmful_count}")
    stats.append(f"- **Helpful Prompts**: {helpful_count}")
    stats.append(f"- **Samples with Critique Applied**: {critiqued_count}")
    stats.append(f"- **Samples without Critique**: {len(samples) - critiqued_count}")
    
    # Average response lengths
    avg_initial_len = sum(len(s.get('initial_response', '')) for s in samples) / len(samples) if samples else 0
    avg_final_len = sum(len(s.get('response', s.get('final_response', ''))) for s in samples) / len(samples) if samples else 0
    
    stats.append(f"- **Avg Initial Response Length**: {avg_initial_len:.0f} chars")
    stats.append(f"- **Avg Final Response Length**: {avg_final_len:.0f} chars")
    
    # Revision stats
    samples_with_revisions = [s for s in samples if s.get('revisions')]
    if samples_with_revisions:
        avg_revisions = sum(len(s['revisions']) for s in samples_with_revisions) / len(samples_with_revisions)
        stats.append(f"- **Avg Revisions per Critiqued Sample**: {avg_revisions:.1f}")
    
    return '\n'.join(stats)

def main():
    # Find datasets
    base_dir = Path("/Users/leeyoungchan/development/AI_LAB/Constitutional_AI_Project_v2")
    
    datasets = [
        (base_dir / "deontological_sl_dataset.jsonl", "Deontological"),
        (base_dir / "consequentialist_sl_dataset.jsonl", "Consequentialist")
    ]
    
    all_stats = []
    
    for dataset_path, name in datasets:
        if dataset_path.exists():
            # Parse ALL samples
            output_path = dataset_path.parent / f"{dataset_path.stem}_FULL_readable.md"
            samples = parse_jsonl_to_markdown(dataset_path, output_path, max_samples=None)
            
            # Generate stats
            stats = generate_summary_stats(samples, name)
            all_stats.append(stats)
            print(stats)
        else:
            print(f"❌ Not found: {dataset_path}")
    
    # Save combined statistics
    stats_path = base_dir / "dataset_statistics.md"
    with open(stats_path, 'w') as f:
        f.write("# Constitutional AI Dataset Statistics\n\n")
        f.write('\n'.join(all_stats))
    print(f"\n📊 Statistics saved to {stats_path}")

if __name__ == "__main__":
    main()