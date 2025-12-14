#!/usr/bin/env python3
"""
Parse the Claude-generated deontological dataset into readable format
"""

import json
from pathlib import Path

def parse_claude_dataset(jsonl_path, output_path, sample_limit=None):
    """Parse Claude-generated JSONL dataset into readable markdown format"""
    
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
            if sample_limit and len(samples) >= sample_limit:
                break
    
    # Calculate statistics
    harmful_count = sum(1 for s in samples if s.get('critique_applied', False))
    helpful_count = sum(1 for s in samples if not s.get('critique_applied', False))
    
    markdown = []
    markdown.append(f"# Claude-Generated {jsonl_path.stem.replace('_', ' ').title()}")
    markdown.append(f"\n## Dataset Statistics")
    markdown.append(f"- **Total Samples**: {len(samples)}")
    markdown.append(f"- **Harmful (with critique)**: {harmful_count}")
    markdown.append(f"- **Helpful (no critique)**: {helpful_count}")
    
    # Check revision counts
    revision_counts = {}
    for s in samples:
        rev_count = len(s.get('revisions', []))
        revision_counts[rev_count] = revision_counts.get(rev_count, 0) + 1
    
    markdown.append(f"\n### Revision Distribution:")
    for count, freq in sorted(revision_counts.items()):
        markdown.append(f"- {count} revisions: {freq} samples")
    
    # Calculate average lengths
    avg_initial = sum(len(s.get('initial_response', '')) for s in samples) / len(samples) if samples else 0
    avg_final = sum(len(s.get('response', '')) for s in samples) / len(samples) if samples else 0
    
    markdown.append(f"\n### Response Lengths:")
    markdown.append(f"- **Avg Initial Response**: {avg_initial:.0f} chars")
    markdown.append(f"- **Avg Final Response**: {avg_final:.0f} chars")
    markdown.append(f"- **Length Increase**: {((avg_final/avg_initial - 1) * 100):.1f}%")
    
    markdown.append("\n---\n")
    markdown.append("## Sample Entries\n")
    
    # Show some examples
    num_to_show = min(10, len(samples)) if not sample_limit else min(sample_limit, len(samples))
    
    # Show some harmful examples
    harmful_samples = [s for s in samples if s.get('critique_applied', False)][:5]
    helpful_samples = [s for s in samples if not s.get('critique_applied', False)][:5]
    
    markdown.append("### Harmful Prompts (with critique)\n")
    for i, sample in enumerate(harmful_samples, 1):
        markdown.append(f"#### Example {i}")
        markdown.append(f"\n**Prompt:**")
        markdown.append(f"```")
        prompt = sample.get('prompt', 'N/A')
        markdown.append(prompt[:300] + ('...' if len(prompt) > 300 else ''))
        markdown.append(f"```")
        
        if sample.get('initial_response'):
            markdown.append(f"\n**Initial Response:**")
            markdown.append(f"```")
            initial = sample['initial_response']
            markdown.append(initial[:250] + ('...' if len(initial) > 250 else ''))
            markdown.append(f"```")
        
        markdown.append(f"\n**Final Response (after {len(sample.get('revisions', []))} revisions):**")
        markdown.append(f"```")
        final = sample.get('response', 'N/A')
        markdown.append(final[:400] + ('...' if len(final) > 400 else ''))
        markdown.append(f"```")
        
        # Show revision principles used
        if sample.get('revisions'):
            principles = [rev.get('principle_used', 'N/A') for rev in sample['revisions']]
            markdown.append(f"\n**Principles Applied:** {', '.join(principles)}")
        
        markdown.append("\n---\n")
    
    markdown.append("### Helpful Prompts (no critique needed)\n")
    for i, sample in enumerate(helpful_samples, 1):
        markdown.append(f"#### Example {i}")
        markdown.append(f"\n**Prompt:**")
        markdown.append(f"```")
        prompt = sample.get('prompt', 'N/A')
        markdown.append(prompt[:300] + ('...' if len(prompt) > 300 else ''))
        markdown.append(f"```")
        
        markdown.append(f"\n**Response:**")
        markdown.append(f"```")
        response = sample.get('response', 'N/A')
        markdown.append(response[:400] + ('...' if len(response) > 400 else ''))
        markdown.append(f"```")
        
        markdown.append("\n---\n")
    
    # Save markdown
    with open(output_path, 'w') as f:
        f.write('\n'.join(markdown))
    
    print(f"✅ Parsed {len(samples)} samples -> {output_path}")
    return samples

def main():
    # Parse the Claude-generated deontological dataset
    dataset_path = Path("/Users/leeyoungchan/development/AI_LAB/Constitutional_AI_Project_v2/deontological_sl_dataset.jsonl")
    
    if dataset_path.exists():
        # First create a summary
        output_path = dataset_path.parent / "claude_deontological_summary.md"
        samples = parse_claude_dataset(dataset_path, output_path, sample_limit=None)
        
        # Also create a full parse if you want
        full_output = dataset_path.parent / "claude_deontological_FULL.md"
        print(f"\nDataset has {len(samples)} total samples")
        print(f"Summary saved to: {output_path}")
        
        # Quick quality check
        print("\n📊 Quality Analysis:")
        
        # Check for any weird patterns
        empty_responses = sum(1 for s in samples if not s.get('response'))
        short_responses = sum(1 for s in samples if len(s.get('response', '')) < 50)
        very_long = sum(1 for s in samples if len(s.get('response', '')) > 1000)
        
        print(f"- Empty responses: {empty_responses}")
        print(f"- Very short (<50 chars): {short_responses}")
        print(f"- Very long (>1000 chars): {very_long}")
        
        # Check revision effectiveness
        harmful_with_revisions = [s for s in samples if s.get('critique_applied') and s.get('revisions')]
        if harmful_with_revisions:
            improved = 0
            for s in harmful_with_revisions:
                initial = s.get('initial_response', '')
                final = s.get('response', '')
                # Simple check: did it add ethical language?
                ethical_terms = ['ethical', 'moral', 'principle', 'duty', 'right', 'wrong', 'cannot', 'refuse']
                initial_ethical = sum(1 for term in ethical_terms if term in initial.lower())
                final_ethical = sum(1 for term in ethical_terms if term in final.lower())
                if final_ethical > initial_ethical:
                    improved += 1
            
            print(f"- Harmful prompts with improved ethical language: {improved}/{len(harmful_with_revisions)} ({improved/len(harmful_with_revisions)*100:.1f}%)")
    else:
        print(f"❌ Dataset not found at {dataset_path}")

if __name__ == "__main__":
    main()