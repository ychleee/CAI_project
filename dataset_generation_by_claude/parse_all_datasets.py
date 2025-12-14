#\!/usr/bin/env python3
"""
Parse Claude-generated Constitutional AI datasets into readable markdown format
"""

import json
from pathlib import Path
from datetime import datetime

def parse_claude_dataset(jsonl_path: Path, output_md_path: Path, full_output_path: Path = None):
    """Parse Claude JSONL dataset to readable markdown"""
    
    print(f"📖 Parsing {jsonl_path.name}...")
    
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Calculate statistics
    harmful_count = sum(1 for s in samples if s.get('critique_applied', False))
    helpful_count = sum(1 for s in samples if not s.get('critique_applied', False))
    
    # Start markdown content for summary
    md_lines = []
    md_lines.append(f"# Claude-Generated {jsonl_path.stem.replace('_', ' ').title()}")
    md_lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    
    md_lines.append(f"\n## 📊 Dataset Statistics\n")
    md_lines.append(f"- **Total Samples**: {len(samples)}")
    md_lines.append(f"- **Harmful Prompts (with critique)**: {harmful_count}")
    md_lines.append(f"- **Helpful Prompts (no critique)**: {helpful_count}")
    
    # Analyze revisions
    if harmful_count > 0:
        revision_counts = {}
        for s in samples:
            if s.get('critique_applied', False):
                rev_count = len(s.get('revisions', []))
                revision_counts[rev_count] = revision_counts.get(rev_count, 0) + 1
        
        md_lines.append(f"\n### Revision Distribution:")
        for count in sorted(revision_counts.keys()):
            md_lines.append(f"- {count} revisions: {revision_counts[count]} samples")
    
    # Response length analysis
    initial_lengths = [len(s.get('initial_response', '')) for s in samples if s.get('initial_response')]
    final_lengths = [len(s.get('response', '')) for s in samples if s.get('response')]
    
    if initial_lengths:
        avg_initial = sum(initial_lengths) / len(initial_lengths)
        avg_final = sum(final_lengths) / len(final_lengths) if final_lengths else 0
        
        md_lines.append(f"\n### Response Lengths:")
        md_lines.append(f"- **Average Initial Response**: {avg_initial:.0f} characters")
        md_lines.append(f"- **Average Final Response**: {avg_final:.0f} characters")
        if avg_initial > 0:
            change = ((avg_final / avg_initial - 1) * 100)
            md_lines.append(f"- **Length Change**: {change:+.1f}%")
    
    # Sample entries - show first 5 of each type
    harmful_samples = [s for s in samples if s.get('critique_applied', False)][:5]
    helpful_samples = [s for s in samples if not s.get('critique_applied', False)][:5]
    
    md_lines.append(f"\n## 📝 Sample Entries (First 5 of each type)\n")
    
    if harmful_samples:
        md_lines.append(f"### 🚫 Harmful Prompts (with Constitutional Critique)\n")
        
        for i, sample in enumerate(harmful_samples, 1):
            md_lines.append(f"#### Harmful Example {i}\n")
            md_lines.append("**Prompt:**")
            md_lines.append("```")
            prompt = sample.get('prompt', 'N/A')
            md_lines.append(prompt[:400] + ('...' if len(prompt) > 400 else ''))
            md_lines.append("```\n")
            
            if sample.get('initial_response'):
                md_lines.append("**Initial Response (Before Critique):**")
                md_lines.append("```")
                initial = sample['initial_response']
                md_lines.append(initial[:300] + ('...' if len(initial) > 300 else ''))
                md_lines.append("```\n")
            
            # Show number of revisions
            revisions = sample.get('revisions', [])
            if revisions:
                md_lines.append(f"**Applied {len(revisions)} rounds of critique and revision**\n")
            
            md_lines.append("**Final Response (After Constitutional Revision):**")
            md_lines.append("```")
            final = sample.get('response', 'N/A')
            md_lines.append(final[:400] + ('...' if len(final) > 400 else ''))
            md_lines.append("```\n")
            md_lines.append("---\n")
    
    if helpful_samples:
        md_lines.append(f"### ✅ Helpful Prompts (No Critique Needed)\n")
        
        for i, sample in enumerate(helpful_samples, 1):
            md_lines.append(f"#### Helpful Example {i}\n")
            md_lines.append("**Prompt:**")
            md_lines.append("```")
            prompt = sample.get('prompt', 'N/A')
            md_lines.append(prompt[:300] + ('...' if len(prompt) > 300 else ''))
            md_lines.append("```\n")
            
            md_lines.append("**Response:**")
            md_lines.append("```")
            response = sample.get('response', 'N/A')
            md_lines.append(response[:400] + ('...' if len(response) > 400 else ''))
            md_lines.append("```\n")
            md_lines.append("---\n")
    
    # Save summary to file
    with open(output_md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    
    print(f"✅ Summary saved to {output_md_path}")
    
    return len(samples)

def main():
    # Parse both datasets
    datasets = [
        ("dataset_generation_by_claude/cai_claude_output/deontological_sl_dataset.jsonl", 
         "deontological_sl_dataset_readable.md"),
        ("dataset_generation_by_claude/cai_claude_output/consequentialist_sl_dataset.jsonl",
         "consequentialist_sl_dataset_readable.md")
    ]
    
    total_samples = 0
    for input_path, output_path in datasets:
        input_file = Path(input_path)
        output_file = Path(output_path)
        
        if input_file.exists():
            count = parse_claude_dataset(input_file, output_file)
            total_samples += count
        else:
            print(f"❌ File not found: {input_file}")
    
    print(f"\n✨ Total samples parsed: {total_samples}")

if __name__ == "__main__":
    main()
EOFILE < /dev/null