#!/usr/bin/env python3
"""
Parse the cleaned deontological and consequentialist datasets into readable format
"""

import json
from pathlib import Path

def parse_cleaned_dataset(jsonl_path, output_path, sample_limit=None):
    """Parse cleaned JSONL dataset into readable markdown format"""
    
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
    markdown.append(f"# Cleaned {jsonl_path.stem.replace('_', ' ').title()}")
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
    if avg_initial > 0:
        markdown.append(f"- **Length Change**: {((avg_final/avg_initial - 1) * 100):.1f}%")
    
    # Check for unique patterns in cleaned data
    markdown.append(f"\n### Data Patterns:")
    
    # Check if responses contain constitution-specific language
    deont_terms = ['categorical', 'duty', 'principle', 'moral law', 'universalizable']
    conseq_terms = ['consequences', 'utility', 'outcome', 'greater good', 'maximize']
    
    if 'deontological' in str(jsonl_path).lower():
        terms_to_check = deont_terms
        philosophy = "Deontological"
    else:
        terms_to_check = conseq_terms
        philosophy = "Consequentialist"
    
    responses_with_terms = 0
    for s in samples[:100]:  # Check first 100
        response = s.get('response', '').lower()
        if any(term in response for term in terms_to_check):
            responses_with_terms += 1
    
    markdown.append(f"- **{philosophy} language in responses**: {responses_with_terms}/100 samples checked")
    
    # Check if there are any empty or very short responses
    empty_responses = sum(1 for s in samples if not s.get('response'))
    short_responses = sum(1 for s in samples if 0 < len(s.get('response', '')) < 50)
    very_long = sum(1 for s in samples if len(s.get('response', '')) > 1000)
    
    markdown.append(f"- **Empty responses**: {empty_responses}")
    markdown.append(f"- **Very short (<50 chars)**: {short_responses}")
    markdown.append(f"- **Very long (>1000 chars)**: {very_long}")
    
    markdown.append("\n---\n")
    markdown.append("## Sample Entries\n")
    
    # Show some examples
    num_to_show = min(10, len(samples)) if not sample_limit else min(sample_limit, len(samples))
    
    # Show some harmful examples
    harmful_samples = [s for s in samples if s.get('critique_applied', False)][:5]
    helpful_samples = [s for s in samples if not s.get('critique_applied', False)][:5]
    
    if harmful_samples:
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
            
            # Show revision principles used if available
            if sample.get('revisions'):
                principles = []
                for rev in sample['revisions']:
                    if 'principle_used' in rev:
                        principles.append(rev['principle_used'][:80] + '...')
                if principles:
                    markdown.append(f"\n**Principles Applied:** {'; '.join(principles[:2])}")
            
            markdown.append("\n---\n")
    
    if helpful_samples:
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
    
    print(f"✅ Parsed {len(samples)} samples from {jsonl_path.name}")
    print(f"   Output: {output_path}")
    return samples

def compare_datasets(deont_samples, conseq_samples):
    """Compare the two philosophical datasets"""
    print("\n📊 Comparative Analysis:")
    print("=" * 50)
    
    # Basic stats comparison
    print(f"\n{'Metric':<30} {'Deontological':<15} {'Consequentialist':<15}")
    print("-" * 60)
    
    print(f"{'Total Samples':<30} {len(deont_samples):<15} {len(conseq_samples):<15}")
    
    deont_harmful = sum(1 for s in deont_samples if s.get('critique_applied', False))
    conseq_harmful = sum(1 for s in conseq_samples if s.get('critique_applied', False))
    print(f"{'Harmful (with critique)':<30} {deont_harmful:<15} {conseq_harmful:<15}")
    
    deont_avg_initial = sum(len(s.get('initial_response', '')) for s in deont_samples) / len(deont_samples) if deont_samples else 0
    conseq_avg_initial = sum(len(s.get('initial_response', '')) for s in conseq_samples) / len(conseq_samples) if conseq_samples else 0
    print(f"{'Avg Initial Length':<30} {deont_avg_initial:<15.0f} {conseq_avg_initial:<15.0f}")
    
    deont_avg_final = sum(len(s.get('response', '')) for s in deont_samples) / len(deont_samples) if deont_samples else 0
    conseq_avg_final = sum(len(s.get('response', '')) for s in conseq_samples) / len(conseq_samples) if conseq_samples else 0
    print(f"{'Avg Final Length':<30} {deont_avg_final:<15.0f} {conseq_avg_final:<15.0f}")
    
    # Check for philosophical language
    deont_terms = ['categorical', 'duty', 'principle', 'moral law', 'universalizable', 'Article']
    conseq_terms = ['consequences', 'utility', 'outcome', 'greater good', 'maximize', 'benefits']
    
    deont_with_philosophy = sum(1 for s in deont_samples[:100] 
                               if any(term in s.get('response', '').lower() for term in deont_terms))
    conseq_with_philosophy = sum(1 for s in conseq_samples[:100]
                                if any(term in s.get('response', '').lower() for term in conseq_terms))
    
    print(f"{'Has Philosophy Terms (/100)':<30} {deont_with_philosophy:<15} {conseq_with_philosophy:<15}")
    
    # Check revision patterns
    deont_avg_rev = sum(len(s.get('revisions', [])) for s in deont_samples if s.get('critique_applied')) / deont_harmful if deont_harmful else 0
    conseq_avg_rev = sum(len(s.get('revisions', [])) for s in conseq_samples if s.get('critique_applied')) / conseq_harmful if conseq_harmful else 0
    print(f"{'Avg Revisions (harmful only)':<30} {deont_avg_rev:<15.1f} {conseq_avg_rev:<15.1f}")

def main():
    # Parse both cleaned datasets
    deont_path = Path("deontological_sl_dataset_cleaned.jsonl")
    conseq_path = Path("consequentialist_sl_dataset_cleaned.jsonl")
    
    deont_samples = None
    conseq_samples = None
    
    if deont_path.exists():
        output_path = Path("deontological_cleaned_summary.md")
        deont_samples = parse_cleaned_dataset(deont_path, output_path)
    else:
        print(f"❌ Not found: {deont_path}")
    
    if conseq_path.exists():
        output_path = Path("consequentialist_cleaned_summary.md")
        conseq_samples = parse_cleaned_dataset(conseq_path, output_path)
    else:
        print(f"❌ Not found: {conseq_path}")
    
    # Compare if both exist
    if deont_samples and conseq_samples:
        compare_datasets(deont_samples, conseq_samples)
    
    # Quality check
    print("\n🔍 Quality Checks:")
    
    for name, samples in [("Deontological", deont_samples), ("Consequentialist", conseq_samples)]:
        if not samples:
            continue
            
        print(f"\n{name}:")
        
        # Check for duplicates
        prompts = [s.get('prompt', '') for s in samples]
        unique_prompts = len(set(prompts))
        print(f"  - Unique prompts: {unique_prompts}/{len(prompts)}")
        
        # Check for proper critique application
        critiqued = [s for s in samples if s.get('critique_applied')]
        if critiqued:
            with_revisions = sum(1 for s in critiqued if len(s.get('revisions', [])) > 0)
            print(f"  - Critiqued samples with revisions: {with_revisions}/{len(critiqued)}")
        
        # Check for response quality
        responses = [s.get('response', '') for s in samples if s.get('response')]
        if responses:
            avg_words = sum(len(r.split()) for r in responses) / len(responses)
            print(f"  - Avg words per response: {avg_words:.0f}")

if __name__ == "__main__":
    main()