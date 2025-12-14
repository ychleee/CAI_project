import json
from pathlib import Path

def parse_full(jsonl_path, output_path):
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    name = "Deontological" if "deont" in jsonl_path else "Consequentialist"
    
    with open(output_path, 'w') as f:
        f.write(f"# FULL Claude-Generated {name} Dataset\n\n")
        f.write(f"**Total Samples:** {len(samples)}\n\n")
        
        harmful = sum(1 for s in samples if s.get('critique_applied'))
        f.write(f"- Harmful (with critique): {harmful}\n")
        f.write(f"- Helpful: {len(samples) - harmful}\n\n")
        
        for i, s in enumerate(samples, 1):
            f.write(f"## Sample {i}/{len(samples)}\n")
            f.write(f"**Type:** {'Harmful (critiqued)' if s.get('critique_applied') else 'Helpful'}\n\n")
            f.write(f"**Prompt:**\n```\n{s['prompt']}\n```\n\n")
            
            if s.get('initial_response'):
                f.write(f"**Initial Response:**\n```\n{s['initial_response']}\n```\n\n")
            
            if s.get('revisions'):
                f.write(f"**{len(s['revisions'])} Revision Rounds Applied**\n\n")
            
            f.write(f"**Final Response:**\n```\n{s['response']}\n```\n\n")
            f.write("---\n\n")
    
    print(f"✅ Created {output_path}: {len(samples)} samples")

# Parse Claude datasets
parse_full("dataset_generation_by_claude/cai_claude_output/deontological_sl_dataset.jsonl",
           "deontological_CLAUDE_FULL.md")
parse_full("dataset_generation_by_claude/cai_claude_output/consequentialist_sl_dataset.jsonl", 
           "consequentialist_CLAUDE_FULL.md")
