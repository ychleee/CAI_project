import json
from pathlib import Path

def parse_dataset(jsonl_path, output_md_path):
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    harmful = sum(1 for s in samples if s.get('critique_applied', False))
    helpful = len(samples) - harmful
    
    md = []
    name = "Deontological" if "deont" in jsonl_path else "Consequentialist"
    md.append(f"# Claude {name} Dataset\n")
    md.append(f"**Total:** {len(samples)} samples")
    md.append(f"**Harmful (critiqued):** {harmful}")
    md.append(f"**Helpful:** {helpful}\n")
    
    # Show 3 harmful examples
    md.append("## Harmful Examples\n")
    harm_samples = [s for s in samples if s.get('critique_applied')][:3]
    for i, s in enumerate(harm_samples, 1):
        md.append(f"### Example {i}")
        md.append(f"**Prompt:** {s['prompt'][:200]}...")
        md.append(f"**Initial:** {s.get('initial_response', '')[:150]}...")
        md.append(f"**Final:** {s['response'][:200]}...")
        md.append(f"**Revisions:** {len(s.get('revisions', []))}\n")
    
    # Show 3 helpful examples
    md.append("## Helpful Examples\n")
    help_samples = [s for s in samples if not s.get('critique_applied')][:3]
    for i, s in enumerate(help_samples, 1):
        md.append(f"### Example {i}")
        md.append(f"**Prompt:** {s['prompt'][:150]}...")
        md.append(f"**Response:** {s['response'][:200]}...\n")
    
    with open(output_md_path, 'w') as f:
        f.write('\n'.join(md))
    
    print(f"✅ {output_md_path}: {len(samples)} samples")

# Parse both
files = [
    ("dataset_generation_by_claude/cai_claude_output/deontological_sl_dataset.jsonl",
     "deontological_sl_dataset_readable.md"),
    ("dataset_generation_by_claude/cai_claude_output/consequentialist_sl_dataset.jsonl",
     "consequentialist_sl_dataset_readable.md")
]

for inp, out in files:
    if Path(inp).exists():
        parse_dataset(inp, out)
