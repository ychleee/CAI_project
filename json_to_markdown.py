import json
import textwrap

def truncate_text(text, max_length=500):
    """Truncate text to max_length characters with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def format_revision(revision, index):
    """Format a single revision entry"""
    principle = truncate_text(revision.get('principle', ''), 200)
    response = truncate_text(revision.get('response', ''), 400)
    
    return f"""
#### Revision {index}
**Principle Applied:** {principle}

**Revised Response:**
> {response}
"""

def json_to_markdown(json_file, output_file, dataset_type):
    """Convert JSON dataset to readable Markdown format"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    markdown_content = f"""# {dataset_type.title()} Constitutional AI Dataset

## Overview
- **Total Entries:** {len(data)}
- **Constitution Type:** {dataset_type}
- **Dataset Structure:** Each entry contains a prompt, initial response, constitutional revisions, and final response

---

## Dataset Entries

"""
    
    for i, entry in enumerate(data, 1):
        # Extract fields
        prompt = entry.get('prompt', '')
        initial_response = truncate_text(entry.get('initial_response', ''), 600)
        final_response = truncate_text(entry.get('final_response', ''), 800)
        num_revisions = entry.get('num_revisions', 0)
        is_harmful = entry.get('is_harmful_prompt', False)
        source = entry.get('source', 'unknown')
        category = entry.get('category', 'unknown')
        revisions = entry.get('revisions', [])
        
        # Format entry
        entry_md = f"""
### Entry {i}

**Metadata:**
- Source: `{source}`
- Category: `{category}`
- Harmful Prompt: `{is_harmful}`
- Number of Revisions: `{num_revisions}`

**Prompt:**
> {prompt}

<details>
<summary><strong>Initial Response</strong> (Click to expand)</summary>

{initial_response}

</details>

<details>
<summary><strong>Constitutional Revisions</strong> ({num_revisions} rounds) (Click to expand)</summary>
"""
        
        # Add revisions
        for j, revision in enumerate(revisions[:2], 1):  # Show first 2 revisions
            entry_md += format_revision(revision, j)
        
        if len(revisions) > 2:
            entry_md += f"\n*... and {len(revisions) - 2} more revisions*\n"
        
        entry_md += """
</details>

**Final Response:**
> {final_response}

---
""".format(final_response=final_response)
        
        markdown_content += entry_md
        
        # Add table of contents after first 5 entries
        if i == 5:
            markdown_content += f"""
<details>
<summary><strong>Quick Navigation</strong> (Entries 6-{len(data)})</summary>

| Entry | Prompt Preview | Category |
|-------|----------------|----------|
"""
            for j in range(6, min(len(data) + 1, 21)):  # Show next 15 entries in TOC
                if j <= len(data):
                    entry_preview = data[j-1].get('prompt', '')[:50] + "..."
                    entry_category = data[j-1].get('category', 'unknown')
                    markdown_content += f"| [{j}](#entry-{j}) | {entry_preview} | {entry_category} |\n"
            
            if len(data) > 20:
                markdown_content += f"\n*... and {len(data) - 20} more entries*\n"
            
            markdown_content += "\n</details>\n\n---\n"
    
    # Add statistics section at the end
    harmful_count = sum(1 for entry in data if entry.get('is_harmful_prompt', False))
    categories = {}
    sources = {}
    
    for entry in data:
        cat = entry.get('category', 'unknown')
        src = entry.get('source', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
        sources[src] = sources.get(src, 0) + 1
    
    markdown_content += f"""

## Dataset Statistics

### Overall Metrics
- **Total Entries:** {len(data)}
- **Harmful Prompts:** {harmful_count} ({harmful_count/len(data)*100:.1f}%)
- **Safe Prompts:** {len(data) - harmful_count} ({(len(data) - harmful_count)/len(data)*100:.1f}%)

### Categories Distribution
| Category | Count | Percentage |
|----------|-------|------------|
"""
    
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        markdown_content += f"| {cat} | {count} | {count/len(data)*100:.1f}% |\n"
    
    markdown_content += """

### Sources Distribution
| Source | Count | Percentage |
|--------|-------|------------|
"""
    
    for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        markdown_content += f"| {src} | {count} | {count/len(data)*100:.1f}% |\n"
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(markdown_content)
    
    print(f"Successfully created {output_file}")
    print(f"  - Processed {len(data)} entries")
    print(f"  - File size: {len(markdown_content):,} characters")

if __name__ == "__main__":
    # Convert consequentialist dataset
    json_to_markdown(
        'consequentialist_formatted.json',
        'consequentialist_dataset.md',
        'consequentialist'
    )
    
    # Convert deontological dataset
    json_to_markdown(
        'deontological_formatted.json',
        'deontological_dataset.md',
        'deontological'
    )