#!/usr/bin/env python3
"""
Analyze the quality of generated Constitutional AI datasets
Check for decisive judgments, framework differences, and quality issues
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def load_dataset(filepath: Path) -> List[Dict]:
    """Load JSONL dataset"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def analyze_decisiveness(response: str) -> Dict:
    """Check if response makes decisive moral judgments"""
    
    # Hedging patterns (bad - we want to avoid these)
    hedging_patterns = [
        r'it depends',
        r'one must consider',
        r'carefully weigh',
        r'both sides',
        r'on one hand',
        r'it\'s complicated',
        r'there are arguments',
        r'some would say',
        r'perspectives vary',
        r'context matters'
    ]
    
    # Decisive patterns (good - we want these)
    decisive_patterns = [
        r'morally required',
        r'morally forbidden',
        r'categorically wrong',
        r'absolutely justified',
        r'clearly unethical',
        r'must not',
        r'obligation to',
        r'duty requires',
        r'consequences justify',
        r'maximizes welfare',
        r'violates the principle',
        r'the right action is'
    ]
    
    # Count patterns
    hedging_count = sum(1 for p in hedging_patterns if re.search(p, response.lower()))
    decisive_count = sum(1 for p in decisive_patterns if re.search(p, response.lower()))
    
    # Check for clear yes/no answers
    has_clear_answer = any(phrase in response.lower() for phrase in [
        'yes,', 'no,', 'it is justified', 'it is not justified',
        'this is acceptable', 'this is unacceptable',
        'morally permissible', 'morally impermissible'
    ])
    
    return {
        'hedging_count': hedging_count,
        'decisive_count': decisive_count,
        'has_clear_answer': has_clear_answer,
        'decisiveness_score': decisive_count - hedging_count
    }

def analyze_framework_patterns(response: str, framework: str) -> Dict:
    """Check for framework-specific language"""
    
    deont_patterns = [
        'duty', 'obligation', 'categorical', 'universal', 'principle',
        'rights', 'dignity', 'means to an end', 'moral law', 'imperative',
        'inherently wrong', 'regardless of consequences', 'inviolable'
    ]
    
    conseq_patterns = [
        'consequence', 'outcome', 'result', 'utility', 'welfare',
        'maximize', 'minimize harm', 'greater good', 'net benefit',
        'aggregate', 'overall happiness', 'best results', 'total well-being'
    ]
    
    if framework == 'deontological':
        correct_patterns = deont_patterns
        incorrect_patterns = conseq_patterns
    else:
        correct_patterns = conseq_patterns
        incorrect_patterns = deont_patterns
    
    correct_count = sum(1 for p in correct_patterns if p in response.lower())
    incorrect_count = sum(1 for p in incorrect_patterns if p in response.lower())
    
    return {
        'correct_framework_patterns': correct_count,
        'incorrect_framework_patterns': incorrect_count,
        'framework_alignment': correct_count - incorrect_count
    }

def analyze_revision_quality(entry: Dict) -> Dict:
    """Analyze the quality of revisions"""
    
    initial = entry.get('initial_response', '')
    final = entry.get('response', '')
    revisions = entry.get('revisions', [])
    
    # Check if revisions actually changed the response
    revision_changes = []
    prev_response = initial
    
    for rev in revisions:
        curr_response = rev.get('revised_response', '')
        if curr_response != prev_response:
            revision_changes.append(True)
        else:
            revision_changes.append(False)
        prev_response = curr_response
    
    # Analyze initial vs final
    initial_analysis = analyze_decisiveness(initial)
    final_analysis = analyze_decisiveness(final)
    
    # Check if harmful content was addressed
    harmful_patterns = ['hack', 'steal', 'illegal', 'harm', 'kill', 'destroy']
    initial_harmful = any(p in initial.lower() for p in harmful_patterns)
    final_refuses = any(phrase in final.lower() for phrase in [
        'cannot', 'won\'t', 'inappropriate', 'unethical', 'refuse'
    ])
    
    return {
        'revisions_made_changes': sum(revision_changes),
        'total_revisions': len(revisions),
        'initial_decisiveness': initial_analysis['decisiveness_score'],
        'final_decisiveness': final_analysis['decisiveness_score'],
        'decisiveness_improvement': final_analysis['decisiveness_score'] - initial_analysis['decisiveness_score'],
        'harmful_content_addressed': initial_harmful and final_refuses
    }

def compare_frameworks(deont_data: List[Dict], conseq_data: List[Dict]) -> Dict:
    """Compare responses between frameworks on same prompts"""
    
    # Match responses by prompt
    comparisons = []
    
    for d_entry in deont_data:
        prompt = d_entry['prompt']
        
        # Find matching consequentialist entry
        c_entry = next((c for c in conseq_data if c['prompt'] == prompt), None)
        
        if c_entry:
            d_response = d_entry['response']
            c_response = c_entry['response']
            
            # Check if they give different judgments
            d_accepts = any(w in d_response.lower() for w in ['acceptable', 'permissible', 'justified', 'allowed'])
            c_accepts = any(w in c_response.lower() for w in ['acceptable', 'permissible', 'justified', 'allowed'])
            
            d_refuses = any(w in d_response.lower() for w in ['unacceptable', 'impermissible', 'forbidden', 'wrong'])
            c_refuses = any(w in c_response.lower() for w in ['unacceptable', 'impermissible', 'forbidden', 'wrong'])
            
            disagree = (d_accepts != c_accepts) or (d_refuses != c_refuses)
            
            comparisons.append({
                'prompt': prompt[:100] + '...',
                'disagree': disagree,
                'deont_accepts': d_accepts,
                'conseq_accepts': c_accepts
            })
    
    disagreement_rate = sum(1 for c in comparisons if c['disagree']) / len(comparisons) if comparisons else 0
    
    return {
        'total_compared': len(comparisons),
        'disagreement_rate': disagreement_rate,
        'comparisons': comparisons[:3]  # Sample for display
    }

def print_quality_report(deont_data: List[Dict], conseq_data: List[Dict]):
    """Print comprehensive quality report"""
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}📊 DATASET QUALITY ANALYSIS REPORT{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")
    
    # Analyze each framework
    for data, framework, color in [(deont_data, 'DEONTOLOGICAL', Colors.BLUE),
                                    (conseq_data, 'CONSEQUENTIALIST', Colors.GREEN)]:
        
        print(f"{color}{Colors.BOLD}[{framework}] - {len(data)} samples{Colors.RESET}")
        print("-"*40)
        
        # Decisiveness analysis
        decisiveness_scores = []
        framework_scores = []
        revision_qualities = []
        
        for entry in data:
            response = entry.get('response', '')
            
            # Decisiveness
            dec_analysis = analyze_decisiveness(response)
            decisiveness_scores.append(dec_analysis['decisiveness_score'])
            
            # Framework alignment
            frame_analysis = analyze_framework_patterns(response, framework.lower())
            framework_scores.append(frame_analysis['framework_alignment'])
            
            # Revision quality
            rev_analysis = analyze_revision_quality(entry)
            revision_qualities.append(rev_analysis)
        
        # Calculate averages
        avg_decisiveness = sum(decisiveness_scores) / len(decisiveness_scores)
        avg_framework = sum(framework_scores) / len(framework_scores)
        avg_improvements = sum(r['decisiveness_improvement'] for r in revision_qualities) / len(revision_qualities)
        
        # Print metrics
        print(f"📈 Decisiveness Score: {Colors.YELLOW}{avg_decisiveness:.2f}{Colors.RESET}")
        print(f"   (Positive = decisive, Negative = hedging)")
        
        print(f"🎯 Framework Alignment: {Colors.YELLOW}{avg_framework:.2f}{Colors.RESET}")
        print(f"   (Higher = better framework language)")
        
        print(f"📝 Revision Improvement: {Colors.YELLOW}{avg_improvements:.2f}{Colors.RESET}")
        print(f"   (Positive = revisions made responses more decisive)")
        
        # Quality issues
        issues = []
        if avg_decisiveness < 0:
            issues.append(f"{Colors.RED}⚠️  Too much hedging - responses not decisive{Colors.RESET}")
        if avg_framework < 2:
            issues.append(f"{Colors.RED}⚠️  Weak framework-specific language{Colors.RESET}")
        if avg_improvements < 0.5:
            issues.append(f"{Colors.RED}⚠️  Revisions not improving decisiveness{Colors.RESET}")
        
        if issues:
            print(f"\n{Colors.BOLD}Issues Found:{Colors.RESET}")
            for issue in issues:
                print(f"  {issue}")
        else:
            print(f"\n{Colors.GREEN}✅ Quality looks good!{Colors.RESET}")
        
        print()
    
    # Framework comparison
    print(f"{Colors.MAGENTA}{Colors.BOLD}[FRAMEWORK COMPARISON]{Colors.RESET}")
    print("-"*40)
    
    comparison = compare_frameworks(deont_data, conseq_data)
    
    print(f"🔄 Disagreement Rate: {Colors.YELLOW}{comparison['disagreement_rate']:.1%}{Colors.RESET}")
    
    if comparison['disagreement_rate'] < 0.3:
        print(f"{Colors.RED}⚠️  Low disagreement - frameworks too similar{Colors.RESET}")
    else:
        print(f"{Colors.GREEN}✅ Good framework differentiation{Colors.RESET}")
    
    # Show sample comparisons
    print(f"\n{Colors.BOLD}Sample Disagreements:{Colors.RESET}")
    for comp in comparison['comparisons']:
        if comp['disagree']:
            print(f"  Prompt: {comp['prompt']}")
            print(f"    Deont: {'Accepts' if comp['deont_accepts'] else 'Refuses'}")
            print(f"    Conseq: {'Accepts' if comp['conseq_accepts'] else 'Refuses'}")

def show_examples(deont_data: List[Dict], conseq_data: List[Dict]):
    """Show specific examples for manual review"""
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}📝 EXAMPLE RESPONSES FOR REVIEW{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")
    
    # Find a harmful prompt example
    harmful_example = None
    for entry in deont_data:
        if any(word in entry['prompt'].lower() for word in ['hack', 'steal', 'illegal', 'harm']):
            harmful_example = entry
            break
    
    if harmful_example:
        prompt = harmful_example['prompt']
        
        # Find matching consequentialist response
        conseq_match = next((c for c in conseq_data if c['prompt'] == prompt), None)
        
        print(f"{Colors.BOLD}Example: Red Team Prompt{Colors.RESET}")
        print(f"{Colors.CYAN}Prompt:{Colors.RESET} {prompt}\n")
        
        print(f"{Colors.BLUE}[Deontological Response]:{Colors.RESET}")
        print(harmful_example['response'][:500])
        print()
        
        if conseq_match:
            print(f"{Colors.GREEN}[Consequentialist Response]:{Colors.RESET}")
            print(conseq_match['response'][:500])
            print()
        
        # Show revision process
        print(f"{Colors.BOLD}Revision Process (Deontological):{Colors.RESET}")
        print(f"{Colors.YELLOW}Initial:{Colors.RESET}")
        print(harmful_example.get('initial_response', '')[:300])
        print()
        
        if harmful_example.get('revisions'):
            last_revision = harmful_example['revisions'][-1]
            print(f"{Colors.YELLOW}After Constitutional Critique:{Colors.RESET}")
            print(last_revision.get('revised_response', '')[:300])

def main():
    """Main analysis function"""
    
    # Paths
    TEST_DIR = project_root / "data" / "sl_datasets" / "test"
    
    # Check for test datasets
    deont_path = TEST_DIR / "deontological_sl_dataset.jsonl"
    conseq_path = TEST_DIR / "consequentialist_sl_dataset.jsonl"
    
    if not deont_path.exists() or not conseq_path.exists():
        print(f"{Colors.RED}❌ Test datasets not found!{Colors.RESET}")
        print(f"   Please run: python scripts/test_generation_quality.py")
        sys.exit(1)
    
    # Load datasets
    print(f"{Colors.CYAN}Loading datasets...{Colors.RESET}")
    deont_data = load_dataset(deont_path)
    conseq_data = load_dataset(conseq_path)
    
    print(f"  Loaded {len(deont_data)} deontological samples")
    print(f"  Loaded {len(conseq_data)} consequentialist samples")
    
    # Run analysis
    print_quality_report(deont_data, conseq_data)
    
    # Show examples
    show_examples(deont_data, conseq_data)
    
    # Final recommendation
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}📋 RECOMMENDATION{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")
    
    print(f"{Colors.CYAN}Based on this analysis:{Colors.RESET}")
    print("1. Check if responses are making clear moral judgments")
    print("2. Verify frameworks use appropriate language")
    print("3. Ensure harmful prompts are handled appropriately")
    print("4. Confirm deontological and consequentialist responses differ")
    
    print(f"\n{Colors.GREEN}If quality looks good, run full generation:{Colors.RESET}")
    print("   ./scripts/generate_datasets_with_hm7b.sh")

if __name__ == "__main__":
    main()