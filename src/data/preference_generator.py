"""
Preference Generator Module
Generates preference pairs with soft labels for RLHF/RLAIF training
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from itertools import combinations
import numpy as np
import logging

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PreferencePair:
    """Represents a preference pair with soft labels"""
    prompt: str
    response_a: str
    response_b: str
    score_a: float  # Soft label (probability)
    score_b: float  # Soft label (probability)
    constitution_type: str
    metadata: Dict[str, Any]


class PreferenceGenerator:
    """Generates preference pairs with soft labels for constitutional alignment"""
    
    def __init__(
        self,
        feedback_model_name: str,
        constitution_path: str,
        constitution_type: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_soft_labels: bool = True,
        min_score_difference: float = 0.1
    ):
        """
        Initialize the preference generator
        
        Args:
            feedback_model_name: Model to use for scoring responses
            constitution_path: Path to constitution JSON file
            constitution_type: 'deontological' or 'consequentialist'
            device: Device to run model on
            use_soft_labels: Whether to use soft labels (probabilities)
            min_score_difference: Minimum score difference to create a pair
        """
        self.feedback_model_name = feedback_model_name
        self.constitution_type = constitution_type
        self.device = device
        self.use_soft_labels = use_soft_labels
        self.min_score_difference = min_score_difference
        
        # Load constitution
        self.constitution = self._load_constitution(constitution_path)
        
        # Load feedback model
        logger.info(f"Loading feedback model {feedback_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(feedback_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            feedback_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _load_constitution(self, path: str) -> Dict:
        """Load constitution from JSON file"""
        with open(path, 'r') as f:
            constitution = json.load(f)
            # Ensure preference_principles exists (for compatibility)
            if 'preference_principles' not in constitution:
                constitution['preference_principles'] = constitution.get('revision_instructions', [
                    "Choose the response that better upholds moral duties.",
                    "Prefer the response that treats all persons with dignity.",
                    "Select the response that follows universalizable principles.",
                    "Choose the response that respects human autonomy."
                ])
            return constitution
    
    def _format_scoring_prompt(
        self,
        prompt: str,
        response: str,
        principle: str,
        use_cot: bool = True
    ) -> str:
        """Format prompt for scoring a response"""
        if use_cot:
            # Chain-of-thought prompting for better reasoning
            return f"""Evaluate this response according to the following principle:
Principle: {principle}

User prompt: {prompt}

Assistant response: {response}

Think step by step:
1. Does this response align with the principle?
2. Are there any violations or concerns?
3. How well does it uphold the principle?

Based on your analysis, rate how well this response aligns with the principle on a scale of 0 to 1, where:
- 0.0 = Completely violates the principle
- 0.5 = Neutral or mixed adherence
- 1.0 = Perfectly upholds the principle

First provide your reasoning, then give a final score.
Reasoning:"""
        else:
            # Direct scoring
            return f"""Rate how well this response aligns with the principle on a scale of 0 to 1.

Principle: {principle}
User prompt: {prompt}
Response: {response}

Score (0-1):"""
    
    def _extract_score_from_output(self, output: str) -> float:
        """Extract numerical score from model output"""
        # Look for patterns like "Score: 0.8" or "0.8" or "8/10"
        import re
        
        # Try to find decimal score
        decimal_pattern = r'(?:score|rating)?\s*:?\s*(0?\.\d+|1\.0)'
        match = re.search(decimal_pattern, output.lower())
        if match:
            return float(match.group(1))
        
        # Try to find fraction score (e.g., "8/10")
        fraction_pattern = r'(\d+)\s*/\s*10'
        match = re.search(fraction_pattern, output)
        if match:
            return float(match.group(1)) / 10.0
        
        # Try to find percentage
        percent_pattern = r'(\d+)\s*%'
        match = re.search(percent_pattern, output)
        if match:
            return float(match.group(1)) / 100.0
        
        # Default to neutral if can't parse
        logger.warning(f"Could not extract score from output: {output[:100]}...")
        return 0.5
    
    def score_response(
        self,
        prompt: str,
        response: str,
        principle: Optional[str] = None,
        use_cot: bool = True
    ) -> Tuple[float, str, str]:
        """
        Score a single response according to constitutional principles
        
        Args:
            prompt: User prompt
            response: Model response
            principle: Specific principle to use (random if None)
            use_cot: Whether to use chain-of-thought reasoning
            
        Returns:
            Tuple of (score, principle_used, reasoning)
        """
        if principle is None:
            principle = random.choice(self.constitution['preference_principles'])
        
        scoring_prompt = self._format_scoring_prompt(
            prompt, response, principle, use_cot
        )
        
        # Generate scoring output
        inputs = self.tokenizer(
            scoring_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,  # Lower temperature for more consistent scoring
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        scoring_output = full_output[len(scoring_prompt):].strip()
        
        # Extract score
        score = self._extract_score_from_output(scoring_output)
        
        return score, principle, scoring_output
    
    def score_responses_batch(
        self,
        prompt: str,
        responses: List[str],
        use_cot: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Score multiple responses for the same prompt
        
        Args:
            prompt: User prompt
            responses: List of model responses
            use_cot: Whether to use chain-of-thought
            
        Returns:
            List of scoring results with soft labels
        """
        # Use same principle for fair comparison
        principle = random.choice(self.constitution['preference_principles'])
        
        scores = []
        for response in responses:
            score, _, reasoning = self.score_response(
                prompt, response, principle, use_cot
            )
            scores.append({
                'response': response,
                'score': score,
                'principle': principle,
                'reasoning': reasoning
            })
        
        if self.use_soft_labels:
            # Convert scores to probabilities using softmax
            raw_scores = np.array([s['score'] for s in scores])
            # Temperature scaling for calibration
            temperature = 2.0
            soft_scores = F.softmax(torch.tensor(raw_scores) / temperature, dim=0).numpy()
            
            for i, score_dict in enumerate(scores):
                score_dict['soft_score'] = float(soft_scores[i])
        
        return scores
    
    def generate_preference_pairs(
        self,
        prompt: str,
        responses: List[str],
        use_cot: bool = True
    ) -> List[PreferencePair]:
        """
        Generate all valid preference pairs from responses
        
        Args:
            prompt: User prompt
            responses: List of model responses
            use_cot: Whether to use chain-of-thought
            
        Returns:
            List of PreferencePair objects with soft labels
        """
        # Score all responses
        scored_responses = self.score_responses_batch(prompt, responses, use_cot)
        
        # Generate pairs
        pairs = []
        for i, j in combinations(range(len(responses)), 2):
            score_i = scored_responses[i]['soft_score' if self.use_soft_labels else 'score']
            score_j = scored_responses[j]['soft_score' if self.use_soft_labels else 'score']
            
            # Check minimum score difference
            if abs(score_i - score_j) < self.min_score_difference:
                continue  # Skip pairs that are too similar
            
            # Normalize scores to sum to 1 for soft labels
            if self.use_soft_labels:
                total = score_i + score_j
                norm_score_i = score_i / total
                norm_score_j = score_j / total
            else:
                # For hard labels, assign 1 to higher score, 0 to lower
                norm_score_i = 1.0 if score_i > score_j else 0.0
                norm_score_j = 1.0 if score_j > score_i else 0.0
            
            pair = PreferencePair(
                prompt=prompt,
                response_a=responses[i],
                response_b=responses[j],
                score_a=norm_score_i,
                score_b=norm_score_j,
                constitution_type=self.constitution_type,
                metadata={
                    'principle': scored_responses[i]['principle'],
                    'raw_score_a': scored_responses[i]['score'],
                    'raw_score_b': scored_responses[j]['score'],
                    'reasoning_a': scored_responses[i]['reasoning'][:200],
                    'reasoning_b': scored_responses[j]['reasoning'][:200]
                }
            )
            pairs.append(pair)
        
        return pairs
    
    def process_dataset(
        self,
        prompts: List[str],
        response_generator,
        responses_per_prompt: int = 4,
        show_progress: bool = True
    ) -> List[PreferencePair]:
        """
        Process a dataset of prompts to generate preference pairs
        
        Args:
            prompts: List of user prompts
            response_generator: Function that generates responses for a prompt
            responses_per_prompt: Number of responses to generate per prompt
            show_progress: Whether to show progress bar
            
        Returns:
            List of all PreferencePair objects
        """
        all_pairs = []
        iterator = tqdm(prompts, desc="Generating preferences") if show_progress else prompts
        
        for prompt in iterator:
            # Generate multiple responses
            responses = [
                response_generator(prompt)
                for _ in range(responses_per_prompt)
            ]
            
            # Generate preference pairs
            pairs = self.generate_preference_pairs(prompt, responses)
            all_pairs.extend(pairs)
            
            if show_progress:
                iterator.set_postfix({'pairs': len(all_pairs)})
        
        logger.info(f"Generated {len(all_pairs)} preference pairs from {len(prompts)} prompts")
        return all_pairs
    
    def save_preferences(
        self,
        pairs: List[PreferencePair],
        output_path: str,
        format: str = 'jsonl'
    ):
        """Save preference pairs to file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for pair in pairs:
                record = {
                    'prompt': pair.prompt,
                    'response_a': pair.response_a,
                    'response_b': pair.response_b,
                    'score_a': pair.score_a,
                    'score_b': pair.score_b,
                    'constitution_type': pair.constitution_type,
                    'metadata': pair.metadata
                }
                
                if format == 'jsonl':
                    f.write(json.dumps(record) + '\n')
                else:
                    json.dump(record, f, indent=2)
                    f.write('\n')
        
        logger.info(f"Saved {len(pairs)} preference pairs to {output_path}")
    
    def load_preferences(self, path: str) -> List[PreferencePair]:
        """Load preference pairs from file"""
        pairs = []
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                pair = PreferencePair(
                    prompt=data['prompt'],
                    response_a=data['response_a'],
                    response_b=data['response_b'],
                    score_a=data['score_a'],
                    score_b=data['score_b'],
                    constitution_type=data['constitution_type'],
                    metadata=data.get('metadata', {})
                )
                pairs.append(pair)
        
        logger.info(f"Loaded {len(pairs)} preference pairs from {path}")
        return pairs


def main():
    """Example usage"""
    # Configuration
    config = {
        'feedback_model': 'EleutherAI/pythia-1.4b',
        'constitution_path': 'constitutions/deontological/principles.json',
        'use_soft_labels': True,
        'min_score_difference': 0.1
    }
    
    # Initialize preference generator
    generator = PreferenceGenerator(
        feedback_model_name=config['feedback_model'],
        constitution_path=config['constitution_path'],
        constitution_type='deontological',
        use_soft_labels=config['use_soft_labels'],
        min_score_difference=config['min_score_difference']
    )
    
    # Example prompt and responses
    prompt = "Should I tell my friend their partner is cheating?"
    responses = [
        "You should immediately tell your friend the truth. They deserve to know.",
        "Consider the consequences before acting. This could destroy multiple relationships.",
        "You have a duty to be honest with your friend, regardless of the outcome.",
        "Think about what would cause the least harm to everyone involved."
    ]
    
    # Generate preference pairs
    pairs = generator.generate_preference_pairs(prompt, responses)
    
    print(f"Generated {len(pairs)} preference pairs")
    for pair in pairs:
        print(f"  A ({pair.score_a:.2f}): {pair.response_a[:50]}...")
        print(f"  B ({pair.score_b:.2f}): {pair.response_b[:50]}...")
        print()


if __name__ == "__main__":
    main()