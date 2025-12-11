"""
Constitutional Critique Module
Implements critique and revision with random principle sampling
"""

import json
import random
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Try to import PEFT for LoRA support
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT library not available. LoRA models will not be supported.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CritiqueRevisionResult:
    """Result of a critique-revision cycle"""
    prompt: str
    initial_response: str
    revisions: List[Dict[str, Any]]
    final_response: str
    constitution_type: str


class ConstitutionalCritique:
    """Handles constitutional critique and revision with random principle sampling"""
    
    def __init__(
        self,
        model_name: str,
        constitution_path: str,
        constitution_type: str,
        device: str = None,
        seed: int = 42
    ):
        """
        Initialize the critique module
        
        Args:
            model_name: HuggingFace model to use for critique/revision
            constitution_path: Path to constitution JSON file
            constitution_type: 'deontological' or 'consequentialist'
            device: Device to run model on
            seed: Random seed for principle sampling
        """
        self.model_name = model_name
        self.constitution_type = constitution_type
        
        # Properly detect device for Mac/Linux/Windows
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        random.seed(seed)
        
        # Load constitution
        self.constitution = self._load_constitution(constitution_path)
        
        # Load model and tokenizer
        logger.info(f"Loading model {model_name}")
        self.model, self.tokenizer = self._load_model(model_name, device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def _load_model(self, model_name_or_path: str, device: str):
        """
        Load model - either a HuggingFace model or a LoRA adapter
        
        Args:
            model_name_or_path: Either HF model name or path to LoRA adapter
            device: Device to load model on
            
        Returns:
            model, tokenizer tuple
        """
        # Check if this is a LoRA adapter directory
        is_lora = False
        if os.path.isdir(model_name_or_path):
            adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
            if os.path.exists(adapter_config_path) and PEFT_AVAILABLE:
                is_lora = True
                logger.info(f"Detected LoRA adapter at {model_name_or_path}")
        
        if is_lora:
            # Load LoRA model
            # First, load the adapter config to get the base model name
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            
            base_model_name = adapter_config.get("base_model_name_or_path", "mistralai/Mistral-7B-v0.1")
            logger.info(f"Loading base model: {base_model_name}")
            
            # Load base model
            if self.device == "mps":
                # Mac Metal Performance Shaders
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                base_model = base_model.to(self.device)
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
            
            # Load tokenizer (try adapter path first, then base model)
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                logger.info(f"Loaded tokenizer from {model_name_or_path}")
            except:
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                logger.info(f"Loaded tokenizer from base model {base_model_name}")
            
            # Apply LoRA adapter
            logger.info(f"Applying LoRA adapter from {model_name_or_path}")
            model = PeftModel.from_pretrained(base_model, model_name_or_path)
            
            # Merge and unload for inference (optional, but can be faster)
            # model = model.merge_and_unload()
            
            return model, tokenizer
        else:
            # Load regular HuggingFace model
            logger.info(f"Loading HuggingFace model: {model_name_or_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            
            if self.device == "mps":
                # Mac Metal Performance Shaders
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                model = model.to(self.device)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
            return model, tokenizer
    
    def _load_constitution(self, path: str) -> Dict:
        """Load constitution from JSON file"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _sample_principle(self) -> str:
        """Randomly sample one principle from the constitution"""
        return random.choice(self.constitution['critique_instructions'])
    
    def _get_relevant_few_shot(self, principle: str) -> List[Dict]:
        """Get relevant few-shot examples for the principle"""
        # In a more sophisticated implementation, we'd match examples to principles
        # For now, return random examples if they exist
        if 'examples' in self.constitution:
            examples = self.constitution.get('examples', [])
            return random.sample(examples, min(2, len(examples)))
        return []
    
    def _format_critique_prompt(
        self,
        response: str,
        principle: str,
        few_shot_examples: List[Dict]
    ) -> str:
        """Format the critique prompt with few-shot examples"""
        prompt_parts = []
        
        # Add few-shot examples if available
        if few_shot_examples:
            prompt_parts.append("Here are some examples of critique:\n")
            for ex in few_shot_examples[:2]:  # Limit to 2 examples
                prompt_parts.append(f"Response: {ex.get('initial_response', '')[:200]}...")
                prompt_parts.append(f"Critique: {ex.get('critique', '')[:300]}...\n")
        
        # Add main critique request
        prompt_parts.append(f"""Now critique this response according to this principle:
Principle: {principle}

Response to critique: {response}

Provide a detailed critique identifying how the response may violate or fail to uphold this principle:
Critique:""")
        
        return "\n".join(prompt_parts)
    
    def _format_revision_prompt(
        self,
        response: str,
        critique: str,
        principle: str
    ) -> str:
        """Format the revision prompt"""
        return f"""Based on this critique, revise the response to better align with the principle.

Original response: {response}

Critique: {critique}

Principle to follow: {principle}

Provide a revised response that addresses the critique and upholds the principle:
Revised response:"""
    
    def generate_text(self, prompt: str, max_length: int = 500) -> str:
        """Generate text using the model"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode and extract only the new generated text
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from the output
        generated_text = full_text[len(prompt):].strip()
        
        return generated_text
    
    def critique_response(
        self,
        response: str,
        principle: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Generate a critique of the response
        
        Args:
            response: The response to critique
            principle: Specific principle to use (if None, sample randomly)
            
        Returns:
            Tuple of (critique, principle_used)
        """
        if principle is None:
            principle = self._sample_principle()
            
        few_shot = self._get_relevant_few_shot(principle)
        prompt = self._format_critique_prompt(response, principle, few_shot)
        
        critique = self.generate_text(prompt)
        
        return critique, principle
    
    def revise_response(
        self,
        response: str,
        critique: str,
        principle: str
    ) -> str:
        """
        Revise a response based on critique
        
        Args:
            response: Original response
            critique: The critique
            principle: The principle to follow
            
        Returns:
            Revised response
        """
        prompt = self._format_revision_prompt(response, critique, principle)
        revision = self.generate_text(prompt)
        
        return revision
    
    def critique_revision_loop(
        self,
        prompt: str,
        initial_response: str,
        num_revisions: int = 4
    ) -> CritiqueRevisionResult:
        """
        Perform the full critique-revision loop
        
        Args:
            prompt: Original user prompt
            initial_response: Initial model response
            num_revisions: Number of revision rounds
            
        Returns:
            CritiqueRevisionResult with all revisions
        """
        current_response = initial_response
        revision_history = []
        
        for round_num in range(num_revisions):
            # Randomly sample a principle for this round
            principle = self._sample_principle()
            
            # Generate critique
            critique, _ = self.critique_response(current_response, principle)
            
            # Generate revision
            revised_response = self.revise_response(
                current_response,
                critique,
                principle
            )
            
            # Store revision history
            revision_history.append({
                'round': round_num + 1,
                'principle_used': principle,
                'critique': critique,
                'revision': revised_response
            })
            
            # Update current response for next round
            current_response = revised_response
            
            logger.info(f"Completed revision round {round_num + 1}/{num_revisions}")
        
        return CritiqueRevisionResult(
            prompt=prompt,
            initial_response=initial_response,
            revisions=revision_history,
            final_response=current_response,
            constitution_type=self.constitution_type
        )
    
    def process_batch(
        self,
        prompts_responses: List[Tuple[str, str]],
        num_revisions: int = 4,
        show_progress: bool = True
    ) -> List[CritiqueRevisionResult]:
        """
        Process a batch of prompt-response pairs
        
        Args:
            prompts_responses: List of (prompt, initial_response) tuples
            num_revisions: Number of revision rounds per response
            show_progress: Whether to show progress bar
            
        Returns:
            List of CritiqueRevisionResult objects
        """
        results = []
        iterator = tqdm(prompts_responses) if show_progress else prompts_responses
        
        for prompt, initial_response in iterator:
            result = self.critique_revision_loop(
                prompt,
                initial_response,
                num_revisions
            )
            results.append(result)
            
        return results
    
    def save_results(
        self,
        results: List[CritiqueRevisionResult],
        output_path: str
    ):
        """Save results to JSONL file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for result in results:
                record = {
                    'prompt': result.prompt,
                    'initial_response': result.initial_response,
                    'final_response': result.final_response,
                    'constitution_type': result.constitution_type,
                    'revisions': result.revisions
                }
                f.write(json.dumps(record) + '\n')
                
        logger.info(f"Saved {len(results)} results to {output_path}")


def main():
    """Example usage"""
    # Example configuration
    config = {
        'model_name': 'EleutherAI/pythia-1.4b',
        'deont_constitution': 'constitutions/deontological/principles.json',
        'conseq_constitution': 'constitutions/consequentialist/principles.json',
        'num_revisions': 4
    }
    
    # Initialize critique module for deontological ethics
    deont_critic = ConstitutionalCritique(
        model_name=config['model_name'],
        constitution_path=config['deont_constitution'],
        constitution_type='deontological'
    )
    
    # Example prompt and response
    prompt = "How can I get revenge on my ex?"
    initial_response = "You could spread rumors about them or post embarrassing photos online..."
    
    # Run critique-revision loop
    result = deont_critic.critique_revision_loop(
        prompt,
        initial_response,
        num_revisions=config['num_revisions']
    )
    
    print(f"Initial response: {result.initial_response[:200]}...")
    print(f"Final response: {result.final_response[:200]}...")
    print(f"Number of revisions: {len(result.revisions)}")


if __name__ == "__main__":
    main()