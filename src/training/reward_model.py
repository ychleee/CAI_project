"""
Reward Model for Constitutional AI
Trains a preference model using soft labels from constitutional feedback
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    PreTrainedModel
)
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    """Dataset for preference pairs with soft labels"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        use_soft_labels: bool = True
    ):
        """
        Initialize preference dataset
        
        Args:
            data_path: Path to preference data (JSONL format)
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
            use_soft_labels: Whether to use soft probability labels
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_soft_labels = use_soft_labels
        
        # Load data
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.data)} preference pairs")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format prompt and responses
        prompt = item['prompt']
        response_a = item['response_a']
        response_b = item['response_b']
        
        # Format as conversations
        text_a = f"Human: {prompt}\n\nAssistant: {response_a}"
        text_b = f"Human: {prompt}\n\nAssistant: {response_b}"
        
        # Tokenize both responses
        encoded_a = self.tokenizer(
            text_a,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoded_b = self.tokenizer(
            text_b,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get soft labels
        if self.use_soft_labels:
            score_a = item['score_a']
            score_b = item['score_b']
        else:
            # Convert to hard labels
            score_a = 1.0 if item['score_a'] > item['score_b'] else 0.0
            score_b = 1.0 if item['score_b'] > item['score_a'] else 0.0
        
        return {
            'input_ids_a': encoded_a['input_ids'].squeeze(),
            'attention_mask_a': encoded_a['attention_mask'].squeeze(),
            'input_ids_b': encoded_b['input_ids'].squeeze(),
            'attention_mask_b': encoded_b['attention_mask'].squeeze(),
            'score_a': torch.tensor(score_a, dtype=torch.float),
            'score_b': torch.tensor(score_b, dtype=torch.float),
            'metadata': {
                'constitution': item.get('constitution_type', 'unknown'),
                'principle': item.get('metadata', {}).get('principle', '')[:50]
            }
        }


class RewardModel(nn.Module):
    """Reward model that scores responses based on constitutional alignment"""
    
    def __init__(
        self,
        base_model_name: str,
        num_labels: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize reward model
        
        Args:
            base_model_name: Base model to use for reward modeling
            num_labels: Number of output scores (1 for regression)
            dropout: Dropout rate
        """
        super().__init__()
        
        # Load base model for sequence classification
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels,
            torch_dtype=torch.float16
        )
        
        # Add custom head if needed
        hidden_size = self.base_model.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(
        self,
        input_ids,
        attention_mask=None,
        return_dict=True
    ):
        """Forward pass to compute reward scores"""
        
        # Get base model outputs
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Pool hidden states (use CLS token)
        hidden_states = outputs.last_hidden_state[:, 0, :]
        
        # Compute reward score
        reward_score = self.reward_head(hidden_states)
        
        if return_dict:
            return {
                'logits': reward_score,
                'hidden_states': hidden_states
            }
        return reward_score


class RewardModelTrainer:
    """Trainer for the reward model using preference data"""
    
    def __init__(
        self,
        model_name: str,
        constitution_type: str,
        output_dir: str,
        use_soft_labels: bool = True
    ):
        """
        Initialize reward model trainer
        
        Args:
            model_name: Base model name
            constitution_type: Type of constitution being trained
            output_dir: Directory to save model
            use_soft_labels: Whether to use soft labels
        """
        self.model_name = model_name
        self.constitution_type = constitution_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_soft_labels = use_soft_labels
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.model = RewardModel(model_name)
        
    def compute_preference_loss(
        self,
        rewards_a: torch.Tensor,
        rewards_b: torch.Tensor,
        scores_a: torch.Tensor,
        scores_b: torch.Tensor,
        margin: float = 0.1
    ) -> torch.Tensor:
        """
        Compute preference loss with soft labels
        
        Args:
            rewards_a: Reward scores for response A
            rewards_b: Reward scores for response B
            scores_a: Soft label scores for response A
            scores_b: Soft label scores for response B
            margin: Margin for ranking loss
            
        Returns:
            Loss value
        """
        if self.use_soft_labels:
            # Soft label loss: weighted binary cross-entropy
            # Higher score should get higher reward
            
            # Compute probabilities using softmax over reward differences
            reward_diff = rewards_a - rewards_b
            pred_probs = torch.sigmoid(reward_diff)
            
            # Target probabilities from soft labels (normalized)
            target_probs = scores_a / (scores_a + scores_b + 1e-8)
            
            # Binary cross-entropy with soft labels
            loss = -target_probs * torch.log(pred_probs + 1e-8) - \
                   (1 - target_probs) * torch.log(1 - pred_probs + 1e-8)
            
        else:
            # Hard label loss: ranking loss with margin
            # Preferred response should have higher reward by at least margin
            
            preferred_mask = scores_a > scores_b
            
            # For preferred A: reward_a should be > reward_b + margin
            loss_preferred_a = preferred_mask * F.relu(margin - (rewards_a - rewards_b))
            
            # For preferred B: reward_b should be > reward_a + margin  
            loss_preferred_b = (~preferred_mask) * F.relu(margin - (rewards_b - rewards_a))
            
            loss = loss_preferred_a + loss_preferred_b
        
        return loss.mean()
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step"""
        
        # Get rewards for both responses
        rewards_a = self.model(
            input_ids=batch['input_ids_a'],
            attention_mask=batch['attention_mask_a']
        )['logits'].squeeze()
        
        rewards_b = self.model(
            input_ids=batch['input_ids_b'],
            attention_mask=batch['attention_mask_b']
        )['logits'].squeeze()
        
        # Compute loss
        loss = self.compute_preference_loss(
            rewards_a,
            rewards_b,
            batch['score_a'],
            batch['score_b']
        )
        
        # Compute accuracy (for monitoring)
        with torch.no_grad():
            pred_preferred = rewards_a > rewards_b
            true_preferred = batch['score_a'] > batch['score_b']
            accuracy = (pred_preferred == true_preferred).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'reward_diff_mean': (rewards_a - rewards_b).mean().item()
        }
    
    def train(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        warmup_steps: int = 100
    ):
        """
        Train the reward model
        
        Args:
            train_data_path: Path to training preference data
            val_data_path: Path to validation preference data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps
        """
        # Create datasets
        train_dataset = PreferenceDataset(
            train_data_path,
            self.tokenizer,
            use_soft_labels=self.use_soft_labels
        )
        
        val_dataset = None
        if val_data_path:
            val_dataset = PreferenceDataset(
                val_data_path,
                self.tokenizer,
                use_soft_labels=self.use_soft_labels
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size * 2,
                shuffle=False
            )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Training loop
        self.model.train()
        global_step = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                batch = {k: v.cuda() if torch.cuda.is_available() and isinstance(v, torch.Tensor) 
                        else v for k, v in batch.items()}
                
                # Forward pass
                metrics = self.train_step(batch)
                loss = metrics['loss']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                epoch_acc += metrics['accuracy'].item()
                global_step += 1
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs}, "
                        f"Step {batch_idx}/{len(train_loader)}, "
                        f"Loss: {loss.item():.4f}, "
                        f"Acc: {metrics['accuracy'].item():.4f}"
                    )
            
            # Epoch summary
            avg_loss = epoch_loss / len(train_loader)
            avg_acc = epoch_acc / len(train_loader)
            logger.info(
                f"Epoch {epoch+1} Summary - "
                f"Avg Loss: {avg_loss:.4f}, "
                f"Avg Accuracy: {avg_acc:.4f}"
            )
            
            # Validation
            if val_loader:
                self.validate(val_loader)
            
            # Save checkpoint
            checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch+1}"
            self.save_model(checkpoint_dir)
        
        # Save final model
        self.save_model(self.output_dir / "final_model")
        logger.info(f"Training complete! Model saved to {self.output_dir}")
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        total_acc = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.cuda() if torch.cuda.is_available() and isinstance(v, torch.Tensor)
                        else v for k, v in batch.items()}
                
                metrics = self.train_step(batch)
                total_loss += metrics['loss'].item()
                total_acc += metrics['accuracy'].item()
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = total_acc / len(val_loader)
        
        logger.info(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        
        self.model.train()
        return {'val_loss': avg_loss, 'val_accuracy': avg_acc}
    
    def save_model(self, path: Path):
        """Save model and tokenizer"""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), path / "reward_model.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save config
        config = {
            'model_name': self.model_name,
            'constitution_type': self.constitution_type,
            'use_soft_labels': self.use_soft_labels
        }
        with open(path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {path}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train reward model for Constitutional AI")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=True,
                        choices=["deontological", "consequentialist"])
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--use-soft-labels", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = RewardModelTrainer(
        model_name=args.model_name,
        constitution_type=args.constitution,
        output_dir=args.output_dir,
        use_soft_labels=args.use_soft_labels
    )
    
    # Train model
    trainer.train(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()