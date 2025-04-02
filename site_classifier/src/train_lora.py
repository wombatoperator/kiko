#!/usr/bin/env python
"""
Final LoRA Trainer for SUV Numerical Scores
- Specifically uses the numerical data files
- Simple and reliable for overnight training
- Handles JSON serialization properly
- Produces scores between 0-1

Usage:
python final_lora_trainer.py
"""

import argparse
import json
import logging
import os
import sys
import re
import gc
from typing import Dict, List, Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def safe_json_dump(obj, file_path):
    """Safely save JSON by handling non-serializable objects"""
    def convert_to_serializable(item):
        if isinstance(item, set):
            return list(item)
        elif isinstance(item, dict):
            return {k: convert_to_serializable(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [convert_to_serializable(i) for i in item]
        else:
            return item
    
    with open(file_path, 'w') as f:
        json.dump(convert_to_serializable(obj), f, indent=2)

def load_jsonl_data(file_path):
    """Load JSONL data, handling any errors"""
    data = []
    
    try:
        logger.info(f"Loading data from: {file_path}")
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {i+1}: {e}")
                    continue
        
        logger.info(f"Successfully loaded {len(data)} examples")
        
        # Show a sample for verification
        if data:
            prompt_sample = data[0]["prompt"]
            completion_sample = data[0]["completion"]
            logger.info(f"Sample prompt: {prompt_sample[:100]}...")
            logger.info(f"Sample completion: {completion_sample}")
            
        return data
        
    except Exception as e:
        logger.error(f"Error reading JSONL file {file_path}: {e}")
        raise

def train_lora(
    train_file: str = "/Users/madiisa-real/Desktop/Kiko/data/processed/lora_data_numerical/train.jsonl",
    test_file: str = "/Users/madiisa-real/Desktop/Kiko/data/processed/lora_data_numerical/test.jsonl",
    output_dir: str = "./output",
    model_name: str = "google/gemma-2-2b",
    seed: int = 42,
    num_train_epochs: int = 5,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
):
    """Main training function with absolute minimal complexity"""
    try:
        # Set random seed for reproducibility
        set_seed(seed)
        logger.info(f"Training with seed: {seed}")
        logger.info(f"Using model: {model_name}")
        
        # Detect device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple MPS")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU (training will be slow)")
        
        # Load model
        logger.info("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure LoRA
        logger.info("Setting up LoRA configuration")
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, peft_config)
        
        # Enable gradient checkpointing to save memory
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        
        # Load data
        logger.info(f"Loading training data from: {train_file}")
        logger.info(f"Loading test data from: {test_file}")
        
        train_data = load_jsonl_data(train_file)
        eval_data = load_jsonl_data(test_file)
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)
        
        logger.info(f"Training examples: {len(train_dataset)}")
        logger.info(f"Evaluation examples: {len(eval_dataset)}")
        
        # Tokenize function
        def tokenize_data(examples):
            prompts = []
            for p, c in zip(examples['prompt'], examples['completion']):
                # Format with human/assistant roles
                prompts.append(f"<s>Human: {p}\n\nAssistant: {c}</s>")
            
            # Tokenize with padding and truncation
            tokenized = tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt"
            )
            
            # Set labels equal to input_ids for causal language modeling
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        # Process data
        logger.info("Tokenizing datasets...")
        train_dataset = train_dataset.map(
            tokenize_data,
            batched=True,
            batch_size=8,
            remove_columns=train_dataset.column_names,
        )
        
        eval_dataset = eval_dataset.map(
            tokenize_data,
            batched=True,
            batch_size=8,
            remove_columns=eval_dataset.column_names,
        )
        
        # Set up training arguments
        logger.info("Setting up training arguments")
        
        # Disable fp16 on MPS (Mac) as it's unstable
        use_fp16 = device.type != "mps"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=1,    # Small batch size for memory
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,   # Compensate for small batch size
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=3,
            remove_unused_columns=False,
            fp16=use_fp16,
            optim="adamw_torch",
            report_to="none",  # Disable reporting to simplify
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_checkpointing=True,
            dataloader_num_workers=1,
            group_by_length=True,
        )
        
        # Create trainer
        logger.info("Creating trainer")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Train
        logger.info("Starting training")
        trainer.train()
        
        # Save the final model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save adapter configuration safely
        adapter_config_path = os.path.join(output_dir, "adapter_config.json")
        safe_json_dump(peft_config.to_dict(), adapter_config_path)
        
        logger.info("Training complete!")
        
        # Test model performance
        logger.info("Testing model on sample prompts...")
        model.eval()
        
        test_prompts = [
            """You are Alex Carter, a 38-year-old middle school teacher and family man looking for a Honda Pilot SUV.

Rate the following website on a scale from 0 to 1 based on relevance to your car-buying interests:
URL: autotrader.com

Content preview: Cars for Sale - Used Cars, New Cars, SUVs, and Trucks - Autotrader""",

            """You are Alex Carter, a 38-year-old middle school teacher and family man looking for a Honda Pilot SUV.

Rate the following website on a scale from 0 to 1 based on relevance to your car-buying interests:
URL: espn.com

Content preview: ESPN: Serving sports fans. Anytime. Anywhere."""
        ]
        
        for prompt in test_prompts:
            formatted_prompt = f"<s>Human: {prompt}\n\nAssistant:"
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = result[len(formatted_prompt):].strip()
                logger.info(f"URL: {prompt.split('URL: ')[1].split('\n\n')[0]}")
                logger.info(f"Response: {response}")
        
        # Return success
        return True
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return False
    finally:
        # Clean up
        if 'model' in locals():
            del model
        if 'trainer' in locals():
            del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

if __name__ == "__main__":
    # For MPS optimization on Mac
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    # For memory efficiency
    torch.set_num_threads(4)
    
    parser = argparse.ArgumentParser(description="Final LoRA Trainer for SUV Numerical Scores")
    parser.add_argument(
        "--train_file",
        type=str,
        default="/Users/madiisa-real/Desktop/Kiko/data/processed/lora_data_numerical/train.jsonl",
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="/Users/madiisa-real/Desktop/Kiko/data/processed/lora_data_numerical/test.jsonl",
        help="Path to test JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-2b",
        help="Name or path of the base model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    success = train_lora(
        args.train_file,
        args.test_file,
        args.output_dir,
        args.model_name,
        args.seed,
        args.num_train_epochs,
        args.learning_rate,
        args.max_seq_length
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)