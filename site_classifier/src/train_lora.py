#!/usr/bin/env python
# Final working version of train_lora.py with all issues fixed

import argparse
import json
import logging
import os
import sys
import gc
from typing import Dict, List, Optional

import numpy as np
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def main(
    data_dir: str,
    output_dir: str,
    model_name: str = "google/gemma-2-2b",
    seed: int = 42,
    num_train_epochs: int = 3,
    learning_rate: float = 5e-5,
    max_seq_length: int = 512,
):
    """Main training function optimized for M1 Pro."""
    set_seed(seed)
    logger.info(f"Loading model: {model_name}")
    
    # M1 optimization: Use MPS (Metal Performance Shaders) when available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model in float16 for memory efficiency
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure LoRA with smaller parameters for M1 Pro
    logger.info("Configuring LoRA (memory-optimized for M1 Pro)")
    peft_config = LoraConfig(
        r=8,                # Reduced rank for memory efficiency
        lora_alpha=16,      # Adjusted alpha
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Reduced target modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    # Load and prepare the data from JSONL files
    logger.info(f"Loading and preparing data from directory: {data_dir}")
    
    # Define paths to the train and test JSONL files
    train_file = os.path.join(data_dir, "train.jsonl")
    test_file = os.path.join(data_dir, "test.jsonl")
    
    logger.info(f"Training file: {train_file}")
    logger.info(f"Testing file: {test_file}")
    
    # Function to load and process JSONL files with prompt and completion fields
    def load_jsonl_data(file_path):
        data = []
        try:
            with open(file_path, 'r') as f:
                # Process the file
                for line in f:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                        
                    json_obj = json.loads(line)
                    
                    # Extract prompt and completion fields
                    data.append({
                        "prompt": json_obj.get('prompt', ''),
                        "completion": json_obj.get('completion', '')
                    })
            
            logger.info(f"Loaded {len(data)} examples from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error reading JSONL file {file_path}: {e}")
            raise e
    
    # Load the data
    try:
        train_data = load_jsonl_data(train_file)
        eval_data = load_jsonl_data(test_file)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise e
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Evaluation examples: {len(eval_dataset)}")
    
    # Tokenize the data
    def tokenize_function(examples):
        # Format the prompts - accessing by column name
        prompts = []
        for p, c in zip(examples['prompt'], examples['completion']):
            prompts.append(f"<s>Human: {p}\n\nAssistant: {c}</s>")
        
        # Tokenize with reduced max length to save memory
        tokenized = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )
        
        # Set labels equal to input_ids (for causal language modeling)
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # Process data with error handling
    try:
        logger.info("Starting tokenization of training dataset")
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=8,
            remove_columns=train_dataset.column_names,
        )
        
        logger.info("Starting tokenization of evaluation dataset")
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=8,
            remove_columns=eval_dataset.column_names,
        )
    except Exception as e:
        logger.error(f"Error during tokenization: {e}")
        raise
    
    # Set up training arguments optimized for M1 Pro with 16GB RAM
    logger.info("Setting up training arguments (M1 Pro optimized)")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,    # Minimal batch size for M1
        per_device_eval_batch_size=1,     # Minimal batch size for M1
        gradient_accumulation_steps=16,   # Increased to compensate for smaller batch size
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        remove_unused_columns=False,
        fp16=False,                       # Disable fp16 as it can be unstable on MPS
        optim="adamw_torch",
        report_to="none",                 # Changed from tensorboard to none
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
    
    logger.info("Training complete!")

if __name__ == "__main__":
    # Enable memory profiling for M1 Pro optimization
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Lower MPS memory usage
    
    # Disable bitsandbytes errors
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    
    # Limit number of threads to prevent memory bloat
    torch.set_num_threads(4)
    
    parser = argparse.ArgumentParser(description="Train LoRA on Gemma 2 2b (M1 Pro Optimized)")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/madiisa-real/Desktop/Kiko/site_classifier/data/processed/cleaned_site_data.csv",
        help="Directory containing train.jsonl and test.jsonl files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-2b",
        help="Name or path of the base model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,  # Reduced for M1 Pro
        help="Maximum sequence length",
    )
    
    args = parser.parse_args()
    
    main(
        args.data_dir,
        args.output_dir,
        args.model_name,
        args.seed,
        args.num_train_epochs,
        args.learning_rate,
        args.max_seq_length
    )