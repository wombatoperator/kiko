import os
import argparse
import json
import torch
import logging
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(data_dir, output_dir, model_name="google/gemma-2-2b", lora_r=8, 
         lora_alpha=16, lora_dropout=0.05, epochs=3, batch_size=2, learning_rate=2e-4,
         gradient_accumulation_steps=8, max_seq_length=384):
    """Main function to train the LoRA adapter optimized for M1."""
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.info(f"Loading model: {model_name}")
    
    # Load model with fp16 instead of quantization (better for M1)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure LoRA - specific target modules for Gemma models
    # Using fewer target modules to reduce memory usage
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],  # Reduced target modules for M1
        bias="none"
    )
    
    # Get the PEFT model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load and prepare the dataset
    logger.info(f"Loading dataset from {data_dir}")
    
    # ===== DATA LOADING SECTION =====
    # Check if the input is a CSV file or a directory containing JSONL files
    if data_dir.endswith('.csv'):
        # For CSV input, we need to process it to the format we need
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        # Read the CSV file
        df = pd.read_csv(data_dir)
        
        # Prepare for input
        def prepare_example(row):
            # Format based on the alex_persona score
            label = float(row.get('alex_persona_Gemini_2.0', 0))
            label_text = "high_quality" if label >= 0.5 else "low_quality"
            
            content = row.get('cleaned_text', row.get('text_content', ''))
            
            # Create prompt-completion pair
            prompt = f"Content: {content}\n\nClassify this website content as high_quality or low_quality."
            completion = label_text
            
            return {
                "prompt": prompt,
                "completion": completion
            }
        
        # Process all rows
        examples = [prepare_example(row) for _, row in df.iterrows()]
        
        # Split into train and test
        train_examples, test_examples = train_test_split(
            examples, test_size=0.1, random_state=42
        )
        
        # Create temporary files
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        train_file = os.path.join(temp_dir, "train.jsonl")
        test_file = os.path.join(temp_dir, "test.jsonl")
        
        # Write to JSONL files
        with open(train_file, 'w') as f:
            for example in train_examples:
                f.write(json.dumps(example) + '\n')
        
        with open(test_file, 'w') as f:
            for example in test_examples:
                f.write(json.dumps(example) + '\n')
        
        # Update data_files to use these temporary files
        data_files = {
            "train": train_file,
            "test": test_file
        }
    else:
        # Otherwise, assuming the path is a directory containing JSONL files
        train_file = os.path.join(data_dir, "train.jsonl")
        test_file = os.path.join(data_dir, "test.jsonl")
        
        # Check if files exist
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(f"Could not find train.jsonl or test.jsonl in {data_dir}")
        
        # Set data files
        data_files = {
            "train": train_file,
            "test": test_file
        }
    
    # Load dataset
    dataset = load_dataset("json", data_files=data_files)
    # ===== END DATA LOADING SECTION =====
    
    # Function to combine prompt and completion for training
    def combine_texts(examples):
        combined = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            combined.append(f"{prompt} {completion}")
        return {"text": combined}
    
    # Combine prompt and completion
    dataset = dataset.map(
        combine_texts,
        batched=True,
        remove_columns=["prompt", "completion"]
    )
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Configure training - M1 optimized parameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        learning_rate=learning_rate,
        weight_decay=0.01,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,  # Save only the best model to save disk space
        remove_unused_columns=False,
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory savings
        optim="adamw_torch",  # Use PyTorch's native AdamW
        dataloader_num_workers=1,  # Reduce to 1 to save memory
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the fine-tuned model
    logger.info(f"Saving fine-tuned model to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save LoRA configuration
    lora_config_dict = {
        "base_model": model_name,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "gradient_accumulation_steps": gradient_accumulation_steps
    }
    
    config_file = os.path.join(output_dir, "lora_config.json")
    with open(config_file, 'w') as f:
        json.dump(lora_config_dict, f, indent=2)
    
    logger.info("LoRA fine-tuning completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Gemma model with LoRA for site classification (M1 optimized)")
    parser.add_argument("--data-dir", type=str, default="data/processed/train.jsonl",
                        help="Directory containing prepared data OR path to CSV file")
    parser.add_argument("--output-dir", type=str, default="models/lora_site_classifier",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--model-name", type=str, default="google/gemma-2-2b",
                        help="Base model to fine-tune")
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA attention dimension")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--max-seq-length", type=int, default=384,
                        help="Maximum sequence length")
    
    args = parser.parse_args()
    main(
        args.data_dir, 
        args.output_dir, 
        args.model_name, 
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.gradient_accumulation_steps,
        args.max_seq_length
    )