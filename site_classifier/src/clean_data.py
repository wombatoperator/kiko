import pandas as pd
import re
import json
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

def clean_text(text):
    """Clean text by removing special characters, extra spaces, etc."""
    if not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace but preserve paragraph structure
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove common web elements and navigation text
    text = re.sub(r'cookie[s]? policy|privacy policy|terms of (use|service)', '', text, flags=re.IGNORECASE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Clean up any remaining whitespace issues
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def truncate_text(text, max_length=512):
    """Truncate text to maximum length while preserving complete sentences."""
    if len(text) <= max_length:
        return text
    
    # Try to find a sentence end within 10% of the max_length
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    
    if last_period != -1 and last_period > max_length * 0.9:
        return text[:last_period + 1]
    
    # If no good sentence break, just truncate at max_length
    return truncated

def format_for_lora_numerical(row, include_url=True):
    """Format data for LoRA training with numerical scores."""
    # Keep the original numerical score
    score = float(row['alex_persona_Gemini_2.0'])
    # Ensure the score is between 0 and 1
    score = max(0.0, min(1.0, score))
    
    # Prepare the text content
    content = clean_text(row['text_content'])
    content = truncate_text(content)
    
    # Prepare the formatted example with Alex persona and SUV context
    if include_url:
        prompt = f"""You are Alex Carter, a 38-year-old middle school teacher and family man looking for a Honda Pilot SUV.

Rate the following website on a scale from 0 to 1 based on relevance to your car-buying interests:
URL: {row['url']}

Content preview: {content}"""
    else:
        prompt = f"""You are Alex Carter, a 38-year-old middle school teacher and family man looking for a Honda Pilot SUV.

Rate the following website on a scale from 0 to 1 based on relevance to your car-buying interests:
Content preview: {content}"""
    
    # Use the numerical score as the completion
    completion = f"{score:.2f}"
    
    return {
        "prompt": prompt,
        "completion": completion
    }

def main(input_file, output_dir, test_size=0.1, include_url=True, numerical=True):
    """Main function to prepare data for LoRA training."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Reading data from {input_file}")
    
    # Try reading with different delimiters
    try:
        df = pd.read_csv(input_file)
    except:
        try:
            df = pd.read_csv(input_file, delimiter='\t')
            print("Using tab delimiter for CSV")
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Basic cleaning
    print("Cleaning and preparing data...")
    
    # Drop rows with missing values
    df = df.dropna(subset=['text_content', 'alex_persona_Gemini_2.0'])
    
    # Filter out rows with very short text content
    df = df[df['text_content'].str.len() > 100]
    
    # Convert score column to float if it's not already
    df['alex_persona_Gemini_2.0'] = df['alex_persona_Gemini_2.0'].astype(float)
    
    # Split data into train and test sets
    train_df, test_df = train_test_split(
        df, test_size=test_size, 
        stratify=df['alex_persona_Gemini_2.0'].apply(lambda x: 1 if float(x) >= 0.5 else 0),
        random_state=42
    )
    
    print(f"Training set size: {train_df.shape[0]}")
    print(f"Test set size: {test_df.shape[0]}")
    
    # Format data for LoRA training
    train_examples = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing training data"):
        example = format_for_lora_numerical(row, include_url=include_url)
        train_examples.append(example)
    
    test_examples = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing test data"):
        example = format_for_lora_numerical(row, include_url=include_url)
        test_examples.append(example)
    
    # Save the formatted data
    train_file = os.path.join(output_dir, "train.jsonl")
    test_file = os.path.join(output_dir, "test.jsonl")
    
    print(f"Saving training data to {train_file}")
    with open(train_file, 'w') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Saving test data to {test_file}")
    with open(test_file, 'w') as f:
        for example in test_examples:
            f.write(json.dumps(example) + '\n')
    
    # Save distribution statistics
    train_labels = [1 if float(row['alex_persona_Gemini_2.0']) >= 0.5 else 0 for _, row in train_df.iterrows()]
    test_labels = [1 if float(row['alex_persona_Gemini_2.0']) >= 0.5 else 0 for _, row in test_df.iterrows()]
    
    stats = {
        "total_examples": len(df),
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "train_high_quality": sum(train_labels),
        "train_low_quality": len(train_labels) - sum(train_labels),
        "test_high_quality": sum(test_labels),
        "test_low_quality": len(test_labels) - sum(test_labels),
        "score_range": [float(df['alex_persona_Gemini_2.0'].min()), float(df['alex_persona_Gemini_2.0'].max())]
    }
    
    stats_file = os.path.join(output_dir, "data_stats.json")
    print(f"Saving data statistics to {stats_file}")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("Data preparation for LoRA training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare site data for LoRA training with numerical outputs")
    parser.add_argument("--input", type=str, default="/Users/madiisa-real/Desktop/Kiko/site_classifier/data/raw/rank_master_site_data_master_v1 - Site List.csv",
                        help="Path to input CSV file")
    parser.add_argument("--output-dir", type=str, default="data/processed/lora_data_numerical",
                        help="Directory to save the prepared data")
    parser.add_argument("--test-size", type=float, default=0.1,
                        help="Fraction of data to use for testing")
    parser.add_argument("--include-url", action="store_true", default=True,
                        help="Include URL in the prompt")
    
    args = parser.parse_args()
    main(args.input, args.output_dir, args.test_size, args.include_url)