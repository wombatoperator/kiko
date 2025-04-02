#!/usr/bin/env python
"""
Training-Aligned Evaluator for Honda site classification
- Uses exactly the same prompt format as in training
- Matches the Human/Assistant pattern used during LoRA fine-tuning
- Provides detailed analysis of model behavior
"""

import torch
import pandas as pd
import numpy as np
import re
import time
import logging
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def load_model(model_dir, force_cpu=False):
    """Load the fine-tuned model with LoRA weights with Mac compatibility"""
    if force_cpu:
        device = torch.device("cpu")
        logger.info(f"Forcing CPU usage as requested")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info(f"Using MPS (Metal Performance Shaders) on Mac")
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    else:
        device = torch.device("cpu")
        logger.info(f"Using CPU (no GPU available)")
    
    try:
        # Load exactly the same model as in training
        logger.info("Loading model google/gemma-2-2b (matching training)...")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        base_model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b",
            torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
            device_map="auto" if device.type == "cuda" else None
        )

        # Move base_model to the appropriate device if not using device_map="auto"
        if device.type != "cuda":
            base_model = base_model.to(device)
            
        # Try to load the LoRA adapter
        logger.info(f"Loading LoRA adapter from {model_dir}...")
        model = PeftModel.from_pretrained(base_model, model_dir)
        model.eval()
        
        return model, tokenizer, device
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        if device.type == "mps":
            logger.info("MPS device error detected. Retrying with CPU...")
            return load_model(model_dir, force_cpu=True)
        raise e

def load_data(file_path, sample_size=15):
    """Load data from CSV and take a random sample"""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} sites from {file_path}")
        
        # Take a random sample
        sample = df.sample(min(sample_size, len(df)), random_state=43)
        logger.info(f"Selected {len(sample)} random sites for evaluation")
        
        return sample
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e

def extract_score(text):
    """Extract numerical score from model output text with enhanced extraction"""
    if not text:
        return None
    
    # Try multiple regex patterns for better extraction
    # First look for a decimal between 0 and 1
    patterns = [
        r'(?<![a-zA-Z])(\d+\.\d+|\d+)(?![a-zA-Z\d])',  # Standalone number
        r'(\d+\.\d+|\d+)',                             # Any number
        r'score[:\s]+(\d+\.\d+|\d+)',                  # "score: X" 
        r'rating[:\s]+(\d+\.\d+|\d+)',                 # "rating: X"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                # Only accept scores in the 0-1 range
                if 0 <= score <= 1:
                    return score
            except:
                pass
    
    return None

def get_training_aligned_prompt(url, content):
    """Create a prompt using exactly the same format as in training"""
    # Truncate content if too long (just like in training)
    content_preview = content[:1000] + "..." if len(content) > 1000 else content
    
    # Use EXACTLY the same prompt format as in training
    prompt = f"""You are Alex Carter, a 38-year-old middle school teacher and family man looking for a Honda Pilot SUV.

Rate the following website on a scale from 0 to 1 based on relevance to your car-buying interests:
URL: {url}

Content preview: {content_preview}"""
    
    return prompt

def evaluate_model(model, tokenizer, data, device):
    """Evaluate model on sample data with training-aligned prompting"""
    results = []
    
    for i, (_, row) in enumerate(data.iterrows()):
        url = row['url']
        content = row['text_content']
        expected = row.get('alex_persona_Gemini_2.0')
        
        # Get prompt in training format
        prompt = get_training_aligned_prompt(url, content)
        
        logger.info(f"Evaluating site {i+1}/{len(data)}: {url}")
        
        try:
            # Format prompt exactly as in training
            formatted_prompt = f"<s>Human: {prompt}\n\nAssistant:"
            
            # Encode prompt and move to device
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            
            # Special handling for MPS
            if device.type == "mps":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            start_time = time.time()
            
            with torch.no_grad():
                # Use the same generation parameters as in your training test
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            inference_time = time.time() - start_time
            
            # Decode response - important to get the exact response as in training
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = result[len(formatted_prompt):].strip()
            score = extract_score(response)
            
            # If no score found, try with slightly more tokens
            if score is None:
                logger.info("  No score found in short response, trying with more tokens...")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=15,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = result[len(formatted_prompt):].strip()
                score = extract_score(response)
            
            # Log result
            if expected is not None:
                logger.info(f"  Expected: {expected:.2f}")
            else:
                logger.info("  Expected: N/A")
                
            if score is not None:
                logger.info(f"  Predicted: {score:.2f}")
            else:
                logger.info("  Predicted: N/A")
                
            logger.info(f"  Response: '{response}'")
            logger.info(f"  Time: {inference_time:.2f}s\n")
            
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            if "MPS" in str(e) and device.type == "mps":
                logger.info("Encountered MPS-specific error. Falling back to CPU...")
                model = model.to("cpu")
                device = torch.device("cpu")
                i -= 1
                continue
                
            response = "ERROR"
            score = None
            inference_time = 0
        
        results.append({
            'url': url,
            'expected_score': expected,
            'predicted_score': score,
            'model_response': response,
            'inference_time': inference_time
        })
    
    return pd.DataFrame(results)

def calculate_metrics(results):
    """Calculate basic evaluation metrics"""
    # Filter out rows with missing scores
    valid_results = results.dropna(subset=['predicted_score'])
    valid_comparisons = valid_results.dropna(subset=['expected_score', 'predicted_score'])
    
    metrics = {
        'total_sites': len(results),
        'valid_predictions': len(valid_results),
        'avg_inference_time': results['inference_time'].mean() if len(results) > 0 else 0
    }
    
    if len(valid_results) > 0:
        metrics['avg_predicted'] = valid_results['predicted_score'].mean()
    
    if len(valid_comparisons) > 0:
        metrics['avg_expected'] = valid_comparisons['expected_score'].mean()
        metrics['mse'] = ((valid_comparisons['predicted_score'] - valid_comparisons['expected_score']) ** 2).mean()
        
        if len(valid_comparisons) >= 3:  # Need at least 3 points for correlation
            try:
                from scipy.stats import pearsonr
                pearson, p_value = pearsonr(valid_comparisons['predicted_score'], valid_comparisons['expected_score'])
                metrics['correlation'] = pearson
            except Exception as e:
                logger.warning(f"Could not calculate correlation: {e}")
                metrics['correlation'] = "N/A (calculation error)"
    
    return metrics

def main(model_dir, csv_path, sample_size=15, force_cpu=False):
    """Main evaluation function"""
    try:
        # Load model
        model, tokenizer, device = load_model(model_dir, force_cpu)
        
        # Load data
        data = load_data(csv_path, sample_size)
        
        # Evaluate model
        results = evaluate_model(model, tokenizer, data, device)
        
        # Calculate metrics
        metrics = calculate_metrics(results)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total sites evaluated: {metrics['total_sites']}")
        logger.info(f"Valid predictions: {metrics['valid_predictions']}")
        logger.info(f"Prediction success rate: {metrics['valid_predictions']/metrics['total_sites']*100:.1f}%")
        logger.info(f"Average inference time: {metrics['avg_inference_time']:.2f}s")
        
        if 'avg_expected' in metrics:
            logger.info(f"Average expected score: {metrics['avg_expected']:.2f}")
            logger.info(f"Average predicted score: {metrics['avg_predicted']:.2f}")
            logger.info(f"Mean squared error: {metrics['mse']:.4f}")
        
        if 'correlation' in metrics:
            logger.info(f"Correlation with expected scores: {metrics['correlation']}")
        
        # Print problematic sites
        if 'expected_score' in results.columns and len(results.dropna(subset=['predicted_score', 'expected_score'])) > 0:
            results['error'] = abs(results['predicted_score'] - results['expected_score'])
            problem_sites = results.dropna(subset=['error']).nlargest(3, 'error')
            
            if len(problem_sites) > 0:
                logger.info("\nLargest prediction errors:")
                for _, site in problem_sites.iterrows():
                    logger.info(f"URL: {site['url']}")
                    logger.info(f"  Expected: {site['expected_score']:.2f}, Predicted: {site['predicted_score']:.2f}")
                    logger.info(f"  Error: {site['error']:.2f}")
        
        logger.info("="*50)
        
        # Also output missing prediction sites
        missing_predictions = results[results['predicted_score'].isna()]
        if len(missing_predictions) > 0:
            logger.info("\nSites without valid predictions:")
            for _, site in missing_predictions.iterrows():
                logger.info(f"URL: {site['url']}")
                logger.info(f"  Model response: '{site['model_response']}'")
                logger.info("")
        
        # Update inference service code
        print_inference_service_update_instructions()
        
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return False

def print_inference_service_update_instructions():
    """Print instructions to update the inference service code"""
    logger.info("\n" + "="*50)
    logger.info("INSTRUCTIONS TO UPDATE INFERENCE SERVICE")
    logger.info("="*50)
    logger.info("Based on the evaluation results, here's how to update your inference service:")
    
    logger.info("\n1. Update classify.py to match the training format:")
    logger.info("   Change this function:")
    logger.info("""
def classify_site(url, content, model, tokenizer):
    \"\"\"Classify a single website\"\"\"
    # Truncate content if too long (adjust as needed)
    content_preview = content[:1000] + "..." if len(content) > 1000 else content
    
    prompt = f\"\"\"You are Alex Carter, a 38-year-old middle school teacher and family man looking for a Honda Pilot SUV.
    
Rate the following website on a scale from 0 to 1 based on relevance to your car-buying interests:
URL: {url}

Content preview: {content_preview}

Only respond with a numerical score.\"\"\"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
            num_return_sequences=1
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part
    full_response = response[len(prompt):].strip()
    print(f"Raw response: {full_response}")
    
    # Extract the rating value using regex
    match = re.search(r'(\d+\.\d+|\d+)', full_response)
    if match:
        return float(match.group(1))
    else:
        print(f"Warning: Could not extract score from response: {full_response}")
        return None""")
    
    logger.info("\nTo this:")
    logger.info("""
def classify_site(url, content, model, tokenizer):
    \"\"\"Classify a single website\"\"\"
    # Truncate content if too long (adjust as needed)
    content_preview = content[:1000] + "..." if len(content) > 1000 else content
    
    # Important: Use exactly the same prompt format as in training
    prompt = f\"\"\"You are Alex Carter, a 38-year-old middle school teacher and family man looking for a Honda Pilot SUV.
    
Rate the following website on a scale from 0 to 1 based on relevance to your car-buying interests:
URL: {url}

Content preview: {content_preview}\"\"\"
    
    # Format with Human/Assistant prefix exactly as in training
    formatted_prompt = f"<s>Human: {prompt}\\n\\nAssistant:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,  # More deterministic
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract response after the Assistant: part
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = result[len(formatted_prompt):].strip()
    print(f"Raw response: {response}")
    
    # Enhanced regex extraction for better reliability
    for pattern in [r'(\\d+\\.\\d+|\\d+)', r'score[:\\s]+(\\d+\\.\\d+|\\d+)', r'rating[:\\s]+(\\d+\\.\\d+|\\d+)']:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 1:  # Validate score range
                    return score
            except:
                continue
    
    print(f"Warning: Could not extract score from response: {response}")
    return None""")
    
    logger.info("\n2. Make sure you're using the right model version:")
    logger.info("""
def load_model(weights_dir="models/current"):
    \"\"\"Load base model with LoRA adapter weights\"\"\"
    print(f"Loading model from {weights_dir}...")
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")  # Note the dash
    
    # Optional: Use lower precision for faster inference
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",  # Note the dash
        torch_dtype=torch.float16,
        device_map="auto"  # Will use GPU if available
    )
    
    # Apply LoRA weights
    model = PeftModel.from_pretrained(base_model, weights_dir)
    
    print("Model loaded successfully!")
    return model, tokenizer""")
    
    logger.info("="*50)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Honda site classifier model")
    parser.add_argument("--model_dir", default="./output", help="Directory with model files")
    parser.add_argument("--csv_path", default="rank_master_site_data_master_v1 - Site List.csv", 
                       help="Path to test CSV")
    parser.add_argument("--sample_size", type=int, default=15, help="Number of sites to evaluate")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage (skip MPS)")
    
    args = parser.parse_args()
    success = main(args.model_dir, args.csv_path, args.sample_size, args.force_cpu)
    sys.exit(0 if success else 1)