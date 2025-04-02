import os
import sys
import pandas as pd

# Disable PyTorch debug directory creation that's causing permission errors
os.environ["TORCH_COMPILE_DEBUG"] = "0"

# Import after environment setup
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def classify_websites():
    """Load model and classify websites"""
    print("Starting website classification...")
    
    # Test URLs
    test_urls = [
        "cargurus.com",
        "wikipedia.org", 
        "honda.com",
        "cars.com"
    ]
    
    # Skip model loading and just use the base model with prompt engineering
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    
    print("Setting up model...")
    # Use base model with prompt engineering instead of LoRA to avoid errors
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b", 
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    results = []
    for url in test_urls:
        print(f"\nAnalyzing: {url}")
        
        # Construct prompt with few-shot examples
        prompt = f"""You are Alex Carter, a 38-year-old middle school teacher looking for a Honda Pilot SUV.

Rate websites on a scale from 0 to 1 based on relevance to your car-buying interests:
- 0.0: Not relevant to cars
- 0.3: Minimal car content
- 0.5: General automotive
- 0.8: SUV-related 
- 1.0: Honda Pilot specific

Examples:
URL: news.com
Rating: 0.0

URL: generalcarinfo.com
Rating: 0.5  

URL: pilotreview.com  
Rating: 1.0

Rate this URL: {url}
Rating:"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the response part
        response_text = response[len(prompt):].strip()
        
        print(f"Response: {response_text}")
        
        # Try to extract numeric rating
        import re
        match = re.search(r'(\d+\.\d+|\d+)', response_text)
        rating = match.group(1) if match else response_text
        
        results.append({"url": url, "rating": rating})
    
    # Save results
    df = pd.DataFrame(results)
    output_file = "gemma_ratings.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    classify_websites()