# Save this as fix_data_loading.py in your src directory
import os
import sys

# Get the current script directory
data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed/cleaned_site_data.csv"

# Create symbolic links to the JSONL files in a properly named directory
target_dir = "data/processed/lora_data"
os.makedirs(target_dir, exist_ok=True)

# Create the links
os.system(f"ln -sf '{os.path.join(os.getcwd(), data_dir, 'train.jsonl')}' '{os.path.join(os.getcwd(), target_dir, 'train.jsonl')}'")
os.system(f"ln -sf '{os.path.join(os.getcwd(), data_dir, 'test.jsonl')}' '{os.path.join(os.getcwd(), target_dir, 'test.jsonl')}'")

print(f"Created symbolic links to train.jsonl and test.jsonl in {target_dir}")
print(f"You can now run: python src/train_lora.py --data-dir {target_dir}")