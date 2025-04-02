#!/usr/bin/env python
"""
Simple script to run the Honda site classifier evaluation
"""

import os
import sys
from model_evaluator import main as run_evaluation

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Honda site classifier model")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./output",
        help="Directory with trained model files"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/Users/madiisa-real/Desktop/Kiko/site_classifier/data/raw/rank_master_site_data_master_v1 - Site List.csv",
        help="Path to CSV file with test data"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=15,
        help="Number of sites to randomly sample for evaluation"
    )
    
    args = parser.parse_args()
    
    # Make sure output directory exists
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Run the evaluation
    success = run_evaluation(args.model_dir, args.csv_path, args.sample_size)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)