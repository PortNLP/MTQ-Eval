import pandas as pd
from datasets import load_dataset, Dataset
import sys
import os

# Add parent dir to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.deformation import deform_text
from src.utils import format_dpo_entry

def prepare_dataset(output_path="data/dpo_processed"):
    print("Loading Belebele dataset...")
    # Using the HF Hub version for reproducibility
    ds = load_dataset("facebook/belebele", split="default")
    
    # Convert to pandas for easier manipulation as per your original script
    df = ds.to_pandas()
    
    # Filter columns
    df = df[['dialect', 'flores_passage']]
    
    # Optional: Filter specific languages here if needed
    # languages_to_remove = [...]
    # df = df[~df['dialect'].isin(languages_to_remove)]

    print("Generating preference pairs (High Quality vs Low Quality)...")
    
    dpo_data = []
    
    for _, row in df.iterrows():
        original_text = row['flores_passage']
        
        # Create deformed text (The "Rejected" sample)
        deformed_text = deform_text(original_text, method='swap')
        
        # Format for DPO
        # Chosen: The original text, rated [[1]]
        # Rejected: The deformed text, rated [[0]] (or we can flip the ratings logic)
        
        # Logic: We want the model to generate "Rating: [[1]]" for good text
        # and "Rating: [[0]]" for bad text.
        
        # Pair 1: Teach model to rate Good Text as 1
        dpo_data.append({
            "prompt": f"Rate this text:\n{original_text}", # Simplified for readability in logic
            "chosen": format_dpo_entry(original_text, 1), 
            "rejected": format_dpo_entry(original_text, 0) # Evaluating good text as 0 is "rejected" behavior
        })
        
        # Pair 2: Teach model to rate Bad Text as 0
        dpo_data.append({
            "prompt": f"Rate this text:\n{deformed_text}",
            "chosen": format_dpo_entry(deformed_text, 0),
            "rejected": format_dpo_entry(deformed_text, 1) # Evaluating bad text as 1 is "rejected" behavior
        })

    # Convert to HF Dataset
    final_ds = Dataset.from_list(dpo_data)
    
    # Split
    split_ds = final_ds.train_test_split(test_size=0.1, seed=42)
    
    print(f"Saving processed dataset to {output_path}...")
    split_ds.save_to_disk(output_path)
    print("Done.")

if __name__ == "__main__":
    prepare_dataset()