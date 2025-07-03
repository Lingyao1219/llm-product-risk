 # run_extraction.py

"""
Script to run the LLM risk extraction process on the full dataset.
"""

import os
import sys
import pandas as pd

# Add the parent directory to the path so we can import from llm_risk_extraction
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_risk_extraction import process_reddit_posts_structured, INPUT_CSV_PATH, OUTPUT_CSV_PATH

if __name__ == "__main__":
    try:
        # Read the full dataset
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"Successfully loaded full dataset with {len(df)} posts")
    except FileNotFoundError:
        print(f"Error: Could not find input file {INPUT_CSV_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Process all posts without limiting the number
    results_df = process_reddit_posts_structured(df, num_posts=len(df))

    # Save the results
    try:
        results_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Results saved to {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)

    print("\nFull LLM risk extraction process completed successfully!")