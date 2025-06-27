import pandas as pd
from prompt_base_extraction.analyzer import Analyzer, parse_llm_response
from prompt_base_extraction.config import CONFIG
from prompt_base_extraction.prompt import get_system_prompt, get_task_prompt

def main():
    analyzer = Analyzer()

    # Load dataset
    dataset = analyzer.read_dataset()
    
    # Check if model column already exists
    model_col = f"{CONFIG['MODEL_NAME']}"
    if model_col not in dataset.columns:
        dataset[model_col] = None

    # Set up batch processing
    batch_size = CONFIG['BATCH_SIZE']
    total_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        print(f"Processing batch {batch_num+1}/{total_batches}")
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, len(dataset))

        batch_df = dataset.iloc[batch_start:batch_end].copy()

        # Annotate batch
        annotated_batch = analyzer.annotate_dataset(
            dataset=batch_df,
            text_column=CONFIG['TEXT_COLUMN']
        )

        # Save intermediate batch results (backup)
        analyzer.save_dataset(annotated_batch, batch_num)
        dataset.iloc[batch_start:batch_end] = annotated_batch

    # Save the final annotated dataset
    analyzer.save_dataset(dataset, batch_num='final', file_path=CONFIG['OUTPUT_PATH'], file_format=CONFIG['OUTPUT_FORMAT'])
    print(f"Final annotated dataset saved to {CONFIG['OUTPUT_PATH']} in {CONFIG['OUTPUT_FORMAT']} format.")

if __name__ == "__main__":
    main()