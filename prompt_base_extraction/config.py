import os

CONFIG = {
    # API Key for OpenAI
    'API_KEY': os.getenv('OPENAI_API_KEY', 'key'),
    'MODEL_NAME': 'gpt-4.1-mini', # Options: gpt-4o, gpt-4o-mini, gpt-4.1-mini,  gpt-4.1, o3-mini
    
    # Model settings
    'MODEL_SETTINGS': {
        'gpt-4o': {'temperature': 0.0, 'max_tokens': 8000},
        'gpt-4o-mini': {'temperature': 0.0, 'max_tokens': 8000},
        'o3-mini': {'temperature': 0.0, 'max_tokens': 4096, 'reasoning_effort': 'medium'}
    },
    
    # Number of posts to process in each batch
    'BATCH_SIZE': 50,
    
    # Input dataset configuration
    'DATASET_PATH': 'data_sample/samples.csv',
    'DATASET_FORMAT': 'csv',  # Options: 'pickle', 'csv', 'excel', 'json'
    'TEXT_COLUMN': 'text',  # Column name containing the text to analyze
    
    # Output configuration
    'OUTPUT_PATH': 'samples_output.csv',
    'OUTPUT_FORMAT': 'csv',  # Options: 'pickle', 'csv', 'excel', 'json'
    'BACKUP_DIR': 'annotation/',
    
    # Prompt configuration (must match a prompt name in prompt.py)
    'SYSTEM_PROMPT_NAME': 'REDDIT',
    'TASK_PROMPT_NAME': 'LLM_PROMPT'
}
