import os
import time
import json
import pandas as pd
from openai import OpenAI
from typing import Optional, Dict, Any, List, Union
from tqdm import tqdm
from prompt_base_extraction.config import CONFIG
from prompt_base_extraction.prompt import *


def parse_llm_response(response: str, default_value: Optional[Union[Dict, List]] = None) -> Union[Dict, List]:
    """
    Extract JSON content strictly from LLM responses.
    """
    if not response:
        return default_value if default_value is not None else {}
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].strip()
        return json.loads(response)
    except json.JSONDecodeError:
        try:
            if '[' in response and ']' in response:
                start = response.find('[')
                end = response.rfind(']') + 1
                if start >= 0 and end > 0:
                    json_content = response[start:end]
                    return json.loads(json_content)
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > 0:
                    json_content = response[start:end]
                    return json.loads(json_content)
            print(f"No valid JSON content found in response: {response[:100]}...")
            return default_value if default_value is not None else {}
        except Exception as e:
            print(f"JSON parsing failed: {response[:100]}..., error: {e}")
            return default_value if default_value is not None else {}


class Analyzer:
    """A class for text analysis and stance annotation using OpenAI."""

    def __init__(self):
        self.client = OpenAI(api_key=CONFIG['API_KEY'])
        self._ensure_backup_dir()

    def _ensure_backup_dir(self):
        """Ensure backup directory exists."""
        os.makedirs(CONFIG['BACKUP_DIR'], exist_ok=True)

    def read_dataset(self, file_path: Optional[str] = None, file_format: Optional[str] = None) -> pd.DataFrame:
        """Read dataset from specified file format."""
        if file_path is None:
            file_path = CONFIG['DATASET_PATH']
        if file_format is None:
            file_format = CONFIG['DATASET_FORMAT']
        if file_format.lower() == 'pickle':
            df = pd.read_pickle(file_path)
        elif file_format.lower() == 'csv':
            df = pd.read_csv(file_path)
        elif file_format.lower() == 'excel':
            df = pd.read_excel(file_path)
        elif file_format.lower() == 'json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        return df.reset_index(drop=True)


    def save_dataset(self, dataset: pd.DataFrame, batch_num: Any, file_path: Optional[str] = None, file_format: Optional[str] = None):
        """Save dataset with timestamp and batch number."""
        if file_format is None:
            file_format = CONFIG['OUTPUT_FORMAT']
        if file_path is None:
            file_path = CONFIG['OUTPUT_PATH']

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"batch_{batch_num}_{timestamp}.{file_format}"
        full_path = os.path.join(CONFIG['BACKUP_DIR'], filename)

        if file_format.lower() == 'pickle':
            dataset.to_pickle(full_path)
        elif file_format.lower() == 'csv':
            dataset.to_csv(full_path, index=False)
        elif file_format.lower() == 'excel':
            dataset.to_excel(full_path, index=False)
        elif file_format.lower() == 'json':
            dataset.to_json(full_path, orient='records', lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")


    def analyze_text(self, text: str, system_prompt: str = None, task_prompt: str = None) -> Union[Dict[str, Any], List]:
        """Get LLM response for the provided text using the configured model."""

        model_name = CONFIG['MODEL_NAME']
        model_settings = CONFIG['MODEL_SETTINGS'].get(model_name, {})
        
        if system_prompt is None:
            system_prompt = get_system_prompt()
        if task_prompt is None:
            task_prompt = get_task_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt.format(text=text)}
        ]
        
        params = {
            "model": model_name,
            "messages": messages,
            "temperature": model_settings.get('temperature', 0.0),
            "max_tokens": model_settings.get('max_tokens', 8000),
        }
        
        if 'reasoning_effort' in model_settings:
            params["reasoning_effort"] = model_settings['reasoning_effort']
        
        try:
            completion = self.client.chat.completions.create(**params)
            response_text = completion.choices[0].message.content
            is_array_output = '[' in response_text and ']' in response_text
            default_value = [] if is_array_output else {"error": "Failed to parse response"}
            result = parse_llm_response(response_text, default_value)
            if not is_array_output and isinstance(result, dict) and "error" in result and "Failed to parse" in result.get("error", ""):
                return {"error": f"Failed to parse response: {response_text[:100]}..."}
            
            return result
        except Exception as e:
            is_array_output = '[' in response_text if 'response_text' in locals() else False
            
            if is_array_output:
                return []
            else:
                return {"error": str(e)}


    def annotate_dataset(self, dataset: pd.DataFrame, text_column: str, system_prompt: str = None, task_prompt: str = None):
        """Annotate entire dataset with outputs from the selected model."""

        model_name = CONFIG['MODEL_NAME']
        
        for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Annotating dataset"):
            text = row[text_column]
            results_col = f"{model_name}_results"
            if results_col in dataset.columns and not pd.isna(dataset.at[idx, results_col]):
                continue
            result = self.analyze_text(text, system_prompt, task_prompt)
            
            if results_col not in dataset.columns:
                dataset[results_col] = None
            dataset.at[idx, results_col] = json.dumps(result) if result else None
        
            if isinstance(result, list):
                count_col = f"{model_name}_risk_count"
                if count_col not in dataset.columns:
                    dataset[count_col] = 0
                dataset.at[idx, count_col] = len(result)
                
                for risk_idx, risk in enumerate(result, 1):
                    if isinstance(risk, dict):
                        for field, value in risk.items():
                            col_name = f"{model_name}_risk{risk_idx}_{field}"
                            if col_name not in dataset.columns:
                                dataset[col_name] = None
                            dataset.at[idx, col_name] = value
            
            elif isinstance(result, dict) and "error" not in result:
                for field, value in result.items():
                    col_name = f"{model_name}_{field}"
                    if col_name not in dataset.columns:
                        dataset[col_name] = None
                    if isinstance(value, list):
                        dataset.at[idx, col_name] = str(value)
                    else:
                        dataset.at[idx, col_name] = value
            
            elif isinstance(result, dict) and "error" in result:
                error_col = f"{model_name}_error"
                if error_col not in dataset.columns:
                    dataset[error_col] = None
                dataset.at[idx, error_col] = result.get("error", "Unknown error")
        
        return dataset