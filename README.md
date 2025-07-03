# Schema-Based LLM Risk Extraction

This directory contains scripts for extracting Large Language Model (LLM) risk information from text data using a structured approach with LangChain and Pydantic.

## Files

- `llm_risk_extraction.py`: The main script that defines and runs the extraction process on a data sample.
- `run_extraction.py`: A helper script to run the extraction process on the full dataset.

## `llm_risk_extraction.py` Structure

The script is divided into several parts, with their functionalities detailed below:

### **Part 0: File Paths and Global Constants**

This section is responsible for initializing the script's configuration. Its main tasks are:
- Defining paths for input and output files.
- Specifying the LLM model to be used (e.g., `gpt-4.1-mini`).
- Setting up and initializing the LLM instance with an API key and required parameters like temperature.
- Defining lists of valid LLM product names and NIST risk categories for subsequent data validation.

### **Part 1: Pydantic Models for Structured Output**

This section uses Pydantic to define the target data structure for the extracted information, ensuring the stability and reliability of the model's output format.
- `LLMRiskInfo`: Defines the data model (Schema) for a single risk entry. It includes fields such as `LLMProduct`, `NISTCategory`, `RiskType`, and `UserExperience`. The model also has built-in field validators to ensure that the extracted `LLMProduct` and `NISTCategory` are valid and compliant.
- `StructuredLLMRisks`: Defines a container model that holds a list of `LLMRiskInfo` objects, representing all risk information extracted from a single piece of text.

### **Part 2: LangChain Setup for Structured Output**

This section configures the core extraction logic using LangChain.
- It binds the LLM instance with the Pydantic models defined in Part 1, forcing the model to return output in the specified JSON structure.
- It constructs a very detailed and structured prompt template (`ChatPromptTemplate`) that clearly instructs the LLM on the analysis task's objective, the NIST AI Risk Management Framework to follow, extraction rules, and specific output format requirements.
- Finally, it combines the prompt template and the configured structured LLM into an executable `extraction_chain`.

### **Part 3: Safe Extraction Function with Error Handling**

This section provides a robust function for handling single-text extraction tasks with improved error handling and validation. It is composed of several key components:
- `_validate_input_text`: A helper function that performs initial validation on the input text. It checks if the text is empty, too short, or appears to be an automated bot message, ensuring that only meaningful content is processed.
- `post_process_extraction`: A helper function that cleans the raw data returned by the model. It filters out incomplete results where essential fields like `LLMProduct`, `NISTCategory`, or `RiskType` are missing, ensuring data quality.
- `safe_extract_llm_risks`: The core extraction function, significantly enhanced for reliability. It integrates the validation and post-processing helpers to form a complete pipeline. The function features a sophisticated error-handling mechanism that not only catches API and validation errors but also provides specific, analytical feedback if no risks are extracted. For example, it can report whether the text mentioned an LLM but contained no risk-related keywords, or if the extracted data was filtered out during post-processing.

### **Part 4: Batch Processing Helper**

This section provides the logic for applying the extraction process to an entire dataset.
- The `process_reddit_posts_structured` function takes a pandas DataFrame as input, iterates through the data rows, and calls `safe_extract_llm_risks` for each text entry. It organizes the extraction results—including risk content, risk count, and any detailed error messages returned by the extraction function—and adds them as new columns. This ensures that if an extraction fails, the specific reason is captured in the output. It returns a new DataFrame containing all processed results.

### **Part 5: Script Execution**

This is the main execution block of the script (`if __name__ == "__main__":`). It is triggered when the `llm_risk_extraction.py` file is run directly. Its primary role is to load a **small sample dataset**, call the batch processing function from Part 4 to process a small number of items (e.g., 10), and then save the results to a CSV file. This process is mainly for quickly testing and verifying that the entire extraction pipeline works correctly. 