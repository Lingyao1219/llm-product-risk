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

This section aims to provide a robust and reliable function to handle the extraction task for a single piece of text.
- `safe_extract_llm_risks`: This is the core function for performing the extraction. It conducts comprehensive input validation before calling the model, checking for things like empty, too short, or bot-generated messages. It calls the `extraction_chain` to perform the extraction and includes a comprehensive error-handling mechanism to manage various issues like network timeouts, API authentication failures, or data validation errors, returning detailed error messages.
- `post_process_extraction`: This is a helper function for post-processing and cleaning the raw data returned by the model, filtering out meaningless or incomplete results.

### **Part 4: Batch Processing Helper**

This section provides the logic for applying the extraction process to an entire dataset.
- The `process_reddit_posts_structured` function takes a pandas DataFrame as input, iterates through the data rows, and calls the `safe_extract_llm_risks` function for each text entry. It organizes the extraction results (including risk content, risk count, and error information) and adds them as new columns, finally returning a new DataFrame containing all processed results.

### **Part 5: Script Execution**

This is the main execution block of the script (`if __name__ == "__main__":`). It is triggered when the `llm_risk_extraction.py` file is run directly. Its primary role is to load a **small sample dataset**, call the batch processing function from Part 4 to process a small number of items (e.g., 10), and then save the results to a CSV file. This process is mainly for quickly testing and verifying that the entire extraction pipeline works correctly. 