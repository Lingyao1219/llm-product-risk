# llm_risk_extraction_v5.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Optional, Any
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
from openai import RateLimitError, AuthenticationError, APITimeoutError

################################################################################################################
# 0. File path and global constants
################################################################################################################
INPUT_CSV_PATH = "data_sample/samples.csv"
OUTPUT_CSV_PATH = "data_extracted/extracted_llm_risks.csv"

# Model configuration to match prompt_base_extraction
MODEL_NAME = "gpt-4.1-mini"

# Configure logging to capture important events
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the LLM with structured output
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.0,
    # Best practice: do not hardcode API keys. Use environment variable or config file.
    api_key=None  # Set to None to avoid leaking secrets; should be set via environment variable
)

# Valid LLM Products and NIST Categories
VALID_LLM_PRODUCTS = [
    "GPT", "Claude", "Llama", "Gemini", "Mistral", "Deepseek", "Qwen"
]
NIST_CATEGORIES = [
    "Valid and Reliable",
    "Safe",
    "Secure and Resilient",
    "Accountable and Transparent",
    "Explainable and Interpretable",
    "Privacy",
    "Fair"
]

################################################################################################################
# 1. Define Pydantic Models for Structured Output
################################################################################################################
class LLMRiskInfo(BaseModel):
    """LLM risk information extraction model"""

    LLMProduct: Optional[str] = Field(
        None,
        description="Specific product or model name or null if not mentioned"
    )
    NISTCategory: Optional[str] = Field(
        None,
        description="One of the seven NIST AI RMF categories or null if not mentioned"
    )
    RiskType: Optional[str] = Field(
        None,
        description="Specific risk manifestation or null if not mentioned"
    )
    UserExperience: Optional[str] = Field(
        None,
        description="Direct quote from Reddit post supporting the risk or null"
    )

    # --- Validators -------------------------------------------------------------------------------------------------
    @field_validator("LLMProduct", mode="before")
    @classmethod
    def validate_llm_product(cls, v: Any):
        if v is None or v == "null":
            return None
        for product in VALID_LLM_PRODUCTS:
            if product.upper() in str(v).upper():
                return v  # keep original string to preserve version info
        return None

    @field_validator("NISTCategory", mode="before")
    @classmethod
    def validate_nist_category(cls, v: Any):
        if v is None or v == "null":
            return None
        return v if v in NIST_CATEGORIES else None

class StructuredLLMRisks(BaseModel):
    """Container for extracted risks"""

    risks: List[LLMRiskInfo] = Field(default_factory=list, description="List of extracted risks")

################################################################################################################
# 2. Set up LangChain with Structured Output
################################################################################################################
# Build structured output chain
structured_llm = llm.with_structured_output(StructuredLLMRisks)

SYSTEM_MESSAGE = """You are an expert in information extraction. 
Your task is to analyze Reddit post and identify the entities and their associated information."""

HUMAN_MESSAGE = """
Identify and extract the information about the potential risks of using large language models (LLMs) according to the NIST AI Risk Management Framework trustworthy AI characteristics, from the given Reddit post: {text}

Return the extracted information as a Python list of dictionaries:

[
   {{
       "LLMProduct": "specific product or model name, or 'null' if not mentioned",
       "NISTCategory": "one of the 7 NIST categories, or 'null' if not mentioned",
       "RiskType": "specific risk manifestation, or 'null' if not mentioned",
       "UserExperience": "direct quote from a Reddit post that is coded as a RiskType and NISTCategory, or 'null' if not implying risk"
   }}
   ...
]

### Guidelines:
**Extract only these LLM products or families: "GPT", "Claude", "Llama", "Gemini", "Mistral", "Deepseek", "Qwen".**
**Use the exact name mentioned for LLMProduct (e.g., "o3", "Claude 3.5", "GPT-4").**
**The "NISTCategory" must be one of the 7 NIST AI RMF Trustworthy AI Characteristics listed below.**
**The "RiskType" should describe the specific risk manifestation. You should use the examples provided as references and create appropriate descriptions for risks beyond those examples.**
**The "UserExperience" must be the direct and complete textual quote from the input that supports the identified RiskType and NISTCategory.**
**If no relevant information can be extracted, return string "null" for that dimension.**
**Return ONLY a valid Python list of dictionaries.**

**NIST AI RMF Trustworthy AI Characteristics and Example Manifestations:**
**1. Valid and Reliable:** AI systems should be accurate, consistent, and produce dependable results.
   - Example manifestations: Hallucination, inaccuracy, inconsistency, unreliability, degradation over time without notice, lack of performance validation across deployment contexts, or others (please list if others)
**2. Safe:** AI systems should prevent unintended harm for users.
   - Example manifestations: Harmful content, dangerous instructions, physical harm, or others (please list if others)
**3. Secure and Resilient:** AI systems should be protected from threats and maintain function when compromised.
   - Example manifestations: Prompt injection, jailbreaking, data leaks, manipulation, vulnerability, system downtime, service disruption, exploitable vulnerabilities in model or infrastructure, resilience failure, or others (please list if others)
**4. Accountable and Transparent:** There should be clarity about how the AI system operates and responsibility for outcomes.
   - Example manifestations: Opacity, poor documentation, accountability gaps, disclosure issues, or others (please list if others)
**5. Explainable and Interpretable:** AI system decisions should be understandable to humans.
   - Example manifestations: Unexplainable decisions, opaque reasoning, unclear outputs, outputs that are difficult to interpret, inability to provide reasons for critical decisions, or others (please list if others)
**6. Privacy:** AI systems should protect individual privacy and manage data responsibly. AI systems must respect user privacy and adhere to responsible data management practices.
   - Example manifestations: Data exposure, privacy violation, unauthorized collection, inadequate data anonymization, non-compliance with privacy regulations, or others (please list if others)
**7. Fair:** AI systems should treat individuals and groups equitably.
   - Example manifestations: Discrimination, bias, unfair treatment, stereotyping, lack of representation, failure to evaluate, or others (please list if others)

### Examples:

**Input:** "ChatGPT is very prone to failing dont think about elephants lol"
**Output:**
[
   {{
       "LLMProduct": "GPT",
       "NISTCategory": "Valid and Reliable",
       "RiskType": "Prompt failure",
       "UserExperience": "very prone to failing dont think about elephants"
   }}
]

**Input:** "Claude 3.5 > DeepSeek Coder v2 > DeepSeek Coder v2 Lite | Codestral"
**Output:**
[
   {{
       "LLMProduct": "Claude",
       "NISTCategory": "null",
       "RiskType": "null",
       "UserExperience": "null"
   }},
   {{
       "LLMProduct": "DeepSeek",
       "NISTCategory": "null",
       "RiskType": "null",
       "UserExperience": "null"
   }}
]

**Input:** "GPT is often confidently wrong which is why it's useless for cheating. Just fail the cheaters like it's always been."
**Output:**
[
   {{
       "LLMProduct": "GPT",
       "NISTCategory": "Valid and Reliable",
       "RiskType": "Confident inaccuracy",
       "UserExperience": "often confidently wrong"
   }}
]

**Input:** "Testing showed that when asked about hiring decisions, multiple AI models including Gemini consistently rated male candidates higher for technical roles, even with identical qualifications."
**Output:**
[
   {{
       "LLMProduct": "Gemini",
       "NISTCategory": "Fair",
       "RiskType": "Gender bias",
       "UserExperience": "consistently rated male candidates higher for technical roles, even with identical qualifications"
   }}
]
"""

# Define ChatPromptTemplate
extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MESSAGE),
    ("human", HUMAN_MESSAGE)
])

# Compose chain
extraction_chain = extraction_prompt | structured_llm

################################################################################################################
# 3. Safe extraction function with error handling
################################################################################################################
def _validate_input_text(text: str) -> Optional[str]:
    """
    Validates the input text against a series of checks.
    
    Args:
        text: The text to validate.
        
    Returns:
        An error message string if validation fails, otherwise None.
    """
    if not isinstance(text, str):
        return "Input validation failed: Non-string input provided"
    
    if not text.strip():
        return "Input validation failed: Empty or whitespace-only text"
    
    # Check for content that is too short to be meaningful
    if len(text.strip()) < 10:
        return "Content too short: Text contains fewer than 10 characters"
    
    # Filter out common automated messages from Reddit
    lower_text = text.lower()
    automated_patterns = [
        "i am a bot", "this action was performed automatically", 
        "contact the moderators", "your submission has been removed",
        "inadequate account karma"
    ]
    if any(pattern in lower_text for pattern in automated_patterns):
        return "Content type: Automated message or moderation notice"
        
    return None


def post_process_extraction(risks: List[dict], original_text: str) -> List[dict]:
    """Post-process extracted data to enforce guidelines"""

    cleaned_risks: List[dict] = []

    for risk in risks:
        # Check if all three critical fields are non-null and non-empty
        llm_product = risk.get("LLMProduct")
        nist_category = risk.get("NISTCategory") 
        risk_type = risk.get("RiskType")
        
        # All three fields must be present, non-null, not "null" string, and not empty
        if (llm_product and llm_product != "null" and str(llm_product).strip() and
            nist_category and nist_category != "null" and str(nist_category).strip() and
            risk_type and risk_type != "null" and str(risk_type).strip()):
            cleaned_risks.append(risk)

    return cleaned_risks


def safe_extract_llm_risks(text: str) -> dict:
    """
    Safely extracts LLM risk information from text with improved error handling.
    
    This function performs the following steps:
    1. Validates the input text for basic quality and relevance.
    2. Invokes an LLM chain to extract structured risk information.
    3. Post-processes the extracted data to ensure it meets quality guidelines.
    4. Provides detailed error messages for different failure scenarios,
       including input validation, API errors, and content analysis failures.

    Args:
        text: The input string from which to extract LLM risks.

    Returns:
        A dictionary containing:
        - "risks" (List[dict]): A list of extracted and processed risk dictionaries.
        - "error_message" (Optional[str]): An error message if extraction fails, otherwise None.
    """
    # 1. Input Validation
    validation_error = _validate_input_text(text)
    if validation_error:
        return {"risks": [], "error_message": validation_error}

    try:
        # 2. Invoke LLM and process results
        result = extraction_chain.invoke({"text": text})
        extracted = result.model_dump()
        raw_risks = extracted.get("risks", [])
        processed_risks = post_process_extraction(raw_risks, text)
        
        # 3. Analyze results and generate specific feedback if no risks are found
        if not processed_risks:
            error_message = "Content analysis: No valid risks found after processing."
            
            # If the LLM returned some data but it was filtered out
            if raw_risks:
                error_message = "Content analysis: Extracted data was incomplete and filtered by post-processing."
            else:
                # If LLM returned nothing, analyze why
                lower_text = text.lower()
                has_llm_mention = any(product.lower() in lower_text for product in VALID_LLM_PRODUCTS)
                has_nist_keywords = any(keyword in lower_text for keyword in [
                    "risk", "problem", "issue", "error", "fail", "bug", "wrong", 
                    "dangerous", "harmful", "bias", "privacy", "security", "safe"
                ])
                
                if not has_llm_mention:
                    error_message = "Content analysis: No specified LLM products were mentioned."
                elif not has_nist_keywords:
                    error_message = "Content analysis: Text mentions an LLM but contains no risk-related keywords."
                else:
                    error_message = "Content analysis: No risks meeting the full criteria could be extracted."
            
            return {"risks": [], "error_message": error_message}
        
        # 4. Return successful extraction
        return {"risks": processed_risks, "error_message": None}
        
    # 5. Handle specific exceptions from API and data validation
    except RateLimitError as e:
        error_msg = f"API Error: Rate limit exceeded. Please wait and try again. Details: {str(e)[:100]}"
        logging.warning(error_msg)
        return {"risks": [], "error_message": error_msg}
        
    except AuthenticationError as e:
        error_msg = f"API Error: Authentication failed. Check API key. Details: {str(e)[:100]}"
        logging.error(error_msg)
        return {"risks": [], "error_message": error_msg}
        
    except APITimeoutError as e:
        error_msg = f"API Error: Request timed out. Details: {str(e)[:100]}"
        logging.warning(error_msg)
        return {"risks": [], "error_message": error_msg}
        
    except ValidationError as e:
        error_msg = f"Data Validation Error: LLM output did not match Pydantic schema. Details: {str(e)[:100]}"
        logging.warning(error_msg)
        return {"risks": [], "error_message": error_msg}

    except Exception as e:
        error_type = type(e).__name__
        error_msg = f"Internal Processing Error: An unexpected error occurred. Type: {error_type}. Details: {str(e)[:100]}"
        logging.error(error_msg, exc_info=True) # Log full traceback for unexpected errors
        
        return {"risks": [], "error_message": error_msg}

################################################################################################################
# 4. Batch processing helper
################################################################################################################

def process_reddit_posts_structured(dataframe: pd.DataFrame, num_posts: int = 100) -> pd.DataFrame:
    """Process a dataframe of reddit posts and append extraction results to match prompt_base_extraction format"""

    processed_rows = []
    df_subset = dataframe.head(num_posts)

    for _, row in tqdm(
        df_subset.iterrows(),
        total=len(df_subset),
        desc=f"Processing first {num_posts} Reddit posts for LLM risk extraction",
    ):
        # Use 'text' column to match prompt_base_extraction config
        post_text = row.get("text", "")

        extraction_result = safe_extract_llm_risks(post_text)
        risks = extraction_result["risks"]
        error_message = extraction_result["error_message"]

        base_row = {**row.to_dict()}
        
        # Match prompt_base_extraction output format
        model_col = MODEL_NAME
        results_col = f"{MODEL_NAME}_results"
        risk_count_col = f"{MODEL_NAME}_risk_count"
        error_col = f"{MODEL_NAME}_error"
        
        # Initialize the base model column (like prompt_base_extraction does)
        base_row[model_col] = None
        
        if risks and isinstance(risks, list):
            # Add results column (JSON format)
            base_row[results_col] = json.dumps(risks, ensure_ascii=False)
            
            # Add risk count
            base_row[risk_count_col] = len(risks)
            
            # Clear error column for successful extractions
            base_row[error_col] = None
            
            # Add individual risk columns
            for risk_idx, risk in enumerate(risks, 1):
                if isinstance(risk, dict):
                    for field, value in risk.items():
                        col_name = f"{MODEL_NAME}_risk{risk_idx}_{field}"
                        base_row[col_name] = value
        else:
            # Handle case where no extraction was possible
            base_row[results_col] = json.dumps([], ensure_ascii=False)
            base_row[risk_count_col] = 0
            # Use detailed error message instead of generic "Extraction failed"
            base_row[error_col] = error_message if error_message else "Extraction failed: Unknown reason"

        processed_rows.append(base_row)

    return pd.DataFrame(processed_rows)

################################################################################################################
# 5. Execute script
################################################################################################################
if __name__ == "__main__":
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"Successfully loaded dataset with {len(df)} posts")
    except FileNotFoundError:
        print(f"Error: Could not find input file {INPUT_CSV_PATH}")
        exit(1)

    num_posts_to_process = 10
    results_df = process_reddit_posts_structured(df, num_posts=num_posts_to_process)

    # Save
    try:
        results_df.to_csv("data_extracted/extracted_llm_risks_sample.csv", index=False)
        print(f"Results saved to {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"Error saving results: {e}")

    print("\nLLM risk extraction process completed successfully!") 