# llm_risk_extraction_v5.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any
import json
from tqdm import tqdm
import pandas as pd
import numpy as np

################################################################################################################
# 0. File path and global constants
################################################################################################################
INPUT_CSV_PATH = "data_sample/samples.csv"
OUTPUT_CSV_PATH = "data_extracted/extracted_llm_risks.csv"

# Model configuration to match prompt_base_extraction
MODEL_NAME = "gpt-4.1-mini"

# Initialize the LLM with structured output
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.0,
    api_key="[REDACTED]"
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
def post_process_extraction(risks: List[dict], original_text: str) -> List[dict]:
    """Post-process extracted data to enforce guidelines"""

    cleaned_risks: List[dict] = []

    for risk in risks:
        # At least one meaningful field besides quote
        if any(
            risk.get(field)
            for field in ("LLMProduct", "NISTCategory", "RiskType")
        ):
            cleaned_risks.append(risk)

    return cleaned_risks


def safe_extract_llm_risks(text: str) -> dict:
    """Safely extract LLM risk info from text and return detailed results with error information"""

    # Input validation
    if not isinstance(text, str):
        return {
            "risks": [],
            "error_message": "Input validation failed: Non-string input provided"
        }
    
    if not text.strip():
        return {
            "risks": [],
            "error_message": "Input validation failed: Empty or whitespace-only text"
        }
    
    # Check if text is too short for meaningful extraction
    if len(text.strip()) < 10:
        return {
            "risks": [],
            "error_message": "Content too short: Text contains fewer than 10 characters"
        }
    
    # Check for obvious non-content patterns
    lower_text = text.lower()
    if any(pattern in lower_text for pattern in [
        "i am a bot", "this action was performed automatically", 
        "contact the moderators", "your submission has been removed",
        "inadequate account karma"
    ]):
        return {
            "risks": [],
            "error_message": "Content type: Automated message or moderation notice"
        }

    try:
        result = extraction_chain.invoke({"text": text})
        extracted = result.model_dump()
        risks = extracted.get("risks", [])
        processed_risks = post_process_extraction(risks, text)
        
        if not processed_risks:
            # Check for specific reasons why no risks were found
            has_llm_mention = any(product.lower() in lower_text for product in VALID_LLM_PRODUCTS)
            has_nist_keywords = any(keyword in lower_text for keyword in [
                "risk", "problem", "issue", "error", "fail", "bug", "wrong", 
                "dangerous", "harmful", "bias", "privacy", "security", "safe"
            ])
            
            if not has_llm_mention:
                error_msg = "Content analysis: No LLM products mentioned in text"
            elif not has_nist_keywords:
                error_msg = "Content analysis: No risk-related keywords found"
            else:
                error_msg = "Content analysis: No extractable risks meeting NIST framework criteria"
            
            return {
                "risks": [],
                "error_message": error_msg
            }
        
        return {
            "risks": processed_risks,
            "error_message": None
        }
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Categorize different types of API/processing errors
        if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
            detailed_error = f"API rate limit: {error_msg[:80]}..."
        elif "timeout" in error_msg.lower():
            detailed_error = f"API timeout: {error_msg[:80]}..."
        elif "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            detailed_error = f"API authentication error: {error_msg[:80]}..."
        elif "validation" in error_msg.lower():
            detailed_error = f"Data validation error: {error_msg[:80]}..."
        else:
            detailed_error = f"Processing error ({error_type}): {error_msg[:80]}..."
        
        print(f"Extraction failed: {detailed_error}")
        return {
            "risks": [],
            "error_message": detailed_error
        }

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