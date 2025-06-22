from config import CONFIG

# System prompts for specific tasks
SYSTEM_PROMPTS = {
    "REDDIT": """You are an expert in information extraction. 
    Your task is to analyze Reddit post and identify the entities and their associated information."""
}

TASK_PROMPTS = {
    "LLM_PROMPT": """
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
       "NISTCategory": "Fair–with Harmful Bias Managed",
       "RiskType": "Gender bias",
       "UserExperience": "consistently rated male candidates higher for technical roles, even with identical qualifications"
   }}
]
"""
}

def get_system_prompt(prompt_name=None):
    """Get system prompt by name"""
    if prompt_name is None:
        prompt_name = CONFIG.get('SYSTEM_PROMPT_NAME')
    return SYSTEM_PROMPTS.get(prompt_name, '')

def get_task_prompt(prompt_name=None):
    """Get task prompt by name"""
    if prompt_name is None:
        prompt_name = CONFIG.get('TASK_PROMPT_NAME')
    return TASK_PROMPTS.get(prompt_name, '')