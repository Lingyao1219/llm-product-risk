# Schema-Based LLM Risk Extraction (基于 Schema 的 LLM 风险提取)

该目录包含使用 LangChain 和 Pydantic 的结构化方法从文本数据中提取大型语言模型 (LLM) 风险信息的脚本。

## 文件 (Files)

- `llm_risk_extraction.py`: 定义和运行数据样本提取过程的主要脚本。
- `run_extraction.py`: 在整个数据集上运行提取过程的辅助脚本。

## `llm_risk_extraction.py` 结构

该脚本分为几个部分，下面将详细说明每个部分的功能：

### **第 0 部分：文件路径和全局常量 (File Paths and Global Constants)**

此部分负责初始化脚本的配置。它主要进行以下工作：
- 定义输入和输出文件的路径。
- 指定要使用的 LLM 模型（例如 `gpt-4.1-mini`）。
- 使用 API 密钥和所需参数（如温度）来设置和初始化 LLM 实例。
- 定义用于后续数据验证的有效 LLM 产品名称列表和 NIST 风险类别列表。

### **第 1 部分：用于结构化输出的 Pydantic 模型 (Pydantic Models for Structured Output)**

此部分使用 Pydantic 定义了提取信息的目标数据结构，以确保模型输出格式的稳定性和可靠性。
- `LLMRiskInfo`: 定义单个风险条目的数据模型（Schema）。它包含了如 `LLMProduct`（LLM产品）、`NISTCategory`（NIST类别）、`RiskType`（风险类型）和 `UserExperience`（用户体验）等字段。该模型还内置了字段验证器，以确保提取出的 `LLMProduct` 和 `NISTCategory` 是有效和合规的。
- `StructuredLLMRisks`: 定义一个容器模型，其功能是容纳一个由多个 `LLMRiskInfo` 对象组成的列表，用于表示从单个文本中提取出的所有风险信息。

### **第 2 部分：设置 LangChain 以进行结构化输出 (LangChain Setup for Structured Output)**

此部分配置了使用 LangChain 的核心提取逻辑。
- 它将 LLM 实例与在第 1 部分中定义的 Pydantic 模型进行绑定，从而强制模型以指定的 JSON 结构返回输出。
- 它构建了一个非常详细和结构化的提示模板（`ChatPromptTemplate`），该模板清晰地向 LLM 指示了分析任务的目标、需要遵循的NIST AI风险管理框架、提取规则以及具体的输出格式要求。
- 最后，它将提示模板和配置好的结构化 LLM 组合成一个可执行的 `extraction_chain`。

### **第 3 部分：带错误处理的安全提取函数 (Safe Extraction Function with Error Handling)**

此部分提供了一个稳健的函数，通过改进的错误处理和验证机制来处理单个文本的提取任务。它由几个关键组件构成：
- `_validate_input_text`: 一个辅助函数，对输入文本执行初始验证。它会检查文本是否为空、过短或疑似自动机器人消息，以确保只处理有意义的内容。
- `post_process_extraction`: 一个辅助函数，用于清洗模型返回的原始数据。它会过滤掉 `LLMProduct`、`NISTCategory` 或 `RiskType` 等基本字段缺失的不完整结果，以确保数据质量。
- `safe_extract_llm_risks`: 核心提取函数，其可靠性已得到显著增强。它集成了验证和后处理辅助函数，构成一个完整的处理流水线。该函数拥有一个复杂的错误处理机制，不仅能捕获 API 和验证错误，还能在未提取到风险时提供具体的分析性反馈。例如，它可以报告文本是否提到了 LLM 但不包含风险相关关键词，或者提取出的数据是否在后处理过程中被过滤掉了。

### **第 4 部分：批量处理辅助函数 (Batch Processing Helper)**

此部分提供了将提取流程应用于整个数据集的逻辑。
- `process_reddit_posts_structured` 函数接收一个 pandas DataFrame 作为输入，然后遍历数据行，并对每个文本条目调用 `safe_extract_llm_risks` 函数。它会将提取结果——包括风险内容、风险数量以及从提取函数返回的任何详细错误信息——整理并添加为新的列。这确保了在提取失败时，具体原因能够被记录在输出中。最终，该函数返回一个包含所有处理结果的、新的 DataFrame。

### **第 5 部分：脚本执行 (Script Execution)**

这是脚本的主执行模块（`if __name__ == "__main__":`）。当用户直接运行此 `llm_risk_extraction.py` 文件时，该模块会被触发。它的主要作用是加载一个**小规模的样本数据**，调用第 4 部分的批量处理函数来处理少量数据（例如10条），然后将结果保存到CSV文件中。这个过程主要用于快速测试和验证整个提取流程是否能够正常工作。