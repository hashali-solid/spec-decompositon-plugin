# Spec Decomposition Plugin for Dataiku

**ID:** `spec-decomposer`  
**Version:** 0.0.1  
**Author:** Ali.Hashim@solidigmtechnology.com  
**License:** Apache Software License

---

## Overview

The **Specification Decomposition Plugin** is a sophisticated Dataiku custom recipe that automates the extraction and analysis of technical specifications (NVMe, PCI Express, etc.) for SSD (Solid State Drive) validation testing. It employs a multi-agent workflow architecture to intelligently decompose complex specifications into structured, actionable validation modules.

### Core Purpose

This plugin processes technical specification documents and automatically:
1. **Extracts specification structure** - identifies table of contents and relevant sections
2. **Identifies callable commands** - discovers all executable commands within specifications
3. **Assesses command relevance** - determines which commands are critical for RCV (Requirements Coverage Verification) testing
4. **Defines validation modules** - organizes commands into testable modules
5. **Extracts parameters** - identifies all command parameters and constraints
6. **Generates validation rules** - creates structured validation criteria for test coverage documentation

---

## Architecture

### High-Level Design

The plugin implements a **multi-stage LLM-powered workflow** using the following technologies:

```
Specification Documents
        ↓
    [Cortex Search Service] - Semantic Search Layer
        ↓
[LangGraph Multi-Agent Pipeline] - Orchestration
        ↓
[Parallel Worker Functions] - Concurrent Processing
        ↓
Structured Output (JSON) - Organized Results
```

### Key Components

#### 1. **Semantic Search Layer: Snowflake Cortex Search Service**
- Replaces traditional vector stores with native Snowflake semantic search
- Enables natural language queries over specification documents
- Provides metadata-enhanced search results
- Configuration via `cortex_log_dataset` input with `service_fqn` and `attribute_aliases`

#### 2. **Multi-Agent Workflow: LangGraph**
- **StateGraph-based orchestration** manages the entire processing pipeline
- **Send() operations** spawn parallel workers for scalable processing
- **Checkpoint system** saves intermediate results to avoid reprocessing
- **Automatic state merging** via `Annotated[list, operator.add]` for worker results

#### 3. **Worker Architecture**
- Parallel processing of specifications using LangGraph's `Send()` construct
- Independent worker functions handle specific extraction tasks
- **Rate limit resilience** with exponential backoff retry mechanism
- Isolated state management through `CURRENT_SPEC_FULL_PATH` global variable

#### 4. **RAG (Retrieval-Augmented Generation)**
- Combines Cortex Search retrieval with LLM-based analysis
- Chains retrieve specification content → prompt → LLM → structured output
- Multiple retrieval strategies (5/7/15/25 results) for different use cases

---

## Recipe Inputs & Outputs

### Input Datasets/Folders

| Input | Purpose | Schema |
|-------|---------|--------|
| **specs_llm_text_files_dataset** | Specification pages for processing | `document_name`, `page_number`, `content` |
| **cortex_log_dataset** | Cortex Search Service configuration | `service_fqn`, `on_column_alias`, `attribute_aliases` |
| **data_collection_folder** | Checkpoint storage for intermediate results | Organized by `{spec_path}/{stage}/` |

### Output Folder Structure

```
multi_agent_output_folder/
├── {spec_path}/
│   └── {timestamp_YYYYMMDD_HHMMSS}/
│       ├── spec_general_info.json          # Specification metadata
│       ├── content_index_pages.json         # Index page ranges
│       ├── specs_relevant_sections.json     # Relevant section names
│       ├── specs_callable_commands.json     # Extracted commands
│       ├── specs_cmd_modules_definition.json # Module definitions
│       └── {command_name}/
│           ├── command_params.json         # Parameter definitions
│           ├── module_rules.json           # Validation rules
│           └── enhanced_params.json        # Contextually enhanced parameters
```

---

## Recipe Configuration Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| **llmID** | String | LLM model identifier (e.g., `azureopenai:Azure-OpenAI-Prod-4-1:gpt-4.1`) |
| **snowflake_connection** | String | Dataiku Snowflake connection name for Cortex Search |
| **document_name_column** | String | Column name in input dataset containing document paths |
| **page_number_column** | String | Column name containing page numbers |
| **content_column** | String | Column name containing page text content |

### Classification Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| **basestore_paths** | List[String] | Primary specification paths (fully processed through workflow) |
| **other_priorities_paths** | List[String] | Secondary specification paths (captured but not fully processed) |

---

## Workflow Pipeline Stages

### Stage 1: Index Page Extraction
**Function:** `locate_spec_index_pages()`  
**Purpose:** Identify table of contents and index page ranges  
**Output:** `content_index_pages` - maps spec path → page range

```python
# Example output
{"spec_path": {"start_index_page": 5, "end_index_page": 15}}
```

### Stage 2: Relevant Sections Identification
**Function:** `identify_relevant_spec_sections()`  
**Purpose:** Extract relevant section names from index  
**Output:** `specs_relevant_sections` - comma-separated section names

```python
# Example output
{"spec_path": "Admin Commands, NVMe Commands, Features, Configuration"}
```

### Stage 3: Callable Command Extraction
**Function:** `extract_callable_commands()`  
**Purpose:** Identify all executable commands from specification sections  
**Output:** `complete_callable_commands` - ModuleStruct with command details

```python
class ModuleStruct(BaseModel):
    command_names: str      # Comma-separated command names
    keyPhrases: str         # Key phrases for command description
    score: float           # Confidence score (0-1)
```

### Stage 4: Command Pruning
**Function:** `prune_command_list()`  
**Purpose:** Remove duplicate/similar commands using semantic similarity  
**Output:** `complete_unique_callable_commands` - deduplicated command list

### Stage 5: Command Relevance Assessment
**Function:** `retrieve_modules_cmd_info()`  
**Purpose:** Assess which commands are relevant for RCV validation testing  
**Output:** `completed_module_cmds` - CommandRelevanceAssessment per command

```python
class CommandRelevanceAssessment(BaseModel):
    command_name: str
    relevance_score: float
    validation_scope: str   # Single module vs cross-module
    variants: List[str]     # Command variants/aliases
```

### Stage 6: Module Definition
**Function:** `define_cmd_modules()`  
**Purpose:** Define validation modules for each command  
**Output:** `completed_cmd_modules_definition` - ProposedModules per command

```python
class ProposedModules(BaseModel):
    module_names: List[str]
    module_descriptions: List[str]
    cross_module_dependencies: List[str]
```

### Stage 7: Parameter Extraction
**Function:** `built_command_params_modules()`  
**Purpose:** Extract all parameters, types, ranges, and constraints  
**Output:** `completed_module_cmds_param` - CommandParametersDefinition per module

```python
class CommandParametersDefinition(BaseModel):
    parameter_name: str
    parameter_type: str     # e.g., "integer", "enum", "boolean"
    min_value: Optional[float]
    max_value: Optional[float]
    allowed_values: Optional[List[str]]
    description: str
```

### Stage 8: Parameter Enhancement
**Function:** `enhance_parameter_context()`  
**Purpose:** Augment parameter definitions with cross-specification context  
**Output:** `completed_module_cmds_enha_param` - enhanced parameters

### Stage 9: Validation Rules Generation
**Function:** `define_command_rules_modules()`  
**Purpose:** Generate validation rules for each module  
**Output:** `completed_module_cmds_[global_rules|module_rules]`

```python
class ValidationRule(BaseModel):
    rule_name: str
    rule_description: str
    condition: str          # What condition triggers the rule
    expected_behavior: str   # What should happen
    error_handling: str      # How to handle violations
```

### Stage 10: Global Rules Consolidation
**Function:** `define_modules_global_rules()`  
**Purpose:** Merge duplicate rules across modules into global rules  
**Output:** `completed_final_global_rules` - deduplicated rules

---

## Technical Implementation Details

### Rate Limit Handling

The `invoke_with_rate_limit_retry()` function implements sophisticated rate limit management:

```python
def invoke_with_rate_limit_retry(chain, input_data, max_retries=15, 
                                 base_delay=2, max_wait_time=120):
    """
    Retry chain invocation with exponential backoff and jitter.
    
    - Detects HTTP 429 errors and null response errors
    - Parses 'retry-after' headers when available
    - Implements exponential backoff: delay = base_delay * 2^attempt
    - Adds ±20% jitter to prevent thundering herd problem
    - Caps maximum wait time to prevent excessive delays
    """
```

**Key Features:**
- Handles up to 15 concurrent worker retries simultaneously
- Respects API retry-after headers
- Jitter prevents synchronized retry storms
- 2-120 second delay range prevents resource exhaustion

### Checkpoint System

Intermediate results are saved to `data_collection_folder` at each stage:

```python
def dataCollectionIndexExtraction():
    """Load/save checkpoint for index page extraction"""
    checkpoint_path = f"{spec_path}/index_pages.json"
    # If exists: load and skip processing
    # If not: process, save, and continue
```

**Benefits:**
- Resume interrupted processing without restarting
- Skip already-processed specifications
- Reduce API costs and processing time
- Enable iterative refinement

### Parallel Worker Processing

Workers are spawned using `Send()` to enable concurrent processing:

```python
def assign_workers_callable_commands_extraction(state):
    """Spawn parallel workers for command extraction"""
    return {
        "missing_spec_paths_to_process": [
            Send("extract_callable_commands_worker", {"spec_path": path})
            for path in missing_specs
        ]
    }
```

---

## Data Processing Flow

### Input Data Transformation

1. **Load Dataset** → pandas DataFrame with document_name, page_number, content
2. **Classification** → partition into 'baseStore' vs 'other-priorities'
3. **Organization** → create `specsStores_df` indexed by filepath
4. **Cortex Search** → semantic search over organized content

### Specification Classification

```python
# baseSpecPaths: Primary specs (e.g., ["NVMe", "PCI_Express"])
# - Fully processed through entire workflow
# otherPriorSpecPaths: Secondary specs (e.g., ["SAS"])
# - Captured but partially processed
```

Document matching is flexible:
```python
# Match if document_name:
#   - Starts with path: "NVMe_Base_Spec" matches "NVMe"
#   - Contains path: "specs/NVMe/v2_1" contains "NVMe"
```

---

## LLM Integration

### Model Configuration

The plugin uses **Azure OpenAI GPT-4** for all LLM operations:

```python
llm = DKUChatLLM(llm_id="azureopenai:Azure-OpenAI-Prod-4-1:gpt-4.1", temperature=0.2)
```

**Temperature Setting:** 0.2 (low) for consistent, deterministic extraction

### Prompt Templates

Each stage uses specialized prompts via `ChatPromptTemplate`:

- **Extract Commands:** "Identify all executable commands in: {context}"
- **Assess Relevance:** "Is this command relevant for SSD validation testing?"
- **Define Modules:** "Propose validation modules for: {command_name}"
- **Extract Parameters:** "List all parameters for: {command_name}"
- **Generate Rules:** "Create validation rules for: {module_name}"

### RAG Chain Pattern

Standard pattern across all LLM operations:

```python
rag_chain = {
    "context": cortex_retriever,  # Retrieve relevant spec content
    "module": RunnablePassthrough()
} | prompt_template | llm  # Chain → prompt → LLM

result = invoke_with_rate_limit_retry(rag_chain, input_data)
```

---

## State Management

### ModuleContentState TypedDict

Central state structure tracking all workflow data:

```python
class ModuleContentState(TypedDict):
    # Accumulated results from parallel workers
    missing_spec_index_pages: Annotated[list, operator.add]
    missing_spec_relevant_sections: Annotated[list, operator.add]
    complete_callable_commands: Annotated[list, operator.add]
    completed_module_cmds: Annotated[list, operator.add]
    completed_module_cmds_param: Annotated[list, operator.add]
    completed_module_cmds_global_rules: Annotated[list, operator.add]
    
    # Organized results by specification path
    specs_relevant_sections: dict
    specs_callable_unique_cmds: dict
    specs_cmd_modules_definition: dict
    specs_cmd_modules_complete_info: dict
    
    # Workflow control
    missing_spec_paths_to_process: list
    save_dataCollection: bool
```

### State Merging

`Annotated[list, operator.add]` fields automatically merge results from parallel workers:

```python
# Worker 1 returns: [{"spec_a": {...}}]
# Worker 2 returns: [{"spec_b": {...}}]
# State merges: [{"spec_a": {...}}, {"spec_b": {...}}]
```

---

## RCV Integration

### RCV Framework Context

The extracted modules and rules integrate with **RCV (Requirements Coverage Verification)**:

> RCV is a comprehensive validation system designed to automate and enhance testing of enterprise SSDs by managing workloads, test sequences, and verification logic. It integrates configurable workloads (read/write operations, error injections), execution runners (random/sequential test sequences), and modular elements (generators, verifiers, rule sets, test knobs).

### Module to RCV Test Mapping

Each extracted module maps to RCV test components:

```
Command Module → RCV Module
├── Parameters → Configuration (DriveConfig)
├── Validation Rules → Verification Logic
├── Module Rules → Response Checking
└── Dependencies → Cross-module Execution
```

---

## Performance Considerations

### Processing Scalability

- **Parallel Workers:** Concurrent processing of multiple specifications
- **Rate Limiting:** Handles API throttling gracefully
- **Checkpoint Recovery:** Resume from intermediate stages
- **Cortex Search Caching:** Reuses connection across searches

### Memory Efficiency

- Streaming document processing (don't load entire spec into memory)
- Text truncation for logging (reduces log file size)
- Lazy loading of dataframes (only load when needed)

### API Cost Optimization

- Reuse Cortex Search connections (cached)
- Limit number of results per search (5-25 documents depending on use case)
- Checkpoint system prevents reprocessing
- Temperature=0.2 produces consistent outputs (fewer retries needed)

---

## Logging & Debugging

### Debug Output

The recipe provides detailed logging:

```python
print(f"[DEBUG] INIT: baseSpecPaths = {baseSpecPaths}")
print(f"[DEBUG] INIT: specsLLMTextFiles_df shape: {shape}")
print(f"[RETRY] Rate limit error, waiting {wait_time:.1f}s before retry")
print(f"[RETRY] Final error: {error[:500]}")
```

### Log Levels

- **WARNING:** Snowflake SDK logs reduced to avoid verbosity
- **DEBUG:** Detailed processing information
- **RETRY:** Rate limit and error recovery events

### Text Truncation Utility

`truncate_text_for_log()` reduces log file size:
```python
# Long text: "abc...defghijk"
# Shows first/last 100 chars with ellipsis
```

---

## Error Handling

### Rate Limit Errors (HTTP 429)
- Detected and retried automatically
- Exponential backoff with jitter
- Up to 15 retry attempts
- Respects retry-after headers

### Null Response Errors
- Transient API/network issues
- Trigger exponential backoff retry
- Separate handling from rate limits

### Validation Errors
- Column existence checks (raises ValueError)
- Type conversion validation (skips invalid rows)
- Null/None handling (graceful degradation)

---

## Usage Example

### Recipe Configuration

1. **Dataset Input:** Upload specification pages with columns:
   - `document_name`: "NVMe_Base_Spec_v2_1"
   - `page_number`: 1, 2, 3...
   - `content`: "The NVMe specification defines..."

2. **Folder Input:** Configure Cortex Search Service:
   - `service_fqn`: "DATAIKU.LLMS.SPEC_SEARCH"
   - `on_column_alias`: "LLM_OUTPUT"
   - `attribute_aliases`: "DOCUMENT_NAME,DOCUMENT_TYPE,VERSION"

3. **Configuration:**
   - `basestore_paths`: ["NVMe", "PCI_Express"]
   - `llmID`: "azureopenai:Azure-OpenAI-Prod-4-1:gpt-4.1"
   - `snowflake_connection`: "snowflake_prod"

4. **Run Recipe** → Outputs JSON files organized by spec/timestamp/command

---

## Dependencies

- **Dataiku:** Recipe framework & LLM integration (DKUChatLLM)
- **LangGraph:** Multi-agent workflow orchestration (StateGraph, Send)
- **LangChain:** RAG chains, document processing, prompts
- **Snowflake:** Cortex Search Service for semantic search
- **Pydantic:** Structured output validation & schema definition
- **Python:** Standard libraries (json, pandas, datetime, logging)

---

## Future Enhancements

- [ ] Support for multiple LLM providers (Claude, Gemini)
- [ ] Enhanced cross-specification rule merging (beyond string deduplication)
- [ ] Real-time streaming output to dashboards
- [ ] Parameter range inference from constraint specifications
- [ ] Automated test case generation from validation rules
- [ ] Integration with CI/CD pipelines for continuous specification updates

---

## Support & Contributing

For issues, enhancements, or questions:
- **Author:** Ali.Hashim@solidigmtechnology.com
- **Plugin ID:** spec-decomposer
- **License:** Apache Software License

This plugin represents the intersection of LLM-powered analysis, retrieval-augmented generation, and multi-agent workflows to automate specification decomposition at scale.
