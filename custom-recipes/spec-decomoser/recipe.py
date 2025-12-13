# -*- coding: utf-8 -*-
"""
SPECIFICATION DECOMPOSITION RECIPE
==================================

Purpose:
    This Dataiku custom recipe processes technical specification documents (e.g., NVMe,
    PCI Express) and automatically extracts structured information for SSD validation
    testing. It uses Large Language Models (LLMs) and a multi-agent workflow to:
    
    1. Extract specification structure (table of contents, relevant sections)
    2. Identify callable commands from specifications
    3. Assess command relevance for RCV (Requirements Coverage Verification) testing
    4. Define validation modules per command
    5. Extract command parameters and validation rules
    6. Generate test coverage documentation

Architecture:
    - Multi-Agent Workflow: Uses LangGraph to orchestrate parallel processing stages
    - Semantic Search: Uses Snowflake Cortex Search Service (replaces vector stores)
    - Checkpoint System: Saves/loads intermediate results to avoid reprocessing
    - RAG (Retrieval-Augmented Generation): Combines document search with LLM analysis

Key Components:
    1. Document Retrieval: cortex_search_to_documents() - semantic search over specs
    2. Workflow Stages: Index extraction → Sections → Commands → Modules → Parameters → Rules
    3. Worker Functions: Parallel processing of specifications using LangGraph Send()
    4. State Management: ModuleContentState TypedDict tracks all workflow data
    5. Checkpoint System: dataCollection* functions load/save intermediate results

Inputs (Dataiku Recipe):
    - specs_llm_text_files_dataset: Specification pages with document_name, page_number, content
    - data_collection_folder: Checkpoint storage for intermediate results
    - cortex_log_dataset: Cortex Search Service configuration

Outputs (Dataiku Recipe):
    - multi_agent_output_folder: Final results organized by spec path and timestamp
      Structure: {specPath}/{timestamp}/{commandName}/{command_params.json, module_rules.json, etc.}

Workflow Flow:
    START → Check Index Pages → Extract/Load → Check Relevant Sections → Extract/Load →
    → Extract Commands → Prune → Assess Relevance → Define Modules → Extract Parameters →
    → Enhance Parameters → Generate Rules → Consolidate Global Rules → END

Dependencies:
    - Dataiku: Recipe framework and LLM integration
    - LangGraph: Multi-agent workflow orchestration
    - LangChain: RAG chains and document processing
    - Snowflake: Cortex Search Service for semantic search
    - Pydantic: Structured output validation

Author: Dataiku Recipe
Last Updated: 2024
"""

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Configuration and Initialization

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Libraries Import

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Python Libraries
import ast
import pandas as pd, numpy as np
import json
import time
import io
import operator
from datetime import datetime
from uuid import uuid4
from copy import deepcopy
import re
import logging

# Typing
from typing import Annotated, Literal, Sequence, List, Any, Dict
from typing_extensions import TypedDict


# IPython (optional - only available in notebook environments)
try:
    from IPython.display import Image, display, Markdown
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    # Define dummy functions for non-notebook environments
    def display(*args, **kwargs):
        pass
    def Image(*args, **kwargs):
        return None
    def Markdown(*args, **kwargs):
        return None

# Dataiku Libraries
import dataiku
from dataiku import pandasutils as pdu
from dataiku.langchain.dku_llm import DKUChatLLM
from dataiku.snowpark import DkuSnowpark
from dataiku.customrecipe import (
    get_input_names_for_role,
    get_output_names_for_role,
    get_recipe_config,
)

# Snowflake
from snowflake.core import Root

# Pydantic
from pydantic import BaseModel, Field

# LangChain
from langchain import hub
## Chains
from langchain.chains.combine_documents import create_stuff_documents_chain
## DocStore
from langchain.docstore.document import Document
## Embeddings
from langchain.embeddings.base import Embeddings
## Tools Retriever
from langchain.tools.retriever import create_retriever_tool
## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain Core
## Documents
from langchain_core.documents import Document
## Messages
from langchain_core.messages import BaseMessage, HumanMessage
## Output Parsers
from langchain_core.output_parsers import StrOutputParser
## Prompts
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
# Runnables
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# LangChain Community
## VectorStores
# Note: FAISS and DistanceStrategy removed - using Cortex Search Service instead

# LangGraph
## Constants
from langgraph.constants import Send
## Graph
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
## Prebuilt
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Config Values

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Configure logging to reduce Snowflake SDK verbosity
# This reduces the amount of document content logged by Snowflake SDK debug logs
logging.getLogger('snowflake').setLevel(logging.WARNING)
logging.getLogger('snowflake.core').setLevel(logging.WARNING)
logging.getLogger('snowflake.connector').setLevel(logging.WARNING)
logging.getLogger('snowflake.core.cortex').setLevel(logging.WARNING)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def unique_filename():
    """
    Generate a unique timestamp-based filename identifier.
    
    Purpose: Creates a unique identifier for each recipe run to organize output files.
    This prevents overwriting previous runs and allows tracking of different execution sessions.
    
    Returns:
        str: Timestamp string in format YYYYMMDD_HHMMSS (e.g., "20240115_143022")
    
    Used by:
        - ID_DATE_PATH global variable (line 186) - organizes output folder structure
        - Output file paths throughout the workflow to create timestamped directories
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def truncate_text_for_log(text, max_length=100, show_start_end=True):
    """
    Truncate text for logging to reduce log file size.
    
    Shows first and last portions of text with ellipsis in between,
    making it easy to verify against the original document.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of each portion (start/end) to show
        show_start_end: If True, show both start and end; if False, just show start
    
    Returns:
        Truncated text string suitable for logging
    """
    if not text or not isinstance(text, str):
        return str(text)
    
    text_len = len(text)
    if text_len <= max_length * 2 + 20:  # If text is short, return as-is
        return text
    
    if show_start_end:
        start = text[:max_length].strip()
        end = text[-max_length:].strip()
        return f"{start}... [truncated {text_len - max_length * 2} chars] ...{end}"
    else:
        return f"{text[:max_length]}... [truncated {text_len - max_length} chars]"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def invoke_with_rate_limit_retry(chain, input_data, max_retries=15, base_delay=2, max_wait_time=120):
    """
    Invoke LLM chain with rate limit error handling.
    
    Handles HTTP 429 errors by parsing retry-after times and implementing
    exponential backoff with rate limit awareness. Designed to handle concurrent
    workers hitting rate limits simultaneously.
    
    Args:
        chain: LangChain runnable chain to invoke
        input_data: Input data to pass to the chain
        max_retries: Maximum number of retry attempts (default: 15, increased for concurrent scenarios)
        base_delay: Base delay in seconds for exponential backoff (default: 2)
        max_wait_time: Maximum wait time in seconds per retry (default: 120, caps exponential growth)
    
    Returns:
        Result from chain.invoke() if successful
    
    Raises:
        Exception: Original exception if max_retries exceeded or non-rate-limit error
    
    Used by:
        - All LLM invocation points throughout the workflow to handle rate limits
    """
    import re
    import random
    
    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            error_str = str(e)
            # Also check exception args and repr for nested error messages
            error_repr = repr(e)
            
            # Check for rate limit error - check multiple formats
            is_rate_limit = (
                "HTTP code: 429" in error_str or 
                "RateLimitReached" in error_str or
                "429" in error_str or
                "rate limit" in error_str.lower() or
                "HTTP code: 429" in error_repr or
                "RateLimitReached" in error_repr
            )
            
            # Check for null response errors (transient API/network issues)
            is_null_response = (
                ("response" in error_str.lower() and "null" in error_str.lower()) or
                ("Cannot read field" in error_str and "response" in error_str.lower() and "null" in error_str.lower()) or
                ("response" in error_repr.lower() and "null" in error_repr.lower())
            )
            
            # Retry on rate limits or null response errors
            if is_rate_limit or is_null_response:
                # Parse retry-after time from error message (only for rate limit errors)
                retry_after = None
                if is_rate_limit:
                    retry_match = re.search(r"retry after (\d+) seconds?", error_str, re.IGNORECASE)
                    if not retry_match:
                        retry_match = re.search(r"retry after (\d+) seconds?", error_repr, re.IGNORECASE)
                    if not retry_match:
                        retry_match = re.search(r"retry.*?(\d+).*?second", error_str, re.IGNORECASE)
                    
                    if retry_match:
                        retry_after = int(retry_match.group(1))
                
                # Calculate wait time with exponential backoff and jitter
                # For rate limits, use retry_after as minimum; for null responses, use exponential backoff
                exponential_delay = base_delay * (2 ** min(attempt, 6))  # Cap exponential growth at 2^6 = 64s
                if retry_after is not None:
                    wait_time = max(retry_after, exponential_delay)
                else:
                    # For null response errors, use exponential backoff with shorter initial delay
                    wait_time = exponential_delay
                
                # Cap maximum wait time to prevent excessive delays
                wait_time = min(wait_time, max_wait_time)
                # Add jitter (±20%) to prevent thundering herd
                jitter = wait_time * 0.2 * random.uniform(-1, 1)
                wait_time = max(1, wait_time + jitter)
                
                if attempt < max_retries - 1:
                    error_type = "HTTP 429" if is_rate_limit else "Null response"
                    print(f"[RETRY] {error_type} error detected, waiting {wait_time:.1f}s before retry {attempt+1}/{max_retries}")
                    print(f"[RETRY] Error details: {error_str[:200]}...")  # Log first 200 chars of error
                    time.sleep(wait_time)
                    continue
                else:
                    error_type = "Rate limit" if is_rate_limit else "Null response"
                    print(f"[RETRY] Max retries ({max_retries}) exceeded for {error_type} error, raising exception")
                    print(f"[RETRY] Final error: {error_str[:500]}...")  # Log first 500 chars of final error
                    raise
            
            # Re-raise non-rate-limit errors immediately
            raise

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ============================================================================
# RECIPE INPUT/OUTPUT INITIALIZATION
# ============================================================================
# This section connects to Dataiku recipe inputs and outputs.
# Recipe inputs are datasets/folders configured in the Dataiku recipe UI.
# Recipe outputs are where the processed results will be saved.

# Read recipe inputs
# ----------------------------------------------------------------------------
# INPUT 1: Data Collection Folder
# Purpose: Stores intermediate processing results and checkpoints to avoid reprocessing.
# Used by: All dataCollection* functions (e.g., dataCollectionIndexExtraction) to save/load
#          cached results. This enables incremental processing and recovery from failures.
data_collection_folder_name = get_input_names_for_role("data_collection_folder")[0]
dataCollectionAI = dataiku.Folder(data_collection_folder_name)
dataCollectionAI_info = dataCollectionAI.get_info()

# INPUT 2: Specification Text Files Dataset
# Purpose: Contains the specification documents to be processed, with columns for:
#          - document_name_column: Name/path of the specification document
#          - page_number_column: Page number within the document
#          - content_column: Text content of that page
# Used by: Initial data loading (lines 263-371) to build specsStores_df dataframe
#          which organizes all spec pages by document and store type.
specs_llm_text_files_dataset_name = get_input_names_for_role("specs_llm_text_files_dataset")[0]
specsLLMTextFiles = dataiku.Dataset(specs_llm_text_files_dataset_name)

# INPUT 3: Cortex Search Service Log Dataset
# Purpose: Contains configuration for Snowflake Cortex Search Service, including:
#          - service_fqn: Fully qualified name of the Cortex Search service
#          - on_column_alias: Column name containing searchable text
#          - attribute_aliases: Metadata columns to retrieve with search results
# Used by: get_cortex_search_service() function to connect to Snowflake and perform
#          semantic search on specification documents (replaces vector store approach).
cortex_log_dataset_name = get_input_names_for_role("cortex_log_dataset")[0]
cortex_log = dataiku.Dataset(cortex_log_dataset_name)

# Write recipe outputs
# ----------------------------------------------------------------------------
# OUTPUT: Multi-Agent Output Folder
# Purpose: Stores final processed results organized by specification path and timestamp.
# Structure: {specPath}/{timestamp}/{commandName}/{command_params.json, module_rules.json, etc.}
# Used by: All save_* functions (save_modules_param, save_modules_rules, etc.) to write
#          the extracted command modules, parameters, and validation rules.
multi_agent_output_folder_name = get_output_names_for_role("multi_agent_output_folder")[0]
MultiAgentOutput = dataiku.Folder(multi_agent_output_folder_name)
MultiAgentOutput_info = MultiAgentOutput.get_info()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ============================================================================
# RECIPE CONFIGURATION PARAMETERS
# ============================================================================
# These parameters are set in the Dataiku recipe UI and control how the recipe processes specifications.

# Get recipe config (Snowflake connection and other parameters)
recipe_config = get_recipe_config()

# Snowflake Connection
# Purpose: Connection name for accessing Snowflake Cortex Search Service.
# Used by: get_cortex_search_service() to establish connection to Snowflake for document search.
snowflake_connection_name = recipe_config.get("snowflake_connection", None)

# Get configuration from recipe parameters
# ----------------------------------------------------------------------------
# LLM Configuration
# Purpose: Identifier for the Language Model to use for all LLM operations.
# Used by: All LLM-based functions (extract_callable_commands, define_cmd_modules, etc.)
#          via DKUChatLLM(llm_id=llm_id) to perform text analysis and extraction.
llm_id = recipe_config.get("llmID")

# Specification Path Filters
# Purpose: Categorize specifications into priority groups for processing.
# - baseSpecPaths: Primary specifications to process (highest priority)
# - otherPriorSpecPaths: Secondary specifications (lower priority)
# Used by: Initial data loading (lines 317-330) to classify each document page into
#          'baseStore' or 'other-priorities' categories. Only 'baseStore' specs are
#          fully processed through the workflow.
baseSpecPaths = recipe_config.get("basestore_paths", [])
otherPriorSpecPaths = recipe_config.get("other_priorities_paths", [])

# Get column names from recipe parameters
# Purpose: Map recipe parameters to actual column names in the input dataset.
# These columns must exist in specs_llm_text_files_dataset:
# - document_name_column: Identifies which specification document the page belongs to
# - page_number_column: Page number within that document
# - content_column: The actual text content of the page
# Used by: Data loading loop (lines 290-343) to extract and organize specification pages.
document_name_column = recipe_config.get("document_name_column")
page_number_column = recipe_config.get("page_number_column")
content_column = recipe_config.get("content_column")

# Validation: Ensure all required column parameters are provided
if not document_name_column or not page_number_column or not content_column:
    raise ValueError("All column parameters (document_name_column, page_number_column, content_column) must be provided")

# Generate unique identifier for this recipe run (used for output folder organization)
unique_name = unique_filename()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ============================================================================
# CORTEX SEARCH SERVICE CONFIGURATION
# ============================================================================
# Cortex Search Service is Snowflake's semantic search capability that replaces traditional
# vector stores. It allows searching specification documents using natural language queries.

# Load Cortex Search Service configuration from log
# Purpose: Read configuration from cortex_log dataset to connect to the search service.
# The dataset contains metadata about how the Cortex Search Service is set up in Snowflake.
cortex_log_df = cortex_log.get_dataframe()
if len(cortex_log_df) == 0:
    raise ValueError("cortex_log dataset is empty. At least one service configuration is required.")

# Get first row for service configuration
# Note: Only the first row is used. If multiple configurations exist, only the first is active.
cortex_service_config = cortex_log_df.iloc[0]

# Service Fully Qualified Name (FQN)
# Format: DATABASE.SCHEMA.SERVICE_NAME (e.g., "DATAIKU.LLMS.DALA_INTERFACE_SPECS")
# Purpose: Identifies the exact Cortex Search Service in Snowflake to use.
service_fqn = cortex_service_config['service_fqn']

# Search Column Alias
# Purpose: Name of the column in the search service that contains the searchable text.
# This is the column that will be searched when performing semantic queries.
on_column_alias = cortex_service_config['on_column_alias']  # e.g., "LLM_OUTPUT"

# Attribute Aliases
# Purpose: Additional metadata columns to retrieve along with search results.
# These provide context like document name, type, version, etc. that help filter
# and organize search results. Format: comma-separated list of column names.
attribute_aliases_str = cortex_service_config['attribute_aliases']  # e.g., "DOCUMENT_NAME,DOCUMENT_TYPE,DOCUMENT_VERSION,RELEVANT_PRODUCTS"
attribute_aliases = [attr.strip() for attr in attribute_aliases_str.split(',')] if attribute_aliases_str else []

# Parse service_fqn to extract database, schema, and service name
# Purpose: Break down the FQN into components needed to access the service via Snowflake API.
# Format: DATAIKU.LLMS.DALA_INTERFACE_SPECS
service_parts = service_fqn.split('.')
if len(service_parts) != 3:
    raise ValueError(f"Invalid service_fqn format: {service_fqn}. Expected format: DATABASE.SCHEMA.SERVICE_NAME")
cortex_search_database = service_parts[0]      # Database name (e.g., "DATAIKU")
cortex_search_schema = service_parts[1]        # Schema name (e.g., "LLMS")
cortex_search_service_name = service_parts[2]   # Service name (e.g., "DALA_INTERFACE_SPECS")

# These parsed values are used by get_cortex_search_service() to access the service.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Embedding Requirements
# NOTE: Embedding LLM no longer needed for vector stores (replaced with Cortex Search Service)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Get embedding LLM (commented out - no longer needed with Cortex Search Service)
client = dataiku.api_client()
project = client.get_default_project()
#embedding_llm = project.get_llm(embedding_model_id)  # embedding_model_id removed from config

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Global Variables

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ============================================================================
# GLOBAL STATE VARIABLES
# ============================================================================
# These global variables maintain state across the workflow execution.

# CURRENT_SPEC_FULL_PATH
# Purpose: Tracks which specification is currently being processed by worker functions.
# Used by: Worker functions (extract_callable_commands, retrieve_modules_cmd_info, etc.)
#          to filter Cortex Search results to the current specification being processed.
#          This ensures workers only retrieve relevant context for their assigned spec.
# Modified by: Worker functions set this before calling retrieval functions.
CURRENT_SPEC_FULL_PATH = ''

# ID_DATE_PATH
# Purpose: Unique timestamp identifier for organizing output files from this recipe run.
# Format: "YYYYMMDD_HHMMSS" (e.g., "20240115_143022")
# Used by: All save_* functions to create timestamped output directories:
#          {specPath}/{ID_DATE_PATH}/{commandName}/command_params.json
#          This prevents overwriting previous runs and enables version tracking.
ID_DATE_PATH = unique_filename()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Cortex Search Service

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ============================================================================
# CORTEX SEARCH SERVICE CONNECTION MANAGEMENT
# ============================================================================
# This module manages the connection to Snowflake Cortex Search Service, which provides
# semantic search capabilities over specification documents. The service is cached to
# avoid reconnecting on every search operation.

# Global cache for Cortex Search Service
# Purpose: Store the service connection object to reuse across multiple search calls.
# This improves performance by avoiding repeated connection establishment.
_cortex_search_service_cache = None

def get_cortex_search_service(connection_name=None):
    """
    Get or create the Cortex Search Service connection with caching.
    
    Purpose:
        Establishes and caches a connection to Snowflake Cortex Search Service.
        The service enables semantic search over specification documents using natural
        language queries, replacing the need for traditional vector stores.
    
    Args:
        connection_name (str, optional): Name of the Snowflake connection configured
            in Dataiku. If None, uses the connection from recipe config.
    
    Returns:
        CortexSearchService: The cached service object that can perform searches.
    
    Caching Strategy:
        Uses module-level cache (_cortex_search_service_cache) to store the connection.
        First call creates the connection; subsequent calls return the cached object.
        This avoids the overhead of reconnecting for each search operation.
    
    Used by:
        - cortex_search_to_documents(): Performs semantic searches on specifications
        - All retrieval functions (get_spec_info_section, get_retriever_command_info, etc.)
          that need to search specification content
    
    Connection Flow:
        1. Check if cache exists → return cached service
        2. If not cached:
           a. Get Snowflake connection via DkuSnowpark
           b. Create Snowpark session
           c. Navigate to Cortex Search Service using parsed FQN components
           d. Cache and return the service object
    
    Raises:
        ValueError: If connection_name is None and no recipe config connection exists.
    """
    global _cortex_search_service_cache
    
    # Return cached service if available (most common path)
    if _cortex_search_service_cache is None:
        if connection_name is None:
            raise ValueError("snowflake_connection must be provided as a recipe parameter")
        
        # Connect to Snowflake using Dataiku's Snowpark integration
        # DkuSnowpark provides a wrapper around Snowflake's Snowpark API
        dku_snowpark = DkuSnowpark()
        snowpark_session = dku_snowpark.get_session(connection_name=connection_name)
        root = Root(snowpark_session)
        
        # Navigate to the Cortex Search Service using the parsed FQN components
        # Structure: root -> database -> schema -> cortex_search_services -> service_name
        # These values were parsed from service_fqn earlier (lines 162-167)
        _cortex_search_service_cache = root.databases[cortex_search_database].schemas[cortex_search_schema].cortex_search_services[cortex_search_service_name]
    
    return _cortex_search_service_cache

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Vector Stores
# NOTE: Vector store functionality has been replaced with Cortex Search Service
# The following code is commented out but kept for reference

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#class DSSLLMEmbedding(Embeddings):
#    def __init__(self):
#        self.embedding_llm = project.get_llm(embedding_model_id)
#
#    def l2_normalize(self, vector):
#        vector = np.array(vector)
#        norm = np.linalg.norm(vector)
#        if norm == 0:
#            return vector.tolist()  # Avoid divide by zero
#        return (vector / norm).tolist()
#
#    def embed_query(self, text):
#        # Call your model's embedding for a single string
#        embedding_query = self.embedding_llm.new_embeddings()
#        cleaned_text = text.strip()
#        embedding_query.add_text(cleaned_text)
#        embedding_response = embedding_query.execute()
#
#        embedding = embedding_response.get_embeddings()[0]
#
#        # return embedding for l2 is not needed to be normalized
#        return self.l2_normalize(embedding) # for cosine is needed to be normalized
#
#    def embed_documents(self, texts):
#        text_vectors = []
#        for t in texts:
#            emb = self.embed_query(t)
#            text_vectors.append(emb)
#        return text_vectors

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
otherPriorSpecPaths

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ============================================================================
# SPECIFICATION DATA LOADING AND ORGANIZATION
# ============================================================================
# This section loads specification pages from the input dataset and organizes them
# into a structured format for processing. Pages are categorized by specification
# type (baseStore vs other-priorities) and organized by document path.

# Debug logging: Print configuration values for troubleshooting
print(f"[DEBUG] INIT: baseSpecPaths = {baseSpecPaths}")
print(f"[DEBUG] INIT: otherPriorSpecPaths = {otherPriorSpecPaths}")
print(f"[DEBUG] INIT: document_name_column = {document_name_column}")
print(f"[DEBUG] INIT: page_number_column = {page_number_column}")
print(f"[DEBUG] INIT: content_column = {content_column}")

# Load dataset
# Purpose: Read all specification pages from the input dataset into a pandas DataFrame.
# Each row represents one page from a specification document.
specsLLMTextFiles_df = specsLLMTextFiles.get_dataframe()
print(f"[DEBUG] INIT: specsLLMTextFiles_df shape: {specsLLMTextFiles_df.shape}")
print(f"[DEBUG] INIT: specsLLMTextFiles_df columns: {list(specsLLMTextFiles_df.columns)}")

# Validate that required columns exist
# Purpose: Ensure the dataset has the expected columns before processing.
# This prevents runtime errors later when trying to access missing columns.
missing_columns = []
if document_name_column not in specsLLMTextFiles_df.columns:
    missing_columns.append(document_name_column)
if page_number_column not in specsLLMTextFiles_df.columns:
    missing_columns.append(page_number_column)
if content_column not in specsLLMTextFiles_df.columns:
    missing_columns.append(content_column)

if missing_columns:
    raise ValueError(f"Required columns not found in dataset: {missing_columns}. Available columns: {list(specsLLMTextFiles_df.columns)}")

# Initialize data structure for organizing specification pages
# Purpose: Dictionary to collect and organize all specification pages with metadata.
# Structure:
#   - filepath: Constructed path like "{document_name}/page_{page_num}.txt"
#   - store: Category ('baseStore' or 'other-priorities')
#   - specRootPath: Root path identifier for the specification
#   - specRootPageNum: Page number within the specification
#   - text: The actual page content
# Used by: Converted to specsStores_df DataFrame (line 365) which is used throughout
#          the workflow to access specification pages by document and page number.
specsDictDistribution = {'filepath': [],
                         'store': [],
                         'specRootPath': [],
                         'specRootPageNum': [],
                         'text': []}

# Counters for tracking classification results
matched_baseStore = 0      # Pages matched to baseSpecPaths (primary specs)
matched_otherPrior = 0    # Pages matched to otherPriorSpecPaths (secondary specs)
matched_none = 0          # Pages that didn't match any configured path

# Iterate over dataset rows
# Purpose: Process each specification page, classify it by specification type,
#          and add it to the organized data structure.
for idx, row in specsLLMTextFiles_df.iterrows():
    # Extract column values with null handling
    # Purpose: Safely extract document name, page number, and content from each row.
    # Handles missing/null values gracefully to avoid processing errors.
    document_name = str(row[document_name_column]) if pd.notna(row[document_name_column]) else None
    page_num = row[page_number_column] if pd.notna(row[page_number_column]) else None
    specPageText = str(row[content_column]) if pd.notna(row[content_column]) else ""
    
    # Validation: Skip rows with missing critical data
    # Purpose: Ensure we only process complete specification pages.
    # Missing document_name or page_number makes the page unusable for processing.
    if document_name is None:
        print(f"[DEBUG] INIT: Row {idx} has None/null document_name, skipping")
        matched_none += 1
        continue
    
    if page_num is None:
        print(f"[DEBUG] INIT: Row {idx} has None/null page_number, skipping")
        matched_none += 1
        continue
    
    # Try to convert page number to int if it's not already
    # Purpose: Ensure page numbers are integers for proper sorting and comparison.
    # Some datasets may have page numbers as strings that need conversion.
    try:
        page_num = int(page_num)
    except (ValueError, TypeError):
        print(f"[DEBUG] INIT: Row {idx} has invalid page_number '{page_num}', skipping")
        matched_none += 1
        continue
    
    # Classification: Determine which specification category this page belongs to
    # Purpose: Categorize pages into 'baseStore' (primary) or 'other-priorities' (secondary).
    # Only 'baseStore' specifications are fully processed through the workflow.
    rootStore = None
    rootSpecPath = None
    
    # Match document name against baseSpecPaths (primary specifications)
    # Strategy: Check if document_name starts with or contains any baseSpecPath.
    # This allows flexible matching for documents with path prefixes.
    for path in baseSpecPaths:
        if document_name.startswith(path) or path in document_name:
            rootStore = 'baseStore'
            rootSpecPath = path
            matched_baseStore += 1
            break
    
    # If not matched to baseStore, try otherPriorSpecPaths (secondary specifications)
    # Purpose: Still capture secondary specs for potential future processing.
    if rootStore is None:
        for path in otherPriorSpecPaths:
            if document_name.startswith(path) or path in document_name:
                rootStore = 'other-priorities'
                rootSpecPath = path
                matched_otherPrior += 1
                break
    
    # Add matched pages to the organized data structure
    # Purpose: Build the complete dataset of categorized specification pages.
    if rootStore is not None:
        # Construct filepath from document name and page number
        # Format: "{document_name}/page_{page_num}.txt"
        # This path is used later for file lookups and Cortex Search filtering.
        filepath = f"{document_name}/page_{page_num}.txt"
        specsDictDistribution['filepath'].append(filepath)
        specsDictDistribution['store'].append(rootStore)
        specsDictDistribution['specRootPath'].append(rootSpecPath)
        specsDictDistribution['specRootPageNum'].append(page_num)
        specsDictDistribution['text'].append(specPageText)
    else:
        # Track unmatched pages for debugging
        matched_none += 1
        if matched_none <= 5:  # Log first 5 unmatched documents to avoid log spam
            print(f"[DEBUG] INIT: Document '{document_name}' (page {page_num}) did not match any path pattern")

print(f"[DEBUG] INIT: Rows matched baseStore: {matched_baseStore}, other-priorities: {matched_otherPrior}, no match: {matched_none}")
print(f"[DEBUG] INIT: specsDictDistribution has {len(specsDictDistribution['filepath'])} entries")

# Log sample document names and page numbers
if len(specsDictDistribution['filepath']) > 0:
    sample_indices = min(5, len(specsDictDistribution['filepath']))
    print(f"[DEBUG] INIT: Sample document names and page numbers (first {sample_indices}):")
    for i in range(sample_indices):
        doc_name = specsDictDistribution['specRootPath'][i] if i < len(specsDictDistribution['specRootPath']) else "N/A"
        page_num = specsDictDistribution['specRootPageNum'][i] if i < len(specsDictDistribution['specRootPageNum']) else "N/A"
        filepath = specsDictDistribution['filepath'][i] if i < len(specsDictDistribution['filepath']) else "N/A"
        print(f"[DEBUG] INIT:   [{i+1}] Document: {doc_name}, Page: {page_num}, Filepath: {filepath}")

# Log how many rows match each baseSpecPath
if baseSpecPaths:
    print(f"[DEBUG] INIT: Matching breakdown by baseSpecPath:")
    for path in baseSpecPaths:
        count = sum(1 for p in specsDictDistribution['specRootPath'] if p == path)
        print(f"[DEBUG] INIT:   '{path}': {count} rows")

specsStores_df = pd.DataFrame(specsDictDistribution)
print(f"[DEBUG] INIT: specsStores_df shape: {specsStores_df.shape}")
print(f"[DEBUG] INIT: specsStores_df columns: {list(specsStores_df.columns)}")
if len(specsStores_df) > 0:
    print(f"[DEBUG] INIT: specsStores_df['store'].value_counts():")
    print(specsStores_df['store'].value_counts())
    print(f"[DEBUG] INIT: specsStores_df unique specRootPath values: {specsStores_df['specRootPath'].unique().tolist()}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
specsStores_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Spec VectorStore

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Convert Embeding column into list
#specsVectorStores_df['embedding'] = specsVectorStores_df['embedding'].apply(ast.literal_eval)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#specsTargetVecBaseStores_df = specsVectorStores_df[(specsVectorStores_df['specAbbr']==specAbbrName)]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
specsBaseStores_df = specsStores_df[(specsStores_df['store']=='baseStore')]
print(f"[DEBUG] INIT: specsBaseStores_df shape: {specsBaseStores_df.shape}")
print(f"[DEBUG] INIT: specsBaseStores_df unique specRootPath values: {specsBaseStores_df['specRootPath'].unique().tolist() if len(specsBaseStores_df) > 0 else 'EMPTY'}")
specsBaseStores_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#text_embbedings = []
#text_embedded = []
#text_metadata = []

#for _, row in specsVecBaseStores_df.iterrows():
#    text_embbedings.append(row['embedding'])
#    text_embedded.append(row['text'])       # main body text
#    text_metadata.append({          # any extra info
#        'filename': row['filename'],
#        'filepath': row['filepath'],
#        'page': row.get('page', None),
#        'store': row.get('store', None)
#    })
#uuids = [str(uuid4()) for _ in range(len(text_embedded))]
#text_embedding_pairs = zip(text_embedded, text_embbedings)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#embedding_fn = DSSLLMEmbedding()
#vectorBaseStore = FAISS.from_embeddings(text_embeddings=text_embedding_pairs,
#                                        embedding=embedding_fn,
#                                        metadatas=text_metadata,
#                                        ids=uuids,
#                                        distance_strategy=DistanceStrategy.COSINE
#                                       )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
specsAllStores_df = specsStores_df[specsStores_df['store'].isin(['baseStore', 'other-priorities'])]
specsAllStores_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#text_embbedings_all_stores = []
#text_embedded_all_stores = []
#text_metadata_all_stores = []

#for _, row in specsVecAllStores_df.iterrows():
#    text_embbedings_all_stores.append(row['embedding'])
#    text_embedded_all_stores.append(row['text'])       # main body text
#    text_metadata_all_stores.append({          # any extra info
#        'filename': row['filename'],
#        'filepath': row['filepath'],
#        'page': row.get('page', None),
#        'store': row.get('store', None)
#    })
#uuids_all_stores = [str(uuid4()) for _ in range(len(text_embedded_all_stores))]
#text_embedding_pairs_all_stores = zip(text_embedded_all_stores, text_embbedings_all_stores)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#embedding_fn = DSSLLMEmbedding()
#vectorBaseStore_all_stores = FAISS.from_embeddings(text_embeddings=text_embedding_pairs_all_stores,
#                                                   embedding=embedding_fn,
#                                                   metadatas=text_metadata_all_stores,
#                                                   ids=uuids_all_stores,
#                                                   distance_strategy=DistanceStrategy.COSINE)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: RAW
# # Retrieve Info
# # Retrieve more documents with higher diversity
# # Useful if your dataset has many similar documents
# docsearch.as_retriever(
#     search_type="mmr",
#     search_kwargs={'k': 6, 'lambda_mult': 0.25}
# )
# 
# # Fetch more documents for the MMR algorithm to consider
# # But only return the top 5
# docsearch.as_retriever(
#     search_type="mmr",
#     search_kwargs={'k': 5, 'fetch_k': 50}
# )
# 
# # Only retrieve documents that have a relevance score
# # Above a certain threshold
# docsearch.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={'score_threshold': 0.8}
# )
# 
# # Only get the single most similar document from the dataset
# docsearch.as_retriever(search_kwargs={'k': 1})
# 
# # Use a filter to only retrieve documents from a specific paper
# docsearch.as_retriever(
#     search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
# )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Spec VectorStore Contextual
# NOTE: Vector store creation has been replaced with Cortex Search Service
# The following code is commented out but kept for reference

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#contextual_rows = []
#baseStore_rows = []
#otherPriorStore_rows = []
#for filePath in specsContextualEmbeddings.list_paths_in_partition():
#    rootStore = None
#    for path in baseSpecPaths:
#        if filePath.startswith(path):
#            rootStore = 'baseStore'
#            break
#
#    if rootStore is None:
#        for path in otherPriorSpecPaths:
#            if filePath.startswith(path):
#                rootStore = 'other-priorities'
#                break
#    if rootStore is not None:
#        with specsContextualEmbeddings.get_download_stream(filePath) as f:
#            contextualEmbeddings = json.loads(f.read().decode("utf-8"))
#            contextual_rows.extend(contextualEmbeddings)
#            if rootStore == 'baseStore':
#                baseStore_rows.extend(contextualEmbeddings)
#            else:
#                otherPriorStore_rows.extend(contextualEmbeddings)
#spec_df_contxt = pd.DataFrame(contextual_rows)
#spec_df_contxt.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#def create_context_vectorstore(spec_path="all"):
#    text_embbedings = []
#    text_embedded = []
#    text_metadata = []
#
#    spec_dataframe_rows = spec_df_contxt.iterrows() if spec_path == 'all' else spec_df_contxt[spec_df_contxt['specName'].str.contains(spec_path)].iterrows()
#
#    for _, row in spec_dataframe_rows:
#        text_embbedings.append(row['embeddings'])
#        text_embedded.append(row['full_text'])       # main body text
#        text_metadata.append({
#            'pageDocument': row['pageDocument']
#        })
#    uuids = [str(uuid4()) for _ in range(len(text_embedded))]
#    text_embedding_pairs_contxt = zip(text_embedded, text_embbedings)
#
#    embedding_fn = DSSLLMEmbedding()
#    vectoreStore = FAISS.from_embeddings(text_embeddings=text_embedding_pairs_contxt,
#                                         embedding=embedding_fn,
#                                         metadatas=text_metadata,
#                                         ids=uuids,
#                                         distance_strategy=DistanceStrategy.COSINE)
#    return vectoreStore

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#vectorSpec_contxt = create_context_vectorstore()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Retrievers

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#retriever_spec = vectorBaseStore.as_retriever(
    #search_type="mmr",
    #search_kwargs={'k': 25, 'fetch_k': 500}
#    search_type="similarity",
#    search_kwargs={'k': 25}
#)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# NOTE: Retriever initialization has been replaced with Cortex Search Service
# The following code is commented out but kept for reference
#retriever_spec_ctxt = vectorSpec_contxt.as_retriever(
#    #search_type="mmr",
#    #search_kwargs={'k': 25, 'fetch_k': 500}
#    search_type="similarity",
#    search_kwargs={'k': 25}
#)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#retriever_module_info = vectorBaseStore.as_retriever(
    #search_type="mmr",
    #search_kwargs={'k': 7, 'fetch_k': 100}
#    search_type="similarity",
#    search_kwargs={'k': 7}
#)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#retriever_module_info_ctxt = vectorSpec_contxt.as_retriever(
#    #search_type="mmr",
#    #search_kwargs={'k': 7, 'fetch_k': 100}
#    search_type="similarity",
#    search_kwargs={'k': 7}
#)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#retriever_command_info = vectorBaseStore.as_retriever(
    #search_type="mmr",
    #search_kwargs={'k': 15, 'fetch_k': 100}
#    search_type="similarity",
#    search_kwargs={'k': 15}
#)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#retriever_command_info_ctxt = vectorSpec_contxt.as_retriever(
#    #search_type="mmr",
#    #search_kwargs={'k': 15, 'fetch_k': 100}
#    search_type="similarity",
#    search_kwargs={'k': 15}
#)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#retriever_specific_info = vectorBaseStore_all_stores.as_retriever(
    #search_type="mmr",
    #search_kwargs={'k': 15, 'fetch_k': 100}
#    search_type="similarity",
#    search_kwargs={'k': 3}
#)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#retriever_specific_info_ctxt = vectorSpec_contxt.as_retriever(
#    #search_type="mmr",
#    #search_kwargs={'k': 15, 'fetch_k': 100}
#    search_type="similarity",
#    search_kwargs={'k': 3}
#)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#retriever_params_info = vectorBaseStore_all_stores.as_retriever(
    #search_type="mmr",
    #search_kwargs={'k': 15, 'fetch_k': 100}
#    search_type="similarity",
#    search_kwargs={'k': 5}
#)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#retriever_params_info_ctxt = vectorSpec_contxt.as_retriever(
#    #search_type="mmr",
#    #search_kwargs={'k': 15, 'fetch_k': 100}
#    search_type="similarity",
#    search_kwargs={'k': 5}
#)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#retriever_spec_info_contxt = vectorSpec_contxt.as_retriever(
#    #search_type="mmr",
#    #search_kwargs={'k': 15, 'fetch_k': 50}
#    search_type="similarity",
#    search_kwargs={'k': 15}
#    #search_type="mmr",
#    #search_kwargs={'k': 15, 'lambda_mult': 0.25}
#)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#retriever_modules_info_contxt = vectorSpec_contxt.as_retriever(
#    #search_type="mmr",
#    #search_kwargs={'k': 15, 'fetch_k': 50}
#    #search_type="similarity",
#    #search_kwargs={'k': 15}
#    search_type="mmr",
#    search_kwargs={'k': 15, 'lambda_mult': 0.25}
#)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#docs = vectorSpec_contxt.similarity_search('/NVMe/NVMe/2_1/NVM-Express-Base-Specification_page_10.txt') #, filter={"pageDocument": "/NVMe/NVMe/2_1/NVM-Express-Base-Specification_page_10.txt"})

#docs = retriever_command_info_ctxt.invoke('firmware update')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#docs

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## General Spec and RCV

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
rcvDesc = "RCV is a comprehensive validation system designed to automate and enhance the testing of enterprise solid-state drives (SSDs) by managing workloads, test sequences, and verification logic to maximize scenario coverage and reliability. It integrates key components such as configurable workloads (read/write operations, error injections), execution runners that coordinate test sequences (either random or sequential, benefiting from randomness and bias to uncover edge cases), and modular elements like generators, verifiers, rule sets, and test knobs that enable flexible validation criteria and response checking. The framework manages execution through organized cycles and batches to structure and repeat tests, tracks configurations with defined data structures (e.g., DriveConfig, drive shadow states), and maintains a journal for result tracking and overrides. By combining automation with intelligently randomized and biased test logic, RCV ensures robust validation, broad coverage of device behaviors, and adaptability across changing SSD requirements, making it a powerful tool for quality assurance in enterprise storage environments. Running small, specification-targeted modules concurrently with other features is crucial in validation frameworks like RCV because it enables thorough testing of how multiple commands and features interact under real-world conditions. This concurrency not only verifies each feature’s individual compliance with the specification but also checks for potential side effects—such as conflicts, dependencies, or unintended interactions—that may arise when commands operate together."

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Modules Extraction

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### General Common Functions

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def cortex_search_to_documents(query, limit, filters=None):
    """
    Perform semantic search using Cortex Search Service and convert results to LangChain Documents.
    
    Purpose:
        This is the core document retrieval function that replaces traditional vector stores.
        It uses Snowflake's Cortex Search Service to perform semantic search over specification
        documents and returns results in LangChain Document format for use in RAG chains.
    
    How It Works:
        1. Connects to Cortex Search Service (cached connection)
        2. Performs semantic search with the query string
        3. Retrieves search results with metadata (document name, type, etc.)
        4. Optionally enriches results by looking up full text from specsAllStores_df
        5. Converts results to LangChain Document objects with page_content and metadata
    
    Args:
        query (str): Natural language search query (e.g., "NVMe write command parameters")
        limit (int): Maximum number of search results to return
        filters (dict, optional): Additional filters to apply to the search
            (e.g., filter by document type or version)
    
    Returns:
        List[Document]: List of LangChain Document objects, each containing:
            - page_content: The text content from the search result (or enriched from dataframe)
            - metadata: Dictionary with filepath, DOCUMENT_NAME, and other attribute columns
    
    Used by:
        - get_spec_info_section(): Retrieves spec sections for command extraction
        - get_retriever_command_info(): Retrieves command-specific information
        - get_retriever_module_info(): Retrieves module-related specification content
        - All other get_retriever_* functions that need to search specification documents
    
    Integration Points:
        - get_cortex_search_service(): Gets the cached Cortex Search Service connection
        - specsAllStores_df: Optional lookup to enrich results with full page text
        - on_column_alias: Column containing searchable text (from Cortex config)
        - attribute_aliases: Metadata columns to retrieve (from Cortex config)
    
    Fallback Behavior:
        If filepath lookup fails or specsAllStores_df doesn't exist, uses the search result
        content directly. This ensures the function works even if the dataframe isn't available.
    
    Example Usage:
        documents = cortex_search_to_documents("NVMe write command", limit=5)
        # Returns 5 most relevant specification sections about NVMe write commands
    """
    # Get cached Cortex Search Service connection
    # This avoids reconnecting for every search operation
    cortex_service = get_cortex_search_service(snowflake_connection_name)
    
    # Build columns list: search column + attribute columns
    # Purpose: Request both the searchable text content and metadata columns in one query.
    # on_column_alias contains the searchable text; attribute_aliases contain metadata.
    all_columns = [on_column_alias] + attribute_aliases
    
    # Perform search
    # Purpose: Execute semantic search query against the Cortex Search Service.
    # The service uses embeddings to find semantically similar content, not just keyword matches.
    if filters:
        # Search with additional filters (e.g., filter by document type)
        response = cortex_service.search(
            query=query,
            columns=all_columns,
            filter=filters,
            limit=limit
        )
    else:
        # Search without filters (most common case)
        response = cortex_service.search(
            query=query,
            columns=all_columns,
            limit=limit
        )
    
    # Parse results from JSON response
    # Purpose: Extract the search results from the Cortex Search Service response.
    # Response format: {"results": [{"LLM_OUTPUT": "...", "DOCUMENT_NAME": "...", ...}, ...]}
    results = json.loads(response.json())['results']
    
    # Convert to Document format
    # Purpose: Transform Cortex Search results into LangChain Document objects
    # that can be used in RAG chains and prompts.
    documents = []
    for result in results:
        # Extract page content from the search result
        # Use the search column (on_column_alias) as the main content
        page_content = result.get(on_column_alias, "")
        
        # Build metadata dictionary from attribute columns
        # Purpose: Preserve document metadata (name, type, version, etc.) for filtering
        # and context in downstream processing.
        metadata = {}
        for attr in attribute_aliases:
            if attr in result:
                metadata[attr] = result[attr]
        
        # Try to maintain compatibility: check if we can map to filepath
        # Purpose: Extract filepath from metadata to enable optional enrichment lookup.
        # If DOCUMENT_NAME or another attribute contains filepath-like info, use it.
        # Otherwise, use the search content directly without lookup.
        filepath = None
        if 'DOCUMENT_NAME' in metadata:
            filepath = metadata['DOCUMENT_NAME']
        elif len(attribute_aliases) > 0 and attribute_aliases[0] in metadata:
            # Fallback: use first attribute as filepath if DOCUMENT_NAME not available
            filepath = metadata[attribute_aliases[0]]
        
        # Optional enrichment: Look up full text from specsAllStores_df
        # Purpose: If we have a filepath and the dataframe exists, retrieve the complete
        # page text from the dataframe. This provides more context than the search snippet.
        # If lookup fails, use the Cortex Search content directly (graceful degradation).
        if filepath:
            try:
                # Check if specsAllStores_df exists and try lookup
                # Note: specsAllStores_df is created during initialization (line 365)
                # and contains all specification pages organized by filepath.
                if 'specsAllStores_df' in globals() and specsAllStores_df is not None:
                    matching_rows = specsAllStores_df[specsAllStores_df["filepath"] == filepath]
                    if len(matching_rows) > 0:
                        # Sort by page number and join all matching pages
                        # This handles cases where multiple pages match the same filepath
                        text_values = matching_rows.sort_values("specRootPageNum")["text"].tolist()
                        page_content = "\n\n".join(text_values)
            except (KeyError, AttributeError, NameError) as e:
                # If lookup fails, use Cortex Search content directly
                # This is expected if filepath doesn't match or specsAllStores_df structure differs
                # Graceful degradation: still return results even if enrichment fails
                pass
        
        # Add filepath to metadata for downstream processing
        metadata['filepath'] = filepath if filepath else 'cortex_search_result'
        # Also add pageDocument for compatibility with legacy code
        metadata['pageDocument'] = filepath if filepath else 'cortex_search_result'
        
        # Create LangChain Document object
        documents.append(Document(
            page_content=page_content,
            metadata=metadata
        ))
    
    return documents

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def chunk_text_with_overlap(df, chunk_size=5, overlap=1):
    """
    Splits the 'text' column of the dataframe into overlapping chunks by rows.

    Args:
        df (pd.DataFrame): Source dataframe.
        chunk_size (int): Number of rows per chunk (including overlap).
        overlap (int): Number of rows overlapping between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    chunks = []
    start = 0
    n = len(df)
    while start < n:
        end = min(start + chunk_size, n)
        # Collect text for the current chunk
        chunk_text = "\n".join(df.iloc[start:end]["text"].astype(str))
        chunks.append(chunk_text)
        # Advance start by chunk_size - overlap for the next chunk
        start += chunk_size - overlap
    return chunks

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ============================================================================
# RETRIEVAL WRAPPER FUNCTIONS
# ============================================================================
# These functions provide convenient wrappers around cortex_search_to_documents()
# with different result limits optimized for different use cases. They are used
# as RAG chain retrievers in LangChain workflows.
#
# All functions:
#   - Accept a query string (natural language search query)
#   - Return List[Document] from Cortex Search Service
#   - Use CURRENT_SPEC_FULL_PATH global variable (set by workers) for context
#   - Replace legacy vector store retrievers with Cortex Search Service

def get_spec_info_section(query):
    """
    Retrieve specification section information (small result set).
    
    Purpose: Get focused specification sections for initial command extraction.
    Limit: 5 results - small set for focused, high-quality matches.
    Used by: extract_callable_commands() worker to find relevant spec sections.
    """
    global CURRENT_SPEC_FULL_PATH
    # Use Cortex Search Service instead of vector store
    # Note: CURRENT_SPEC_FULL_PATH filtering would need to be implemented via filters if needed
    return cortex_search_to_documents(query, limit=5)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_modules_contxt_text_only(query):
    """
    Retrieve module context with broader coverage (medium result set).
    
    Purpose: Get contextual information about modules across specifications.
    Limit: 15 results - broader coverage for understanding module relationships.
    Used by: Module extraction and analysis workflows.
    Note: MMR (Maximum Marginal Relevance) search type is not directly supported
          by Cortex Search, so standard similarity search is used instead.
    """
    global CURRENT_SPEC_FULL_PATH
    # Use Cortex Search Service instead of vector store
    # Note: MMR search type is not directly supported by Cortex Search, using standard search
    # Note: CURRENT_SPEC_FULL_PATH filtering would need to be implemented via filters if needed
    return cortex_search_to_documents(query, limit=15)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_retriever_spec(query):
    """
    Retrieve general specification information (large result set).
    
    Purpose: Get comprehensive specification content for general analysis.
    Limit: 25 results - large set for comprehensive coverage.
    Used by: General specification analysis and overview generation.
    """
    global CURRENT_SPEC_FULL_PATH
    # Use Cortex Search Service instead of vector store
    # Note: CURRENT_SPEC_FULL_PATH filtering would need to be implemented via filters if needed
    return cortex_search_to_documents(query, limit=25)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_retriever_module_info(query):
    """
    Retrieve module-specific information (small focused set).
    
    Purpose: Get focused information about specific modules.
    Limit: 7 results - small focused set for module details.
    Used by: Module definition and analysis workflows.
    """
    global CURRENT_SPEC_FULL_PATH
    # Use Cortex Search Service instead of vector store
    # Note: CURRENT_SPEC_FULL_PATH filtering would need to be implemented via filters if needed
    return cortex_search_to_documents(query, limit=7)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_retriever_command_info(query):
    """
    Retrieve command-specific information (medium result set).
    
    Purpose: Get detailed information about commands and their specifications.
    Limit: 15 results - medium set for comprehensive command details.
    Used by: 
        - retrieve_modules_cmd_info() - assess command relevance
        - define_cmd_modules() - define modules for commands
        - built_command_params_modules() - extract command parameters
        - define_command_rules_modules() - generate validation rules
    This is one of the most frequently used retrieval functions.
    """
    global CURRENT_SPEC_FULL_PATH
    # Use Cortex Search Service instead of vector store
    # Note: CURRENT_SPEC_FULL_PATH filtering would need to be implemented via filters if needed
    return cortex_search_to_documents(query, limit=15)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_retriever_specific_info(query):
    """
    Retrieve very specific, focused information (minimal result set).
    
    Purpose: Get highly targeted information for specific queries.
    Limit: 3 results - minimal set for very focused searches.
    Used by: Specific detail extraction and targeted lookups.
    """
    global CURRENT_SPEC_FULL_PATH
    # Use Cortex Search Service instead of vector store
    # Note: CURRENT_SPEC_FULL_PATH filtering would need to be implemented via filters if needed
    return cortex_search_to_documents(query, limit=3)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_retriever_params_info(query):
    """
    Retrieve parameter-specific information (small focused set).
    
    Purpose: Get information about command parameters and their specifications.
    Limit: 5 results - small focused set for parameter details.
    Used by: Parameter extraction and enhancement workflows.
    """
    global CURRENT_SPEC_FULL_PATH
    # Use Cortex Search Service instead of vector store
    # Note: CURRENT_SPEC_FULL_PATH filtering would need to be implemented via filters if needed
    return cortex_search_to_documents(query, limit=5)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_specific_spec_info(query):
    """
    Retrieve specific specification information (small result set).
    
    Purpose: Get targeted specification information for specific queries.
    Limit: 5 results - small set for focused specification lookups.
    Used by: Specific specification detail extraction.
    """
    global CURRENT_SPEC_FULL_PATH
    # Use Cortex Search Service instead of vector store
    # Note: CURRENT_SPEC_FULL_PATH filtering would need to be implemented via filters if needed
    return cortex_search_to_documents(query, limit=5)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_retriever_all_spec_info(query):
    """
    Retrieve information across all specifications (medium-large result set).
    
    Purpose: Get comprehensive information across all specifications.
    Limit: 15 results - medium-large set for cross-specification analysis.
    Used by: Cross-specification analysis and global rule generation.
    Note: Does not use CURRENT_SPEC_FULL_PATH filtering (searches all specs).
    """
    # Use Cortex Search Service instead of vector store
    return cortex_search_to_documents(query, limit=15)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_retriever_all_spec_short_info(query):
    """
    Retrieve short information across all specifications (small result set).
    
    Purpose: Get brief information across all specifications for quick lookups.
    Limit: 5 results - small set for quick cross-specification queries.
    Used by: 
        - enhance_parameter_context() - enhance incomplete parameter info
        - define_command_rules_modules() - enhance incomplete rule definitions
        - define_modules_global_rules() - enhance global rule definitions
    Note: Does not use CURRENT_SPEC_FULL_PATH filtering (searches all specs).
    """
    # Use Cortex Search Service instead of vector store
    return cortex_search_to_documents(query, limit=5)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Generate General Specifications Description

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# NOTE: spec_retrievers_dict initialization has been replaced with Cortex Search Service
# The following code is commented out but kept for reference
specs_desc_dict = {}
spec_retrievers_dict = {}
#for path in specsBaseStores_df['specRootPath'].unique().tolist():
#    if path not in specs_desc_dict.keys():
#        specs_desc_dict[path] = None
#        spec_vector_store = create_context_vectorstore(path)
#        spec_vector_store_retriever = spec_vector_store.as_retriever(
#            search_type="similarity",
#            search_kwargs={'k': 25})
#        spec_retrievers_dict[path] = spec_vector_store_retriever
# Note: If path-based filtering is needed, implement via Cortex Search filters
for path in specsBaseStores_df['specRootPath'].unique().tolist():
    if path not in specs_desc_dict.keys():
        specs_desc_dict[path] = None
        # Store path for potential filtering, but use shared Cortex Search Service
        spec_retrievers_dict[path] = None  # Will use cortex_search_to_documents directly

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def create_cortex_retriever_runnable(spec_path):
    """
    Create a Runnable that uses Cortex Search Service filtered by document name.
    
    Args:
        spec_path: Document name/path to filter results by
        
    Returns:
        Runnable that takes a query and returns formatted context string
    """
    def cortex_retriever(query):
        # Get documents from Cortex Search (without filters first)
        documents = cortex_search_to_documents(query, limit=50, filters=None)
        
        # Filter documents by matching document name
        # Match if DOCUMENT_NAME starts with or contains spec_path
        filtered_docs = []
        for doc in documents:
            doc_name = doc.metadata.get('DOCUMENT_NAME', '')
            # Check if document name matches spec_path (handle both exact match and prefix match)
            if doc_name and (doc_name.startswith(spec_path) or spec_path in doc_name or doc_name.startswith(spec_path.replace('_', ' '))):
                filtered_docs.append(doc)
        
        # If no filtered results, use all documents (fallback)
        if len(filtered_docs) == 0:
            filtered_docs = documents[:10]  # Use first 10 as fallback
        
        # Join page_content from filtered documents
        context = "\n\n".join([doc.page_content for doc in filtered_docs[:10]])
        return context
    
    return RunnableLambda(cortex_retriever)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def generate_spec_full_name(spec_path, specAbbrName):
    systemMessage = f"You are an expert of a module for {specAbbrName} specifications"

    message = """
    Identify the full, official specification title from the excerpt below. Return only the most canonical name to reference this specification (no extra commentary). If uncertain, give the best plausible title and append "(uncertain)".

    Specification excerpt:
    {context}
    """
    llm_spec_desc = DKUChatLLM(llm_id="azureopenai:Azure-OpenAI-Prod-4-1:gpt-4.1", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([("system", systemMessage),
                                               ("human", message)])

    # Create Runnable wrapper for Cortex Search
    cortex_retriever = create_cortex_retriever_runnable(spec_path)
    
    rag_chain = {
        "context": cortex_retriever,
    } | prompt | llm_spec_desc

    specDesc = invoke_with_rate_limit_retry(rag_chain, f"What are the processes and commands for {specAbbrName} for SSDs validation").content
    return specDesc

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def generate_spec_description(spec_path, specFullName):
    systemMessage = f"You are an expert of a module for {specFullName} specifications"

    message = """
    Summarize, in one concise paragraph (max 250 words), the validation intent of the specification named {specName}. Focus on what the specification aims to validate: objectives, scope boundaries, key functional and non-functional aspects, required behaviors, interfaces, constraints, and any implicit assumptions. Do not invent details; infer only from the supplied context.
    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages([("system", systemMessage),
                                               ("human", message)])

    llm_spec_desc = DKUChatLLM(llm_id="azureopenai:Azure-OpenAI-Prod-4-1:gpt-4.1", temperature=0.2)
    
    # Create Runnable wrapper for Cortex Search
    cortex_retriever = create_cortex_retriever_runnable(spec_path)
    
    rag_chain = {
        "specName": lambda specName: f"{specFullName}",
        "context": cortex_retriever,
        # "module": RunnablePassthrough(),
    } | prompt | llm_spec_desc

    specDesc = invoke_with_rate_limit_retry(rag_chain, f"What are the processes and commands for {specFullName} for SSDs validation").content
    return specDesc

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for spec_path in specs_desc_dict.keys():
    spec_general_info_path = f"{spec_path}/spec_general_info.json"
    if spec_general_info_path not in dataCollectionAI.list_paths_in_partition():
        print(f'Generating description for: {spec_path}')
        # Handle both old path format and new document name format
        path_parts = spec_path.split('/')
        if len(path_parts) >= 4:
            # Old format: full file path with multiple '/' separators
            root_spec_name = f"{path_parts[-4]}>{path_parts[-3]}>{path_parts[-2]}"
        else:
            # New format: just document name (e.g., "PCI_Express_Base_5r1")
            root_spec_name = spec_path
        spec_full_name = generate_spec_full_name(spec_path, root_spec_name)
        spec_full_desc = generate_spec_description(spec_path, spec_full_name)
        specs_desc_dict[spec_path] = {"spec_full_name": spec_full_name,
                                      "spec_full_description": spec_full_desc}
        json_bytes = json.dumps(specs_desc_dict[spec_path], indent=2).encode("utf-8")

        with dataCollectionAI.get_writer(spec_general_info_path) as w_binary:
            w_binary.write(json_bytes)

    else:
        print(f'Skipping description generation for: {spec_path}')
        with dataCollectionAI.get_download_stream(spec_general_info_path) as f:
            general_info = json.loads(f.read().decode("utf-8"))
            specs_desc_dict[spec_path] = {"spec_full_name": general_info["spec_full_name"],
                                          "spec_full_description": general_info["spec_full_description"]}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Initialize DataFrames for Saving or Retrieving Info

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_sorted = specsStores_df.sort_values(['specRootPath', 'specRootPageNum'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_sorted.head(20)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Initialize the record DataFrame outside of any class or function
# modules_record_df = pd.DataFrame(columns=["modules"])
# modules_record_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Table Index and Modules Identification

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# LLM ID is now configured via recipe parameters (llm_id from recipe_config)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ============================================================================
# WORKFLOW STATE DEFINITION
# ============================================================================
# ModuleContentState defines the shared state structure for the LangGraph workflow.
# This TypedDict specifies all data that flows through the multi-agent processing pipeline.
#
# Key Design Patterns:
#   - Annotated[list, operator.add]: Lists that accumulate results from parallel workers.
#     LangGraph automatically merges results from multiple workers using operator.add.
#   - Annotated[dict, ...]: Dictionaries that store organized results by specification path.
#   - Regular dict/list: Simple state that gets replaced rather than accumulated.
#
# State Flow Through Workflow:
#   1. Initialization: Empty state with default values
#   2. Checkpoint Loading: Load existing results from dataCollectionAI folder
#   3. Worker Processing: Parallel workers add results to Annotated[list, operator.add] fields
#   4. Result Aggregation: Workers' results are automatically merged by LangGraph
#   5. Saving: Results saved to dataCollectionAI and MultiAgentOutput folders
#
# Used by: All workflow nodes and worker functions to read/write state

class ModuleContentState(TypedDict):
    """
    Central state structure for the specification decomposition workflow.
    
    This TypedDict defines all data that flows through the LangGraph multi-agent pipeline.
    The state is shared across all workflow nodes and worker functions.
    """
    
    # ========================================================================
    # CHECKPOINT: AI AGENTS STAGES - Accumulated Results from Workers
    # ========================================================================
    # These fields use Annotated[list, operator.add] to accumulate results from
    # parallel workers. LangGraph automatically merges results using operator.add.
    
    missing_spec_index_pages: Annotated[
        list, operator.add
    ]
    # Purpose: Accumulates index page extraction results from locate_spec_index_pages workers.
    # Format: [{"spec_path": {"start_index_page": N, "end_index_page": M}}, ...]
    # Used by: save_spec_index_pages() to update content_index_pages
    
    missing_spec_relevant_sections: Annotated[
        list, operator.add
    ]
    # Purpose: Accumulates relevant section extraction results from identify_relevant_spec_sections workers.
    # Format: [{"spec_path": "section1, section2, section3"}, ...]
    # Used by: save_spec_most_relavant_sections() to update specs_relevant_sections
    
    complete_callable_commands: Annotated[
        list, operator.add
    ]
    # Purpose: Accumulates callable command extraction results from extract_callable_commands workers.
    # Format: [{"spec_path": ModuleStruct(command_names="...", keyPhrases="...", score="...")}, ...]
    # Used by: assign_workers_prune_commands_list() to group commands by spec before pruning
    # Note: All workers write to this key in parallel
    
    complete_unique_callable_commands: Annotated[
        list, operator.add
    ]
    # Purpose: Accumulates pruned unique command lists from prune_command_list workers.
    # Format: [{"spec_path": "command1, command2, command3"}, ...]
    # Used by: save_spec_unique_callable_commands() to update specs_callable_unique_cmds
    
    completed_module_cmds: Annotated[
        list, operator.add
    ]
    # Purpose: Accumulates command relevance assessment results from retrieve_modules_cmd_info workers.
    # Format: [{"spec_path": {"command_name": CommandRelevanceAssessment(...)}}, ...]
    # Used by: save_spec_most_relavant_info_cmd() to update specs_cmd_modules_complete_info
    
    completed_module_cmds_overview: Annotated[
        list, operator.add
    ]
    # Purpose: Reserved for module overview information (currently unused in workflow).
    
    completed_module_cmds_submodules: Annotated[
        list, operator.add
    ]
    # Purpose: Reserved for submodule information (currently unused in workflow).
    
    completed_module_cmds_param: Annotated[
        list, operator.add
    ]
    # Purpose: Accumulates parameter extraction results from built_command_params_modules workers.
    # Format: [{"spec_path": {"command_name": {"module_name": CommandParametersDefinition(...)}}}, ...]
    # Used by: assign_workers_cmds_enha_params() to enhance incomplete parameters
    
    completed_module_cmds_enha_param: Annotated[
        list, operator.add
    ]
    # Purpose: Accumulates enhanced parameter results from enhance_parameter_context workers.
    # Format: [{"spec_path": {"command_name": {"module_name": {...}}}}, ...]
    # Used by: save_modules_param() to save final parameter definitions
    
    completed_module_cmds_global_rules: Annotated[
        list, operator.add
    ]
    # Purpose: Accumulates global rule extraction results from define_command_rules_modules workers.
    # Format: [{"spec_path": {"command_name": {"module_name": [rule1, rule2, ...]}}}, ...]
    # Used by: define_modules_global_rules() to consolidate global rules across modules
    
    completed_module_cmds_module_rules: Annotated[
        list, operator.add
    ]
    # Purpose: Accumulates module-specific rule extraction results from define_command_rules_modules workers.
    # Format: [{"spec_path": {"command_name": {"module_name": [rule1, rule2, ...]}}}, ...]
    # Used by: save_modules_rules() to save module-specific validation rules
    
    completed_cmd_modules_definition: Annotated[
        list, operator.add
    ]
    # Purpose: Accumulates module definition results from define_cmd_modules workers.
    # Format: [{"spec_path": {"command_name": ProposedModules(...)}}, ...]
    # Used by: save_spec_cmd_modules_definition() to update specs_cmd_modules_definition
    
    completed_final_global_rules: Annotated[list, "Must have the rules information dictionary for each global rule identified."]
    # Purpose: Stores consolidated global rules after merging similar rules across modules.
    # Format: [{"ruleName": "...", "ruleDesc": "...", ...}, ...]
    # Used by: save_global_rules() to save final global validation rules
    # Note: This is NOT Annotated[list, operator.add] because it's set once by define_modules_global_rules()
    
    # ========================================================================
    # ORGANIZED RESULTS - Dictionary Structures by Specification Path
    # ========================================================================
    # These fields store organized results indexed by specification path.
    # They are updated by save_* functions and read by subsequent workflow stages.
    
    specs_relevant_sections: Annotated[dict, "Must have the most relevant Sections Per Spec."]
    # Purpose: Maps specification paths to comma-separated lists of relevant section names.
    # Format: {"spec_path": "section1, section2, section3", ...}
    # Used by: assign_workers_callable_commands_extraction() to get sections for each spec
    
    specs_callable_cmds: Annotated[dict, "Must have callable commands found in the spec sections."]
    # Purpose: Maps specification paths to callable command information (currently unused).
    
    specs_callable_unique_cmds: Annotated[dict, "Must have the unrepeated commands found in the spec sections."]
    # Purpose: Maps specification paths to comma-separated lists of unique command names.
    # Format: {"spec_path": "command1, command2, command3", ...}
    # Used by: assign_workers_unique_cmds() to get commands for relevance assessment
    
    specs_cmd_modules_definition: Annotated[dict, "Must have the unrepeated commands information description."]
    # Purpose: Maps specification paths to command module definitions.
    # Format: {"spec_path": {"command_name": ProposedModules(...)}, ...}
    # Used by: assign_workers_cmds_params() and assign_workers_cmds_rules() to get module info
    
    specs_cmd_modules_complete_info: Annotated[dict, "Must have the unrepeated commands additional information description."]
    # Purpose: Maps specification paths to complete command information including relevance assessment.
    # Format: {"spec_path": {"command_name": CommandRelevanceAssessment(...)}, ...}
    # Used by: assign_workers_cmd_modules_definition() to get command scope and variants
    
    content_index_pages: dict
    # Purpose: Maps specification paths to index page ranges.
    # Format: {"spec_path": {"start_index_page": N, "end_index_page": M}, ...}
    # Used by: identify_relevant_spec_sections() to extract index content
    
    # ========================================================================
    # WORKFLOW CONTROL STATE
    # ========================================================================
    
    missing_spec_paths_to_process: Annotated[list, "Must have a list of the spec paths which are missing to process in the stage."]
    # Purpose: Tracks which specification paths still need processing in the current stage.
    # Format: ["spec_path1", "spec_path2", ...]
    # Used by: assign_workers_* functions to determine which specs need workers
    # Updated by: dataCollection* functions to list missing specs after checkpoint loading
    
    save_dataCollection: Annotated[bool, "Must have an boolean value to track whenever is required to save DataCollection"]
    # Purpose: Controls whether results should be saved to dataCollectionAI folder.
    # True: Save results (new processing or updates)
    # False: Skip saving (results already exist or loaded from checkpoint)
    # Used by: save_load_checkpoint() to route workflow to save vs load paths
    # Updated by: dataCollection* functions based on whether checkpoints exist

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Save or Load CheckPoint Decision

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def save_load_checkpoint(state: ModuleContentState) -> Literal["savable", "loadable"]:
    """
    Decision function: Determine if workflow should process new data or load from checkpoint.
    
    Purpose:
        This is a conditional routing function used by LangGraph workflow edges to decide
        whether to execute processing (savable) or skip to next stage (loadable).
        It enables checkpoint-based workflow optimization - if results already exist,
        skip expensive processing and load existing results instead.
    
    How It Works:
        Checks the save_dataCollection flag in state:
        - True (savable): Results don't exist or need updating → route to processing nodes
        - False (loadable): Results exist in checkpoint → route to next stage, skip processing
    
    Args:
        state (ModuleContentState): Current workflow state containing save_dataCollection flag
    
    Returns:
        Literal["savable", "loadable"]: Routing decision for workflow conditional edges
            - "savable": Route to worker execution nodes (process new data)
            - "loadable": Route to next checkpoint check (skip processing)
    
    Used by:
        Workflow conditional edges throughout the graph to implement checkpoint-based routing:
        - After dataCollectionIndexExtraction
        - After dataCollectionRelevantSections
        - After dataCollectionUniqueCommands
        - After dataCollectionCommandsInfo
        - After dataCollectionModulesExtraction
    
    Integration:
        - Reads: state["save_dataCollection"] (set by dataCollection* functions)
        - Routes to: Worker execution nodes or next checkpoint check node
        - Part of: LangGraph conditional edge routing pattern
    """
    print("---CALL Start Save Check Point Decision---")
    savable = state["save_dataCollection"]

    if savable:
        return "savable"
    return "loadable"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Table Content Pages Extraction

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def index_files_found(state: ModuleContentState) -> Literal["index_found", "terminate"]:
    """
    Decision function: Check if all specifications have index pages extracted.
    
    Purpose:
        Determines whether index page extraction is complete for all specifications.
        Used to route workflow after dataCollectionIndexExtraction - if all index pages
        are found, skip extraction and proceed to next stage.
    
    How It Works:
        1. Gets list of all unique specification paths from specsBaseStores_df
        2. Checks if each spec has both start_index_page and end_index_page in content_index_pages
        3. Returns "index_found" only if ALL specs have complete index page definitions
        4. Otherwise returns "index_not_found" to trigger extraction
    
    Args:
        state (ModuleContentState): Current workflow state containing content_index_pages
    
    Returns:
        Literal["index_found", "index_not_found"]: Routing decision
            - "index_found": All specs have index pages → route to next stage (dataCollectionRelevantSections)
            - "index_not_found": Some specs missing index pages → route to extraction workers
    
    Used by:
        Workflow conditional edge after dataCollectionIndexExtraction node to decide
        whether to extract index pages or proceed to relevant sections extraction.
    
    Integration:
        - Reads: state["content_index_pages"] (populated by dataCollectionIndexExtraction)
        - Uses: specsBaseStores_df to get list of all spec paths
        - Routes to: exec_index_extraction_workers (if missing) or dataCollectionRelevantSections (if found)
    """
    print("---CALL Start Index Found Decision---")
    content_index = state["content_index_pages"]
    unique_spec_root_names = specsBaseStores_df['specRootPath'].unique().tolist()
    defined_index_pages = []

    # Check each specification path to see if it has complete index page information
    for spec_path in unique_spec_root_names:
        if spec_path in content_index:
            index_pages = content_index[spec_path]
            # Both start and end page numbers must be defined for extraction to be complete
            if 'start_index_page' in index_pages and 'end_index_page' in index_pages:
                defined_index_pages.append(True)

    # Only return "index_found" if ALL specs have complete index pages
    if len(unique_spec_root_names) == len(defined_index_pages) and all(defined_index_pages):
        return "index_found"
    return "index_not_found"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def dataCollectionIndexExtraction(state: ModuleContentState):
    """
    Checkpoint loading function: Load existing index pages or mark specifications for extraction.
    
    Purpose:
        Checks if index pages have already been extracted and saved to the dataCollectionAI
        checkpoint folder. If found, loads them into state. If missing, marks specifications
        for processing. This enables incremental processing and recovery from failures.
    
    How It Works:
        1. Gets list of all unique specification paths from specsBaseStores_df
        2. Lists all files in dataCollectionAI checkpoint folder
        3. For each specification, searches for {spec_path}/index_pages.json
        4. If found: Loads index pages into content_index_pages state
        5. If missing: Adds spec path to missing_spec_paths_to_process
        6. Sets save_dataCollection=True if any specs are missing
    
    Args:
        state (ModuleContentState): Workflow state (may contain existing content_index_pages)
    
    Returns:
        dict: Updated state with:
            - content_index_pages: Loaded from checkpoint if found, empty dict if not
            - missing_spec_paths_to_process: List of spec paths needing index extraction
            - save_dataCollection: True if any specs missing, False if all found
    
    Used by:
        - Workflow edge from START (first checkpoint check in workflow)
        - Routes to: index_files_found decision function
        - If all found → proceed to next stage
        - If any missing → route to index extraction workers
    
    Integration:
        - Reads from: dataCollectionAI checkpoint folder (Dataiku Folder)
        - Reads from: specsBaseStores_df to get list of all spec paths
        - Updates: state["content_index_pages"] with loaded data
        - Updates: state["missing_spec_paths_to_process"] with missing specs
        - Updates: state["save_dataCollection"] based on whether checkpoints exist
    
    Checkpoint File Format:
        Expected file: {spec_path}/index_pages.json
        Content format: {"spec_path": {"start_index_page": N, "end_index_page": M}}
        This matches the format saved by save_spec_index_pages.
    
    Workflow Impact:
        This is the first checkpoint check in the workflow. If index pages exist for all specs,
        the workflow skips expensive index extraction and proceeds directly to relevant sections.
        This can save significant time on subsequent runs.
    
    Error Handling:
        If checkpoint file exists but is corrupted/invalid, logs error and marks spec as missing.
        This ensures the workflow can recover by re-extracting rather than failing completely.
    """
    print("--- Loading Specs Index in Data Collection Files---")
    print(f"[DEBUG] dataCollectionIndexExtraction: Function called")
    unique_spec_root_names = specsBaseStores_df['specRootPath'].unique().tolist()
    print(f"[DEBUG] dataCollectionIndexExtraction: Searching for {len(unique_spec_root_names)} spec root paths: {unique_spec_root_names}")
    
    # List all files in checkpoint folder to search for index page files
    all_data_paths = list(dataCollectionAI.list_paths_in_partition())
    print(f"[DEBUG] dataCollectionIndexExtraction: Total paths in dataCollectionAI partition: {len(all_data_paths)}")
    if len(all_data_paths) > 0:
        print(f"[DEBUG] dataCollectionIndexExtraction: Sample paths: {all_data_paths[:5]}")
    
    specIndexDataCollectionPaths = []  # Found checkpoint files
    missingSpecIndexPaths = []  # Specs needing extraction
    # Check each specification for existing index page checkpoint
    for root_path_name in unique_spec_root_names:
        path_found_in_collection = False
        print(f"[DEBUG] dataCollectionIndexExtraction: Searching for root_path_name = {root_path_name}")
        for dataPath in all_data_paths:
            # Look for index_pages.json file for this specification
            if root_path_name in dataPath and 'index_pages.json' in dataPath:
                print(f"---Data Collection for {root_path_name} Found: {dataPath} ---")
                specIndexDataCollectionPaths.append(dataPath)
                path_found_in_collection = True
                break
        if not path_found_in_collection:
            print(f"[DEBUG] dataCollectionIndexExtraction: NOT FOUND for {root_path_name}")
            missingSpecIndexPaths.append(root_path_name)

    print(f"[DEBUG] dataCollectionIndexExtraction: Found {len(specIndexDataCollectionPaths)} files, missing {len(missingSpecIndexPaths)} paths")
    
    spec_index_pages = {}
    for dataCollectionPath in specIndexDataCollectionPaths:
        print(f'Loading Spec Index Data Collection {dataCollectionPath}')
        try:
            with dataCollectionAI.get_download_stream(dataCollectionPath) as f:
                file_content = f.read().decode("utf-8")
                print(f"[DEBUG] dataCollectionIndexExtraction: File size = {len(file_content)} bytes")
                index_pages = json.loads(file_content)
                print(f"[DEBUG] dataCollectionIndexExtraction: Loaded JSON keys = {list(index_pages.keys()) if isinstance(index_pages, dict) else 'N/A'}")
                spec_index_pages.update(index_pages)
                print(f"[DEBUG] dataCollectionIndexExtraction: Updated spec_index_pages, now has {len(spec_index_pages)} keys")
        except Exception as e:
            print(f"[DEBUG] dataCollectionIndexExtraction: ERROR loading {dataCollectionPath}: {str(e)}")
            import traceback
            print(f"[DEBUG] dataCollectionIndexExtraction: Traceback: {traceback.format_exc()}")

    print(f"[DEBUG] dataCollectionIndexExtraction: Final spec_index_pages has {len(spec_index_pages)} keys: {list(spec_index_pages.keys())}")
    print(f"[DEBUG] dataCollectionIndexExtraction: save_dataCollection = {bool(missingSpecIndexPaths)}")
    
    return {'content_index_pages': spec_index_pages,
            'missing_spec_paths_to_process': missingSpecIndexPaths,
            'save_dataCollection': bool(missingSpecIndexPaths)}
# 'index_pages_found': not bool(missingSpecIndexPaths)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
class WorkerIndexExtractionState(TypedDict):
    spec_full_path: str
    missing_spec_index_pages: Annotated[
        list, operator.add
    ]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def locate_spec_index_pages(state: WorkerIndexExtractionState):
    """
    Worker function: Identify table of contents pages in a specification document.
    
    Purpose:
        Analyzes specification pages sequentially to find the start and end of the
        table of contents section. The table of contents is used later to identify
        relevant sections for command extraction.
    
    How It Works:
        1. Gets specification pages from df_sorted filtered by spec_full_path
        2. Iterates through pages sequentially
        3. For each page, uses LLM to classify as "index" (table of contents) or "none"
        4. Records first "index" page as start_index_page
        5. Records page before first "none" after start as end_index_page
        6. Returns index page range for the specification
    
    Args:
        state (WorkerIndexExtractionState): Worker state containing:
            - spec_full_path: Specification path being processed
    
    Returns:
        dict: Updates state["missing_spec_index_pages"] with:
            Format: [{"spec_path": {"start_index_page": N, "end_index_page": M}}]
    
    Used by:
        - assign_workers_index_extraction() dispatches this worker via Send()
        - Called in parallel for each missing specification
        - Each worker processes one specification sequentially (page by page)
    
    Integration:
        - Uses: df_sorted DataFrame (created during initialization, sorted by specRootPath and page number)
        - Uses: llm_t (DKUChatLLM) with structured output (indexFound) to classify pages
        - Uses: specs_desc_dict for specification metadata
        - Writes to: state["missing_spec_index_pages"] (accumulated by LangGraph)
    
    Index Detection Logic:
        - System message defines index page characteristics:
          * List of section headers with page numbers
          * Hierarchical numbering (1, 1.2, 3.4.1)
          * Excludes figure indexes, table indexes, annexes indexes
        - LLM classifies each page as "index" or "none"
        - First "index" page → start_index_page
        - First "none" after start → end_index_page = previous page number
    
    Sequential Processing:
        This worker processes pages sequentially (not in parallel) because:
        - Need to find first index page (start)
        - Need to find first non-index page after start (end)
        - Sequential order is required for correct boundary detection
    
    LLM Configuration:
        - Temperature: 0 (binary classification, needs deterministic output)
        - Retry: 4 attempts with exponential backoff (LLM calls can be flaky)
        - Structured output: indexFound model with pageType field
    
    Workflow Position:
        Runs AFTER dataCollectionIndexExtraction (which checks for existing index pages)
        and routes to save_spec_index_pages (which saves extracted index pages)
    """
    spec_full_path = state["spec_full_path"]
    spec_full_name = specs_desc_dict[spec_full_path]["spec_full_name"]

    systemMessage = f"""You are an expert in {spec_full_name}. Decide whether the provided page is a specification content index or not.

Specification index definition: a list of section headers (optionally numbered hierarchically like 1, 1.2, 3.4.1) each paired with a page number (e.g. "3.2 Power Management ..... 27"). It is primarily header lines plus page numbers. Ignore decorative dots.
Exclude index pages that consist only of figure indexes, table indexes, or annexes indexes (even if they have page numbers). Those count as none.

Return 'index'; if the pattern matches a specification index page.
Return 'none'; otherwise or if ambiguous.

Output must be a single word: 'index' or 'none'. No explanations."""

    message = """
Page content:
{context}
"""
    prompt = ChatPromptTemplate.from_messages([("system", systemMessage),
                                               ("human", message)])

    print(f"---Data Collection not Found for {spec_full_path} Index Pages---")

    class indexFound(BaseModel):
        """Assign module state for table contant detection."""

        pageType: str = Field(description="Set 'index' or 'none' in base of context.")

    # Temperature is 0, as it is expected a binary response
    model = DKUChatLLM(llm_id=llm_id, temperature=0)
    llm_with_tool = model.with_structured_output(indexFound)

    print(f"---- Seeking {spec_full_path} Specification Index -------")

    spec_indexPages = {}
    spec_indexPages[spec_full_path] = {}
    df_specName = df_sorted[df_sorted['specRootPath'] == spec_full_path]

    start_IndexFound = False
    end_IndexFound = False
    for current_file_idx in range(df_specName.shape[0]):
        print(f"---- {spec_full_path} Specification Analizing {current_file_idx} page -------")
        context = df_specName.iloc[current_file_idx].text
        page_num = int(df_specName.iloc[current_file_idx].specRootPageNum)

        rag_chain = {
            "context": RunnablePassthrough(),
        } | prompt | llm_with_tool


        rag_chain_w_retry = rag_chain.with_retry(
            wait_exponential_jitter=True,
            stop_after_attempt=4,
            exponential_jitter_params={"initial": 2}
        )
        response = invoke_with_rate_limit_retry(rag_chain_w_retry, context)
        time.sleep(1)

        if response.pageType == 'index' and start_IndexFound is False:
            print(f"---- {spec_full_path} Specification Start Index Found: {page_num} page -------")
            spec_indexPages[spec_full_path].update({'start_index_page': page_num})
            start_IndexFound = True
        elif response.pageType == 'none' and start_IndexFound is True:
            print(f"---- {spec_full_path} Specification End Index Found: {page_num-1} page -------")
            spec_indexPages[spec_full_path].update({'end_index_page': page_num-1})
            end_IndexFound = True
            break

    return {'missing_spec_index_pages': [spec_indexPages]}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def assign_workers_index_extraction(state: ModuleContentState) -> List[Send]:
    """
    Assigns workers to identify specification index pages.
    """
    spec_missing_paths = state["missing_spec_paths_to_process"]
    print(f"----------{spec_missing_paths} Missing Paths")
    final_output = []
    for spec_path in spec_missing_paths:
        final_output.append(Send(
            "locate_spec_index_pages",
            {"spec_full_path": spec_path}))
        time.sleep(25)
    return final_output

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Update Data Collection for Content Index Pages

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def save_spec_index_pages(state: ModuleContentState):
    if state['save_dataCollection']:
        indexes_spec_pages = state['missing_spec_index_pages']
        spec_missing_paths = state["missing_spec_paths_to_process"]
        content_index_pages = state["content_index_pages"]

        for pathFile in spec_missing_paths:
            for index_pages in indexes_spec_pages:
                if next(iter(index_pages)) == pathFile:
                    print(f"Saving index pages for: {pathFile}")
                    content_index_pages.update(index_pages)
                    output_path = f"{pathFile}/index_pages.json"
                    json_bytes = json.dumps(index_pages, indent=2).encode("utf-8")
                    with dataCollectionAI.get_writer(output_path) as w_binary:
                        w_binary.write(json_bytes)

        # print(f"--- Spec Index Pages: {content_index_pages}")

    return {'save_dataCollection': False, "content_index_pages": content_index_pages, 'missing_spec_paths_to_process': []}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Find Table Content Relevant Sections

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#llm_t = DKUChatLLM(llm_id="azureopenai:Azure-OpenAI-Prod-4-1:gpt-4.1", temperature=0.2)
llm_t = DKUChatLLM(llm_id=llm_id, temperature=0.2)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def dataCollectionRelevantSections(state: ModuleContentState):
    """
    Check if Most Relevant Sections are already in DataCollection

    Args:
        state (messages): The current state

    Returns:
        str: Load Information if exist or keep the proccess for AI Agent
    """
    print("--- Loading Most Relevant Spec Sections in Data Collection Files---")
    print(f"[DEBUG] dataCollectionRelevantSections: Function called")
    unique_spec_root_names = specsBaseStores_df['specRootPath'].unique().tolist()
    print(f"[DEBUG] dataCollectionRelevantSections: Searching for {len(unique_spec_root_names)} spec root paths: {unique_spec_root_names}")
    
    all_data_paths = list(dataCollectionAI.list_paths_in_partition())
    print(f"[DEBUG] dataCollectionRelevantSections: Total paths in dataCollectionAI partition: {len(all_data_paths)}")
    
    specRelSecDataCollectionPaths = []
    missingSpecMostRelSecPaths = []
    for root_path_name in unique_spec_root_names:
        path_found_in_collection = False
        print(f"[DEBUG] dataCollectionRelevantSections: Searching for root_path_name = {root_path_name}")
        for dataPath in all_data_paths:
            if root_path_name in dataPath and 'relevant_sections.json' in dataPath:
                print(f"---Data Collection for {root_path_name} Found: {dataPath} ---")
                specRelSecDataCollectionPaths.append(dataPath)
                path_found_in_collection = True
                break
        if not path_found_in_collection:
            print(f"[DEBUG] dataCollectionRelevantSections: NOT FOUND for {root_path_name}")
            missingSpecMostRelSecPaths.append(root_path_name)

    print(f"[DEBUG] dataCollectionRelevantSections: Found {len(specRelSecDataCollectionPaths)} files, missing {len(missingSpecMostRelSecPaths)} paths")

    spec_relevant_sections = {}
    for dataCollectionPath in specRelSecDataCollectionPaths:
        print(f'Loading Relevant Sections Data Collection {dataCollectionPath}')
        try:
            with dataCollectionAI.get_download_stream(dataCollectionPath) as f:
                file_content = f.read().decode("utf-8")
                print(f"[DEBUG] dataCollectionRelevantSections: File size = {len(file_content)} bytes")
                relevantSectionsJson = json.loads(file_content)
                print(f"[DEBUG] dataCollectionRelevantSections: Loaded JSON keys = {list(relevantSectionsJson.keys()) if isinstance(relevantSectionsJson, dict) else 'N/A'}")
                for spec_path, relevantSections in relevantSectionsJson.items():
                    print(f'--- Loading {spec_path} most relevant sections ---')
                    relevantSectionsList = [relevantSections[key] for key in sorted(relevantSections, key=lambda x: int(x))]
                    relevantSections = ", ".join(relevantSectionsList)
                    spec_relevant_sections.update({spec_path: relevantSections})
                    print(f"[DEBUG] dataCollectionRelevantSections: Loaded {len(relevantSectionsList)} sections for {spec_path}")
        except Exception as e:
            print(f"[DEBUG] dataCollectionRelevantSections: ERROR loading {dataCollectionPath}: {str(e)}")
            import traceback
            print(f"[DEBUG] dataCollectionRelevantSections: Traceback: {traceback.format_exc()}")

    print(f"[DEBUG] dataCollectionRelevantSections: Final spec_relevant_sections has {len(spec_relevant_sections)} keys: {list(spec_relevant_sections.keys())}")
    print(f"[DEBUG] dataCollectionRelevantSections: save_dataCollection = {bool(missingSpecMostRelSecPaths)}")

    return {'specs_relevant_sections': spec_relevant_sections,
            'missing_spec_paths_to_process': missingSpecMostRelSecPaths,
            'save_dataCollection': bool(missingSpecMostRelSecPaths)}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
class WorkerRelevantSectionExtractionState(TypedDict):
    spec_full_path: str
    content_index_pages: dict
    missing_spec_relevant_sections: Annotated[
        list, operator.add
    ]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def identify_relevant_spec_sections(state: WorkerRelevantSectionExtractionState):
    """
    Worker function: Identify relevant specification sections for command extraction.
    
    Purpose:
        Analyzes the table of contents (index pages) of a specification to identify
        sections that relate directly to commands and operational processes. This filters
        out general concepts, annexes, and informative sections that don't contain
        callable commands.
    
    How It Works:
        1. Gets index page range from content_index_pages state
        2. Extracts text content from index pages (start_index_page to end_index_page)
        3. Uses LLM to analyze index content and extract relevant section names
        4. Returns comma-separated list of relevant module/section names
    
    Args:
        state (WorkerRelevantSectionExtractionState): Worker state containing:
            - spec_full_path: Specification path being processed
            - content_index_pages: Dictionary with index page ranges
                Format: {"spec_path": {"start_index_page": N, "end_index_page": M}}
    
    Returns:
        dict: Updates state["missing_spec_relevant_sections"] with:
            Format: [{"spec_path": "section1, section2, section3"}]
    
    Used by:
        - assign_workers_relevant_sections_extraction() dispatches this worker via Send()
        - Called in parallel for each missing specification
        - Each worker processes one specification's index pages
    
    Integration:
        - Uses: df_sorted DataFrame to extract index page content
        - Uses: content_index_pages from state to get page range
        - Uses: llm_t (DKUChatLLM) with structured output (structModulesList)
        - Uses: specs_desc_dict for specification metadata
        - Writes to: state["missing_spec_relevant_sections"] (accumulated by LangGraph)
    
    Section Filtering Criteria:
        Includes sections related to:
        - Commands and their specifications
        - Operational processes and state transitions
        
        Excludes:
        - General concepts, annexes, references, overviews, introductions
        - Informative sections
        - Figure content and descriptions
        - Duplicate subsection names
    
    Output Format:
        Comma-separated list of relevant section names:
        "Section 1, Section 2, Section 3"
        These section names are used later to extract commands from those sections.
    
    LLM Prompt Strategy:
        - System message: Defines validation focus and RCV methodology
        - Input: Index page content (section headers with page numbers)
        - Output: Comma-separated list of relevant module/section names
        - Filtering: LLM applies criteria to exclude non-relevant sections
    
    Workflow Position:
        Runs AFTER dataCollectionRelevantSections (which checks for existing sections)
        and routes to save_spec_most_relavant_sections (which saves extracted sections)
    """
    spec_full_path = state["spec_full_path"]
    spec_full_name = specs_desc_dict[spec_full_path]["spec_full_name"]
    spec_full_desc = specs_desc_dict[spec_full_path]["spec_full_description"]
    systemMessage = f"""You are an SSDs testing validation architect expert for the {spec_full_name} specification. Your primary validation focus is: {spec_full_desc}.
All validation testing follows the RCV (Requirements Coverage Verification) approach with methodology detailed here: {rcvDesc}."""

    message = """Based on the index contents of {specFullName}, extract module names derived from sections and subsections that relate directly to:
- Commands and their specifications
- Operational processes and state transitions

Filtering criteria:
- Exclude: general concepts, annexes, references, overviews, introductions, informative sections
- Exclude: figure content and descriptions
- Avoid: duplicate subsection names
- Format: plain text module names without special characters

Index Content:
{context}

Output: Comma-separated list of relevant module names only.
"""

    print("---Get Relevant Section List ---")

    prompt = ChatPromptTemplate.from_messages([("system", systemMessage),
                                               ("human", message)])

    class structModulesList(BaseModel):
        """Assign module state for module designs check."""

        module_struct: str = Field(description="Provide a single comma-separated list of the module names only, i.e. 'Module Name 1, Module Name 2, Module Name NNN'")

    llm_with_tool = llm_t.with_structured_output(structModulesList)

    rag_chain = {
        "specFullName": lambda specFullName: spec_full_name,
        "context": RunnablePassthrough(),
    } | prompt | llm_with_tool


    rag_chain_w_retry = rag_chain.with_retry(
        wait_exponential_jitter=True,
        stop_after_attempt=4,
        exponential_jitter_params={"initial": 2}
    )

    content_index = state['content_index_pages']
    
    # Validate required keys exist
    if not isinstance(content_index, dict):
        raise ValueError(f"content_index_pages must be a dict, got {type(content_index)}")
    if 'start_index_page' not in content_index:
        raise KeyError(f"start_index_page missing in content_index_pages for {spec_full_path}. Available keys: {list(content_index.keys())}")
    if 'end_index_page' not in content_index:
        raise KeyError(f"end_index_page missing in content_index_pages for {spec_full_path}. Available keys: {list(content_index.keys())}")

    context = ""
    df_specName = df_sorted[
        (df_sorted['store'] == 'baseStore') & (df_sorted['specRootPath'] == spec_full_path)
    ]
    context += f"This is the content for {spec_full_path} \n"
    context += "\n".join(df_specName["text"].iloc[(content_index["start_index_page"] - 1):(content_index["end_index_page"])].tolist())
    context += "\n\n\n"

    print(f"[CONTEXT] {truncate_text_for_log(context, max_length=150)}")

    updated_modules = invoke_with_rate_limit_retry(rag_chain_w_retry, context)

    # print(f'Most Relevant Section {spec_full_path}: {updated_modules.module_struct}')

    return {"missing_spec_relevant_sections": [{spec_full_path: updated_modules.module_struct}]}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def assign_workers_relevant_sections_extraction(state: ModuleContentState) -> List[Send]:
    """
    Assigns workers to identify specification index pages.
    """
    spec_missing_paths = state["missing_spec_paths_to_process"]
    spec_content_index = state['content_index_pages']
    print(f"----------{spec_missing_paths} Relevant Sections Missing Paths")
    final_output = []
    for spec_path in spec_missing_paths:
        # Validate index pages exist and are complete
        if spec_path not in spec_content_index:
            print(f"[WARNING] Skipping {spec_path}: not found in content_index_pages")
            continue
        
        index_pages = spec_content_index[spec_path]
        if not isinstance(index_pages, dict):
            print(f"[WARNING] Skipping {spec_path}: invalid index_pages format")
            continue
        
        if 'start_index_page' not in index_pages or 'end_index_page' not in index_pages:
            print(f"[WARNING] Skipping {spec_path}: incomplete index pages (has: {list(index_pages.keys())})")
            continue
        
        final_output.append(Send(
            "identify_relevant_spec_sections",
            {"spec_full_path": spec_path,
             "content_index_pages": spec_content_index[spec_path]}))
        time.sleep(25)
    return final_output

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Update Data Collection with Most Relevant Sections

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def save_spec_most_relavant_sections(state: ModuleContentState):
    if state['save_dataCollection']:
        missingSpecRelevantSections = state['missing_spec_relevant_sections']
        specMissingPaths = state["missing_spec_paths_to_process"]
        specsRelevantSections = state["specs_relevant_sections"]

        for pathFile in specMissingPaths:
            # missingSpecRelevantSections is a list of dictionaries, check which ones match
            # with the pathFile
            for relevant_section_dict in missingSpecRelevantSections:
                if next(iter(relevant_section_dict)) == pathFile:
                    listRelevantSections = next(iter(relevant_section_dict.values()))
                    dictRelevantSections = {str(section_id): section.strip() for section_id, section in enumerate(listRelevantSections.split(','))}
                    print(f"Saving relevant sections for: {pathFile}")
                    formattedRelevantSections = {pathFile: dictRelevantSections}
                    specsRelevantSections.update(relevant_section_dict)
                    output_path = f"{pathFile}/relevant_sections.json"
                    json_bytes = json.dumps(formattedRelevantSections, indent=2).encode("utf-8")
                    with dataCollectionAI.get_writer(output_path) as w_binary:
                        w_binary.write(json_bytes)
        print(f'specs_relevant_sections: {specsRelevantSections}')
    return {'save_dataCollection': False, "specs_relevant_sections": specsRelevantSections, 'missing_spec_paths_to_process': []}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Seeking Commands to Process the Most Relevant Table Content Sections In Base of Context

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def dataCollectionUniqueCommands(state: ModuleContentState):
    """
    Check if Commands are already in DataCollection

    Args:
        state (messages): The current state

    Returns:
        str: Load Information if exist or keep the proccess for AI Agent
    """
    print("--- Loading Unique Callable Commands in Data Collection Files---")
    print(f"[DEBUG] dataCollectionUniqueCommands: Function called")
    unique_spec_root_names = specsBaseStores_df['specRootPath'].unique().tolist()
    print(f"[DEBUG] dataCollectionUniqueCommands: Searching for {len(unique_spec_root_names)} spec root paths: {unique_spec_root_names}")
    
    all_data_paths = list(dataCollectionAI.list_paths_in_partition())
    print(f"[DEBUG] dataCollectionUniqueCommands: Total paths in dataCollectionAI partition: {len(all_data_paths)}")
    
    specUniqCmdDataCollectionPaths = []
    missingSpecUniqCmdPaths = []
    for root_path_name in unique_spec_root_names:
        path_found_in_collection = False
        print(f"[DEBUG] dataCollectionUniqueCommands: Searching for root_path_name = {root_path_name}")
        for dataPath in all_data_paths:
            if root_path_name in dataPath and 'callable_commands.json' in dataPath:
                print(f"---Data Collection for {root_path_name} Found: {dataPath} ---")
                specUniqCmdDataCollectionPaths.append(dataPath)
                path_found_in_collection = True
                break
        if not path_found_in_collection:
            print(f"[DEBUG] dataCollectionUniqueCommands: NOT FOUND for {root_path_name}")
            missingSpecUniqCmdPaths.append(root_path_name)

    print(f"[DEBUG] dataCollectionUniqueCommands: Found {len(specUniqCmdDataCollectionPaths)} files, missing {len(missingSpecUniqCmdPaths)} paths")

    specUniqCmds = {}
    for dataCollectionPath in specUniqCmdDataCollectionPaths:
        print(f'Loading Unique Callable Commands Collection {dataCollectionPath}')
        try:
            with dataCollectionAI.get_download_stream(dataCollectionPath) as f:
                file_content = f.read().decode("utf-8")
                print(f"[DEBUG] dataCollectionUniqueCommands: File size = {len(file_content)} bytes")
                uniqCmdsJson = json.loads(file_content)
                print(f"[DEBUG] dataCollectionUniqueCommands: Loaded JSON keys = {list(uniqCmdsJson.keys()) if isinstance(uniqCmdsJson, dict) else 'N/A'}")
                for spec_path, uniqCmds in uniqCmdsJson.items():
                    print(f'--- Loading {spec_path} Unique Callable Commands ---')
                    uniqCmdsList = [uniqCmds[key] for key in sorted(uniqCmds, key=lambda x: int(x))]
                    unqCmdsStr = ", ".join(uniqCmdsList)
                    specUniqCmds.update({spec_path: unqCmdsStr})
                    print(f"[DEBUG] dataCollectionUniqueCommands: Loaded {len(uniqCmdsList)} commands for {spec_path}")
        except Exception as e:
            print(f"[DEBUG] dataCollectionUniqueCommands: ERROR loading {dataCollectionPath}: {str(e)}")
            import traceback
            print(f"[DEBUG] dataCollectionUniqueCommands: Traceback: {traceback.format_exc()}")

    print(f"[DEBUG] dataCollectionUniqueCommands: Final specUniqCmds has {len(specUniqCmds)} keys: {list(specUniqCmds.keys())}")
    print(f"[DEBUG] dataCollectionUniqueCommands: save_dataCollection = {bool(missingSpecUniqCmdPaths)}")

    return {'specs_callable_unique_cmds': specUniqCmds,
            'missing_spec_paths_to_process': missingSpecUniqCmdPaths,
            'save_dataCollection': bool(missingSpecUniqCmdPaths)}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
class WorkerState(TypedDict):
    spec_section_name: str
    spec_full_path: str
    complete_callable_commands: Annotated[list, operator.add]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def extract_callable_commands(state: WorkerState):
    """
    Worker function: Extract callable commands from a specification section.
    
    Purpose:
        Analyzes a specific specification section to identify truly callable commands
        (operations that can be invoked, not just descriptive text). This is the first
        step in command extraction - it finds candidate commands from relevant sections.
    
    How It Works:
        1. Sets CURRENT_SPEC_FULL_PATH global to filter Cortex Search results
        2. Uses RAG chain with get_spec_info_section() to retrieve relevant spec content
        3. LLM analyzes the content to identify callable commands with evidence
        4. Returns structured output with command names, key phrases, and scores
        5. Results are accumulated in state["complete_callable_commands"] via operator.add
    
    Args:
        state (WorkerState): Worker state containing:
            - spec_full_path: Specification path being processed
            - spec_section_name: Name of the section to analyze
    
    Returns:
        dict: Updates state["complete_callable_commands"] with:
            Format: [{"spec_path": ModuleStruct(command_names="...", keyPhrases="...", score="...")}]
    
    Used by:
        - assign_workers_callable_commands_extraction() dispatches this worker via Send()
        - Called in parallel for each relevant section across all specifications
    
    Integration:
        - Sets: CURRENT_SPEC_FULL_PATH global (used by get_spec_info_section)
        - Uses: get_spec_info_section() to retrieve spec content via Cortex Search
        - Uses: llm_t (DKUChatLLM) with structured output (ModuleStruct)
        - Writes to: state["complete_callable_commands"] (accumulated by LangGraph)
        - Reads from: specs_desc_dict for spec metadata
    
    LLM Prompt Strategy:
        - System message: Defines callable command criteria (must have invocation mechanics)
        - Validation filters: Callability evidence, specificity, relevance, non-ambiguity
        - Output: Command names, key phrases proving callability, relevance scores (1-100)
    
    Note:
        This worker processes ONE section at a time. Multiple workers run in parallel
        to process all relevant sections across all specifications concurrently.
    """
    specFullPath = state["spec_full_path"]
    print(f'extract callable commands path: {specFullPath}')
    specFullName = specs_desc_dict[specFullPath]["spec_full_name"]
    specFullDesc = specs_desc_dict[specFullPath]["spec_full_description"]
    systemMessage = f"""
You are an SSD validation architect specializing in the {specFullName} specification, designing robust RCV-based test coverage.

Mission:
1. Extract only truly callable commands relevant to validating {specFullName}.
2. Discard generic, narrative, descriptive, or non-invocable terms.
3. Prioritize commands that enable protocol/state/feature verification, security enforcement, configuration, sequencing, error/status handling, or capability negotiation.

Constraints:
- Use only the provided content. No outside assumptions. No hallucinated commands.
- If evidence is insufficient to prove callability, discard.

Definitions:
Callable Command: A named operation explicitly invocable (via opcode, API, method, request/response, message, function, transaction, sequence step) with at least one of:
  a) Parameters, fields, operands, payload, or format specification.
  b) Preconditions, states, triggers, sequencing rules, or execution context.
  c) Observable results, effects, status codes, error reporting, or side effects.

Irrelevant Command: Present in text but not contributing to validation goals (e.g., marketing, legacy-only, informal examples, purely informative, outside scope of {specFullName}'s stated intent).

Output must be objective and evidence-grounded.

Specification Description: {specFullDesc}
RCV Methodology Reference: {rcvDesc}
"""

    message = """
From the following specification page segments, identify candidate callable commands.

Detection Guidance:
1. Scan for keywords indicating operations: command, method, request, response, opcode, function, operation, subcommand, message, payload, sequence, transaction, protocol, action, procedure, routine, invocation, execution.
2. Merge duplicates or variants referring to the same callable entity.

Validation Filters (must pass ALL to retain):
A. Callability Evidence: Text shows invocation mechanics (opcode value, request/response structure, field layout, parameters, sequence position, state dependency, or explicit call semantics).
B. Specificity: Not a broad heading or category (e.g., "Security Commands") unless individual callable members are enumerated.
C. Relevance: Directly supports validation of required behavior, compliance, sequencing, configuration, security, state management, error/status handling, or negotiation as defined in the specification description.
D. Non-Ambiguity: Name maps unambiguously to a single callable artifact.

If any filter fails → discard.

Discard Reasons (Examples):
- Not callable (descriptive, informative, or narrative only).
- Generic group heading without discrete member detail.
- Ambiguous label or placeholder.
- Out of scope for specification validation intent.
- Insufficient evidence in provided text.

When you identify a command, consider the following for the final output:
Commands name: Use the root name of each callable command. Split multi word names with a space. Avoid redundant terms (e.g., command, method, protocol) unless necessary.
Key Phrases: For each kept command gather minimal verbatim key phrases (exact substrings) proving callability/relevance (limit to most salient; avoid entire paragraphs).
Score: Assign each retained command a score (integer 1–100) relative to the command with the richest documented detail (fields, flow, rules, responses, constraints). That command = 100. Others scaled proportionally. If only one command: score 100.

Specification {section} Section Segment:
{context}
"""

    specSectionName = state["spec_section_name"]

    print(f"---Extract Callable Commands on Section {specSectionName} ---")

    prompt = ChatPromptTemplate.from_messages([("system", systemMessage),
                                               ("human", message)])

    class ModuleStruct(BaseModel):
        command_names: str = Field(description="Name of the commands split by a comma. i.e. 'command1, command2, command3'; in case of no commands just set 'no'")
        keyPhrases: str = Field(description="Enlist keyphrases split by command use ::: to split between commands and :: to split between keyphrases, i.e command1: keyphrase1::keyphrase2::keyphase3:::command2: keyphrase1:: keyphrase2:: keyphase3:::command3: keyphrase1, keyphrase2, keyphase3; in case of no command just set 'no'")
        score: str = Field(description="Enlist the commands with a score split a command, i.e command1-score1, command2-score2, command3-score3; in case of no command then set 'no'")

    llm_with_tool = llm_t.with_structured_output(ModuleStruct)

    global CURRENT_SPEC_FULL_PATH
    # This wil set the global variable need it to create the right retriever get_spec_info_section
    CURRENT_SPEC_FULL_PATH = specFullPath

    print(f'Set Retriever Path to {CURRENT_SPEC_FULL_PATH}')

    rag_chain = {
        "section": RunnablePassthrough(),
        "context": get_spec_info_section, #get_retriever_module_info, #retriever_module_info,
    } | prompt | llm_with_tool

    rag_chain_w_retry = rag_chain.with_retry(
        wait_exponential_jitter=True,
        stop_after_attempt=4,
        exponential_jitter_params={"initial": 2}
    )

    response = invoke_with_rate_limit_retry(rag_chain_w_retry, specSectionName)

    time.sleep(2)
    return {"complete_callable_commands": [{specFullPath: response}]}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def assign_workers_callable_commands_extraction(state: ModuleContentState):
    """
    Worker assignment function: Dispatch parallel workers to extract commands from relevant sections.
    
    Purpose:
        Assigns one worker per relevant section across all missing specifications.
        Each worker processes one section to extract callable commands. Workers run
        in parallel to process all sections concurrently.
    
    How It Works:
        1. Gets list of specifications that need command extraction (missing_spec_paths_to_process)
        2. For each specification, gets its relevant sections from specs_relevant_sections
        3. Creates one Send() operation per section to dispatch extract_callable_commands worker
        4. Returns list of Send operations for LangGraph to execute in parallel
    
    Args:
        state (ModuleContentState): Workflow state containing:
            - missing_spec_paths_to_process: List of spec paths needing command extraction
            - specs_relevant_sections: Dictionary mapping spec paths to comma-separated section lists
    
    Returns:
        List[Send]: List of Send operations, each dispatching extract_callable_commands worker
            Format: [Send("extract_callable_commands", {"spec_full_path": "...", "spec_section_name": "..."}), ...]
    
    Used by:
        - Workflow conditional edge from exec_extract_callable_commands_workers node
        - LangGraph executes all Send operations in parallel (up to max_concurrency limit)
        - Each Send dispatches extract_callable_commands worker with section-specific state
    
    Integration:
        - Reads from: state["missing_spec_paths_to_process"] (set by dataCollectionUniqueCommands)
        - Reads from: state["specs_relevant_sections"] (loaded/generated in previous stage)
        - Dispatches to: extract_callable_commands worker function
        - Workers write to: state["complete_callable_commands"] (accumulated by LangGraph)
    
    Parallel Execution:
        If a specification has 10 relevant sections, this creates 10 Send operations.
        LangGraph executes them concurrently (subject to max_concurrency=8 limit),
        significantly speeding up command extraction compared to sequential processing.
    
    Workflow Position:
        Runs AFTER dataCollectionUniqueCommands (which checks for existing commands)
        and routes to extract_callable_commands workers (which extract commands from sections)
    """
    specMissingPaths = state["missing_spec_paths_to_process"]
    specsRelevantSections = state['specs_relevant_sections']
    print(f"----------{specMissingPaths} Callable Commands Missing Paths")
    final_output = []
    for specPath in specMissingPaths:
        specRelevantSections = specsRelevantSections[specPath]
        # Parse comma-separated section list into individual section names
        relevantSectionsList = [section.strip() for section in specRelevantSections.split(", ") if section.strip()]
        print(f"----------Relevant Sections for spec {specPath}: {relevantSectionsList}")
        # Create one worker per section for parallel processing
        for relevantSection in relevantSectionsList:
            final_output.append(Send(
                "extract_callable_commands",
                {"spec_full_path": specPath,
                 "spec_section_name": relevantSection}))
        time.sleep(25)  # Rate limiting between specifications
    return final_output

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Identify Unique Commands

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
class WorkerState(TypedDict):
    callable_commands_info: str
    spec_full_path: str
    complete_unique_callable_commands: Annotated[list, operator.add]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def prune_command_list(state: WorkerState):
    """
    Worker function: Consolidate and deduplicate commands extracted from multiple sections.
    
    Purpose:
        Takes all commands extracted from different sections of a specification and:
        1. Normalizes command names (removes duplicates, variants, aliases)
        2. Merges commands referring to the same operation
        3. Keeps the variant with highest evidence score
        4. Produces final unique command list for the specification
    
    How It Works:
        1. Receives aggregated command extraction results for one specification
        2. LLM analyzes all commands to identify duplicates and variants
        3. Normalizes names (removes numbering, punctuation, collapses whitespace)
        4. Merges duplicates based on key phrases and scores
        5. Returns comma-separated list of unique commands
    
    Args:
        state (WorkerState): Worker state containing:
            - spec_full_path: Specification path being processed
            - callable_commands_info: Aggregated command extraction results from all sections
                Format: Multi-line string with command names, key phrases, and scores
    
    Returns:
        dict: Updates state["complete_unique_callable_commands"] with:
            Format: [{"spec_path": "command1, command2, command3"}]
    
    Used by:
        - assign_workers_prune_commands_list() dispatches this worker via Send()
        - Called once per specification after all section extractions complete
        - Processes aggregated results from multiple extract_callable_commands workers
    
    Integration:
        - Reads from: state["complete_callable_commands"] (via assign_workers function)
        - Uses: llm_t (DKUChatLLM) with structured output (ModuleStruct)
        - Writes to: state["complete_unique_callable_commands"] (accumulated by LangGraph)
        - Reads from: specs_desc_dict for spec metadata
    
    LLM Prompt Strategy:
        - System message: Instructions for deduplication and normalization
        - Input: All candidate commands with their key phrases and scores
        - Output: Single comma-separated list of unique command names
        - Logic: Keep highest-scoring variant when duplicates found
    
    Workflow Position:
        Runs AFTER extract_callable_commands (which processes sections in parallel)
        and BEFORE save_spec_unique_callable_commands (which saves results)
    """
    specFullPath = state["spec_full_path"]
    specFullName = specs_desc_dict[specFullPath]["spec_full_name"]
    systemMessage = f"""You are an SSDs testing validation architect expert for the {specFullName} specification creating a good coverage for module designs

From the candidate modules report, produce a consolidated list of unique callable commands needed to validate {specFullName}.

To describe each command, follow these guidelines:
Command Name: The root name of each possible callable command found.
Key Phrases: List segments of the document content that clearly explain why the command is a callable command. Use this information to help identify whether similarly named commands are duplicates. List the commands and their key phrases using the following format: ‘:::’ to separate each command, and ‘::’ to separate key phrases for each command. For example:
command1: keyphrase1::keyphrase2::keyphrase3:::command2: keyphrase1::keyphrase2::keyphrase3:::command3: keyphrase1::keyphrase2::keyphrase3
Score: List the commands along with a score reflecting how much information in the analyzed section relates to each command. If you identify duplicate commands, keep only the name of the one with the highest score.

Steps:
1. Identify candidate callable commands.
2. Normalize names (remove numbering/punctuation; collapse whitespace; use CamelCase for multi-word names).
3. Collect minimal verbatim key phrases proving callability (evidence substrings only).
4. Merge duplicates (variants of same command); keep the variant with highest evidence.
5. Score each kept command (1–100) relative to the command with the richest detail (that one = 100; scale others). If only one command, score 100.

Rules:
- Use '::' between key phrases for a command.
- Use ':::' between commands.
- No trailing separators.
- Key phrases must be concise and decisive (no full paragraphs).
- Do not invent commands absent from input.
- If no callable commands found output: NONE

Final Output: List of all unique commands split by a comma.
"""

    message = """
Candidature modules report:
{context}
"""
    prompt = ChatPromptTemplate.from_messages([("system", systemMessage),
                                               ("human", message)])

    class ModuleStruct(BaseModel):
        unique_commands: str = Field(description="Names of the unique commands, list of all commands split by a comma, i.e. 'Command 1, Command 2, Command NNN'")

    context = state["callable_commands_info"]
    llm_with_tool = llm_t.with_structured_output(ModuleStruct)

    rag_chain = {
        "context": RunnablePassthrough(),
    } | prompt | llm_with_tool

    rag_chain_w_retry = rag_chain.with_retry(
        wait_exponential_jitter=True,
        stop_after_attempt=4,
        exponential_jitter_params={"initial": 2}
    )
    time.sleep(25)
    response = invoke_with_rate_limit_retry(rag_chain_w_retry, context)
    print(f'{specFullPath}: {response.unique_commands}')

    return {"complete_unique_callable_commands": [{specFullPath: response.unique_commands}]}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def assign_workers_prune_commands_list(state: ModuleContentState):
    """
    Worker assignment function: Dispatch workers to consolidate and deduplicate commands per specification.
    
    Purpose:
        Groups command extraction results by specification and dispatches one worker per spec
        to consolidate and deduplicate commands. This happens after all sections have been
        processed, so each spec's commands from all sections are aggregated before pruning.
    
    How It Works:
        1. Gets aggregated command extraction results from complete_callable_commands
        2. Groups results by specification path (commands from multiple sections per spec)
        3. Formats grouped commands into context string for LLM processing
        4. Creates one Send() operation per specification to dispatch prune_command_list worker
        5. Returns list of Send operations for LangGraph to execute in parallel
    
    Args:
        state (ModuleContentState): Workflow state containing:
            - missing_spec_paths_to_process: List of spec paths needing command pruning
            - complete_callable_commands: Accumulated command extraction results from all sections
                Format: [{"spec_path": ModuleStruct(...)}, ...] (one per section)
    
    Returns:
        List[Send]: List of Send operations, each dispatching prune_command_list worker
            Format: [Send("prune_command_list", {"spec_full_path": "...", "callable_commands_info": "..."}), ...]
    
    Used by:
        - Workflow conditional edge from exec_prune_command_list_workers node
        - LangGraph executes all Send operations in parallel
        - Each Send dispatches prune_command_list worker with spec-specific aggregated commands
    
    Integration:
        - Reads from: state["complete_callable_commands"] (accumulated from extract_callable_commands workers)
        - Reads from: state["missing_spec_paths_to_process"] (set by dataCollectionUniqueCommands)
        - Dispatches to: prune_command_list worker function
        - Workers write to: state["complete_unique_callable_commands"] (accumulated by LangGraph)
    
    Data Grouping:
        Since extract_callable_commands processes sections individually, results are scattered.
        This function groups them by specification so prune_command_list can process all
        commands for a spec together, enabling proper deduplication across sections.
    
    Context Formatting:
        Converts structured ModuleStruct objects into formatted text context:
        - command_names: command1, command2, ...
        - keyPhrases: command1: phrase1::phrase2:::command2: phrase1::phrase2
        - score: command1-score1, command2-score2, ...
        This format allows LLM to identify duplicates and merge variants.
    
    Workflow Position:
        Runs AFTER extract_callable_commands (which processes sections in parallel)
        and routes to prune_command_list workers (which deduplicate commands per spec)
    """
    specMissingPaths = state["missing_spec_paths_to_process"]
    specsCallableCmds = state['complete_callable_commands']
    print("Asigning workers to prune commands list")
    final_output = []
    callableCmdsInfoDict = {}
    # Group previous output to process complete commands found in the same spec
    # Purpose: Aggregate commands from multiple sections into per-specification groups
    for specCallableCmdsDict in specsCallableCmds:
        specPath, specCallableCmdsObj = next(iter(specCallableCmdsDict.items()))
        specCallableCmdsDict = specCallableCmdsObj.model_dump()
        if specPath in callableCmdsInfoDict:
            callableCmdsInfoDict[specPath].append(specCallableCmdsDict)
        else:
            callableCmdsInfoDict.update({specPath: [specCallableCmdsDict]})

    # Formatize the context that will be processed by the llm
    # Purpose: Convert structured command data into formatted text for LLM analysis
    for specPath, callableCommandsInfoList in callableCmdsInfoDict.items():
        context = ""
        for callableCommandsInfo in callableCommandsInfoList:
            for key, content in callableCommandsInfo.items():
                context += f"{key}: {content}\n"

        final_output.append(Send(
            "prune_command_list",
            {"spec_full_path": specPath,
             "callable_commands_info": context}))
        time.sleep(25)  # Rate limiting between specifications
    return final_output

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Save Unique Callable Commands

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def save_spec_unique_callable_commands(state: ModuleContentState):
    """
    Save function: Persist unique callable commands to checkpoint and update state.
    
    Purpose:
        Saves the consolidated unique command lists to the dataCollectionAI checkpoint folder.
        This allows future runs to skip command extraction if results already exist.
        Also updates the workflow state with the saved commands.
    
    How It Works:
        1. Gets unique command lists from complete_unique_callable_commands (accumulated from workers)
        2. Updates specs_callable_unique_cmds state dictionary
        3. For each specification, converts comma-separated command list to numbered dictionary
        4. Saves to checkpoint: {specPath}/callable_commands.json
        5. Returns updated state with commands and cleared missing paths
    
    Args:
        state (ModuleContentState): Workflow state containing:
            - complete_unique_callable_commands: Accumulated unique command lists from prune workers
                Format: [{"spec_path": "command1, command2, command3"}, ...]
            - specs_callable_unique_cmds: Dictionary to update with saved commands
    
    Returns:
        dict: Updated state with:
            - specs_callable_unique_cmds: Updated with all saved commands
            - missing_spec_paths_to_process: Cleared (all commands processed)
            - save_dataCollection: Set to False (checkpoint saved)
    
    Used by:
        - Workflow edge after prune_command_list workers complete
        - Called once after all command pruning is done
        - Updates state before proceeding to next workflow stage
    
    Integration:
        - Reads from: state["complete_unique_callable_commands"] (from prune_command_list workers)
        - Updates: state["specs_callable_unique_cmds"] (used by next stage)
        - Writes to: dataCollectionAI checkpoint folder (for future runs)
        - Format: {"spec_path": {"0": "command1", "1": "command2", ...}}
    
    Checkpoint Format:
        Saves commands as numbered dictionary for easy lookup:
        {"spec_path": {"0": "command1", "1": "command2", "2": "command3", ...}}
        This format is loaded by dataCollectionUniqueCommands in future runs.
    
    Workflow Position:
        Runs AFTER prune_command_list workers complete
        and BEFORE dataCollectionCommandsInfo (which checks for command relevance assessments)
    """
    uniqueCallableCommandsList = state['complete_unique_callable_commands']
    specsCallableUniqueCmds = state['specs_callable_unique_cmds']

    # Process each specification's unique command list
    for uniqueCallableCommands in uniqueCallableCommandsList:
        # Update state dictionary with commands
        specsCallableUniqueCmds.update(uniqueCallableCommands)
        specPath, specUniqueCallableCmds = next(iter(uniqueCallableCommands.items()))
        # Convert comma-separated list to numbered dictionary for checkpoint storage
        uniqueCallableCommandsDict = {str(command_id): command.strip() for command_id, command in enumerate(specUniqueCallableCmds.split(','))}
        specUniqueCallableCommandsDict = {specPath: uniqueCallableCommandsDict}
        # Save to checkpoint folder for future runs
        output_path = f"{specPath}/callable_commands.json"
        json_bytes = json.dumps(specUniqueCallableCommandsDict, indent=2).encode("utf-8")
        with dataCollectionAI.get_writer(output_path) as w_binary:
            w_binary.write(json_bytes)

    print(f'specs Callable: {specsCallableUniqueCmds}')

    return {'specs_callable_unique_cmds': specsCallableUniqueCmds,
            'missing_spec_paths_to_process': [],
            'save_dataCollection': False}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Retrieve Information For Each Unique Command Found

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def dataCollectionCommandsInfo(state: ModuleContentState):
    """
    Check if Commands are already in DataCollection

    Args:
        state (messages): The current state

    Returns:
        str: Load Information if exist or keep the proccess for AI Agent
    """
    print("--- Loading Unique Commands Relevent Info Modules Info in Data Collection Files---")
    print(f"[DEBUG] dataCollectionCommandsInfo: Function called")
    uniqueSpecRootNames = specsBaseStores_df['specRootPath'].unique().tolist()
    print(f"[DEBUG] dataCollectionCommandsInfo: Searching for {len(uniqueSpecRootNames)} spec root paths: {uniqueSpecRootNames}")
    
    all_data_paths = list(dataCollectionAI.list_paths_in_partition())
    print(f"[DEBUG] dataCollectionCommandsInfo: Total paths in dataCollectionAI partition: {len(all_data_paths)}")
    
    specCmdsCompleteInfoDataCollectionPaths = []
    missingSpecCmdsCompleteInfoPaths = []
    for rootPathName in uniqueSpecRootNames:
        pathFoundInCollection = False
        print(f"[DEBUG] dataCollectionCommandsInfo: Searching for rootPathName = {rootPathName}")
        for dataPath in all_data_paths:
            if rootPathName in dataPath and 'relevant_modules_info.json' in dataPath:
                print(f"---Data Collection for {rootPathName} Found: {dataPath} ---")
                specCmdsCompleteInfoDataCollectionPaths.append(dataPath)
                pathFoundInCollection = True
                break
        if not pathFoundInCollection:
            print(f"[DEBUG] dataCollectionCommandsInfo: NOT FOUND for {rootPathName}")
            missingSpecCmdsCompleteInfoPaths.append(rootPathName)

    print(f"[DEBUG] dataCollectionCommandsInfo: Found {len(specCmdsCompleteInfoDataCollectionPaths)} files, missing {len(missingSpecCmdsCompleteInfoPaths)} paths")

    specCmdsCompleteInfoDict = {}
    for dataCollectionPath in specCmdsCompleteInfoDataCollectionPaths:
        print(f'------ Loading Unique Callable Commands Collection {dataCollectionPath} ---------')
        try:
            with dataCollectionAI.get_download_stream(dataCollectionPath) as f:
                file_content = f.read().decode("utf-8")
                print(f"[DEBUG] dataCollectionCommandsInfo: File size = {len(file_content)} bytes")
                specCmdsCompleteInfoInput = json.loads(file_content)
                print(f"[DEBUG] dataCollectionCommandsInfo: Loaded JSON keys = {list(specCmdsCompleteInfoInput.keys()) if isinstance(specCmdsCompleteInfoInput, dict) else 'N/A'}")
                for specFullPath, specCmdsCompleteInfo in specCmdsCompleteInfoInput.items():
                    print(f"[DEBUG] dataCollectionCommandsInfo: Processing specFullPath = {specFullPath}, commands = {len(specCmdsCompleteInfo) if isinstance(specCmdsCompleteInfo, dict) else 'N/A'}")
                    specCmdsCompleteInfoDict.update({specFullPath: {}})
                    relevant_count = 0
                    skipped_count = 0
                    for cmdName, cmdInfoDict in specCmdsCompleteInfo.items():
                        is_relevant = cmdInfoDict.get('is_relevant', True)
                        if is_relevant:
                            specCmdsCompleteInfoDict[specFullPath].update({cmdName: cmdInfoDict})
                            relevant_count += 1
                        else:
                            print(f"---Skipped Tagged Irrelevant Info Module {specFullPath}: {cmdName} ---")
                            skipped_count += 1
                    print(f"[DEBUG] dataCollectionCommandsInfo: specFullPath {specFullPath}: {relevant_count} relevant, {skipped_count} skipped")
        except Exception as e:
            print(f"[DEBUG] dataCollectionCommandsInfo: ERROR loading {dataCollectionPath}: {str(e)}")
            import traceback
            print(f"[DEBUG] dataCollectionCommandsInfo: Traceback: {traceback.format_exc()}")
    
    print(f"[DEBUG] dataCollectionCommandsInfo: Final specCmdsCompleteInfoDict has {len(specCmdsCompleteInfoDict)} spec paths: {list(specCmdsCompleteInfoDict.keys())}")
    total_commands = sum(len(cmds) for cmds in specCmdsCompleteInfoDict.values())
    print(f"[DEBUG] dataCollectionCommandsInfo: Total commands loaded = {total_commands}")
    print(f"[DEBUG] dataCollectionCommandsInfo: save_dataCollection = {bool(missingSpecCmdsCompleteInfoPaths)}")
                        
    return {'specs_cmd_modules_complete_info': specCmdsCompleteInfoDict,
            'missing_spec_paths_to_process': missingSpecCmdsCompleteInfoPaths,
            'save_dataCollection': bool(missingSpecCmdsCompleteInfoPaths)}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
class WorkerCmdState(TypedDict):
    module_cmd_name: str
    spec_full_path: str
    completed_module_cmds: Annotated[list, operator.add]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def retrieve_modules_cmd_info(state: WorkerCmdState):
    """
    Worker function: Assess command relevance and sufficiency for RCV module creation.
    
    Purpose:
        Evaluates whether each unique command has enough specification detail to create
        a meaningful RCV validation module. This is a quality filter that discards
        commands that are insufficiently documented or not relevant to the specification.
    
    How It Works:
        1. Sets CURRENT_SPEC_FULL_PATH global to filter Cortex Search results
        2. Uses RAG chain with get_retriever_command_info() to retrieve command context
        3. LLM analyzes command context against strict relevance criteria:
           - Primary command status (defined in spec, not just referenced)
           - Parameter specification sufficiency (enough detail for test generation)
           - Validation rule existence (testable requirements present)
           - Specification alignment (directly supports validation intent)
        4. Returns structured assessment with is_relevant flag and detailed reasoning
        5. Commands marked is_relevant=False are discarded from further processing
    
    Args:
        state (WorkerCmdState): Worker state containing:
            - spec_full_path: Specification path being processed
            - module_cmd_name: Command name to assess
    
    Returns:
        dict: Updates state["completed_module_cmds"] with:
            Format: [{"spec_path": {"command_name": CommandRelevanceAssessment(...)}}]
            The CommandRelevanceAssessment includes is_relevant boolean and detailed fields
    
    Used by:
        - assign_workers_unique_cmds() dispatches this worker via Send()
        - Called in parallel for each unique command across all specifications
        - Only commands with is_relevant=True proceed to module definition stage
    
    Integration:
        - Sets: CURRENT_SPEC_FULL_PATH global (used by get_retriever_command_info)
        - Uses: get_retriever_command_info() to retrieve command context via Cortex Search
        - Uses: llm_t (DKUChatLLM) with structured output (CommandRelevanceAssessment)
        - Writes to: state["completed_module_cmds"] (accumulated by LangGraph)
        - Reads from: specs_desc_dict for spec metadata
        - Reads from: state["specs_callable_unique_cmds"] (via assign_workers function)
    
    Assessment Criteria (ALL must pass for is_relevant=True):
        1. Primary Command: Explicitly defined in spec, not just referenced externally
        2. Parameter Sufficiency: At least one documented parameter with types/ranges
        3. Validation Rules: Testable requirements (SHALL/MUST) present
        4. Specification Alignment: Directly supports spec's validation intent
    
    Discard Reasons (if is_relevant=False):
        - insufficient_parameter_specification: No parameters documented
        - no_validation_rules: No testable requirements found
        - external_specification_reference: Defined in different spec
        - informational_only: Reserved/future use, not currently testable
        - weak_specification_alignment: Tangential to core validation focus
        - duplicate_or_alias: Synonym of another command
        - missing_critical_details: Only name mentioned, no implementation details
    
    Workflow Position:
        Runs AFTER prune_command_list (which produces unique commands)
        and BEFORE define_cmd_modules (which only processes relevant commands)
    
    Quality Impact:
        This is a critical quality gate. Commands that pass this assessment proceed
        to module definition. Commands that fail are logged but excluded from
        further processing, ensuring only well-documented commands become modules.
    """
    specFullPath = state["spec_full_path"]
    specFullName = specs_desc_dict[specFullPath]["spec_full_name"]
    specFullDesc = specs_desc_dict[specFullPath]["spec_full_description"]

    systemMessage = f"""You are an SSD validation architect evaluating whether a command has sufficient specification detail to warrant RCV module creation for {specFullName}.

**Context: RCV Module Requirements**
RCV modules validate testable features through randomized parameter execution. A command qualifies for module creation ONLY if specification content provides:
1. **Parameter Specifications**: Defined fields, values, ranges, formats, or configurations
2. **Validation Rules**: Testable behaviors, state transitions, constraints, or requirements
3. **Specification Alignment**: Direct relevance to {specFullName}'s core validation intent

**Your Mission**
Analyze the command context to determine:
- Is this a PRIMARY command for {specFullName} validation?
- Does specification content provide SUFFICIENT detail for module design?
- Can RCV randomization meaningfully exercise this command's features?

**Command Relevance Criteria (ALL must be satisfied):**

1. **Primary Command Status:**
   - Command is explicitly defined within {specFullName} scope
   - Command directly implements specification-defined functionality
   - Command is NOT merely referenced from external specifications
   - Command is NOT informational/descriptive only

2. **Parameter Specification Sufficiency:**
   - At least one documented parameter, field, operand, or configuration option
   - Parameter types, ranges, valid values, or formats are specified
   - Parameter combinations or constraints are described
   - Sufficient detail exists to generate randomized test inputs

3. **Validation Rule Existence:**
   - Normative requirements (SHALL/MUST/SHOULD/MAY) are present
   - Testable behaviors, outcomes, or state transitions are defined
   - Success/error conditions are specified
   - Compliance criteria can be verified through testing

4. **Specification Core Alignment:**
   - Command directly validates {specFullName}'s stated purpose of the given specification description.
   - Command exercises features central to specification compliance
   - Command impact is observable and verifiable

**Discard Command If ANY Apply:**

**Insufficient Information:**
- No parameter specifications (only command name mentioned)
- No validation rules or testable requirements
- Only high-level descriptions without implementation details
- Missing critical fields: opcodes, formats, behaviors, constraints
- Purely informational (e.g., "reserved for future use")

**External/Referenced Commands:**
- Command defined in different specification (only referenced here)
- Command belongs to external protocol/interface layer
- Command is vendor-specific extension outside {specFullName} scope

**Non-Primary Commands:**
- Duplicate of command with different name (alias/variant)
- Subcommand better covered by parent command module
- Helper/utility function without independent validation value

**Weak Specification Alignment:**
- Tangential to {specFullName}'s core validation focus
- Ancillary feature with minimal compliance impact
- Legacy/deprecated functionality

**Decision Framework:**

**ACCEPT Examples (Sufficient for RCV Module):**
Command: "Write Data"
- Parameters: LBA, transfer length, data buffer, flags
- Rules: SHALL write data atomically, MUST validate LBA range
- Alignment: Core data integrity validation for storage specification → ACCEPT: Complete parameter spec + validation rules + core alignment
Command: "Get Log Page"
- Parameters: Log Page ID (LID), offset, length
- Rules: SHALL return log data per LID, MUST handle invalid LID
- Alignment: Essential for health/error monitoring validation → ACCEPT: Testable parameters + normative requirements + specification purpose

**DISCARD Examples:**
Command: "Authenticate User"
- Context: "See Specification section 5.2 for details"
- Issue: Defined in external spec, only referenced here → DISCARD: external_specification_reference
Command: "Initialize Session"
- Context: "This command initializes communication"
- Issue: No parameters, no requirements, no format specified → DISCARD: insufficient_parameter_specification
Command: "Reserved Command 0xFF"
- Context: "Reserved for future use"
- Issue: Not currently testable, no validation criteria → DISCARD: informational_only
Command: "Update Metadata"
- Context: NVMe spec focuses on data path; metadata is ancillary
- Issue: Weak alignment with core specification validation intent → DISCARD: weak_specification_alignment

**Required Analysis Tasks:**
1. Extract command name from context
2. Identify parameter specifications (fields, ranges, formats, constraints)
3. Locate normative sentences (SHALL/MUST/SHOULD/MAY)
4. Assess alignment with {specFullName} validation purpose
5. Determine if RCV can meaningfully randomize parameters
6. Make binary decision: ACCEPT (sufficient) or DISCARD (insufficient)
7. If DISCARD, provide specific reason from predefined categories

**Output Schema: CommandRelevanceAssessment**

**Fields:**
- `command_name`: Exact command identifier from context (string)
- `is_relevant`: Boolean - true if command qualifies for RCV module; false if should be discarded
- `primary_command_status`: Is this a primary command defined in {specFullName}? (boolean)
- `command`: Canonical command name or 'none' (string)
- `command_name_variants`: Observed naming variants; empty if none (list of strings)
- `scope`: Single paragraph (<=250 words) describing validation focus (string)
- `parameter_sufficiency`: Are parameters adequately specified for test generation? (boolean)
- `validation_rules_present`: Do testable requirements exist? (boolean)
- `specification_alignment`: Does command directly support {specFullName} validation intent? (boolean)
- `discard_reason`: If is_relevant=false, select ONE from: 'insufficient_parameter_specification', 'no_validation_rules', 'external_specification_reference', 'informational_only', 'weak_specification_alignment', 'duplicate_or_alias', 'missing_critical_details' (string or null)
- `discard_explanation`: If is_relevant=false, detailed explanation citing missing elements (string or null)
- `key_parameters_found`: If is_relevant=true, list critical parameters enabling RCV randomization (list of strings)
- `normative_requirements_summary`: If is_relevant=true, brief summary of testable SHALL/MUST requirements (string or null)
- `module_feasibility_notes`: If is_relevant=true, notes on how RCV can exercise this command (string or null)

**Quality Guidelines:**
- Be strict: When in doubt about sufficiency, DISCARD
- Prioritize quality over quantity: Better to skip weak commands than create incomplete modules
- Evidence-based decisions: Cite specific missing elements in discard_explanation
- Focus on RCV feasibility: Can parameters be randomized? Can behaviors be verified?
- Align with specification purpose: Does command validate {specFullDesc}?

**Special Cases:**
- If context mentions "see external specification", mark external_specification_reference
- If only command name appears without details, mark missing_critical_details
- If marked "reserved" or "future use", mark informational_only
- If command is synonym/alias of another, mark duplicate_or_alias

Specification Description: {specFullDesc}
"""

    message = """
Command Name: {command}

Command context:
{context}
"""
    commandName = state["module_cmd_name"]
    print(f"---Assess Command Relevance and Info Sufficiency: {commandName} ---")

    prompt = ChatPromptTemplate.from_messages([("system", systemMessage),
                                               ("human", message)])

    class CommandRelevanceAssessment(BaseModel):
        name: str = Field(description="Exact command identifier")
        is_relevant: bool = Field(description="True if command qualifies for RCV module; false if should be discarded")
        primary_command_status: bool = Field(
            description="Is this a primary command defined in the target specification?")
        command: str = Field(description="Canonical command name or 'none'.")
        command_name_variants: List[str] = Field(description="Observed naming variants; empty if none.")
        scope: str = Field(description="Single paragraph (<=250 words) describing validation focus.")
        parameter_sufficiency: bool = Field(description="Are parameters adequately specified for test generation?")
        validation_rules_present: bool = Field(description="Do testable requirements exist?")
        specification_alignment: bool = Field(
            description="Does command directly support specification validation intent?")
        discard_reason: str | None = Field(
            description="If is_relevant=false: 'insufficient_parameter_specification', 'no_validation_rules', 'external_specification_reference', 'informational_only', 'weak_specification_alignment', 'duplicate_or_alias', 'missing_critical_details'")
        discard_explanation: str | None = Field(
            description="If is_relevant=false, detailed explanation citing missing elements")
        key_parameters_found: List[str] = Field(description="If is_relevant=true, critical parameters for RCV")
        normative_requirements_summary: str | None = Field(
            description="If is_relevant=true, summary of testable requirements")
        module_feasibility_notes: str | None = Field(
            description="If is_relevant=true, notes on RCV exercising approach")

    global CURRENT_SPEC_FULL_PATH
    # This wil set the global variable need it to create the right retriever get_spec_info_section
    CURRENT_SPEC_FULL_PATH = specFullPath

    llm_with_tool = llm_t.with_structured_output(CommandRelevanceAssessment)

    rag_chain = {
                    "command": RunnablePassthrough(),
                    "context": get_retriever_command_info  # retriever_command_info,
                } | prompt | llm_with_tool

    rag_chain_w_retry = rag_chain.with_retry(
        wait_exponential_jitter=True,
        stop_after_attempt=2,
        exponential_jitter_params={"initial": 2}
    )
    response = invoke_with_rate_limit_retry(rag_chain_w_retry, f'{commandName} command')
    time.sleep(25)

    return {"completed_module_cmds": [{specFullPath: {commandName: response.model_dump()}}]}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def assign_workers_unique_cmds(state: ModuleContentState) -> List[Send]:
    """
    Worker assignment function: Dispatch workers to assess command relevance for RCV module creation.
    
    Purpose:
        Assigns one worker per unique command across all missing specifications to assess
        whether each command has sufficient specification detail to warrant RCV module creation.
        This is a quality filter that discards insufficiently documented commands.
    
    How It Works:
        1. Gets list of specifications that need relevance assessment (missing_spec_paths_to_process)
        2. For each specification, gets its unique command list from specs_callable_unique_cmds
        3. Parses comma-separated command list into individual command names
        4. Creates one Send() operation per command to dispatch retrieve_modules_cmd_info worker
        5. Returns list of Send operations for LangGraph to execute in parallel
    
    Args:
        state (ModuleContentState): Workflow state containing:
            - missing_spec_paths_to_process: List of spec paths needing relevance assessment
            - specs_callable_unique_cmds: Dictionary mapping spec paths to comma-separated command lists
                Format: {"spec_path": "command1, command2, command3", ...}
    
    Returns:
        List[Send]: List of Send operations, each dispatching retrieve_modules_cmd_info worker
            Format: [Send("retrieve_modules_cmd_info", {"spec_full_path": "...", "module_cmd_name": "..."}), ...]
    
    Used by:
        - Workflow conditional edge from exec_relevance_workers node
        - LangGraph executes all Send operations in parallel
        - Each Send dispatches retrieve_modules_cmd_info worker with command-specific state
    
    Integration:
        - Reads from: state["missing_spec_paths_to_process"] (set by dataCollectionCommandsInfo)
        - Reads from: state["specs_callable_unique_cmds"] (loaded/generated in previous stage)
        - Dispatches to: retrieve_modules_cmd_info worker function
        - Workers write to: state["completed_module_cmds"] (accumulated by LangGraph)
    
    Quality Filtering:
        Workers assess each command against strict criteria:
        - Primary command status (defined in spec, not just referenced)
        - Parameter specification sufficiency
        - Validation rule existence
        - Specification alignment
        Commands with is_relevant=False are discarded and won't proceed to module definition.
    
    Parallel Execution:
        If a specification has 20 unique commands, this creates 20 Send operations.
        LangGraph executes them concurrently, significantly speeding up relevance assessment.
    
    Workflow Position:
        Runs AFTER dataCollectionCommandsInfo (which checks for existing relevance assessments)
        and routes to retrieve_modules_cmd_info workers (which assess command relevance)
    """
    # Assign a worker to each unique command across all missing specifications
    specMissingPaths = state["missing_spec_paths_to_process"]
    specsCmdModule = state['specs_callable_unique_cmds']

    print(f"----------{specMissingPaths} Relevant Info Missing Paths")
    final_output = []
    for specPath in specMissingPaths:
        specUniqueCmds = specsCmdModule[specPath]
        # Parse comma-separated command list into individual command names
        specUniqueCmdsList = [section.strip() for section in specUniqueCmds.split(", ") if section.strip()]
        print(f"---------- Unique Command List for spec {specPath}: {specUniqueCmdsList}")
        # Create one worker per command for parallel relevance assessment
        for uniqueCmd in specUniqueCmdsList:
            final_output.append(Send(
                "retrieve_modules_cmd_info",
                {"spec_full_path": specPath,
                 "module_cmd_name": uniqueCmd}))
        time.sleep(25)  # Rate limiting between specifications
    return final_output

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Save Most Relevant Information per Command

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def save_spec_most_relavant_info_cmd(state: ModuleContentState):
    """
    Save function: Persist command relevance assessments and filter out irrelevant commands.
    
    Purpose:
        Saves command relevance assessments to checkpoint and updates state with only
        relevant commands (is_relevant=True). Irrelevant commands are logged but excluded
        from further processing. This is a critical quality gate.
    
    How It Works:
        1. Gets relevance assessments from completed_module_cmds (accumulated from workers)
        2. Separates commands into relevant (is_relevant=True) and irrelevant (is_relevant=False)
        3. Updates specs_cmd_modules_complete_info with only relevant commands
        4. Saves ALL assessments (including irrelevant) to checkpoint for reference
        5. Returns updated state with only relevant commands
    
    Args:
        state (ModuleContentState): Workflow state containing:
            - completed_module_cmds: Accumulated relevance assessments from retrieve_modules_cmd_info workers
                Format: [{"spec_path": {"command_name": CommandRelevanceAssessment(...)}}, ...]
            - specs_cmd_modules_complete_info: Dictionary to update with relevant commands only
    
    Returns:
        dict: Updated state with:
            - specs_cmd_modules_complete_info: Updated with only relevant commands (is_relevant=True)
            - missing_spec_paths_to_process: Cleared (all commands processed)
            - save_dataCollection: Set to False (checkpoint saved)
    
    Used by:
        - Workflow edge after retrieve_modules_cmd_info workers complete
        - Called once after all relevance assessments are done
        - Updates state before proceeding to module definition stage
    
    Integration:
        - Reads from: state["completed_module_cmds"] (from retrieve_modules_cmd_info workers)
        - Updates: state["specs_cmd_modules_complete_info"] (used by module definition stage)
        - Writes to: dataCollectionAI checkpoint folder (for future runs)
        - Format: {"spec_path": {"command_name": CommandRelevanceAssessment(...), ...}}
    
    Quality Filtering:
        Only commands with is_relevant=True proceed to module definition.
        Irrelevant commands are:
        - Logged with discard reason
        - Saved to checkpoint for reference
        - Excluded from specs_cmd_modules_complete_info (won't get modules defined)
    
    Checkpoint Content:
        Saves ALL assessments (both relevant and irrelevant) to checkpoint.
        This preserves the full assessment history, including why commands were discarded.
        Future runs can load this to skip relevance assessment if needed.
    
    Workflow Position:
        Runs AFTER retrieve_modules_cmd_info workers complete
        and BEFORE dataCollectionModulesExtraction (which checks for module definitions)
    """
    print(f"---Saving Most Relevant Modules Data per Command---")

    cmdsRelevantInfoList = state['completed_module_cmds']
    specsCmdsCompleteInfoState = state['specs_cmd_modules_complete_info']

    specCmdsRelevantInfoDict = {}  # All commands (relevant + irrelevant) for checkpoint
    specCmdsRelevantSubmodulesDict= {}  # Unused variable (legacy)
    specCmdsRelevantInfoDictLoadable = {}  # Only relevant commands for state update
    
    # Process each command's relevance assessment
    for cmdsRelevantInfo in cmdsRelevantInfoList:
        specPath, moduleCmdInfo = next(iter(cmdsRelevantInfo.items()))
        cmdName, moduleInfo = next(iter(moduleCmdInfo.items()))
        # Add to all commands dict (for checkpoint - includes irrelevant)
        if specPath in specCmdsRelevantInfoDict:
            specCmdsRelevantInfoDict[specPath].update(moduleCmdInfo)
        else:
            specCmdsRelevantInfoDict.update({specPath: moduleCmdInfo})
        # Filter: Only add relevant commands to loadable dict (for state update)
        is_relevant = moduleInfo['is_relevant'] 
        if is_relevant:
            if specPath in specCmdsRelevantInfoDictLoadable:
                specCmdsRelevantInfoDictLoadable[specPath].update(moduleCmdInfo)
            else:
                specCmdsRelevantInfoDictLoadable.update({specPath: moduleCmdInfo})
        else:
            # Log discarded commands for visibility
            print(f"---Discard {cmdName} command for {specPath} ---")
                                            
    # Update state with only relevant commands (irrelevant excluded from further processing)
    specsCmdsCompleteInfoState.update(specCmdsRelevantInfoDictLoadable)

    # Save ALL assessments (including irrelevant) to checkpoint for reference
    for specPath, cmdsRelevantInfoDict in specCmdsRelevantInfoDict.items():
        specCmdsRelInfoDictOutput = {specPath: cmdsRelevantInfoDict}
        output_path = f"{specPath}/relevant_modules_info.json"
        json_bytes = json.dumps(specCmdsRelInfoDictOutput, indent=2).encode("utf-8")
        with dataCollectionAI.get_writer(output_path) as w_binary:
            w_binary.write(json_bytes)

    return {'specs_cmd_modules_complete_info': specsCmdsCompleteInfoState,
            'missing_spec_paths_to_process': [],
            'save_dataCollection': False}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Retrieve Submodules per command

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def dataCollectionModulesExtraction(state: ModuleContentState):
    """
    Check if Modules are already in DataCollection

    Args:
        state (messages): The current state

    Returns:
        str: Load Information if exist or keep the proccess for AI Agent
    """
    print("--- Loading Unique Callable Commands Modules Info in Data Collection Files---")
    print(f"[DEBUG] dataCollectionModulesExtraction: Function called")
    uniqueSpecRootNames = specsBaseStores_df['specRootPath'].unique().tolist()
    print(f"[DEBUG] dataCollectionModulesExtraction: Searching for {len(uniqueSpecRootNames)} spec root paths: {uniqueSpecRootNames}")
    
    all_data_paths = list(dataCollectionAI.list_paths_in_partition())
    print(f"[DEBUG] dataCollectionModulesExtraction: Total paths in dataCollectionAI partition: {len(all_data_paths)}")
    
    specCmdsDefinitionDataCollectionPaths = []
    missingSpecCmdsDefinitionPaths = []
    for rootPathName in uniqueSpecRootNames:
        pathFoundInCollection = False
        print(f"[DEBUG] dataCollectionModulesExtraction: Searching for rootPathName = {rootPathName}")
        for dataPath in all_data_paths:
            if rootPathName in dataPath and 'defined_modules_info.json' in dataPath:
                print(f"---Data Collection for {rootPathName} Found: {dataPath} ---")
                specCmdsDefinitionDataCollectionPaths.append(dataPath)
                pathFoundInCollection = True
                break
        if not pathFoundInCollection:
            print(f"[DEBUG] dataCollectionModulesExtraction: NOT FOUND for {rootPathName}")
            missingSpecCmdsDefinitionPaths.append(rootPathName)

    print(f"[DEBUG] dataCollectionModulesExtraction: Found {len(specCmdsDefinitionDataCollectionPaths)} files, missing {len(missingSpecCmdsDefinitionPaths)} paths")

    specUniqCmdsInfo = {}
    for dataCollectionPath in specCmdsDefinitionDataCollectionPaths:
        print(f'Loading Unique Callable Commands Modules Info Collection {dataCollectionPath}')
        try:
            with dataCollectionAI.get_download_stream(dataCollectionPath) as f:
                file_content = f.read().decode("utf-8")
                print(f"[DEBUG] dataCollectionModulesExtraction: File size = {len(file_content)} bytes")
                specUniqCmdsInfoInput = json.loads(file_content)
                print(f"[DEBUG] dataCollectionModulesExtraction: Loaded JSON keys = {list(specUniqCmdsInfoInput.keys()) if isinstance(specUniqCmdsInfoInput, dict) else 'N/A'}")
                specUniqCmdsInfo.update(specUniqCmdsInfoInput)
                print(f"[DEBUG] dataCollectionModulesExtraction: Updated specUniqCmdsInfo, now has {len(specUniqCmdsInfo)} keys")
        except Exception as e:
            print(f"[DEBUG] dataCollectionModulesExtraction: ERROR loading {dataCollectionPath}: {str(e)}")
            import traceback
            print(f"[DEBUG] dataCollectionModulesExtraction: Traceback: {traceback.format_exc()}")

    print(f"[DEBUG] dataCollectionModulesExtraction: Final specUniqCmdsInfo has {len(specUniqCmdsInfo)} keys: {list(specUniqCmdsInfo.keys())}")
    print(f"[DEBUG] dataCollectionModulesExtraction: save_dataCollection = {bool(missingSpecCmdsDefinitionPaths)}")

    return {'specs_cmd_modules_definition': specUniqCmdsInfo,
            'missing_spec_paths_to_process': missingSpecCmdsDefinitionPaths,
            'save_dataCollection': bool(missingSpecCmdsDefinitionPaths)}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
class WorkerCmdModulesDefinitionState(TypedDict):
    module_cmd_name: str
    spec_full_path: str
    command_scope: str
    command_alternative_names: list
    completed_cmd_modules_definition: Annotated[list, operator.add]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def define_cmd_modules(state: WorkerCmdModulesDefinitionState):
    """
    Worker function: Define validation modules for a command based on distinct features.
    
    Purpose:
        Analyzes a command and defines the minimal set of validation modules needed to
        comprehensively test the command's distinct features. This is a critical design
        step that determines how commands are broken down into testable modules.
    
    Key Design Principle:
        In RCV, positive and negative test cases for the SAME feature belong to the
        SAME module. RCV's randomized parameter generation naturally covers both valid
        and invalid scenarios through concurrent execution. Modules are split only when
        they validate FUNDAMENTALLY DIFFERENT features, not different test outcomes.
    
    How It Works:
        1. Sets CURRENT_SPEC_FULL_PATH global to filter Cortex Search results
        2. Uses RAG chain with get_retriever_command_info() to retrieve command context
        3. LLM analyzes command to identify distinct features requiring validation
        4. For each distinct feature, defines ONE module covering both positive/negative cases
        5. Returns ProposedModules with module definitions, normative sentences, and rationale
    
    Args:
        state (WorkerCmdModulesDefinitionState): Worker state containing:
            - spec_full_path: Specification path being processed
            - module_cmd_name: Command name to define modules for
            - command_scope: Command's validation scope (from relevance assessment)
            - command_alternative_names: List of command name variants
    
    Returns:
        dict: Updates state["completed_cmd_modules_definition"] with:
            Format: [{"spec_path": {"command_name": ProposedModules(...)}}]
            ProposedModules contains list of ModuleDefinition objects
    
    Used by:
        - assign_workers_cmd_modules_definition() dispatches this worker via Send()
        - Called in parallel for each relevant command across all specifications
        - Only commands that passed relevance assessment (is_relevant=True) reach this stage
    
    Integration:
        - Sets: CURRENT_SPEC_FULL_PATH global (used by get_retriever_command_info)
        - Uses: get_retriever_command_info() to retrieve command context via Cortex Search
        - Uses: llm_t (DKUChatLLM) with structured output (ProposedModules)
        - Writes to: state["completed_cmd_modules_definition"] (accumulated by LangGraph)
        - Reads from: specs_desc_dict for spec metadata
        - Reads from: state["specs_cmd_modules_complete_info"] for command scope and variants
    
    Module Definition Criteria:
        Create NEW module ONLY when command configuration enables:
        - Different operational modes (immediate vs deferred execution)
        - Different security boundaries (admin vs locking security provider)
        - Different data structure families (different table/log types)
        - Different state transitions (power states, session states)
        - Mutually exclusive capabilities requiring different hardware support
    
    DO NOT Split for:
        - Positive vs negative test cases (same feature, different outcomes)
        - Valid vs invalid parameter values (same feature, different inputs)
        - Success vs error paths (same feature, different results)
        - Different scalar values of same parameter type
        - Range boundaries (min/max/mid values)
        - Different error codes from same feature
    
    Output Structure:
        Each ModuleDefinition includes:
        - name: Unique, feature-descriptive module name
        - feature_enabled: Specific feature/capability being validated
        - command_configuration: Representative configuration pattern
        - alt_configurations_same_intent: Other configs (including error cases) for same feature
        - scope: Validation focus description (≤250 words)
        - rationale_boundary: Why this is a distinct feature
        - interaction_elements: Tables, logs, memory regions affected
        - parameter_dimensions: Key parameters driving variation
        - normative_sentences: Parsed SHALL/MUST/SHOULD/MAY requirements
        - missing_information: Gaps preventing complete coverage
    
    Workflow Position:
        Runs AFTER retrieve_modules_cmd_info (which filters relevant commands)
        and BEFORE built_command_params_modules (which extracts parameters for modules)
    
    Quality Impact:
        This function determines the granularity of test modules. Too many modules
        creates unnecessary complexity; too few modules misses distinct features.
        The LLM is instructed to create the MINIMAL set while maintaining complete
        feature coverage.
    """
    specFullPath = state["spec_full_path"]
    specFullName = specs_desc_dict[specFullPath]["spec_full_name"]
    systemMessage = f"""You are an SSD testing and validation architect specializing in module design coverage for the {specFullName} specification.

**Context: RCV Validation Framework**
RCV is a validation framework for enterprise solid-state drives (SSDs) that tests concurrent loads by executing randomized commands with varied input parameters. Commands are issued at random times with optional predefined sequences, enabling comprehensive scenario coverage through randomness and expanding the combinations of command interactions tested. RCV comprises two main components: modules and workloads.

**Critical RCV Design Principle:**
In RCV, positive and negative test cases for the SAME feature belong to the SAME module. RCV's randomized parameter generation naturally covers both valid and invalid scenarios through concurrent execution. DO NOT create separate modules for positive vs. negative cases, valid vs. error conditions, or success vs. failure paths of the same feature.

**Your Mission**
Given a command name and its context, define the MINIMAL set of modules essential for validating the command's DISTINCT features according to the {specFullName} specification.

**Module Definition Philosophy**
A module in RCV represents a DISTINCT FEATURE or FUNCTIONAL CAPABILITY, not a test case. Each module:
- Validates ONE specific feature, operational mode, security boundary, data structure family, or drive state transition
- Handles BOTH positive and negative scenarios for that feature through randomized parameters
- Executes concurrently with other modules to uncover interaction effects

**Critical Module Identification Criteria**
Create a NEW module ONLY when the command configuration enables a FUNDAMENTALLY DIFFERENT feature or functional behavior:

1. **Feature-Based Splitting (CORRECT Reasons to Split):**
   - Different operational modes (e.g., immediate vs. deferred execution)
   - Different security boundaries (e.g., admin security provider vs. locking security provider)
   - Different data structure families (e.g., different table types, different log types)
   - Different state transitions (e.g., power states, session states)
   - Mutually exclusive capabilities requiring different hardware/firmware support

2. **DO NOT Split for These Reasons (INCORRECT):**
   - Positive vs. negative test cases (same feature, different outcomes)
   - Valid vs. invalid parameter values (same feature, different inputs)
   - Success vs. error paths (same feature, different results)
   - Different scalar values of the same parameter type (e.g., LID 0x01 vs 0x02 if validating same mechanism)
   - Range boundaries (min/max/mid values of same parameter)
   - Different error codes from the same feature
   - Different return statuses from the same operation

**Decision Framework Examples:**

**CORRECT - Single Module (Positive + Negative Together):**
- Module: "Basic Read Operation"
- Feature: Block-level data retrieval
- Parameter Dimensions: LBA range, transfer length
- Covers: Valid LBAs, invalid LBAs, out-of-range, overlapping, concurrent reads
- Rationale: All variations test the SAME read feature; RCV randomization handles edge cases

**CORRECT - Multiple Modules (Different Features):**
Module 1: "Error Information Log Retrieval" (LID=0x01)
- Feature: Error tracking and reporting mechanism
- Different validation: Error counters, timestamps, error types
Module 2: "SMART/Health Log Retrieval" (LID=0x02)
- Feature: Device health monitoring mechanism
- Different validation: Temperature, wear level, power cycles

**INCORRECT - Unnecessary Split:**
Incorrect:
Module 1: "Set Authority - Disable Authority"
Module 2: "Set Authority - Enable Authority"
CORRECT:
Module 1: Single "Authority Management" module covering both

**Module Split Decision Criteria (ALL must apply for NEW module):**
1. **Distinct Feature Test:** The configuration enables validation of a materially different feature/capability
2. **Different Validation Logic:** Success criteria, assertions, and verification points differ fundamentally
3. **Different Prerequisites:** Preconditions, dependencies, or drive states differ substantially
4. **Different Interaction Scope:** Affects different tables, logs, memory regions, or state machines
5. **RCV Concurrency Benefit:** Running concurrently with other modules reveals unique interaction effects

**Unification Principle:**
Merge configurations when they validate the SAME feature effect, even if:
- Parameter values differ
- Some inputs are valid, others invalid
- Error codes differ
- Return paths vary
- Boundary conditions exist

List alternate configurations under `alt_configurations_same_intent` rather than creating separate modules.

**Required Tasks:**
1. Identify the command name (confirm or infer from context)
2. Determine the MINIMAL set of DISTINCT features requiring validation
3. For each feature (not test case), define ONE module covering both positive and negative scenarios
4. Explicitly justify why modules cannot be merged
5. List discarded splits with detailed reasoning emphasizing RCV principles
6. Provide `module_unification_principle` explaining how you grouped positive/negative cases
7. Provide `duplication_avoidance_checks` documenting decisions against unnecessary splits

**Output Schema: ProposedModules**

**Top-Level Fields:**
- `command_name`: Exact command identifier (string)
- `module_unification_principle`: Rule explaining how positive/negative cases, error conditions, and parameter variations are unified within modules for RCV concurrent testing (string)
- `duplication_avoidance_checks`: Specific heuristics applied to prevent splits based on test case outcomes rather than feature differences (list of strings)
- `modules`: List of ModuleDefinition objects (one per DISTINCT feature)
- `discarded_module_rationales`: Detailed explanations of why potential feature-based splits were rejected, emphasizing RCV's ability to cover variations through randomization (list of strings)

**ModuleDefinition Fields:**
- `name`: Unique, feature-descriptive module name (string)
- `feature_enabled`: The specific feature, capability, or behavioral facet this module validates - NOT "positive test" or "negative test" (string)
- `command_configuration`: Representative configuration pattern enabling the target feature (string)
- `alt_configurations_same_intent`: Other configurations (including error cases, boundary values, invalid inputs) that validate the SAME feature and belong in this module (list of strings)
- `scope`: One paragraph (≤250 words) describing the feature validation focus, explicitly noting that both positive and negative scenarios are covered through RCV randomization (string)
- `rationale_boundary`: Why this is a DISTINCT FEATURE requiring a separate module, not just a different test case outcome (string)
- `interaction_elements`: Tables, logs, memory regions, namespaces, attributes, authority scopes, flags, ranges affected by this FEATURE (list of strings)
- `parameter_dimensions`: Key parameter dimensions driving feature variation (e.g., LID for different log types, authority levels for different security boundaries) - NOT positive/negative distinction (list of strings)
- `normative_sentences`: List of NormativeSentence objects relevant to this feature validation (list)
- `missing_information`: Gaps preventing complete coverage; use 'well_defined' if none (list of strings)

**NormativeSentence Fields:**
- `text`: Exact sentence containing SHALL/SHOULD/MAY/MUST (string)
- `verb`: One of: SHALL, SHOULD, MAY, MUST (string)
- `intent`: Plain language meaning of the requirement/recommendation/permission (string)
- `validation_implication`: How this impacts feature validation design (not individual test cases) (string)

**Special Case:**
If the command is not testable (informational only), return one module with name 'informational_only' and explain in `rationale_boundary`.

**Quality Guidelines:**
- Think in terms of FEATURES to validate, not test cases to execute
- Remember: RCV randomization handles parameter variations, edge cases, and error conditions
- One feature = one module, regardless of how many test scenarios it encompasses
- Justify splits based on fundamentally different validation objectives, not different input/output combinations
- Be precise and evidence-based; avoid speculation
- Prioritize minimal module count while maintaining complete feature coverage

RCV Methodology Reference: {rcvDesc}
"""

    message = """
Command Name: {command_name}
Alternative Command Names: {alt_command_name}
Specification Name: {spec_full_name}

Command Scope:
{command_scope}

Command Context:
{context}
"""
    commandName = state["module_cmd_name"]
    commandAlternativeNamesState = state["command_alternative_names"]
    commandAlternativeNames = ", ".join(commandAlternativeNamesState)
    commandScope = state['command_scope']
    print(f"---Retrive CMD Info For Module {specFullPath}: {commandName} ---")

    prompt = ChatPromptTemplate.from_messages([("system", systemMessage),
                                               ("human", message)])

    class NormativeSentence(BaseModel):
        text: str = Field(description="Exact sentence containing SHALL/SHOULD/MAY/MUST.")
        verb: str = Field(description="One of SHALL, SHOULD, MAY, MUST.")
        intent: str = Field(description="Plain meaning of requirement/recommendation/permission.")
        validation_implication: str = Field(description="How this impacts test design for this command/module.")

    class ModuleDefinition(BaseModel):
        name: str = Field(description="Unique module name.")
        feature_enabled: str = Field(description="Feature, capability, or behavioral facet this module validates.")
        command_configuration: str = Field(description="Minimal positive configuration: parameter/value pattern enabling target behavior.")
        alt_configurations_same_intent: List[str] = Field(description="Other configs that do NOT require a new module because intent stays identical.")
        scope: str = Field(description="One paragraph (<=250 words) describing validation focus and boundaries.")
        rationale_boundary: str = Field(description="Why this is a distinct module (different enabled feature, data structure, semantic effect, or drive state transition).")
        interaction_elements: List[str] = Field(description="Tables, logs, memory regions, namespaces, attributes, authority scopes, flags, ranges affected.")
        parameter_dimensions: List[str] = Field(description="Key parameter dimensions driving variation (e.g., LID, key size, boolean flag, range, selector).")
        normative_sentences: List[NormativeSentence] = Field(description="Parsed normative sentences relevant to this module.")
        missing_information: List[str] = Field(description="Gaps preventing complete coverage; 'well_defined' if none.")

    class ProposedModules(BaseModel):
        command_name: str = Field(description="Exact command identifier extracted or inferred.")
        module_unification_principle: str = Field(description="Rule applied to decide when multiple parameter sets stay in one module.")
        modules: List[ModuleDefinition] = Field(description="List of proposed modules.")

    global CURRENT_SPEC_FULL_PATH
    # This wil set the global variable need it to create the right retriever get_spec_info_section
    CURRENT_SPEC_FULL_PATH = specFullPath

    llm_with_tool = llm_t.with_structured_output(ProposedModules)

    rag_chain = {
        "alt_command_name": RunnablePassthrough(),
        "command_name": lambda _: commandName,
        "command_scope": lambda _: commandScope,
        "context": get_retriever_command_info, #retriever_command_info,
        "spec_full_name": lambda _: specFullName,
    } | prompt | llm_with_tool

    rag_chain_w_retry = rag_chain.with_retry(
        wait_exponential_jitter=True,
        stop_after_attempt=2,
        exponential_jitter_params={"initial": 2}
    )
    
    response = invoke_with_rate_limit_retry(rag_chain_w_retry, f'{commandName} - {commandAlternativeNames}')
    time.sleep(25)

    return {"completed_cmd_modules_definition": [{specFullPath: {commandName: response.model_dump()}}]}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def assign_workers_cmd_modules_definition(state: ModuleContentState) -> List[Send]:
    """
    Worker assignment function: Dispatch workers to define validation modules for relevant commands.
    
    Purpose:
        Assigns one worker per relevant command across all missing specifications to define
        the minimal set of validation modules needed to test each command's distinct features.
        Only commands that passed relevance assessment (is_relevant=True) reach this stage.
    
    How It Works:
        1. Gets list of specifications that need module definition (missing_spec_paths_to_process)
        2. For each specification, gets its relevant commands from specs_cmd_modules_complete_info
        3. For each relevant command, extracts command scope and name variants
        4. Creates one Send() operation per command to dispatch define_cmd_modules worker
        5. Returns list of Send operations for LangGraph to execute in parallel
    
    Args:
        state (ModuleContentState): Workflow state containing:
            - missing_spec_paths_to_process: List of spec paths needing module definition
            - specs_cmd_modules_complete_info: Dictionary of relevant commands with their assessments
                Format: {"spec_path": {"command_name": CommandRelevanceAssessment(...)}, ...}
                Only commands with is_relevant=True are included
    
    Returns:
        List[Send]: List of Send operations, each dispatching define_cmd_modules worker
            Format: [Send("define_cmd_modules", {"spec_full_path": "...", "module_cmd_name": "...", ...}), ...]
    
    Used by:
        - Workflow conditional edge from exec_modules_extraction node
        - LangGraph executes all Send operations in parallel
        - Each Send dispatches define_cmd_modules worker with command-specific state
    
    Integration:
        - Reads from: state["missing_spec_paths_to_process"] (set by dataCollectionModulesExtraction)
        - Reads from: state["specs_cmd_modules_complete_info"] (from relevance assessment stage)
        - Dispatches to: define_cmd_modules worker function
        - Workers write to: state["completed_cmd_modules_definition"] (accumulated by LangGraph)
    
    Command Filtering:
        Only processes commands where is_relevant=True from retrieve_modules_cmd_info.
        Commands that failed relevance assessment are excluded from module definition.
    
    Module Definition:
        Each worker defines modules based on distinct features, not test cases.
        Positive and negative scenarios for the same feature belong to the same module.
        Modules are split only when they validate fundamentally different features.
    
    Parallel Execution:
        If a specification has 15 relevant commands, this creates 15 Send operations.
        LangGraph executes them concurrently, significantly speeding up module definition.
    
    Workflow Position:
        Runs AFTER dataCollectionModulesExtraction (which checks for existing module definitions)
        and routes to define_cmd_modules workers (which define modules for commands)
    """
    specMissingPaths = state["missing_spec_paths_to_process"]
    specsCmdModule = state['specs_cmd_modules_complete_info']


    print(f"----------{specMissingPaths} Callable Commands Info Missing Paths")
    final_output = []
    for specPath in specMissingPaths:
        specCmdsModulesInfo = specsCmdModule[specPath]
        # Process each relevant command (only is_relevant=True commands are in this dict)
        for cmdName, cmdInfoDict in specCmdsModulesInfo.items():
            cmdNameVariants = cmdInfoDict['command_name_variants']
            cmdScope = cmdInfoDict['scope']
            print(f"---------- Generating {specPath} - {cmdName} - {cmdNameVariants} modules")
            # Create one worker per command for parallel module definition
            final_output.append(Send(
                "define_cmd_modules",
                {"spec_full_path": specPath,
                 "module_cmd_name": cmdName,
                 "command_scope": cmdScope,
                 "command_alternative_names": cmdNameVariants
                }))
        time.sleep(5)  # Rate limiting between specifications (shorter delay for module definition)
    return final_output

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Save Modules definition per Command

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def save_spec_cmd_modules_definition(state: ModuleContentState):
    print(f"---Saving Defined Submodules per Command---")

    uniqueCmdsInfoList = state['completed_cmd_modules_definition']
    specsCmdsInfoState = state['specs_cmd_modules_definition']

    specCmdsInfoDict = {}
    for uniqueCmdsInfo in uniqueCmdsInfoList:
        specPath, moduleCmdInfo = next(iter(uniqueCmdsInfo.items()))
        if specPath in specCmdsInfoDict:
            specCmdsInfoDict[specPath].update(moduleCmdInfo)
        else:
            specCmdsInfoDict.update({specPath: moduleCmdInfo})

    specsCmdsInfoState.update(specCmdsInfoDict)

    for specPath, uniqueCmdsInfoDict in specCmdsInfoDict.items():
        specCmdsInfoDictOutput = {specPath: uniqueCmdsInfoDict}
        output_path = f"{specPath}/defined_modules_info.json"
        json_bytes = json.dumps(specCmdsInfoDictOutput, indent=2).encode("utf-8")
        with dataCollectionAI.get_writer(output_path) as w_binary:
            w_binary.write(json_bytes)

    return {'specs_cmd_modules_definition': specsCmdsInfoState,
            'missing_spec_paths_to_process': [],
            'save_dataCollection': False}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def save_modules_info(state: ModuleContentState):
    global ID_DATE_PATH
    print(f"---Generate Modules Overview ---")
    print(f"[DEBUG] save_modules_info: Function called")

#Summary Table: Create a table that synthesizes all the covered commands and the most relevant information for each command.
#* Command Name: Specify the command to use in order to execute the module.
#* Command Coverage: Obtain all information related to the functionalities or features of the command, and generate a concise explantion of the command.
#* Module Name: Provide a name for the module based on the content of each command.
#* Scope and Validation: Define the general scope of the command based on the provided information.
#* Criteria: Define the validation criteria that need to be considered for testing the command.
#* Conclusion: Provide a general conclusion to emphasize the covered commands and the specification.

#    class ModuleDescription(BaseModel):
        #ModuleName: str = Field(description="Name of the module for this section.")
#        CommandName: str = Field(description="Name of the command.")
#        CommandCoverage: str = Field(description="Functionalities and features of the command.")
        #ScopeNValidation: str = Field(description="Scope and validation coverage of the command.")
        #Criteria: str = Field(description="Validation Criteria Addressed.")

    class SummaryBody(BaseModel):
        Title: str = Field(description=f"Complete name of the specification")
        Coverage: str = Field(description="Commands coverage content.") #List[ModuleDescription] = Field(description="Commands coverage in all content.")
        #SummaryTable: str = Field(description=f"Summary Table: Modules-to-Validation Criteria Mapping")
        Conclusion: str = Field(description=f"Brief conclusion of all content.")


    message = """
    Modules context: \n
    {context}
    """
    completedCmdModulesInfoDictState = state["specs_cmd_modules_complete_info"]
    print(f"[DEBUG] save_modules_info: ID_DATE_PATH = {ID_DATE_PATH}")
    print(f"[DEBUG] save_modules_info: completedCmdModulesInfoDictState type = {type(completedCmdModulesInfoDictState)}, keys = {list(completedCmdModulesInfoDictState.keys()) if isinstance(completedCmdModulesInfoDictState, dict) else 'N/A'}")
    
    if not completedCmdModulesInfoDictState or (isinstance(completedCmdModulesInfoDictState, dict) and len(completedCmdModulesInfoDictState) == 0):
        print(f"[DEBUG] save_modules_info: WARNING - specs_cmd_modules_complete_info is empty, no modules info to save")
        return {}

    print(f"[DEBUG] save_modules_info: Processing {len(completedCmdModulesInfoDictState)} spec paths")

    for specPath, completeCmdModuleInfo in completedCmdModulesInfoDictState.items():
        specFullName = specs_desc_dict[specPath]["spec_full_name"]
        systemMessage = f"""You are an SSD testing validation architect expert for the {specFullName} specification, creating comprehensive coverage for module designs.

Given the module list—which includes both the command and the scope of each module—generate a comprehensive report that details all the modules covered, explicitly outlining what functionalities and features are included.

This is the expected structure of the report body:

Title: Specify the complete specification name and version from which this information is retrieved.
Coverage: Resume the coverage in base of the given commands and their specific coverage for the specification.
Conclusion: Provide a brief conclusion on how these commands help to validate the specification.
"""
        final_content = []
        commands_desc_content = {}
        for command_name, fields in completeCmdModuleInfo.items():
            name_abbr = command_name
            module_name = fields['name']
            #final_content.append(f"## Module: {module_name}\n")
            final_content.append(f"**Command:** > {fields['command']}\n")
            final_content.append(f"**Scope:** > {fields['scope']}\n")
            #dependencies_value = fields['dependencies']
            #if dependencies_value.strip().lower() == "no":
            #    continue
            #else:
            #    final_content.append(f"**Dependencies:**  \n> {dependencies_value}\n")
            commands_desc_content.update({command_name: fields['scope']})

        prompt = ChatPromptTemplate.from_messages([("system", systemMessage),
                                                   ("human", message)])

        finalAllModulesOutputMD = '\n'.join(final_content)
        # print(finalAllModulesOutputMD)

        llm_with_tool = llm_t.with_structured_output(SummaryBody)

        rag_chain = {
            "context": RunnablePassthrough(),
        } | prompt | llm_with_tool

        rag_chain_w_retry = rag_chain.with_retry(
            wait_exponential_jitter=True, # Add jitter to the exponential backoff
            stop_after_attempt=2, # Try twice
            exponential_jitter_params={"initial": 4}  # if desired, customize backoff
        )
        time.sleep(2)
        response = invoke_with_rate_limit_retry(rag_chain_w_retry, finalAllModulesOutputMD)

        summaryBodyResponse = response.model_dump()

        summaryBodyResponse.update({"modules_overview": commands_desc_content})

        finalContent = json.dumps(summaryBodyResponse, indent=2).encode("utf-8")

        output_path = f"{specPath}/{ID_DATE_PATH}/general_info/generalFeatureBreakerOverview.json"
        print(f"[DEBUG] save_modules_info: Writing to path = {output_path}")
        print(f"[DEBUG] save_modules_info: specPath = {specPath}, commands processed = {len(completeCmdModuleInfo)}")
        try:
            print(f"[DEBUG] save_modules_info: JSON size = {len(finalContent)} bytes")
            with MultiAgentOutput.get_writer(output_path) as w:
                w.write(finalContent)
            print(f"[DEBUG] save_modules_info: SUCCESS - File written: {output_path}")
        except Exception as e:
            print(f"[DEBUG] save_modules_info: ERROR - Failed to write {output_path}: {str(e)}")
            import traceback
            print(f"[DEBUG] save_modules_info: Traceback: {traceback.format_exc()}")

    print(f"[DEBUG] save_modules_info: Function complete")
    return {}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Get Most Relevant Parameter For Command Module

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
class WorkerCmdParamsState(TypedDict):
    spec_full_path: str
    module_cmd_name: str
    module_cmd_scope: str
    submodule_name: str
    submodule_feature_enabled: str
    submodule_scope: str
    completed_module_cmds_param: Annotated[list, operator.add]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def built_command_params_modules(state: WorkerCmdParamsState):
    """
    Worker function: Extract input parameters for a command module.
    
    Purpose:
        Analyzes specification content to extract all input parameters relevant to a
        specific command module configuration. Parameters are the inputs that RCV will
        randomize to generate test scenarios. This is a critical step for test generation.
    
    How It Works:
        1. Sets CURRENT_SPEC_FULL_PATH global to filter Cortex Search results
        2. Uses RAG chain with get_retriever_command_info() to retrieve command context
        3. LLM analyzes context to identify input parameters with:
           - Parameter name, description, type, valid ranges
           - Dependencies on other parameters/features
           - Successful input values (for positive testing)
           - Failure-inducing input values (for negative testing)
           - Specification references and object dependencies
        4. Returns CommandParametersDefinition with complete parameter definitions
    
    Args:
        state (WorkerCmdParamsState): Worker state containing:
            - spec_full_path: Specification path being processed
            - module_cmd_name: Command name
            - module_cmd_scope: Command's validation scope
            - submodule_name: Module name (from module definition)
            - submodule_feature_enabled: Feature this module validates
            - submodule_scope: Module's validation scope
    
    Returns:
        dict: Updates state["completed_module_cmds_param"] with:
            Format: [{"spec_path": {"command_name": {"module_name": CommandParametersDefinition(...)}}}]
    
    Used by:
        - assign_workers_cmds_params() dispatches this worker via Send()
        - Called in parallel for each module across all commands and specifications
        - Each command may have multiple modules, each processed separately
    
    Integration:
        - Sets: CURRENT_SPEC_FULL_PATH global (used by get_retriever_command_info)
        - Uses: get_retriever_command_info() to retrieve command context via Cortex Search
        - Uses: llm_t (DKUChatLLM) with structured output (CommandParametersDefinition)
        - Writes to: state["completed_module_cmds_param"] (accumulated by LangGraph)
        - Reads from: specs_desc_dict for spec metadata
        - Reads from: state["specs_cmd_modules_definition"] for module information
        - Reads from: state["specs_cmd_modules_complete_info"] for command scope
    
    Parameter Identification:
        Extracts parameters that:
        - Define what command operates on (target identifiers, resource selectors)
        - Configure operational mode (action types, feature flags, modes)
        - Specify data formats (buffer sizes, data patterns, range limits)
        - Control access/security (authority levels, encryption modes)
        - Define sequencing/timing (order requirements, prerequisite states)
    
    Excludes:
        - Output parameters or response fields (validation targets, not inputs)
        - Internal implementation details not exposed in command interface
        - Informative or example-only content
        - Descriptive text that doesn't define configurable parameters
    
    Output Structure:
        CommandParametersDefinition contains:
        - Name: Module name
        - Description: General information about module coverage
        - CoverageFocus: Input configuration theme
        - Parameters: List of ParameterDefinition objects, each with:
          * paramName: Exact parameter name from specification
          * paramDesc: Complete description (max 150 words)
          * paramType: Data type and valid value range
          * paramDependencyMatrix: Dependencies on other parameters/features
          * paramSuccessfulInputs: Values for positive test scenarios
          * paramFailureInducingInputs: Values for negative test scenarios
          * paramMissingInfo: Questions if incomplete, "well_defined" if complete
          * paramReferences: Specification file paths where parameter is documented
          * paramObjectReference: Referenced tables/figures
    
    Workflow Position:
        Runs AFTER define_cmd_modules (which defines modules)
        and BEFORE enhance_parameter_context (which enhances incomplete parameters)
    
    Quality Impact:
        Complete parameter definitions enable RCV to generate comprehensive test
        scenarios. Incomplete parameters are enhanced in the next stage.
    """
    specFullPath = state["spec_full_path"]
    specFullName = specs_desc_dict[specFullPath]["spec_full_name"]
    systemMessage = f"""You are an SSD testing validation architect expert for the {specFullName} specification, extracting parameter details for a specific module configuration.

Your task is to extract all input parameters relevant to the given module configuration:

Parameter Identification Guidelines:

Extract parameters that:
* Define what the command operates on (target identifiers, resource selectors, scope controls)
* Configure the command's operational mode or behavior (action types, feature flags, operational modes)
* Specify data formats, lengths, or boundaries (buffer sizes, data patterns, range limits)
* Control access, security, or permissions (authority levels, encryption modes, access controls)
* Define sequencing, timing, or state dependencies (order requirements, prerequisite states)

For each identified parameter, provide:
* Parameter Name: The exact parameter name as defined in the specification
* Parameter Description: Complete explanation of the parameter's purpose and how it affects command execution (max 150 words)
* Parameter Type: Data type and valid value range (e.g., uint8, boolean, enum, bit field, byte array)
* Parameter Dependency Matrix: List other parameters or drive features this parameter depends on or interacts with; empty list if independent
* Successful Input Values: Specific values or value ranges that represent valid, positive test scenarios (these should execute successfully)
* Failure-Inducing Input Values: Specific values that represent invalid scenarios for negative testing (reserved values, out-of-range, unsupported combinations)
* Missing Information: Questions needed to fully define validation criteria for this parameter; use "well_defined" if complete
* Parameter References: Full file paths from the specification where this parameter is documented (comma-separated, exact paths)
* Object References: Names of tables, figures, or data structures referenced by this parameter (comma-separated)

Exclusion Criteria:
Do NOT extract:
* Output parameters or response fields (these are validation targets, not inputs)
* Internal implementation details not exposed in the command interface
* Informative or example-only content
* Descriptive text that doesn't define configurable parameters

Validation Context:
* RCV executes commands with randomized parameter combinations to maximize scenario coverage
* Parameters enable testing different drive behaviors, features, and error conditions
* Each parameter variation may trigger different validation rules and expected responses.

RCV Validation Context:
{rcvDesc}
"""

    message = """
Command Module Request: {command_request}

Command Info: {command_scope}

Command Module Scope: {module_scope}

Command context:
{context}
"""
    class ParameterDefinition(BaseModel):
        paramName: str = Field(description="Name of the parameter.")
        paramDesc: str = Field(description="Complete description of the parameter's usage in the command.")
        paramType: str = Field(description="Type of value that can be used for the parameter.")
        paramDependencyMatrix: List[str] = Field(description="Dependencies on other parameters/features; empty if none.")
        paramSuccessfulInputs: str = Field(
            description="Values to be used when testing positive scenarios for this parameter;"
                        "these inputs should be accepted and result in successful command execution.")
        paramFailureInducingInputs: str = Field(
            description="Values to be used when testing negative scenarios for this parameter;"
                        "these inputs are intentionally invalid or unsupported and should trigger command errors or rejection.")
        paramMissingInfo: str = Field(
            description="Questions needed to fully define validation criteria for this parameter, separated by commas (e.g., question1, question2, question3)."
                        "If no additional information is needed, set as 'well_defined'.")
        paramReferences: str = Field(description="List all spec references text file exact path where relevant information about the parameter can be found, separate them with commas (e.g., /path/reference_1.txt, /path/reference_2.txt, /path/reference_NNN.txt).")
        paramObjectReference: str = Field(
            description="Specify the name of any reference dependency related to this parameter, separated by commas (e.g., table1, figureN, tableN).")

    class CommandParametersDefinition(BaseModel):
        Name: str = Field(description=f"Name of the submodule for testing with the command")
        Description: str = Field(description=f"General Information about what is covering the module for command usage.")
        CoverageFocus: str = Field(description="Input configuration theme only.")
        Parameters: List[ParameterDefinition] = Field(description="Full definitions of parameters for this module partition.")

    commandName = state["module_cmd_name"]
    commandScope = state["module_cmd_scope"]
    moduleName = state["submodule_name"]
    moduleFeatureEnabled = state["submodule_feature_enabled"]
    moduleScope = state["submodule_scope"]
        
    command_question = f"Command: {commandName} covering {moduleName} which intention is {moduleFeatureEnabled}"
    

    print(f"---Retrieving Parameters Info {commandName} -> {moduleName} ---")

    prompt = ChatPromptTemplate.from_messages([("system", systemMessage),
                                               ("human", message)])
    global CURRENT_SPEC_FULL_PATH
    # This wil set the global variable need it to create the right retriever get_spec_info_section
    CURRENT_SPEC_FULL_PATH = specFullPath
    
    llm_with_tool = llm_t.with_structured_output(CommandParametersDefinition)

    rag_chain = {
        "command_request": RunnablePassthrough(),
        "command_scope": lambda _: commandScope,
        "module_scope": lambda _: moduleScope,
        "context": get_retriever_command_info#get_retriever_all_spec_info, #retriever_command_info,
    } | prompt | llm_with_tool
    
    rag_chain_w_retry = rag_chain.with_retry(
        wait_exponential_jitter=True, # Add jitter to the exponential backoff
        stop_after_attempt=2, # Try twice
        exponential_jitter_params={"initial": 2},  # if desired, customize backoff
    )
    time.sleep(20)
    response = invoke_with_rate_limit_retry(rag_chain_w_retry, command_question)

    command_param_vals = response.model_dump()
    
    return {'completed_module_cmds_param': [{specFullPath: {commandName: {moduleName: command_param_vals}}}]}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def assign_workers_cmds_params(state: ModuleContentState):
    """
    Worker assignment function: Dispatch workers to extract parameters for each command module.
    
    Purpose:
        Assigns one worker per module across all commands and specifications to extract
        input parameters. Since each command may have multiple modules (one per distinct feature),
        this creates workers for all modules across all commands.
    
    How It Works:
        1. Gets module definitions from specs_cmd_modules_definition (all specs, not just missing)
        2. For each specification, iterates through all commands and their modules
        3. For each module, extracts module metadata (name, scope, feature_enabled)
        4. Creates one Send() operation per module to dispatch built_command_params_modules worker
        5. Returns list of Send operations for LangGraph to execute in parallel
    
    Args:
        state (ModuleContentState): Workflow state containing:
            - specs_cmd_modules_definition: Dictionary of module definitions for all specs
                Format: {"spec_path": {"command_name": ProposedModules(...)}, ...}
            - specs_cmd_modules_complete_info: Dictionary of command scopes
                Format: {"spec_path": {"command_name": CommandRelevanceAssessment(...)}, ...}
    
    Returns:
        List[Send]: List of Send operations, each dispatching built_command_params_modules worker
            Format: [Send("built_command_params_modules", {"spec_full_path": "...", "module_cmd_name": "...", ...}), ...]
    
    Used by:
        - Workflow conditional edge from exec_build_params node
        - LangGraph executes all Send operations in parallel
        - Each Send dispatches built_command_params_modules worker with module-specific state
    
    Integration:
        - Reads from: state["specs_cmd_modules_definition"] (from module definition stage)
        - Reads from: state["specs_cmd_modules_complete_info"] (for command scope)
        - Uses: specsBaseStores_df to get all spec paths (processes all specs, not just missing)
        - Dispatches to: built_command_params_modules worker function
        - Workers write to: state["completed_module_cmds_param"] (accumulated by LangGraph)
    
    Module Iteration:
        Each command can have multiple modules (one per distinct feature).
        This function creates workers for ALL modules across ALL commands and specs,
        not just missing ones, since parameter extraction is always needed for new modules.
    
    Parallel Execution:
        If there are 50 modules across all commands and specs, this creates 50 Send operations.
        LangGraph executes them concurrently, significantly speeding up parameter extraction.
    
    Workflow Position:
        Runs AFTER save_modules_info (which generates module overview)
        and routes to built_command_params_modules workers (which extract parameters)
    """
    print(f"---Assign Workers for Params ---")
    # Get all specification paths (process all specs, not just missing ones)
    uniqueSpecRootPaths = specsBaseStores_df['specRootPath'].unique().tolist()
    specsCompleteCmdModuleInfo = state['specs_cmd_modules_definition']
    specsCompleteCmdCompleteInfo = state['specs_cmd_modules_complete_info']
    
    final_output = []
    # Iterate through all specifications, commands, and modules
    for specPath, moduleInfoDict in specsCompleteCmdModuleInfo.items():
        print(f"---------- Create Params and Module Submodules {specPath} -------")
        for cmdName, moduleRelevantInfoDict in moduleInfoDict.items():
            # Get list of modules defined for this command
            cmdDefinedModules = moduleRelevantInfoDict['modules']
            # Get command scope from complete info
            cmdInfo = specsCompleteCmdCompleteInfo[specPath][cmdName]["scope"]
            # Create one worker per module
            for cmdDefinedModule in cmdDefinedModules:
                submoduleName = cmdDefinedModule['name']
                submoduleScope = cmdDefinedModule['scope']
                submoduleFeatureEnable = cmdDefinedModule['feature_enabled']
                print(f"---------- Processing Params for {cmdName}: {submoduleName} -------")
                final_output.append(Send(
                    "built_command_params_modules",
                    {"spec_full_path": specPath,
                     "module_cmd_name": cmdName,
                     "module_cmd_scope": cmdInfo,
                     "submodule_feature_enabled": submoduleFeatureEnable,
                     "submodule_name": submoduleName,
                     "submodule_scope": submoduleScope}))
        
    return final_output

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Enhance Parameters For Command Module

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
class WorkerEnhaCmdParamsState(TypedDict):
    spec_full_path: str
    module_params_info: str
    completed_module_cmds_enha_param: Annotated[list, operator.add]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def enhance_parameter_context(state: WorkerEnhaCmdParamsState):
    """
    Worker function: Enhance incomplete parameter information with additional context.
    
    Purpose:
        Takes parameter definitions that have missing information (paramMissingInfo != "well_defined")
        and uses additional specification context to fill gaps. This ensures all parameters
        have complete definitions needed for RCV test generation.
    
    How It Works:
        1. Iterates through all parameters in the module
        2. For parameters with paramMissingInfo != "well_defined":
           - Uses RAG chain with get_retriever_all_spec_short_info() to get additional context
           - LLM analyzes context to fill missing fields (description, type, dependencies, test values)
           - Replaces incomplete parameter with enhanced version
        3. For parameters already "well_defined", keeps original
        4. Returns enhanced parameter definitions
    
    Args:
        state (WorkerEnhaCmdParamsState): Worker state containing:
            - spec_full_path: Specification path being processed
            - module_params_info: Dictionary of command modules with their parameters
                Format: {"command_name": {"module_name": CommandParametersDefinition(...)}}
    
    Returns:
        dict: Updates state["completed_module_cmds_enha_param"] with:
            Format: [{"spec_path": {"command_name": {"module_name": {...}}}}]
            Contains enhanced parameter definitions with missing info filled
    
    Used by:
        - assign_workers_cmds_enha_params() dispatches this worker via Send()
        - Called once per specification to enhance all incomplete parameters
        - Processes all commands and modules for that specification together
    
    Integration:
        - Uses: get_retriever_all_spec_short_info() to retrieve additional context (searches all specs)
        - Uses: llm_t (DKUChatLLM) with structured output (ParameterDefinition)
        - Reads from: state["completed_module_cmds_param"] (via assign_workers function)
        - Writes to: state["completed_module_cmds_enha_param"] (accumulated by LangGraph)
        - Reads from: specs_desc_dict for spec metadata
    
    Enhancement Process:
        For each parameter with incomplete info:
        1. Builds prompt with current parameter info and missing info questions
        2. Retrieves additional context via Cortex Search
        3. LLM fills missing fields based on context:
           - Missing Description: Complete explanation of parameter purpose
           - Missing Type: Exact data type, ranges, enumeration values
           - Missing Dependencies: Other parameters/features this depends on
           - Missing Test Values: Successful and failure-inducing input values
        4. Updates paramMissingInfo to "well_defined" if complete, or refines questions
    
    Constraints:
        - Uses ONLY provided specification context (no assumptions)
        - If evidence insufficient, marks field as "insufficient_context"
        - Preserves existing correct information
        - Maintains consistency with RCV validation objectives
    
    Workflow Position:
        Runs AFTER built_command_params_modules (which extracts initial parameters)
        and BEFORE save_modules_param (which saves final parameter definitions)
    
    Quality Impact:
        Ensures all parameters have complete definitions needed for test generation.
        Parameters that remain incomplete after enhancement are marked as such,
        allowing downstream processes to handle them appropriately.
    """
    specFullPath = state["spec_full_path"]
    specFullName = specs_desc_dict[specFullPath]["spec_full_name"]
    systemMessage = f"""You are an SSD testing validation architect expert for the {specFullName} specification, refining parameter definitions to achieve complete RCV validation coverage.

Your task is to enhance incomplete parameter information by:
1. Filling missing or incomplete fields with precise, specification-grounded details
2. Correcting ambiguous or generic information
3. Expanding validation test values (successful and failure-inducing inputs)
4. Identifying additional dependencies and references

Enhancement Guidelines:

For each parameter with incomplete information:

**Missing Description**: If paramDesc is generic or empty, provide a complete explanation (max 150 words) covering:
- Parameter's purpose in command execution
- How it affects drive behavior or validation scope
- Valid value interpretation and constraints

**Missing Type Information**: If paramType lacks specificity, define:
- Exact data type (uint8, uint16, uint32, boolean, enum, bit field, byte array)
- Valid value range or enumeration values
- Bit positions if applicable

**Missing Dependency Matrix**: If paramDependencyMatrix is empty, identify:
- Other parameters this depends on (e.g., "requires paramX to be set first")
- Drive features that must be enabled
- State prerequisites or sequencing requirements

**Missing Test Values**: If paramSuccessfulInputs or paramFailureInducingInputs are incomplete:
- Successful inputs: Specify exact values or representative ranges for positive scenarios (boundary values, typical values, max/min valid)
- Failure inputs: Specify invalid values for negative testing (reserved values, out-of-range, unsupported combinations, conflicting states)

Constraints:
- Use ONLY the provided specification context—no assumptions or external knowledge
- If evidence is insufficient, mark field as "insufficient_context" rather than inventing details
- Preserve existing correct information; only enhance incomplete fields
- Maintain consistency with RCV validation objectives
"""

    message = """
Command Name:
{command}

Parameter Current Info:
{current_info}

Parameter Missing Info:
{missing_info}

Command context:
{context}
"""
    prompt = ChatPromptTemplate.from_messages([("system", systemMessage),
                                               ("human", message)])

    class ParameterDefinition(BaseModel):
        paramName: str = Field(description="Name of the parameter.")
        paramDesc: str = Field(description="Complete description of the parameter's usage in the command.")
        paramType: str = Field(description="Type of value that can be used for the parameter.")
        paramDependencyMatrix: List[str] = Field(description="Dependencies on other parameters/features; empty if none.")
        paramSuccessfulInputs: str = Field(
            description="Values to be used when testing positive scenarios for this parameter;"
                        "these inputs should be accepted and result in successful command execution.")
        paramFailureInducingInputs: str = Field(
            description="Values to be used when testing negative scenarios for this parameter;"
                        "these inputs are intentionally invalid or unsupported and should trigger command errors or rejection.")
        paramMissingInfo: str = Field(
            description="Questions needed to fully define validation criteria for this parameter, separated by commas (e.g., question1, question2, question3)."
                        "If no additional information is needed, set as 'well_defined'.")
        paramReferences: str = Field(description="List all spec references text file exact path where relevant information about the parameter can be found, separate them with commas (e.g., /path/reference_1.txt, /path/reference_2.txt, /path/reference_NNN.txt).")
        paramObjectReference: str = Field(
            description="Specify the name of any reference dependency related to this parameter, separated by commas (e.g., table1, figureN, tableN).")

    llm_with_tool = llm_t.with_structured_output(ParameterDefinition)

    modules_cmd_info = state['module_params_info']

    modules_new_info = deepcopy(modules_cmd_info)
    
    for commandName, module_cmd_info in modules_cmd_info.items():
        moduleName, submodule_cmd_info = next(iter(module_cmd_info.items()))
        enriched_param_details = []
        for param in submodule_cmd_info['Parameters']:
            print(f"---Enhancing {commandName} -> {moduleName} -> {param} Parameter---")
            if param['paramMissingInfo'] == 'well_defined':
                enriched_param_details.append(param)
            else:
                rag_chain = {
                    "command": lambda nameCommand: submodule_cmd_info['Name'],
                    "missing_info": RunnablePassthrough(),
                    "current_info": lambda currInfo: param,
                    "context": get_retriever_all_spec_short_info #get_retriever_all_spec_short_info, #retriever_params_info,
                } | prompt | llm_with_tool

                rag_chain_w_retry = rag_chain.with_retry(
                    wait_exponential_jitter=True,
                    stop_after_attempt=2,
                    exponential_jitter_params={"initial": 2},
                    )
                time.sleep(20)
                response = invoke_with_rate_limit_retry(rag_chain_w_retry, "Missing info for command: " + moduleName + ">" + param['paramMissingInfo'])
                enriched_param_details.append(response.model_dump())
        modules_new_info[commandName][moduleName]['Parameters'] = enriched_param_details

    result = {'completed_module_cmds_enha_param': [{specFullPath: modules_new_info}]}
    print(f"[DEBUG] enhance_parameter_context: Returning data for specPath = {specFullPath}")
    print(f"[DEBUG] enhance_parameter_context: modules_new_info has {len(modules_new_info)} commands")
    for cmd_name in modules_new_info.keys():
        print(f"[DEBUG] enhance_parameter_context: Command {cmd_name} has {len(modules_new_info[cmd_name])} modules")
    return result

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def assign_workers_cmds_enha_params(state: ModuleContentState):
    """Assign a worker to each section in the modules iterable"""
    print(f"---Assign Workers for Enhacement Params ---")
    print(f"[DEBUG] assign_workers_cmds_enha_params: Function called")
    final_output = []
    cmdModulesParam = state["completed_module_cmds_param"]
    print(f"[DEBUG] assign_workers_cmds_enha_params: cmdModulesParam type = {type(cmdModulesParam)}, length = {len(cmdModulesParam) if hasattr(cmdModulesParam, '__len__') else 'N/A'}")
    
    if not cmdModulesParam or len(cmdModulesParam) == 0:
        print(f"[DEBUG] assign_workers_cmds_enha_params: WARNING - completed_module_cmds_param is empty, no workers to assign")
        return final_output
    
    worker_count = 0
    for cmdModuleParamDict in cmdModulesParam:
        for specPath, cmdModuleParamInfoDict in cmdModuleParamDict.items():
            print(f"[DEBUG] assign_workers_cmds_enha_params: Assigning worker for specPath = {specPath}, commands = {len(cmdModuleParamInfoDict) if isinstance(cmdModuleParamInfoDict, dict) else 'N/A'}")
            final_output.append(Send("enhance_parameter_context",
                                     {"spec_full_path": specPath,
                                      "module_params_info": cmdModuleParamInfoDict}))
            worker_count += 1
    
    print(f"[DEBUG] assign_workers_cmds_enha_params: Assigned {worker_count} workers, returning {len(final_output)} Send operations")
    return final_output

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Save Parameters in Json Files

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def save_modules_param(state: ModuleContentState):
    print(f"[DEBUG] save_modules_param: Function called")
    modulesCmdInfoState = state["completed_module_cmds_enha_param"]
    global ID_DATE_PATH
    
    print(f"[DEBUG] save_modules_param: ID_DATE_PATH = {ID_DATE_PATH}")
    print(f"[DEBUG] save_modules_param: modulesCmdInfoState type = {type(modulesCmdInfoState)}, length = {len(modulesCmdInfoState) if hasattr(modulesCmdInfoState, '__len__') else 'N/A'}")
    
    if not modulesCmdInfoState or len(modulesCmdInfoState) == 0:
        print(f"[DEBUG] save_modules_param: WARNING - modulesCmdInfoState is empty, no data to save")
        return {}
    
    groupedCommandsInfo = {}
    print(f"[DEBUG] save_modules_param: Starting grouping loop, processing {len(modulesCmdInfoState)} items")

    for idx, moduleCmdInfoState in enumerate(modulesCmdInfoState):
        print(f"[DEBUG] save_modules_param: Processing item {idx+1}/{len(modulesCmdInfoState)}")
        specPath, modulesCmdInfoDict = next(iter(moduleCmdInfoState.items()))
        print(f"[DEBUG] save_modules_param: specPath = {specPath}, commands count = {len(modulesCmdInfoDict) if isinstance(modulesCmdInfoDict, dict) else 'N/A'}")
        if specPath not in groupedCommandsInfo:
                groupedCommandsInfo[specPath] = {}

        for commandName, moduleParamsInfo in modulesCmdInfoDict.items():
            if commandName in groupedCommandsInfo[specPath]:
                groupedCommandsInfo[specPath][commandName].update(moduleParamsInfo)
            else:
                groupedCommandsInfo[specPath].update({commandName: moduleParamsInfo})

    print(f"[DEBUG] save_modules_param: Grouping complete, groupedCommandsInfo has {len(groupedCommandsInfo)} spec paths")
    total_files_to_write = sum(len(cmds) for cmds in groupedCommandsInfo.values())
    print(f"[DEBUG] save_modules_param: Total files to write = {total_files_to_write}")

    files_written = 0
    for specPath, modulesCmdModulesParamsInfo in groupedCommandsInfo.items():
        print(f"[DEBUG] save_modules_param: Processing specPath = {specPath}, commands = {len(modulesCmdModulesParamsInfo)}")
        for cmdName, modulesCmdModuleParamsInfo in modulesCmdModulesParamsInfo.items():
            print(f"---Save {cmdName} Module Params in path {specPath}---")
            safeCmdName = cmdName.replace("/", "_").replace(" ", "_")
            output_path = f"{specPath}/{ID_DATE_PATH}/{safeCmdName}/command_params.json"
            print(f"[DEBUG] save_modules_param: Writing to path = {output_path}")
            try:
                json_bytes = json.dumps(modulesCmdModuleParamsInfo, indent=2).encode("utf-8")
                print(f"[DEBUG] save_modules_param: JSON size = {len(json_bytes)} bytes")
                with MultiAgentOutput.get_writer(output_path) as w_binary:
                    w_binary.write(json_bytes)
                files_written += 1
                print(f"[DEBUG] save_modules_param: SUCCESS - File written: {output_path}")
            except Exception as e:
                print(f"[DEBUG] save_modules_param: ERROR - Failed to write {output_path}: {str(e)}")
                import traceback
                print(f"[DEBUG] save_modules_param: Traceback: {traceback.format_exc()}")
    
    print(f"[DEBUG] save_modules_param: Complete - {files_written}/{total_files_to_write} files written successfully")
    return {}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Key Phrase Extraction from Identified Commands

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
class WorkerCmdRulesState(TypedDict):
    spec_full_path: str
    module_cmd_name: str
    module_cmd_scope: str
    submodule_name: str
    submodule_feature_enabled: str
    submodule_scope: str
    completed_module_cmds_module_rules: Annotated[list, operator.add]
    completed_module_cmds_global_rules: Annotated[list, operator.add]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def define_command_rules_modules(state: WorkerCmdRulesState):
    """
    Worker function: Generate validation rules (module-specific and global) for a command module.
    
    Purpose:
        Creates validation rules that verify SSD drive responses and outputs based on
        specific command configurations. Rules define what to expect from the drive
        when commands are executed under various conditions. This is critical for
        automated test validation.
    
    How It Works:
        1. Sets CURRENT_SPEC_FULL_PATH global to filter Cortex Search results
        2. Uses RAG chain with get_retriever_command_info() to retrieve command context
        3. LLM generates initial rules covering:
           - Valid configurations and expected successful responses
           - Invalid configurations and expected error responses
           - State dependencies and their effects
           - Cross-cutting validation patterns
        4. For rules with missing information, enhances them with additional context
        5. Separates rules into "module" (command-specific) and "global" (cross-cutting) types
        6. Returns both module rules and global rules
    
    Args:
        state (WorkerCmdRulesState): Worker state containing:
            - spec_full_path: Specification path being processed
            - module_cmd_name: Command name
            - module_cmd_scope: Command's validation scope
            - submodule_name: Module name
            - submodule_feature_enabled: Feature this module validates
            - submodule_scope: Module's validation scope
    
    Returns:
        dict: Updates two state fields:
            - state["completed_module_cmds_module_rules"]: Module-specific rules
            - state["completed_module_cmds_global_rules"]: Global/cross-cutting rules
            Format: [{"spec_path": {"command_name": {"module_name": [rule1, rule2, ...]}}}]
    
    Used by:
        - assign_workers_cmds_rules() dispatches this worker via Send()
        - Called in parallel for each module across all commands and specifications
        - Each command may have multiple modules, each processed separately
    
    Integration:
        - Sets: CURRENT_SPEC_FULL_PATH global (used by get_retriever_command_info)
        - Uses: get_retriever_command_info() for initial rule generation
        - Uses: get_retriever_all_spec_short_info() for rule enhancement
        - Uses: llm_t (DKUChatLLM) with structured output (RulesDefinition, RuleDefinition)
        - Writes to: state["completed_module_cmds_module_rules"] and state["completed_module_cmds_global_rules"]
        - Reads from: specs_desc_dict for spec metadata
        - Reads from: state["specs_cmd_modules_definition"] for module information
        - Reads from: state["specs_cmd_modules_complete_info"] for command scope
    
    Rule Categories:
        1. MODULE RULES: Validate drive responses specific to this command/module
           - Positive cases: Expected successful effects with valid parameters
           - Negative cases: Expected error responses with invalid parameters
           - State validation: Correct state transitions and persistent effects
           - Output validation: Command outputs, status codes, returned data
        
        2. GLOBAL RULES: Cross-cutting validations applicable to multiple commands/modules
           - Drive state constraints (write protection, feature enablement)
           - Resource availability conditions
           - Concurrent operation limitations
           - General error handling patterns
    
    Rule Definition Structure:
        Each RuleDefinition includes:
        - ruleName: Concise, descriptive name (avoid generic terms)
        - ruleType: "module" or "global"
        - ruleDesc: Clear explanation of what drive behavior is validated
        - ruleCondition: Explicit conditions/parameters that trigger scenario
        - ruleResponse: Specific response expected from drive (status codes, outputs, state changes)
        - ruleMissingInfo: Questions if clarification needed, "well_defined" if complete
        - ruleReferences: Specification file paths (comma-separated)
        - ruleDependencies: Referenced tables/figures or "no_additions"
    
    Enhancement Process:
        For rules with ruleMissingInfo != "well_defined":
        1. Uses get_retriever_all_spec_short_info() to get additional context
        2. LLM enhances rule with missing information
        3. Updates ruleMissingInfo to "well_defined" if complete
    
    Workflow Position:
        Runs AFTER save_modules_param (which saves parameter definitions)
        and BEFORE define_modules_global_rules (which consolidates global rules)
    
    Quality Impact:
        Comprehensive validation rules enable automated test verification. Rules must
        be specific enough to verify drive behavior but general enough to apply across
        parameter variations. Global rules are later consolidated to avoid duplication.
    """
    specFullPath = state["spec_full_path"]
    specFullName = specs_desc_dict[specFullPath]["spec_full_name"]
    systemMessage = f"""You are an SSD testing validation architect expert for the {specFullName} specification, creating comprehensive validation rules for module testing in the RCV framework.

Your task is to define validation rules that verify the SSD drive's response and output based on specific command configurations and scenarios. Focus on what the drive returns or how its state changes when commands are executed under various conditions.

Rule Categories:

1. MODULE RULES - Validate drive responses specific to this command and module:
   - Positive cases: Expected successful effects and responses when valid parameters are used
   - Negative cases: Expected error responses when invalid/unsupported parameters are used
   - State validation: Verify correct state transitions and persistent effects
   - Output validation: Check command outputs, status codes, and returned data

2. GLOBAL RULES - Cross-cutting validations that apply to multiple commands/modules:
   - Drive state constraints (e.g., write protection, feature enablement)
   - Resource availability conditions
   - Concurrent operation limitations
   - General error handling patterns

Rule Definition Requirements:
Each rule must clearly specify:
- The exact scenario/condition being tested
- The command configuration that triggers this scenario
- The expected drive response (status codes, outputs, state changes)
- How to verify the response is correct

For each rule, provide:
* Rule Name: Concise, descriptive name reflecting the validation scenario (avoid generic terms like "rule", "validation", "check")
* Rule Type: "module" or "global"
* Rule Description: Clear explanation of what drive behavior is being validated
* Trigger Condition: Explicit conditions/parameters that cause this scenario (e.g., "Write command issued when write protection is enabled")
* Expected Drive Response: Specific response expected from the drive (e.g., "Command Status: Write Protected Error (0x02)", "No data written to media", "Error log entry created")
* Missing Information: Questions if clarification needed, or "well_defined" if complete
* Rule References: Full file paths from context, comma-separated (e.g., /path/reference_1.txt, /path/reference_2.txt)
* Dependencies: Referenced tables/figures, or "no_additions" if none

Important Guidelines:
- You main focus is only in the given command and its interactions.
- Create separate rules for each distinct drive response scenario
- Be specific about command parameters and drive state conditions
- Focus on verifiable drive outputs and behaviors
- Identify reusable patterns as global rules
- Cover both success paths and error conditions
- Ensure all specification requirements (SHALL/MUST) are covered by rules


RCV Validation Context:
{rcvDesc}
"""
    #"Name the rule. For Module rules, use the format moduleAbbr_NNNN; for global rules, use {specAbbrName}_NNNN.
 # Concurrency Rules: These rules are based on other specifications that are not directly defined in {specFullName}, but could still affect the command’s execution.

    message = """
Generate comprehensive validation rules covering:
1. All valid command configurations and their expected successful responses
2. All invalid configurations and their expected error responses
3. State dependencies and their effects on command execution
4. Cross-cutting validation patterns applicable beyond this module

Command Module Request: {command_request}

Command Info: {command_scope}

Command Module Scope: {module_scope}

Command context:
{context}
"""

    prompt = ChatPromptTemplate.from_messages([("system", systemMessage),
                                               ("human", message)])

    class RuleDefinition(BaseModel):
        ruleName: str = Field(description=f"Give a short name for the rule in base of its intention.")#Name the rule. For module rules, use the format commandName_NNNN; and for Global rules, use {specAbbrName}_NNNN") # ; and for concurrency rules, use other_NNNN.
        ruleType: str = Field(description="Classification of the rule, according to the type of rule 'module', or 'global'") # , or concurrency.
        ruleDesc: str = Field(description="Clear explanation of what drive behavior is being validated.")
        ruleCondition: str = Field(description="Trigger Condition, explicit conditions/parameters that cause this scenario. If more than one statement is listed, then separate them with commas like: statement1, statement2, statementN.")
        ruleResponse: str = Field(description="Specific response expected from the drive when the condition is met.")
        ruleMissingInfo: str = Field(description="Questions that can be done in order to clarify the rule split by a comma in example question1, question2, question3; In case it is not needed more information, set as 'well_defined'.")
        ruleReferences:str = Field(description="List all spec references text file exact path where relevant information about the rule can be found, separate them with commas (e.g., /path/reference_1.txt, /path/reference_2.txt, /path/reference_NNN.txt).")
        ruleDependencies: str = Field(description="Referenced tables/figures or 'no_additions'")

    class RulesDefinition(BaseModel):
        rules: List[RuleDefinition] = Field(
            description="Rules of the module"
        )

    llm_with_tool = llm_t.with_structured_output(RulesDefinition)
    
    commandName = state["module_cmd_name"]
    commandScope = state["module_cmd_scope"]
    moduleName = state["submodule_name"]
    moduleFeatureEnabled = state["submodule_feature_enabled"]
    moduleScope = state["submodule_scope"]
    
    command_question = f"Command: {commandName} covering {moduleName} which intention is {moduleFeatureEnabled}"

    global CURRENT_SPEC_FULL_PATH
    # This wil set the global variable need it to create the right retriever get_spec_info_section
    CURRENT_SPEC_FULL_PATH = specFullPath

    rag_chain = {
        "command_request": RunnablePassthrough(),
        "command_scope": lambda _: commandScope,
        "module_scope": lambda _: moduleScope,
        "context": get_retriever_command_info
    } | prompt | llm_with_tool
    
    rag_chain_w_retry = rag_chain.with_retry(
        wait_exponential_jitter=True,
        stop_after_attempt=2,
        exponential_jitter_params={"initial": 2}
    )

    time.sleep(20)
    response = invoke_with_rate_limit_retry(rag_chain_w_retry, command_question)

    systemMessage =f"""You are an SSD testing validation architect expert for the {specFullName} specification, refining validation rules for module testing in the RCV framework.

Your task is to enhance generated validation rules by filling in missing information based on additional context. The original rules were designed to verify SSD drive responses and outputs for specific command configurations and scenarios.

Rule Categories (from original generation):

1. MODULE RULES - Validate drive responses specific to this command and module:
   - Positive cases: Expected successful effects and responses when valid parameters are used
   - Negative cases: Expected error responses when invalid/unsupported parameters are used
   - State validation: Verify correct state transitions and persistent effects
   - Output validation: Check command outputs, status codes, and returned data

2. GLOBAL RULES - Cross-cutting validations that apply to multiple commands/modules:
   - Drive state constraints (e.g., write protection, feature enablement)
   - Resource availability conditions
   - Concurrent operation limitations
   - General error handling patterns

Enhancement Instructions:
Review each rule's missing information and use the provided context to complete or clarify the following fields:

* Rule Name: Keep the original name unless it was generic; if generic, improve it to be more descriptive
* Rule Type: Keep as "module" or "global"
* Rule Description: Enhance with additional details from context if the original was incomplete
* Trigger Condition: Fill in any missing details about conditions/parameters that trigger this scenario
* Expected Drive Response: Complete any gaps in the expected response specification (status codes, outputs, state changes)
* Validation Method: Add or clarify how to verify the response is correct
* Missing Information: Update to "well_defined" if context now provides complete information, otherwise refine the questions
* Rule References: Add any new reference file paths from the context, maintaining full paths separated by commas (e.g., /path/reference_1.txt, /path/reference_2.txt)
* Dependencies: Add any newly identified table/figure references, or keep as "no_additions"

Enhancement Guidelines:
- Only modify fields that had missing information or were incomplete
- Maintain consistency with the original rule's intent and validation scope
- Add specific details from the context that address the missing information
- Ensure enhanced rules remain focused on verifiable drive outputs and behaviors
- Keep the distinction clear between module-specific and global rules
- Do not create new rules; only enhance existing ones

RCV Validation Context:
{rcvDesc}
""" # Concurrency Rules: These rules are based on other specifications that are not directly defined in {specFullName}, but could still affect the command’s execution.
# Name the rule. For module rules, use the format commandName_NNNN; and for global rules, use {specAbbrName}_NNNN.
    message = """
Command Name: {command}
Module: {module}

Missing Rule Info: {rule_missing_info}

Current Rule Content info: {rule_current_info}

Command context:
{context}
"""
    prompt = ChatPromptTemplate.from_messages([("system", systemMessage),
                                               ("human", message)])

    cmd_rules_type = {}
    cmd_rules_type[moduleName] = {'module': [], 'global': []}

    llm_with_tool = llm_t.with_structured_output(RuleDefinition)

    rules_dict = response.model_dump()
    rules_list = rules_dict['rules']

    # Collect rules grouped by type
    for rule_dict in rules_list:
        rule_missing_info = rule_dict['ruleMissingInfo']
        rule_name = rule_dict['ruleName']
        rule_desc = rule_dict['ruleDesc']
        if rule_missing_info == 'well_defined':
            cmd_rules_type[moduleName][rule_dict['ruleType']].append(rule_dict)
            continue

        print(f'-------Module {moduleName}: Enhancing Rule {rule_name} as the following info was not well defined: {rule_missing_info}')

        rule_content = json.dumps(rule_dict, indent=2)

        rag_chain = {
            "rule_current_info": lambda rule_all_info: f"{rule_content}",
            "rule_missing_info": RunnablePassthrough(),
            "command": lambda command_name: f"{commandName}",
            "module": lambda submodule_name: f"{moduleName}",
            "context": get_retriever_all_spec_short_info #retriever_specific_info
        } | prompt | llm_with_tool

        rag_chain_w_retry = rag_chain.with_retry(
            wait_exponential_jitter=True,
            stop_after_attempt=4,
            exponential_jitter_params={"initial": 2}
        )

        time.sleep(25)

        response = invoke_with_rate_limit_retry(rag_chain_w_retry, f"Module {moduleName} for command {commandName} with the following definition: {rule_desc}" + rule_missing_info)

        rule_complete_definition = response.model_dump()

        cmd_rules_type[moduleName][rule_dict['ruleType']].append(rule_complete_definition)

    result = {'completed_module_cmds_module_rules': [{specFullPath: {commandName: {moduleName: cmd_rules_type[moduleName]['module']}}}],
               'completed_module_cmds_global_rules': [{specFullPath: {commandName: {moduleName: cmd_rules_type[moduleName]['global']}}}]}
    print(f"[DEBUG] define_command_rules_modules: Returning data for specPath = {specFullPath}, commandName = {commandName}, moduleName = {moduleName}")
    print(f"[DEBUG] define_command_rules_modules: Module rules count = {len(cmd_rules_type[moduleName]['module'])}, Global rules count = {len(cmd_rules_type[moduleName]['global'])}")
    return result

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def assign_workers_cmds_rules(state: ModuleContentState):
    """
    Worker assignment function: Dispatch workers to generate validation rules for each command module.
    
    Purpose:
        Assigns one worker per module across all commands and specifications to generate
        validation rules. Each module needs both module-specific rules (for that command/module)
        and global rules (cross-cutting validations). Workers generate both types.
    
    How It Works:
        1. Gets module definitions from specs_cmd_modules_definition (all specs)
        2. For each specification, iterates through all commands and their modules
        3. For each module, extracts module metadata (name, scope, feature_enabled)
        4. Creates one Send() operation per module to dispatch define_command_rules_modules worker
        5. Returns list of Send operations for LangGraph to execute in parallel
    
    Args:
        state (ModuleContentState): Workflow state containing:
            - specs_cmd_modules_definition: Dictionary of module definitions for all specs
                Format: {"spec_path": {"command_name": ProposedModules(...)}, ...}
            - specs_cmd_modules_complete_info: Dictionary of command scopes
                Format: {"spec_path": {"command_name": CommandRelevanceAssessment(...)}, ...}
    
    Returns:
        List[Send]: List of Send operations, each dispatching define_command_rules_modules worker
            Format: [Send("define_command_rules_modules", {"spec_full_path": "...", "module_cmd_name": "...", ...}), ...]
    
    Used by:
        - Workflow conditional edge from exec_cmd_rules node
        - LangGraph executes all Send operations in parallel
        - Each Send dispatches define_command_rules_modules worker with module-specific state
    
    Integration:
        - Reads from: state["specs_cmd_modules_definition"] (from module definition stage)
        - Reads from: state["specs_cmd_modules_complete_info"] (for command scope)
        - Dispatches to: define_command_rules_modules worker function
        - Workers write to: 
          * state["completed_module_cmds_module_rules"] (module-specific rules)
          * state["completed_module_cmds_global_rules"] (global/cross-cutting rules)
    
    Rule Generation:
        Each worker generates both:
        - Module rules: Specific to the command/module being processed
        - Global rules: Cross-cutting validations applicable to multiple modules
        Global rules are later consolidated by define_modules_global_rules to avoid duplication.
    
    Parallel Execution:
        If there are 50 modules across all commands and specs, this creates 50 Send operations.
        LangGraph executes them concurrently, significantly speeding up rule generation.
    
    Workflow Position:
        Runs AFTER save_modules_param (which saves parameter definitions)
        and routes to define_command_rules_modules workers (which generate validation rules)
    """
    print(f"---Assign Workers for Rules ---")
    print(f"[DEBUG] assign_workers_cmds_rules: Function called")
    # Get module definitions for all specifications (process all specs, not just missing ones)
    specsCompleteCmdModuleInfo = state['specs_cmd_modules_definition']
    specsCompleteCmdCompleteInfo = state['specs_cmd_modules_complete_info']
    
    print(f"[DEBUG] assign_workers_cmds_rules: specs_cmd_modules_definition type = {type(specsCompleteCmdModuleInfo)}, keys = {list(specsCompleteCmdModuleInfo.keys()) if isinstance(specsCompleteCmdModuleInfo, dict) else 'N/A'}")
    print(f"[DEBUG] assign_workers_cmds_rules: specs_cmd_modules_complete_info type = {type(specsCompleteCmdCompleteInfo)}, keys = {list(specsCompleteCmdCompleteInfo.keys()) if isinstance(specsCompleteCmdCompleteInfo, dict) else 'N/A'}")
    
    # Validate that module definitions exist
    if not specsCompleteCmdModuleInfo or (isinstance(specsCompleteCmdModuleInfo, dict) and len(specsCompleteCmdModuleInfo) == 0):
        print(f"[DEBUG] assign_workers_cmds_rules: WARNING - specs_cmd_modules_definition is empty, no workers to assign")
        return []
    
    final_output = []
    worker_count = 0
    # Iterate through all specifications, commands, and modules
    for specPath, moduleInfoDict in specsCompleteCmdModuleInfo.items():
        print(f"---------- Generating Module Rules {specPath} -------")
        print(f"[DEBUG] assign_workers_cmds_rules: Processing specPath = {specPath}, commands = {len(moduleInfoDict) if isinstance(moduleInfoDict, dict) else 'N/A'}")
        for cmdName, moduleRelevantInfoDict in moduleInfoDict.items():
            # Get list of modules defined for this command
            cmdDefinedModules = moduleRelevantInfoDict['modules']
            # Get command scope from complete info
            cmdInfo = specsCompleteCmdCompleteInfo[specPath][cmdName]["scope"]
            print(f"[DEBUG] assign_workers_cmds_rules: Processing command = {cmdName}, modules = {len(cmdDefinedModules) if isinstance(cmdDefinedModules, list) else 'N/A'}")
            # Create one worker per module
            for cmdDefinedModule in cmdDefinedModules:
                submoduleName = cmdDefinedModule['name']
                submoduleScope = cmdDefinedModule['scope']
                submoduleFeatureEnable = cmdDefinedModule['feature_enabled']
                print(f"---------- Processing Rules for {cmdName}: {submoduleName} -------")
                final_output.append(Send(
                    "define_command_rules_modules",
                    {"spec_full_path": specPath,
                     "module_cmd_name": cmdName,
                     "module_cmd_scope": cmdInfo,
                     "submodule_feature_enabled": submoduleFeatureEnable,
                     "submodule_name": submoduleName,
                     "submodule_scope": submoduleScope}))
                worker_count += 1
    
    print(f"[DEBUG] assign_workers_cmds_rules: Assigned {worker_count} workers, returning {len(final_output)} Send operations")
    return final_output

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def define_modules_global_rules(state: WorkerCmdRulesState):
    """
    Consolidation function: Merge and enhance global rules across all modules.
    
    Purpose:
        Takes global rules generated by individual modules and consolidates similar rules
        into reusable rules that apply across multiple modules. This reduces duplication
        and creates a unified set of cross-cutting validation rules.
    
    How It Works:
        1. Collects all global rules from state["completed_module_cmds_global_rules"]
        2. Formats rules into prompt for LLM consolidation
        3. LLM identifies similar rules and merges them based on:
           - Same drive state constraint (write protection, feature enablement)
           - Same resource limitation
           - Similar error conditions across commands
           - Same cross-cutting behavior
        4. Creates unified rules with applicable modules listed
        5. Enhances incomplete rules with additional context
        6. Returns consolidated global rules list
    
    Args:
        state (ModuleContentState): Workflow state containing:
            - completed_module_cmds_global_rules: All global rules from individual modules
                Format: [{"spec_path": {"command_name": {"module_name": [rule1, rule2, ...]}}}, ...]
    
    Returns:
        dict: Updates state["completed_final_global_rules"] with:
            Format: [{"ruleName": "...", "ruleDesc": "...", "ruleApplicableModules": [...], ...}, ...]
            List of consolidated global rule definitions
    
    Used by:
        - Workflow edge after save_modules_rules (sequential, not parallel)
        - Called once after all module rules are generated and saved
        - Processes all global rules across all commands, modules, and specifications
    
    Integration:
        - Reads from: state["completed_module_cmds_global_rules"] (accumulated from all modules)
        - Uses: get_retriever_all_spec_short_info() for rule enhancement
        - Uses: llm_t (DKUChatLLM) with structured output (RulesDefinition, RuleDefinition)
        - Writes to: state["completed_final_global_rules"] (replaces, not accumulates)
        - Note: This is NOT a worker function - it runs sequentially on all rules
    
    Consolidation Criteria:
        Merge rules when they:
        - Test the same drive state constraint
        - Validate the same resource limitation
        - Check similar error conditions across commands
        - Verify the same cross-cutting behavior
    
    Do NOT Merge when:
        - Rules have command-specific validations
        - Rules have different expected responses
        - Rules have conflicting trigger conditions
        - Rules are too specific to generalize
    
    Output Structure:
        Each consolidated RuleDefinition includes:
        - ruleName: Short name for the consolidated rule
        - ruleDesc: Comprehensive description covering all merged cases
        - ruleDomain: General section name grouping related rules
        - ruleApplicableModules: List of module names where rule applies (comma-separated)
        - ruleCondition: Generalized condition applicable across modules
        - ruleResponse: Expected SSD response for global condition
        - ruleMissingInfo: Questions if incomplete, "well_defined" if complete
        - ruleReferences: Specification file paths (comma-separated)
        - ruleDependencies: Referenced tables/figures
    
    Enhancement Process:
        For consolidated rules with ruleMissingInfo != "well_defined":
        1. Uses get_retriever_all_spec_short_info() to get additional context
        2. LLM enhances rule with missing information
        3. Maintains rule's cross-cutting nature and applicability
    
    Workflow Position:
        Runs AFTER save_modules_rules (which saves module-specific and initial global rules)
        and BEFORE save_global_rules (which saves final consolidated global rules)
    
    Quality Impact:
        Consolidation reduces rule duplication and creates reusable validation patterns.
        Rules remain generic enough to apply across all linked modules while being specific
        enough to enable automated verification.
    """
    systemMessage = f"""You are an SSD testing validation architect expert for technical specification, consolidating global validation rules across modules in the RCV framework.

Your task is to identify similar global rules and merge them into reusable rules that apply across multiple modules. Some rules may be too specific to merge and should remain separate.

Consolidation Criteria:
- Rules testing the same drive state constraint (e.g., write protection, feature enablement)
- Rules validating the same resource limitation
- Rules checking similar error conditions across commands
- Rules verifying the same cross-cutting behavior

When merging rules:
* Create a unified rule that covers all merged scenarios
* List all source modules in the rule metadata
* Ensure the merged rule is broadly applicable
* Keep the most comprehensive description and validation method

When NOT to merge:
- Rules with command-specific validations
- Rules with different expected responses
- Rules with conflicting trigger conditions
- Rules too specific to generalize

For each consolidated or individual rule output:
* Rule Name: Give a short name for the rule in base of its intention. Avoid generic terms like "rule", "validation", "check", etc.
* Rule Description: Comprehensive description covering all merged cases
* Rule Domain: General section name grouping related rules
* Applicable Modules: List of module names where this rule applies
* Trigger Condition: Generalized condition applicable across modules
* Expected Drive Response: Expected response for the global condition
* Missing Information: Update to "well_defined" if context now provides complete information, otherwise refine the questions
* Rule References: Add any new reference file paths from the context, maintaining full paths separated by commas (e.g., /path/reference_1.txt, /path/reference_2.txt)
* Dependencies: Add any newly identified table/figure references, or keep as "no_additions"

RCV Validation Context:
{rcvDesc}
"""
    message = """
Global Rules:
{global_rules_description}
"""

    class RuleDefinition(BaseModel):
        ruleName: str = Field(description=f"Name for the consolidated global rule.") #f"Name the rule format of the naming: {specAbbrName}_NNNN")
        ruleDesc: str = Field(description="Provide a complete description of the validation intent or propose of the Rule")
        ruleDomain: str = Field(description="Provide a general section name that can be used to group this rule with other related rules, based on the rule's description and intent.")
        ruleApplicableModules: List[str] = Field(description="List of modules where this rule applies, enlist as a comma-separated list e.g. command1, command2, command3")
        ruleCondition: str = Field(description="Specify the condition(s) used to validate the rule. Generalized condition that triggers this validation. If more than one statement is listed, then separate them with commas like: statement1, statement2, statementN.")
        ruleResponse: str = Field(description="Expected SSD response for this global condition.")
        ruleMissingInfo: str = Field(description="Questions that can be done in order to clarify the rule split by a comma in example question1, question2, question3; In case it is not needed more information, set as 'well_defined'.")
        ruleReferences:str = Field(description="List all spec references text file exact path where relevant information about the rule can be found, separate them with commas (e.g., /path/reference_1.txt, /path/reference_2.txt, /path/reference_NNN.txt).")
        ruleDependencies: str = Field(description="List all references to table or figures needed for the rule.")

    class RulesDefinition(BaseModel):
        rules: List[RuleDefinition] = Field(
            description="Rules of the module"
        )

    llm_with_tool = llm_t.with_structured_output(RulesDefinition)

    global_modules_rules = state['completed_module_cmds_global_rules']

    # rule_content = json.dumps(global_modules_rules, indent=2)


    prompt_sections = []
    prompt_sections.append("=== GLOBAL RULES SUMMARY ===\n")
    prompt_sections.append("The following global rules have been defined across all command modules:\n")

    # Process each command's global rules
    for globalRulesOutput in global_modules_rules:
        specPath, cmd_dict = next(iter(globalRulesOutput.items()))
        for command_name, modules in cmd_dict.items():
            for module_name, global_rules in modules.items():
                if not global_rules:  # Skip if no global rules
                    continue

                prompt_sections.append(f"\n--- COMMAND: {command_name} ---")
                prompt_sections.append(f"--- Module Name: {module_name} ---")

                for rule in global_rules:
                    prompt_sections.append(f"Rule Name: {rule.get('ruleName', 'N/A')}")
                    prompt_sections.append(f"Rule Type: {rule.get('ruleType', 'N/A')}")
                    prompt_sections.append(f"Description: {rule.get('ruleDesc', 'N/A')}")
                    prompt_sections.append(f"Condition: {rule.get('ruleCondition', 'N/A')}")
                    prompt_sections.append(f"Expected Response: {rule.get('ruleResponse', 'N/A')}")
                    prompt_sections.append(f"Missing Info: {rule.get('ruleMissingInfo', 'N/A')}")
                    prompt_sections.append(f"References: {rule.get('ruleReferences', 'N/A')}")
                    # prompt_sections.append("")

    rule_content = "\n".join(prompt_sections)

    # print(f'Global Rules per command\n{rule_content}')

    prompt = ChatPromptTemplate.from_messages([("system", systemMessage),
                                               ("human", message)])

    rag_chain = {
        "global_rules_description": RunnablePassthrough()
    } | prompt | llm_with_tool

    time.sleep(25)
    rag_chain_w_retry = rag_chain.with_retry(
        wait_exponential_jitter=True,
        stop_after_attempt=4,
        exponential_jitter_params={"initial": 2}
    )

    response = invoke_with_rate_limit_retry(rag_chain_w_retry, rule_content)

    systemMessage = f"""You are an SSD testing validation architect expert for technical specifications, refining consolidated global validation rules for the RCV framework.

Your task is to enhance previously consolidated global rules by resolving missing information using the provided context. These rules were designed to capture cross-cutting validations applicable across multiple commands and modules.

Global Rule Enhancement Context:
The rules you are enhancing have already been consolidated from module-specific global rules. Your focus is to:
1. Fill gaps in rule definitions where information was incomplete
2. Clarify ambiguous trigger conditions or expected responses
3. Add missing specification references and dependencies
4. Ensure rules remain broadly applicable across their linked modules

Enhancement Instructions:
Review each consolidated global rule and use the provided context to complete or clarify the following fields:

* Rule Name: Keep the format CommonRule_NNNN; improve name if it was too generic
* Rule Description: Enhance with specific details about what drive behavior is being validated across modules
* Rule Domain: Refine the domain classification if context provides clearer grouping
* Linked Modules: Maintain the list of modules where this rule applies (do not modify unless context shows missing modules)
* Trigger Condition: Fill in missing details about conditions that trigger this validation across different commands (use pseudocode where applicable)
* Expected Drive Response: Complete any gaps in the expected response specification (status codes, state changes, error behaviors)
* Missing Information: Update to "well_defined" if context now provides complete information, otherwise refine the clarification questions
* Rule References: Add any new reference file paths from the context, maintaining full paths separated by commas (e.g., /path/reference_1.txt, /path/reference_2.txt)
* Dependencies: Add any newly identified table/figure references, or keep as "no_additions"

Enhancement Guidelines:
- Only modify fields that had missing information or were incomplete
- Maintain the rule's cross-cutting nature and applicability across multiple modules
- Add specific details from the context that address the missing information
- Ensure enhanced rules remain generic enough to apply to all linked modules
- Do not create new rules; only enhance existing consolidated rules
- Preserve the consolidation logic that merged similar module-level global rules
- Keep validation criteria focused on verifiable drive outputs and behaviors

Important Constraints:
- Use ONLY the provided context to fill gaps
- Do not hallucinate values, conditions, or responses
- If context doesn't answer a missing information question, refine the question or mark as still undefined
- Maintain consistency with RCV validation principles

RCV Validation Context:
{rcvDesc}
""" # Concurrency Rules: These rules are based on other specifications that are not directly defined in {specFullName}, but could still affect the command’s execution.

    message = """
Linked Modules: {linked_modules}

Missing Rule Info: {rule_missing_info}

Current Rule Content info: {rule_current_info}

Command context:
{context}
"""
    prompt = ChatPromptTemplate.from_messages([("system", systemMessage),
                                               ("human", message)])

    global_rules_final = []

    global_rules_dict = response.model_dump()
    global_rules_list = global_rules_dict['rules']

    for global_rule_dict in global_rules_list:
        rule_missing_info = global_rule_dict['ruleMissingInfo']
        rule_name = global_rule_dict['ruleName']
        rule_linked_modules = global_rule_dict['ruleApplicableModules']
        rule_desc = global_rule_dict['ruleDesc']

        if rule_missing_info == 'well_defined':
            global_rules_final.append(global_rule_dict)
            continue

        print(f'-------Enhancing Global Rule {rule_name} as the following info was not well defined: {rule_missing_info}')

        rule_content = json.dumps(global_rule_dict, indent=2)

        rag_chain = {
            "rule_current_info": lambda rule_all_info: f"{rule_content}",
            "rule_missing_info": RunnablePassthrough(),
            "linked_modules": lambda command_name: f"{rule_linked_modules}",
            "context": get_retriever_all_spec_short_info #retriever_specific_info,
        } | prompt | llm_with_tool

        time.sleep(25)
        rag_chain_w_retry = rag_chain.with_retry(
            wait_exponential_jitter=True,
            stop_after_attempt=4,
            exponential_jitter_params={"initial": 2}
        )

        response = invoke_with_rate_limit_retry(rag_chain_w_retry, f"{rule_linked_modules} with the following definition: {rule_desc}" + rule_missing_info)
        rule_complete_definition = response.model_dump()
        global_rules_final.append(rule_complete_definition)

    result = {'completed_final_global_rules': global_rules_final}
    print(f"[DEBUG] define_modules_global_rules: Returning {len(global_rules_final)} global rules")
    return result

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Save Rules In Json Files

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def save_modules_rules(state: ModuleContentState):
    print(f"[DEBUG] save_modules_rules: Function called")
    modulesCmdRulesOutput = state["completed_module_cmds_module_rules"]
    global ID_DATE_PATH
    
    print(f"[DEBUG] save_modules_rules: ID_DATE_PATH = {ID_DATE_PATH}")
    print(f"[DEBUG] save_modules_rules: modulesCmdRulesOutput type = {type(modulesCmdRulesOutput)}, length = {len(modulesCmdRulesOutput) if hasattr(modulesCmdRulesOutput, '__len__') else 'N/A'}")
    
    if not modulesCmdRulesOutput or len(modulesCmdRulesOutput) == 0:
        print(f"[DEBUG] save_modules_rules: WARNING - modulesCmdRulesOutput is empty, no module rules to save")
    else:
        groupedCommandsInfo = {}
        print(f"[DEBUG] save_modules_rules: Starting grouping loop for module rules, processing {len(modulesCmdRulesOutput)} items")

        for idx, moduleCmdRulesOutput in enumerate(modulesCmdRulesOutput):
            print(f"[DEBUG] save_modules_rules: Processing module rules item {idx+1}/{len(modulesCmdRulesOutput)}")
            specPath, moduleCmdRules = next(iter(moduleCmdRulesOutput.items()))
            print(f"[DEBUG] save_modules_rules: specPath = {specPath}, rules count = {len(moduleCmdRules) if isinstance(moduleCmdRules, dict) else 'N/A'}")
            if specPath not in groupedCommandsInfo:
                    groupedCommandsInfo[specPath] = {}

            for commandName, moduleRulesInfo in moduleCmdRules.items():
                if commandName in groupedCommandsInfo[specPath]:
                    groupedCommandsInfo[specPath][commandName].update(moduleRulesInfo)
                else:
                    groupedCommandsInfo[specPath].update({commandName: moduleRulesInfo})

        print(f"[DEBUG] save_modules_rules: Module rules grouping complete, {len(groupedCommandsInfo)} spec paths")
        total_module_rules_files = sum(len(cmds) for cmds in groupedCommandsInfo.values())
        print(f"[DEBUG] save_modules_rules: Total module rules files to write = {total_module_rules_files}")

        files_written = 0
        for specPath, modulesCmdRulesInfo in groupedCommandsInfo.items():
            print(f"[DEBUG] save_modules_rules: Processing module rules for specPath = {specPath}, commands = {len(modulesCmdRulesInfo)}")
            for cmdName, moduleRulesInfo in modulesCmdRulesInfo.items():
                print(f"---Save {cmdName} Rules in path {specPath}---")
                safeCmdName = cmdName.replace("/", "_").replace(" ", "_")
                output_path = f"{specPath}/{ID_DATE_PATH}/{safeCmdName}/module_rules.json"
                print(f"[DEBUG] save_modules_rules: Writing module rules to path = {output_path}")
                try:
                    json_bytes = json.dumps(moduleRulesInfo, indent=2).encode("utf-8")
                    print(f"[DEBUG] save_modules_rules: JSON size = {len(json_bytes)} bytes")
                    with MultiAgentOutput.get_writer(output_path) as w_binary:
                        w_binary.write(json_bytes)
                    files_written += 1
                    print(f"[DEBUG] save_modules_rules: SUCCESS - Module rules file written: {output_path}")
                except Exception as e:
                    print(f"[DEBUG] save_modules_rules: ERROR - Failed to write {output_path}: {str(e)}")
                    import traceback
                    print(f"[DEBUG] save_modules_rules: Traceback: {traceback.format_exc()}")
        
        print(f"[DEBUG] save_modules_rules: Module rules complete - {files_written}/{total_module_rules_files} files written")

    modulesGlobalRulesCmdInfo = state["completed_module_cmds_global_rules"]
    print(f"[DEBUG] save_modules_rules: modulesGlobalRulesCmdInfo type = {type(modulesGlobalRulesCmdInfo)}, length = {len(modulesGlobalRulesCmdInfo) if hasattr(modulesGlobalRulesCmdInfo, '__len__') else 'N/A'}")
    
    if not modulesGlobalRulesCmdInfo or len(modulesGlobalRulesCmdInfo) == 0:
        print(f"[DEBUG] save_modules_rules: WARNING - modulesGlobalRulesCmdInfo is empty, no global rules to save")
    else:
        groupedGlobalCmdsGlobalRulesInfo = {}
        print(f"[DEBUG] save_modules_rules: Starting grouping loop for global rules, processing {len(modulesGlobalRulesCmdInfo)} items")
        
        for idx, modulesGlobalRulesCmdInfoOutput in enumerate(modulesGlobalRulesCmdInfo):
            print(f"[DEBUG] save_modules_rules: Processing global rules item {idx+1}/{len(modulesGlobalRulesCmdInfo)}")
            specPath, modulesGlobalRulesCmdInfo = next(iter(modulesGlobalRulesCmdInfoOutput.items()))
            print(f"[DEBUG] save_modules_rules: specPath = {specPath}, global rules count = {len(modulesGlobalRulesCmdInfo) if isinstance(modulesGlobalRulesCmdInfo, dict) else 'N/A'}")
            if specPath not in groupedGlobalCmdsGlobalRulesInfo:
                    groupedGlobalCmdsGlobalRulesInfo[specPath] = {}

            for cmdName, moduleGlobalRulesCmdInfo in modulesGlobalRulesCmdInfo.items():
                if cmdName in groupedGlobalCmdsGlobalRulesInfo[specPath]:
                    groupedGlobalCmdsGlobalRulesInfo[cmdName].update(moduleGlobalRulesCmdInfo)
                else:
                    groupedGlobalCmdsGlobalRulesInfo.update({cmdName: moduleGlobalRulesCmdInfo})

        print(f"[DEBUG] save_modules_rules: Global rules grouping complete, {len(groupedGlobalCmdsGlobalRulesInfo)} entries")
        total_global_rules_files = len(groupedGlobalCmdsGlobalRulesInfo)
        print(f"[DEBUG] save_modules_rules: Total global rules files to write = {total_global_rules_files}")

        files_written = 0
        for specPath, modulesCmdGlobalRulesInfo in groupedCommandsInfo.items():
            print(f"[DEBUG] save_modules_rules: Processing global rules for specPath = {specPath}")
            for cmdName, moduleGlobalRulesInfo in modulesCmdGlobalRulesInfo.items():
                print(f"---Save Generic Global {specPath} -> {cmdName} Rules---")
                safeCmdName = cmdName.replace("/", "_").replace(" ", "_")
                output_path = f"{specPath}/{ID_DATE_PATH}/{safeCmdName}/global_rules.json"
                print(f"[DEBUG] save_modules_rules: Writing global rules to path = {output_path}")
                try:
                    json_bytes = json.dumps(moduleGlobalRulesInfo, indent=2).encode("utf-8")
                    print(f"[DEBUG] save_modules_rules: JSON size = {len(json_bytes)} bytes")
                    with MultiAgentOutput.get_writer(output_path) as w_binary:
                        w_binary.write(json_bytes)
                    files_written += 1
                    print(f"[DEBUG] save_modules_rules: SUCCESS - Global rules file written: {output_path}")
                except Exception as e:
                    print(f"[DEBUG] save_modules_rules: ERROR - Failed to write {output_path}: {str(e)}")
                    import traceback
                    print(f"[DEBUG] save_modules_rules: Traceback: {traceback.format_exc()}")
        
        print(f"[DEBUG] save_modules_rules: Global rules complete - {files_written}/{total_global_rules_files} files written")
    
    print(f"[DEBUG] save_modules_rules: Function complete")
    return {}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def save_global_rules(state: ModuleContentState):
    print(f"[DEBUG] save_global_rules: Function called")
    uniqueSpecRootPaths = specsBaseStores_df['specRootPath'].unique().tolist()
    print(f"[DEBUG] save_global_rules: Found {len(uniqueSpecRootPaths)} unique spec root paths: {uniqueSpecRootPaths}")
    
    global ID_DATE_PATH
    print(f"[DEBUG] save_global_rules: ID_DATE_PATH = {ID_DATE_PATH}")
    
    module_rules_info = state['completed_final_global_rules']
    print(f"[DEBUG] save_global_rules: module_rules_info type = {type(module_rules_info)}, length = {len(module_rules_info) if hasattr(module_rules_info, '__len__') else 'N/A'}")
    
    if not module_rules_info or (hasattr(module_rules_info, '__len__') and len(module_rules_info) == 0):
        print(f"[DEBUG] save_global_rules: WARNING - completed_final_global_rules is empty, no global rules to save")
    else:
        files_written = 0
        for specPath in uniqueSpecRootPaths:
            print(f"---Save Global Rules for {specPath} ---")
            output_path = f"{specPath}/{ID_DATE_PATH}/commands_global_rules/global_rules.json"
            print(f"[DEBUG] save_global_rules: Writing to path = {output_path}")
            try:
                json_bytes = json.dumps(module_rules_info, indent=2).encode("utf-8")
                print(f"[DEBUG] save_global_rules: JSON size = {len(json_bytes)} bytes")
                with MultiAgentOutput.get_writer(output_path) as w_binary:
                    w_binary.write(json_bytes)
                files_written += 1
                print(f"[DEBUG] save_global_rules: SUCCESS - File written: {output_path}")
            except Exception as e:
                print(f"[DEBUG] save_global_rules: ERROR - Failed to write {output_path}: {str(e)}")
                import traceback
                print(f"[DEBUG] save_global_rules: Traceback: {traceback.format_exc()}")
        
        print(f"[DEBUG] save_global_rules: Complete - {files_written}/{len(uniqueSpecRootPaths)} files written successfully")
    
    return {}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### LangGraph Framework Architecture

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ============================================================================
# WORKFLOW GRAPH DEFINITION
# ============================================================================
# This section defines the LangGraph workflow that orchestrates the multi-agent
# specification decomposition process. The workflow uses a state machine pattern
# where each node processes the shared ModuleContentState and routes to the next stage.
#
# Workflow Pattern:
#   1. Checkpoint Loading: Check if results exist in dataCollectionAI folder
#   2. Conditional Routing: If missing, process; if exists, load and skip
#   3. Worker Assignment: Assign parallel workers using Send() for concurrent processing
#   4. Result Aggregation: LangGraph automatically merges worker results
#   5. Saving: Save results to both dataCollectionAI (checkpoint) and MultiAgentOutput (final)
#
# Node Types:
#   - dataCollection*: Checkpoint loading nodes (check if results exist)
#   - exec_*: Worker assignment nodes (dispatch parallel workers)
#   - Worker nodes: Process individual tasks (extract, define, retrieve, etc.)
#   - save_*: Save results to checkpoints and final output
#   - Join nodes: Empty nodes that aggregate worker results before next stage

# Define a new graph with ModuleContentState as the shared state
# Purpose: Create the LangGraph workflow that will orchestrate all processing stages.
# ModuleContentState defines the structure of data flowing through the workflow.
workflow = StateGraph(ModuleContentState)

# ========================================================================
# WORKER NODES - Process Individual Tasks
# ========================================================================
# These nodes perform the actual work of extracting, analyzing, and generating
# specification decomposition results. They are called by exec_* nodes via Send().

# Index Extraction Stage
workflow.add_node("locate_spec_index_pages",  locate_spec_index_pages)
# Purpose: Worker that identifies table of contents pages in specifications.
# Called by: exec_index_extraction_workers via Send()

workflow.add_node("save_spec_index_pages",  save_spec_index_pages)
# Purpose: Saves extracted index pages to dataCollectionAI checkpoint folder.
# Called by: Workflow after locate_spec_index_pages completes

# Relevant Sections Stage
workflow.add_node("identify_relevant_spec_sections", identify_relevant_spec_sections)
# Purpose: Worker that identifies relevant specification sections for command extraction.
# Called by: exec_relevant_sections_workers via Send()

workflow.add_node("save_spec_most_relavant_sections", save_spec_most_relavant_sections)
# Purpose: Saves relevant sections to dataCollectionAI checkpoint folder.
# Called by: Workflow after identify_relevant_spec_sections completes

# Command Extraction Stage
workflow.add_node("extract_callable_commands", extract_callable_commands)
# Purpose: Worker that extracts callable commands from relevant specification sections.
# Called by: exec_extract_callable_commands_workers via Send()

workflow.add_node("prune_command_list", prune_command_list)
# Purpose: Worker that removes duplicate commands and normalizes command names.
# Called by: exec_prune_command_list_workers via Send()

workflow.add_node("save_spec_unique_callable_commands", save_spec_unique_callable_commands)
# Purpose: Saves unique command lists to dataCollectionAI checkpoint folder.
# Called by: Workflow after prune_command_list completes

# Command Relevance Assessment Stage
workflow.add_node("retrieve_modules_cmd_info", retrieve_modules_cmd_info)
# Purpose: Worker that assesses if commands have sufficient detail for RCV module creation.
# Called by: exec_relevance_workers via Send()

workflow.add_node("save_spec_most_relavant_info_cmd", save_spec_most_relavant_info_cmd)
# Purpose: Saves command relevance assessments to dataCollectionAI checkpoint folder.
# Called by: Workflow after retrieve_modules_cmd_info completes

# Module Definition Stage
workflow.add_node("define_cmd_modules", define_cmd_modules)
# Purpose: Worker that defines validation modules for each command.
# Called by: exec_modules_extraction via Send()

workflow.add_node("save_spec_cmd_modules_definition", save_spec_cmd_modules_definition)
# Purpose: Saves module definitions to dataCollectionAI checkpoint folder.
# Called by: Workflow after define_cmd_modules completes

workflow.add_node("save_modules_info", save_modules_info)
# Purpose: Generates and saves module overview summary to MultiAgentOutput.
# Called by: Workflow after module definitions are complete

# Parameter Extraction Stage
workflow.add_node("built_command_params_modules", built_command_params_modules)
# Purpose: Worker that extracts parameters for each command module.
# Called by: exec_build_params via Send()

workflow.add_node("enhance_parameter_context", enhance_parameter_context)
# Purpose: Worker that enhances incomplete parameter information with additional context.
# Called by: exec_enhance_params via Send()

workflow.add_node("save_modules_param", save_modules_param)
# Purpose: Saves parameter definitions to MultiAgentOutput folder.
# Called by: Workflow after parameter extraction and enhancement complete

# Rule Generation Stage
workflow.add_node("define_command_rules_modules", define_command_rules_modules)
# Purpose: Worker that generates validation rules (module-specific and global) for commands.
# Called by: exec_cmd_rules via Send()

workflow.add_node("define_modules_global_rules", define_modules_global_rules)
# Purpose: Consolidates and enhances global rules across all modules.
# Called by: Workflow after all module rules are generated

workflow.add_node("save_modules_rules", save_modules_rules)
# Purpose: Saves module-specific validation rules to MultiAgentOutput folder.
# Called by: Workflow after define_command_rules_modules completes

workflow.add_node("save_global_rules", save_global_rules)
# Purpose: Saves consolidated global validation rules to MultiAgentOutput folder.
# Called by: Workflow after define_modules_global_rules completes

# ========================================================================
# CHECKPOINT LOADING NODES - Load Existing Results
# ========================================================================
# These nodes check if results already exist in dataCollectionAI folder.
# If results exist, they load them into state and skip processing.
# If missing, they set save_dataCollection=True to trigger processing.

workflow.add_node("dataCollectionIndexExtraction", dataCollectionIndexExtraction)
# Purpose: Check if index pages already exist, load if found, mark missing if not.
# Updates: content_index_pages, missing_spec_paths_to_process, save_dataCollection

workflow.add_node("dataCollectionRelevantSections", dataCollectionRelevantSections)
# Purpose: Check if relevant sections already exist, load if found, mark missing if not.
# Updates: specs_relevant_sections, missing_spec_paths_to_process, save_dataCollection

workflow.add_node("dataCollectionUniqueCommands", dataCollectionUniqueCommands)
# Purpose: Check if unique commands already exist, load if found, mark missing if not.
# Updates: specs_callable_unique_cmds, missing_spec_paths_to_process, save_dataCollection

workflow.add_node("dataCollectionCommandsInfo", dataCollectionCommandsInfo)
# Purpose: Check if command relevance assessments already exist, load if found, mark missing if not.
# Updates: specs_cmd_modules_complete_info, missing_spec_paths_to_process, save_dataCollection

workflow.add_node("dataCollectionModulesExtraction", dataCollectionModulesExtraction)
# Purpose: Check if module definitions already exist, load if found, mark missing if not.
# Updates: specs_cmd_modules_definition, missing_spec_paths_to_process, save_dataCollection

# ========================================================================
# WORKER ASSIGNMENT NODES (Join Nodes) - Dispatch Parallel Workers
# ========================================================================
# These are empty nodes that serve as join points. They use conditional_edges
# with assign_workers_* functions that return List[Send()] to dispatch parallel workers.
# LangGraph automatically waits for all workers to complete before proceeding.

workflow.add_node("exec_index_extraction_workers", lambda state: {})
# Purpose: Join point before dispatching index extraction workers.
# Routes to: assign_workers_index_extraction() which returns Send() to locate_spec_index_pages

workflow.add_node("exec_relevant_sections_workers", lambda state: {})
# Purpose: Join point before dispatching relevant sections workers.
# Routes to: assign_workers_relevant_sections_extraction() which returns Send() to identify_relevant_spec_sections

workflow.add_node("exec_extract_callable_commands_workers", lambda state: {})
# Purpose: Join point before dispatching command extraction workers.
# Routes to: assign_workers_callable_commands_extraction() which returns Send() to extract_callable_commands

workflow.add_node("exec_prune_command_list_workers", lambda state: {})
# Purpose: Join point before dispatching command pruning workers.
# Routes to: assign_workers_prune_commands_list() which returns Send() to prune_command_list

workflow.add_node("exec_relevance_workers", lambda state: {})
# Purpose: Join point before dispatching command relevance assessment workers.
# Routes to: assign_workers_unique_cmds() which returns Send() to retrieve_modules_cmd_info

workflow.add_node("exec_modules_extraction", lambda state: {})
# Purpose: Join point before dispatching module definition workers.
# Routes to: assign_workers_cmd_modules_definition() which returns Send() to define_cmd_modules

workflow.add_node("exec_build_params", lambda state: {})
# Purpose: Join point before dispatching parameter extraction workers.
# Routes to: assign_workers_cmds_params() which returns Send() to built_command_params_modules

workflow.add_node("exec_enhance_params", lambda state: {})
# Purpose: Join point before dispatching parameter enhancement workers.
# Routes to: assign_workers_cmds_enha_params() which returns Send() to enhance_parameter_context

workflow.add_node("exec_cmd_rules", lambda state: {})
# Purpose: Join point before dispatching rule generation workers.
# Routes to: assign_workers_cmds_rules() which returns Send() to define_command_rules_modules


# ========================================================================
# WORKFLOW EDGES - Define Execution Flow
# ========================================================================
# Edges connect nodes to define the execution order. Conditional edges use
# decision functions to route based on state, enabling checkpoint-based skipping.

# Start the workflow by checking for existing index pages
# Purpose: Begin with checkpoint loading to avoid reprocessing existing results.
workflow.add_edge(START, "dataCollectionIndexExtraction")

# Route based on whether index pages were found
# Purpose: If index pages exist, skip extraction and proceed to next stage.
#          If missing, extract index pages first.
workflow.add_conditional_edges(
    "dataCollectionIndexExtraction",
    index_files_found,  # Decision function: checks if all specs have index pages
    {
        "index_not_found": "exec_index_extraction_workers",  # Missing → extract
        "index_found": "dataCollectionRelevantSections"     # Found → skip to next stage
    }
)

workflow.add_conditional_edges(
    "exec_index_extraction_workers",
    assign_workers_index_extraction,
    ["locate_spec_index_pages"]
)


workflow.add_conditional_edges(
    "locate_spec_index_pages",
    save_load_checkpoint,
    {
         "loadable": "dataCollectionRelevantSections",
         "savable": "save_spec_index_pages"
     })

workflow.add_edge("save_spec_index_pages", "dataCollectionRelevantSections")

workflow.add_conditional_edges(
    "dataCollectionRelevantSections",
    save_load_checkpoint,
    {
        "savable": "exec_relevant_sections_workers",
        "loadable": "dataCollectionUniqueCommands"
    }
)

workflow.add_conditional_edges(
    "exec_relevant_sections_workers",
    assign_workers_relevant_sections_extraction,
    ["identify_relevant_spec_sections"]
)

workflow.add_edge("identify_relevant_spec_sections", "save_spec_most_relavant_sections")

workflow.add_edge("save_spec_most_relavant_sections", "dataCollectionUniqueCommands")


workflow.add_conditional_edges(
    "dataCollectionUniqueCommands",
    save_load_checkpoint,
    {
        "savable": "exec_extract_callable_commands_workers",
        "loadable": "dataCollectionCommandsInfo"
    }
)

workflow.add_conditional_edges(
    "exec_extract_callable_commands_workers",
    assign_workers_callable_commands_extraction,
    ["extract_callable_commands"]
)

workflow.add_edge("extract_callable_commands", "exec_prune_command_list_workers")


workflow.add_conditional_edges(
    "exec_prune_command_list_workers",
    assign_workers_prune_commands_list,
    ["prune_command_list"]
)

workflow.add_edge("prune_command_list", "save_spec_unique_callable_commands")

workflow.add_edge("save_spec_unique_callable_commands", "dataCollectionCommandsInfo")


#################

workflow.add_conditional_edges(
    "dataCollectionCommandsInfo",
    save_load_checkpoint,
    {
        "savable": "exec_relevance_workers",
        "loadable": "dataCollectionModulesExtraction"
    }
)

# Save Most Relevant Info
workflow.add_conditional_edges(
    "exec_relevance_workers",
    assign_workers_unique_cmds,
    ["retrieve_modules_cmd_info"]
)

workflow.add_edge("retrieve_modules_cmd_info", "save_spec_most_relavant_info_cmd")
workflow.add_edge("save_spec_most_relavant_info_cmd", "dataCollectionModulesExtraction")

###############################
workflow.add_conditional_edges(
    "dataCollectionModulesExtraction",
    save_load_checkpoint,
    {
        "savable": "exec_modules_extraction",
        "loadable": "save_modules_info"
    }
)

workflow.add_conditional_edges(
    "exec_modules_extraction",
    assign_workers_cmd_modules_definition,
    ["define_cmd_modules"]
)

workflow.add_edge("define_cmd_modules", "save_spec_cmd_modules_definition")
workflow.add_edge("save_spec_cmd_modules_definition", "save_modules_info")
####################################


workflow.add_edge("save_modules_info", "exec_build_params")

# Command Parameters
workflow.add_conditional_edges(
    "exec_build_params",
    assign_workers_cmds_params,
    ["built_command_params_modules"]
)

workflow.add_edge("built_command_params_modules", "exec_enhance_params")


workflow.add_conditional_edges(
    "exec_enhance_params",
    assign_workers_cmds_enha_params,
    ["enhance_parameter_context"]
)

workflow.add_edge("enhance_parameter_context", "save_modules_param")
"""
workflow.add_edge("save_modules_param", END)
"""
# Rules

workflow.add_edge("save_modules_param", "exec_cmd_rules")
workflow.add_conditional_edges(
    "exec_cmd_rules",
    assign_workers_cmds_rules,
    ["define_command_rules_modules"]
)
# workflow.add_edge("define_command_rules_modules", "save_modules_rules")

# workflow.add_edge("save_modules_rules", END)

workflow.add_edge("define_command_rules_modules", "save_modules_rules")

workflow.add_edge("save_modules_rules", "define_modules_global_rules")

workflow.add_edge("define_modules_global_rules", "save_global_rules")


workflow.add_edge("save_global_rules", END)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compile
graph = workflow.compile()

# Display graph visualization if IPython is available (notebook environment)
if IPYTHON_AVAILABLE:
    try:
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except Exception as e:
        print(f"Could not display graph visualization: {e}")
else:
    print("Graph visualization skipped (not in notebook environment)")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(f"[DEBUG] WORKFLOW: Starting graph execution")
print(f"[DEBUG] WORKFLOW: ID_DATE_PATH = {ID_DATE_PATH}")
print(f"[DEBUG] WORKFLOW: MultiAgentOutput folder name = {multi_agent_output_folder_name}")
print(f"[DEBUG] WORKFLOW: MultiAgentOutput info = {MultiAgentOutput_info}")

initial_state = {"specs_relevant_sections": "",
                 "modules_unique_cmds": "",
                 "complete_callable_commands": [],
                 "completed_module_cmds": [],
                 "completed_module_cmds_submodules": [],
                 "completed_module_cmds_param": [],
                 "completed_module_cmds_enha_param": [],
                 "completed_module_cmds_global_rules": [],
                 "completed_module_cmds_module_rules": [],
                 "completed_final_global_rules": [],
                 "completed_cmd_modules_definition": [],
                 "content_index_pages": {},
                 # "index_pages_found": False,
                 "save_dataCollection": False}

print(f"[DEBUG] WORKFLOW: Initial state keys = {list(initial_state.keys())}")
for key, value in initial_state.items():
    if isinstance(value, (list, dict)):
        print(f"[DEBUG] WORKFLOW: Initial state[{key}] = {type(value).__name__} with length {len(value)}")
    else:
        print(f"[DEBUG] WORKFLOW: Initial state[{key}] = {value}")

GraphResult_ModuleHunt = graph.invoke(initial_state,
                                      {"recursion_limit": 3000,
                                       "max_concurrency": 8})

print(f"[DEBUG] WORKFLOW: Graph execution completed")
print(f"[DEBUG] WORKFLOW: Final state keys = {list(GraphResult_ModuleHunt.keys()) if isinstance(GraphResult_ModuleHunt, dict) else 'N/A'}")
if isinstance(GraphResult_ModuleHunt, dict):
    for key, value in GraphResult_ModuleHunt.items():
        if isinstance(value, (list, dict)):
            print(f"[DEBUG] WORKFLOW: Final state[{key}] = {type(value).__name__} with length {len(value)}")
        else:
            print(f"[DEBUG] WORKFLOW: Final state[{key}] = {type(value).__name__}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Only for testing