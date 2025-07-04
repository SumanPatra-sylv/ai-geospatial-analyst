#!/usr/bin/env python3
"""
Query Parser for AI-GIS Analyst (Refined Version)

This module provides functionality to parse natural language queries into structured
spatial analysis requests using Pydantic models and a real LLM-based extraction.
"""

import json
import logging
from enum import Enum
from typing import List, Optional
import os
import requests

from pydantic import BaseModel, Field, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FIX 1: Define a new, more specific custom exception for invalid user queries ---
class InvalidQueryError(Exception):
    """Custom exception for queries that are invalid or non-geospatial."""
    pass

# --- This exception is for system/network errors ---
class QueryParserError(Exception):
    """Custom exception for system/network errors during query parsing."""
    pass

# --- Pydantic Models (Unchanged) ---
class SpatialRelationship(str, Enum):
    WITHIN = "within"
    NOT_WITHIN = "not within"
    NEAR = "near"
    FAR_FROM = "far from"

class SpatialConstraint(BaseModel):
    feature_type: str = Field(..., description="Type of spatial feature")
    relationship: SpatialRelationship = Field(..., description="The spatial relationship")
    distance_meters: Optional[int] = Field(None, description="Optional distance in meters")

class ParsedQuery(BaseModel):
    target: str = Field(..., description="The main feature being analyzed")
    location: str = Field(..., description="Geographic location for the analysis")
    constraints: List[SpatialConstraint]

# --- Real LLM Caller (Unchanged) ---
def call_llm(prompt: str) -> str:
    """
    Calls the Ollama LLM service to get a JSON response.
    
    Raises:
        QueryParserError: If the OLLAMA_BASE_URL is not set or if there is a network error.
    """
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if not ollama_url:
        raise QueryParserError("OLLAMA_BASE_URL environment variable is not set.")
        
    full_api_url = f"{ollama_url}/api/generate"
    logger.info(f"Connecting to LLM at: {full_api_url}")
    
    payload = {
        "model": "mistral", 
        "prompt": prompt, 
        "stream": False, 
        "format": "json"
    }
    
    try:
        response = requests.post(full_api_url, json=payload, timeout=90)
        response.raise_for_status()
        # The 'response' field in Ollama's output contains the actual JSON string
        return response.json().get('response', '{}')
    except requests.exceptions.RequestException as e:
        msg = f"Failed to connect to Ollama service at {full_api_url}. Is it running?"
        logger.error(msg, exc_info=True)
        raise QueryParserError(msg) from e


class QueryParser:
    """
    Parses natural language queries into structured spatial analysis requests.
    """
    
    # --- FIX 2: A much more robust system prompt with rules and negative examples ---
    SYSTEM_PROMPT = """You are a highly specialized AI assistant for a geospatial analysis system. Your ONLY task is to convert a user's query into a structured JSON object.

**CRITICAL RULES:**
1.  Your output MUST be a single, raw JSON object. DO NOT provide any other text, explanations, or markdown like ```json ... ```.
2.  If the user's query contains a clear, real-world geographic location and a specific analysis task, create a standard JSON object according to the required schema.
3.  **If the query lacks a clear, real-world location, you MUST return this specific JSON error object:** `{"error": "Missing Location", "message": "I can't perform a geospatial analysis without a location. Please specify a city or region."}`
4.  **If the query is a generic command or greeting (e.g., 'hello', 'what can you do', 'load the data'), you MUST return this specific JSON error object:** `{"error": "Non-Geospatial Query", "message": "This does not seem to be a geospatial task. Please ask me to find or analyze something in a specific location."}`

**EXAMPLE 1 (Good Query):**
User Query: "Find suitable locations for a community park in Potsdam that is within 800m of residential areas."
Your JSON Response:
{
  "target": "community park",
  "location": "Potsdam, Germany",
  "constraints": [{"feature_type": "residential", "relationship": "within", "distance_meters": 800}]
}

**EXAMPLE 2 (Bad Query - Generic Command):**
User Query: "Load and analyze the sample data"
Your JSON Response:
{"error": "Non-Geospatial Query", "message": "This does not seem to be a geospatial task. Please ask me to find or analyze something in a specific location."}

**EXAMPLE 3 (Bad Query - No Location):**
User Query: "Find a place to build a hospital near a school."
Your JSON Response:
{"error": "Missing Location", "message": "I can't perform a geospatial analysis without a location. Please specify a city or region."}

Now, apply these rules strictly. Convert the following user query into a single JSON object.
"""

    def parse(self, user_query: str) -> ParsedQuery:
        """
        Parses a natural language query into a structured ParsedQuery object.
        
        Raises:
            InvalidQueryError: If the user's query is non-geospatial or lacks a location.
            QueryParserError: If there is a system-level error (LLM connection, bad JSON).
        """
        full_prompt = f"{self.SYSTEM_PROMPT}\n\nUser Query: \"{user_query}\""
        logger.info(f"Parsing query: '{user_query}'")
        
        try:
            llm_response = call_llm(full_prompt)
            logger.info(f"Raw LLM Response: {llm_response}")
            parsed_json = json.loads(llm_response)

            # --- FIX 3: Check for the error key before validation ---
            if "error" in parsed_json:
                error_message = parsed_json.get("message", "The query is not a valid geospatial request.")
                logger.warning(f"LLM identified an invalid query: {error_message}")
                # Raise the specific exception for invalid user queries
                raise InvalidQueryError(error_message)

            # If no error key, proceed with Pydantic validation
            return ParsedQuery(**parsed_json)
            
        # --- Catch the specific InvalidQueryError and re-raise it ---
        except InvalidQueryError:
            # This allows the calling application to handle user errors gracefully
            raise
        
        # --- These exceptions now represent system/LLM failures ---
        except (json.JSONDecodeError, ValidationError) as e:
            msg = (f"LLM response could not be parsed or validated. This is a system-level issue. "
                   f"Response was: '{llm_response}'. Details: {e}")
            logger.error(msg, exc_info=True)
            raise QueryParserError(msg) from e
            
        # Re-raise network or config errors from call_llm
        except QueryParserError: 
            raise

        except Exception as e:
            msg = f"An unexpected error occurred during parsing: {e}"
            logger.error(msg, exc_info=True)
            raise QueryParserError(msg) from e

# --- Main block for testing ---
# This block demonstrates how to use the QueryParser and handle the different exceptions.
if __name__ == '__main__':
    # To run this, ensure the OLLAMA_BASE_URL environment variable is set, or it will default.
    # For example:
    # $ export OLLAMA_BASE_URL="http://192.168.1.10:11434"
    # $ python3 ./src/core/planners/query_parser.py

    parser = QueryParser()
    test_queries = [
        # Valid queries
        "Find suitable locations for a new hospital in Berlin, Germany, that is near a major highway but not within 500m of another hospital.",
        "Show me industrial zones in Munich.",
        # Invalid queries that the LLM should reject
        "Find a park near my house.", # Missing a clear location
        "Hello, what can you do?", # Non-geospatial query
    ]

    print("=== AI-GIS Query Parser Demo (Live LLM) ===")
    print("This test connects to a live Ollama instance.")
    print("Ensure Ollama is running with the 'mistral' model (ollama pull mistral).\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"--- Test Query {i} ---\nInput: '{query}'")
        try:
            result = parser.parse(query)
            print(f"Status: SUCCESS\nResult:\n{result.model_dump_json(indent=2)}")
        # Catch the specific error types to show different handling
        except InvalidQueryError as e:
            print(f"Status: REJECTED (User Error)\n  Reason: {e}")
        except QueryParserError as e:
            print(f"Status: FAILED (System Error)\n  Reason: {e}")
        print("\n" + "="*50 + "\n")