#!/usr/bin/env python3
"""
Query Parser for AI-GIS Analyst (Refined Version)

This module provides functionality to parse natural language queries into structured
spatial analysis requests using Pydantic models and a real LLM-based extraction.
"""

import json
import logging
from enum import Enum
from typing import List, Optional, Union
import os
import jinja2
import requests

from pydantic import BaseModel, Field, ValidationError, field_validator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This exception is for system/network errors
class QueryParserError(Exception):
    """Custom exception for system/network errors during query parsing."""
    pass

# --- Pydantic Models (Updated) ---
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
    constraints: Optional[List[SpatialConstraint]] = Field(default=None, description="Optional list of spatial constraints")
    # This field is new, added to match the upgraded prompt
    summary_required: bool = Field(..., description="True if the user asks for a summary, count, or list")

    @field_validator('target', mode='before')
    @classmethod
    def normalize_target(cls, v):
        """
        ENHANCEMENT: Support multi-target queries by converting lists to comma-separated string.
        E.g., ["hospital", "metro_station", "drinking_fountain"] -> "hospital, metro_station, drinking_fountain"
        
        This allows the LLM to extract multiple targets, which we then process as a unified query.
        Single targets pass through unchanged.
        """
        if isinstance(v, list):
            # Join multiple targets with comma for unified processing
            return ', '.join([str(item).strip() for item in v])
        elif isinstance(v, str):
            return v.strip()
        else:
            return str(v).strip()

    @field_validator('constraints', mode='before')
    @classmethod
    def set_constraints_default(cls, v):
        """
        FIX: Bulletproof protection against None/Null from LLM.
        If the LLM returns "constraints": null, this converts it to [].
        This prevents "'NoneType' object is not iterable" crashes in workflow_generator.py
        when code tries to iterate: [c.feature_type for c in parsed_query.constraints]
        """
        if v is None:
            return []
        return v
    
    @field_validator('location', mode='before')
    @classmethod
    def normalize_location(cls, v):
        """
        Normalize location strings to help with geocoding.
        E.g., "Westbengal" -> "West Bengal" for better Nominatim matching
        This is backwards-compatible - already-correct locations pass through unchanged.
        """
        if not v:
            return v
        
        # Common location fixes (handles typos, missing spaces, shorthand)
        fixes = {
            # India
            "westbengal": "West Bengal",
            "west bengal": "West Bengal",
            "westbengal, india": "West Bengal, India",
            "west bengal, india": "West Bengal, India",
            "maharashtra": "Maharashtra",
            "karnataka": "Karnataka",
            "tamilnadu": "Tamil Nadu",
            "tamil nadu": "Tamil Nadu",
            "uttarpradesh": "Uttar Pradesh",
            "uttar pradesh": "Uttar Pradesh",
            "rajasthan": "Rajasthan",
            
            # Major Indian Cities
            "mumbai": "Mumbai, India",
            "delhi": "Delhi, India",
            "bangalore": "Bangalore, India",
            "bengaluru": "Bangalore, India",
            "kolkata": "Kolkata, India",
            "pune": "Pune, India",
            "hyderabad": "Hyderabad, India",
            "jaipur": "Jaipur, India",
            "ahmedabad": "Ahmedabad, India",
            "lucknow": "Lucknow, India",
            
            # UK & Europe (Central/West variants)
            "london": "London, UK",
            "central london": "London, UK",
            "west london": "London, UK",
            "east london": "London, UK",
            "paris": "Paris, France",
            "central paris": "Paris, France",
            "berlin": "Berlin, Germany",
            "central berlin": "Berlin, Germany",
            "amsterdam": "Amsterdam, Netherlands",
            "prague": "Prague, Czech Republic",
            "budapest": "Budapest, Hungary",
            
            # USA
            "newyork": "New York, USA",
            "new york": "New York, USA",
            "nyc": "New York, USA",
            "losangeles": "Los Angeles, USA",
            "los angeles": "Los Angeles, USA",
            "sanfrancisco": "San Francisco, USA",
            "san francisco": "San Francisco, USA",
            "chicago": "Chicago, USA",
            "boston": "Boston, USA",
            "seattle": "Seattle, USA",
            "denver": "Denver, USA",
            
            # Asia
            "tokyo": "Tokyo, Japan",
            "bangkok": "Bangkok, Thailand",
            "singapore": "Singapore",
            "hongkong": "Hong Kong",
            "hong kong": "Hong Kong",
            "shanghai": "Shanghai, China",
            "beijing": "Beijing, China",
            "seoul": "Seoul, South Korea",
            
            # Australia
            "sydney": "Sydney, Australia",
            "melbourne": "Melbourne, Australia",
            "brisbane": "Brisbane, Australia",
            "perth": "Perth, Australia"
        }
        
        v_lower = str(v).lower().strip()
        if v_lower in fixes:
            normalized = fixes[v_lower]
            logger.debug(f"Location normalized: '{v}' → '{normalized}'")
            return normalized
        
        return v


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
    
    # --- ACTION 1: Upgraded Few-Shot SYSTEM_PROMPT ---
    SYSTEM_PROMPT = """You are an expert GIS query analyst. Your task is to convert a user's natural language query into a structured, machine-readable JSON object.

The JSON output MUST conform to the following Pydantic model structure:
- `target`: (string) The main feature the user wants to find.
- `location`: (string) The geographic area for the analysis.
- `constraints`: (list of objects) An optional list of spatial conditions. Each constraint has:
  - `feature_type`: (string) The feature type of the constraint.
  - `relationship`: (string) One of ["within", "not within", "near", "far from"].
  - `distance_meters`: (integer) An optional distance for "near" or "far from".
- `summary_required`: (boolean) Set to true if the user asks for a summary, count, or list.

--- EXAMPLES ---

[EXAMPLE 1]
User Query: "Show me parks near residential areas in Berlin"
JSON Output:
{
  "target": "park",
  "location": "Berlin, Germany",
  "constraints": [
    {
      "feature_type": "residential",
      "relationship": "near",
      "distance_meters": null
    }
  ],
  "summary_required": true
}

[EXAMPLE 2]
User Query: "Find schools in London that are not within 500m of a highway"
JSON Output:
{
  "target": "school",
  "location": "London, UK",
  "constraints": [
    {
      "feature_type": "highway",
      "relationship": "not within",
      "distance_meters": 500
    }
  ],
  "summary_required": false
}

[EXAMPLE 3]
User Query: "Summarize landuse in Potsdam"
JSON Output:
{
  "target": "landuse",
  "location": "Potsdam, Germany",
  "constraints": null,
  "summary_required": true
}
--- END EXAMPLES ---

Now, analyze the following user query and provide ONLY the JSON output.

User Query: "{{user_query}}"
"""

    # --- ACTION 2: Simplified 'parse' method ---
    def parse(self, user_query: str) -> ParsedQuery:
        """
        Parses a natural language query into a structured ParsedQuery object
        using a high-reliability few-shot prompt.
        """
        template = jinja2.Environment().from_string(self.SYSTEM_PROMPT)
        prompt = template.render(user_query=user_query)
        logger.info(f"Parsing query: '{user_query}'")
        
        try:
            # We expect the LLM to get this right on the first try now.
            llm_response_str = call_llm(prompt)
            logger.info(f"Raw LLM Response: {llm_response_str}")

            parsed_data = json.loads(llm_response_str)
            return ParsedQuery(**parsed_data)

        except (ValidationError, json.JSONDecodeError) as e:
            logger.error(f"LLM output failed validation or parsing. Error: {e}\nOutput: {llm_response_str}", exc_info=True)
            raise QueryParserError("I'm having trouble understanding the structure of your request. Please try rephrasing.") from e
        
        # Re-raise network or config errors from call_llm to preserve the specific message
        except QueryParserError: 
            raise

        except Exception as e:
            logger.error(f"An unexpected error occurred during query parsing: {e}", exc_info=True)
            raise QueryParserError("An unexpected error occurred while I was trying to understand your query.") from e

# --- Main block for testing (Simplified) ---
# This block demonstrates how to use the new, more reliable QueryParser.
if __name__ == '__main__':
    # To run this, ensure the OLLAMA_BASE_URL environment variable is set, or it will default.
    # For example:
    # $ export OLLAMA_BASE_URL="http://192.168.1.10:11434"
    # $ python3 ./src/core/planners/query_parser.py

    parser = QueryParser()
    test_queries = [
        # Valid queries matching the examples
        "Show me parks near residential areas in Berlin",
        "Find schools in London that are not within 500m of a highway",
        "Summarize landuse in Potsdam",
        # Another valid query
        "Find hospitals in Paris far from industrial zones",
    ]

    print("=== AI-GIS Query Parser Demo (Few-Shot Prompt) ===")
    print("This test connects to a live Ollama instance.")
    print("Ensure Ollama is running with the 'mistral' model (ollama pull mistral).\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"--- Test Query {i} ---\nInput: '{query}'")
        try:
            result = parser.parse(query)
            print(f"Status: SUCCESS\nResult:\n{result.model_dump_json(indent=2)}")
        # With the new prompt, we only expect system/parsing errors, not user-level errors.
        except QueryParserError as e:
            print(f"Status: FAILED (System or Parsing Error)\n  Reason: {e}")
        print("\n" + "="*50 + "\n")