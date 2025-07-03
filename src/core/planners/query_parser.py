#!/usr/bin/env python3
"""
Query Parser for AI-GIS Analyst (Refined Version)

This module provides functionality to parse natural language queries into structured
spatial analysis requests using Pydantic models and LLM-based extraction.
"""

import json
import logging
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Custom Exception for Better Error Handling ---
class QueryParserError(Exception):
    """Custom exception for errors during query parsing."""
    pass

# --- Pydantic Models with Enum for Robustness ---
class SpatialRelationship(str, Enum):
    """Enumeration for allowed spatial relationships to enforce consistency."""
    WITHIN = "within"
    NOT_WITHIN = "not within"
    NEAR = "near"
    FAR_FROM = "far from"

class SpatialConstraint(BaseModel):
    """Represents a spatial constraint for GIS analysis."""
    feature_type: str = Field(..., description="Type of spatial feature (e.g., 'residential', 'hospital', 'park')")
    relationship: SpatialRelationship = Field(..., description="The spatial relationship to the feature.")
    distance_meters: Optional[int] = Field(None, description="Optional distance in meters for proximity constraints.")

class ParsedQuery(BaseModel):
    """Represents a validated and structured spatial query."""
    target: str = Field(..., description="The main feature or facility being analyzed.")
    location: str = Field(..., description="Geographic location for the analysis.")
    constraints: List[SpatialConstraint]

# --- Enhanced Mock LLM for Realistic Testing ---
def call_llm(prompt: str) -> str:
    """
    Placeholder function for LLM API calls with dynamic mock responses.
    
    This function simulates an LLM by returning a different hard-coded JSON
    based on keywords in the prompt, making testing more realistic.
    """
    logger.info("LLM called (returning dynamic mock response)")
    prompt_lower = prompt.lower()

    if "potsdam" in prompt_lower:
        response = {
            "target": "community park",
            "location": "Potsdam, Germany",
            "constraints": [
                {"feature_type": "residential", "relationship": "within", "distance_meters": 800},
                {"feature_type": "industrial", "relationship": "not within", "distance_meters": 1500}
            ]
        }
    elif "munich" in prompt_lower:
        response = {
            "target": "school",
            "location": "Munich, Germany",
            "constraints": [
                {"feature_type": "residential", "relationship": "near"}
            ]
        }
    elif "invalid_relationship" in prompt_lower: # For testing validation
        response = {
            "target": "warehouse",
            "location": "Hamburg, Germany",
            "constraints": [
                {"feature_type": "port", "relationship": "adjacent to", "distance_meters": 2000}
            ]
        }
    else: # Default response
        response = {
            "target": "hospital",
            "location": "Frankfurt, Germany",
            "constraints": []
        }
        
    return json.dumps(response, indent=2)


class QueryParser:
    """
    Parses natural language queries into structured spatial analysis requests.
    """
    
    # Define the system prompt as a class attribute for efficiency
    SYSTEM_PROMPT = """You are an expert spatial analyst... 
    (The rest of your detailed prompt text remains unchanged here)
    
    Valid spatial relationships are: "within", "not within", "near", "far from".
    ...
    Now parse this query:"""

    def parse(self, user_query: str) -> ParsedQuery:
        """
        Parses a natural language query into a structured ParsedQuery object.
        
        Raises:
            QueryParserError: If the query cannot be parsed or validated.
        """
        full_prompt = f"{self.SYSTEM_PROMPT}\n\nUser Query: \"{user_query}\""
        
        logger.info(f"Parsing query: '{user_query}'")
        
        try:
            # 1. Call the LLM
            llm_response = call_llm(full_prompt)
            
            # 2. Parse the JSON string
            parsed_json = json.loads(llm_response)
            logger.info("Successfully parsed LLM JSON response.")
            
            # 3. Validate and deserialize using Pydantic models (with Enum)
            parsed_query = ParsedQuery(**parsed_json)
            logger.info(f"Query validation successful for target: '{parsed_query.target}'")
            
            return parsed_query
            
        except json.JSONDecodeError as e:
            msg = "LLM returned invalid JSON. Could not be decoded."
            logger.error(f"{msg} Response: {llm_response}")
            raise QueryParserError(msg) from e
            
        except ValidationError as e:
            msg = "LLM response failed validation against the required schema."
            logger.error(f"{msg}\nDetails: {e}")
            raise QueryParserError(msg) from e
            
        except Exception as e:
            msg = f"An unexpected error occurred during query parsing: {e}"
            logger.error(msg)
            raise QueryParserError(msg) from e


if __name__ == '__main__':
    parser = QueryParser()
    
    test_queries = [
        "Find suitable locations for a community park in Potsdam...",
        "Show me potential sites for a new school in Munich...",
        "Where can I build a hospital in Frankfurt?",
        "Find a warehouse with an invalid_relationship in Hamburg" # Test query for validation
    ]
    
    print("=== AI-GIS Query Parser Demo ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"--- Test Query {i} ---\nInput: {query[:70]}...")
        
        try:
            result = parser.parse(query)
            print(f"Status: SUCCESS")
            print(f"  Target: {result.target}")
            print(f"  Location: {result.location}")
            for constraint in result.constraints:
                dist = f"({constraint.distance_meters}m)" if constraint.distance_meters else "(no distance)"
                # Accessing the enum's value for clean printing
                print(f"  Constraint: {constraint.relationship.value.title()} {constraint.feature_type} {dist}")
            
        except QueryParserError as e:
            print(f"Status: FAILED")
            print(f"  Error: {e}")
        
        print("\n" + "="*50 + "\n")