#!/usr/bin/env python3
"""
AI-GIS Workflow Generator (LLM-Powered Version)
Generates correct, executable spatial analysis workflows from parsed queries using a large language model.
"""

# Step 1: Add New Imports
import json
import os
import requests
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from pprint import pprint

# This try/except block is preserved for standalone testing.
try:
    from src.core.planners.query_parser import ParsedQuery, SpatialConstraint, SpatialRelationship
except ImportError:
    # Fallback for standalone testing
    from enum import Enum
    
    class SpatialRelationship(Enum):
        WITHIN = "within"
        NOT_WITHIN = "not within"
        NEAR = "near"
        FAR_FROM = "far from"
    
    class SpatialConstraint(BaseModel):
        feature_type: str
        relationship: SpatialRelationship
        distance_meters: int = None
    
    class ParsedQuery(BaseModel):
        target: str
        location: str
        constraints: List[SpatialConstraint]


# Step 2: Define New Pydantic Models
class LLMWorkflowResponse(BaseModel):
    """Defines the expected structured response from the LLM for workflow generation."""
    reasoning: str = Field(..., description="A step-by-step explanation of the thought process behind the plan.")
    plan: List[Dict] = Field(..., description="The final, machine-readable list of workflow steps.")


class WorkflowGenerator:
    """
    Generates a logical sequence of spatial operations from a ParsedQuery using an LLM.
    """
    # Step 3: Create the "True CoT" System Prompt
    SYSTEM_PROMPT = """You are a master GIS Workflow Planner. Your job is to convert a user's request into a detailed, step-by-step reasoning process and a final, machine-readable JSON execution plan.

    **AVAILABLE OPERATIONS:**
    - `load_osm_data`: {{ "location": "string", "output_layer": "string" }} - Always the first step.
    - `filter_by_category`: {{ "input_layer": "string", "category": "string", "output_layer": "string" }}
    - `buffer`: {{ "input_layer": "string", "distance": "int_meters", "output_layer": "string" }}
    - `intersect`: {{ "input_layers": ["layer1", "layer2"], "output_layer": "string" }}
    - `difference`: {{ "input_layers": ["layer_to_keep", "layer_to_remove"], "output_layer": "string" }}
    - `rename_layer`: {{ "input_layer": "string", "output_layer": "final_result" }} - Usually the last step.

    **YOUR TASK:**
    1.  First, in a `reasoning` field, think step-by-step. Explain the logic for your plan. Describe why each operation is necessary to satisfy the user's constraints.
    2.  Second, based on your reasoning, construct the `plan` as a JSON list of step dictionaries.

    **RESPONSE FORMAT:**
    Your output MUST be a single, valid JSON object with exactly two keys: "reasoning" and "plan".

    **USER REQUEST ANALYSIS:**
    {parsed_query_json}

    Now, provide your JSON response.
    """

    # Step 4: Overhaul the generate_workflow Method
    def generate_workflow(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """
        Uses an LLM to generate a reasoned workflow plan from a parsed query.
        """
        # 1. Convert the Pydantic object to a nicely formatted JSON string for the prompt
        parsed_query_json = parsed_query.model_dump_json(indent=2)

        # 2. Format the final prompt
        prompt = self.SYSTEM_PROMPT.format(parsed_query_json=parsed_query_json)

        # 3. Make the LLM call
        try:
            llm_response = self._make_llm_call(prompt)
            
            # 4. Validate the response using our new Pydantic model
            validated_response = LLMWorkflowResponse(**llm_response)
            
            # 5. Return the validated data in the desired dictionary format
            return {
                "reasoning": validated_response.reasoning,
                "plan": validated_response.plan
            }

        except Exception as e:
            # Handle LLM call failures or validation errors
            print(f"Error during workflow generation: {e}") # Or use a proper logger
            raise  # Re-raise the exception to be handled by the worker

    def _make_llm_call(self, prompt: str) -> Dict:
        """Helper to make the LLM call and return a JSON dictionary."""
        # This logic can be refactored into a shared LLMClient later.
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if not ollama_base_url:
            raise ValueError("OLLAMA_BASE_URL environment variable is not configured.")
        full_api_url = f"{ollama_base_url}/api/generate"
        payload = {"model": "mistral", "prompt": prompt, "stream": False, "format": "json"}
        
        response = requests.post(full_api_url, json=payload, timeout=120)
        response.raise_for_status()
        
        response_data = response.json()
        return json.loads(response_data.get('response', '{}'))


# Updated test block to work with the new LLM-based implementation
if __name__ == '__main__':
    # This test now requires a running Ollama instance with the 'mistral' model.
    print("--- Running WorkflowGenerator Standalone Test ---")
    
    # Define a sample parsed query
    sample_query = ParsedQuery(
        target='restaurant',
        location='downtown San Francisco, CA',
        constraints=[
            SpatialConstraint(
                feature_type='park',
                relationship=SpatialRelationship.NEAR,
                distance_meters=500
            ),
            SpatialConstraint(
                feature_type='highway',
                relationship=SpatialRelationship.FAR_FROM,
                distance_meters=200
            )
        ]
    )

    generator = WorkflowGenerator()
    
    try:
        print("✅ Generating workflow for sample query...")
        print("=" * 60)
        
        # Call the new workflow generator
        generation_result = generator.generate_workflow(sample_query)

        # Print the results in the new format
        print("\n--- AI Planner's Reasoning ---")
        print(generation_result.get("reasoning", "No reasoning provided."))
        
        print("\n--- Generated Machine-Readable Plan ---")
        pprint(generation_result.get("plan", []), width=100)
        
        print("\n" + "=" * 60)
        print("✅ Test completed successfully.")

    except requests.exceptions.ConnectionError as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED: Could not connect to the Ollama service.")
        print(f"   Please ensure Ollama is running and the OLLAMA_BASE_URL is set correctly.")
        print(f"   Error: {e}")
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ An unexpected error occurred during the test: {e}")