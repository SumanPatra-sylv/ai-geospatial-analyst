#!/usr/bin/env python3
"""
AI-GIS Workflow Generator (LLM-Powered Version)
Generates correct, executable spatial analysis workflows from parsed queries using a large language model.
"""

# Step 1: Add New Imports
import json
import os
import jinja2
import requests
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from pprint import pprint


# This try/except block is preserved for standalone testing.
try:
    from core.planners.query_parser import ParsedQuery, SpatialConstraint, SpatialRelationship
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
    # SYSTEM_PROMPT Updated to include a section for RAG guidance
    SYSTEM_PROMPT = """You are an expert GIS Workflow Planner. Your sole task is to generate a single, valid JSON object that conforms **perfectly** to the schema provided below. Do not add any commentary or introductory text. Your output must begin with `{` and end with `}`.

**1. Required JSON Output Schema:**
Your entire output MUST be a single JSON object with exactly two keys: "reasoning" and "plan".
- `reasoning`: A single string explaining the step-by-step thought process. Use newline characters (`\\n`) for line breaks within this single string.
- `plan`: A list of dictionaries, where each dictionary is a step in the plan.

**2. Available Operations for the `plan` list:**
Each dictionary in the `plan` list must have an "operation" key and corresponding parameters. Use ONLY the following operations:
- `load_osm_data`: { "operation": "load_osm_data", "location": "string", "output_layer": "string" }
- `filter_by_category`: { "operation": "filter_by_category", "input_layer": "string", "category": "string", "output_layer": "string" }
- `buffer`: { "operation": "buffer", "input_layer": "string", "distance": int_meters, "output_layer": "string" }
- `intersect`: { "operation": "intersect", "input_layers": ["layer1", "layer2"], "output_layer": "string" }
- `difference`: { "operation": "difference", "input_layers": ["layer_to_keep", "layer_to_remove"], "output_layer": "string" }
- `rename_layer`: { "operation": "rename_layer", "input_layer": "string", "output_layer": "final_result" }

**3. GUIDANCE FROM KNOWLEDGE BASE:**
Based on past successful workflows and known patterns, here is some expert guidance. Consider this strongly when forming your plan:
{{ guidance_from_rag }}

**4. User's Request Analysis:**
This is the parsed user request you must create a workflow for:
{{ parsed_query_json }}

**Your Task:**
Based on the user's request AND the expert guidance, provide a single JSON object that strictly follows the schema defined in section 1 and uses only the operations from section 2.
"""

    # Method signature updated to accept guidance
    def generate_workflow(self, parsed_query: ParsedQuery, guidance_from_rag: str) -> Dict[str, Any]:
        """
        Uses an LLM to generate a reasoned workflow plan from a parsed query and expert guidance.
        """
        # 1. Prepare variables for the prompt template
        parsed_query_json = parsed_query.model_dump_json(indent=2)
        
        # 2. Format the final prompt using Jinja2
        template = jinja2.Environment().from_string(self.SYSTEM_PROMPT)
        prompt = template.render(
            parsed_query_json=parsed_query_json,
            guidance_from_rag=guidance_from_rag # <-- Pass the new variable here
        )

        # 3. Make the LLM call
        try:
            llm_response = self._make_llm_call(prompt)

            # 4. Validate the response using our Pydantic model
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
    
    # Sample guidance for the test
    sample_guidance = "For 'near' queries, use a buffer then an intersect. For 'far from' queries, use a buffer then a difference. Always load the main location first."

    generator = WorkflowGenerator()

    try:
        print("✅ Generating workflow for sample query with expert guidance...")
        print("=" * 60)

        # Call the new workflow generator with the guidance
        generation_result = generator.generate_workflow(
            parsed_query=sample_query,
            guidance_from_rag=sample_guidance
        )

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