#!/usr/bin/env python3
"""
AI-GIS Workflow Generator (Enhanced LLM-Powered Version)
Generates correct, executable spatial analysis workflows from parsed queries using a large language model
with Chain-of-Thought reasoning for complex geospatial tasks.
"""

import json
import os
import requests
import jinja2
from typing import List, Dict, Any, Optional
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
        INTERSECTS = "intersects"
        CONTAINS = "contains"

    class SpatialConstraint(BaseModel):
        feature_type: str
        relationship: SpatialRelationship
        distance_meters: Optional[int] = None

    class ParsedQuery(BaseModel):
        target: str
        location: str
        constraints: List[SpatialConstraint]


class LLMWorkflowResponse(BaseModel):
    """Defines the expected structured response from the LLM for workflow generation."""
    reasoning: str = Field(..., description="Chain-of-Thought explanation of the spatial analysis approach.")
    plan: List[Dict] = Field(..., description="Sequential workflow steps with proper data flow.")
    complexity_assessment: str = Field(..., description="Assessment of task complexity and potential challenges.")
    error_handling: List[str] = Field(..., description="Anticipated error scenarios and mitigation strategies.")


class WorkflowGenerator:
    """
    Generates a logical sequence of spatial operations from a ParsedQuery using an LLM
    with enhanced Chain-of-Thought reasoning for complex geospatial workflows.
    """
    
    # Enhanced SYSTEM_PROMPT with Chain-of-Thought reasoning and better structure
    SYSTEM_PROMPT = """You are an expert AI Geospatial Workflow Planner with deep knowledge of GIS operations and spatial analysis. Your task is to convert natural language queries into precise, executable geospatial workflows using Chain-of-Thought reasoning.

**RESPONSE FORMAT:**
You MUST respond with a single JSON object containing four keys: "reasoning", "plan", "complexity_assessment", and "error_handling".

**COMPLETE RESPONSE EXAMPLE:**
```json
{
  "reasoning": "CHAIN-OF-THOUGHT ANALYSIS: The user wants to find restaurants near parks but far from highways in downtown San Francisco. Let me break this down step by step: 1) First, I need to load OSM data for the location to get all spatial features. 2) Then filter for restaurants (amenity=restaurant). 3) Next, I need to handle the 'near parks' constraint by filtering for parks (leisure=park), creating a 500m buffer around them. 4) For the 'far from highways' constraint, I'll filter highways, buffer them by 200m, then exclude restaurants within this buffer. 5) Finally, I'll find restaurants that are within park buffers but outside highway buffers using spatial intersection and difference operations.",
  "plan": [
    {
      "operation": "load_osm_data",
      "location": "downtown San Francisco, CA",
      "output_layer": "all_osm_features"
    },
    {
      "operation": "filter_by_attribute",
      "input_layer": "all_osm_features",
      "key": "amenity",
      "value": "restaurant",
      "output_layer": "restaurants"
    },
    {
      "operation": "filter_by_attribute",
      "input_layer": "all_osm_features",
      "key": "leisure",
      "value": "park",
      "output_layer": "parks"
    },
    {
      "operation": "buffer",
      "input_layer": "parks",
      "distance_meters": 500,
      "output_layer": "park_buffers"
    },
    {
      "operation": "clip",
      "input_layer": "restaurants",
      "clip_layer": "park_buffers",
      "output_layer": "restaurants_near_parks"
    },
    {
      "operation": "rename_layer",
      "input_layer": "restaurants_near_parks",
      "output_layer": "final_result"
    }
  ],
  "complexity_assessment": "Medium complexity - involves multiple spatial relationships and buffer operations. The 'far from highways' constraint is challenging as it requires spatial exclusion operations not directly supported by available tools.",
  "error_handling": [
    "Check for CRS compatibility between all layers",
    "Validate that location exists in OSM data",
    "Handle cases where no restaurants or parks are found",
    "Note: 'far from highways' constraint cannot be fully implemented with available tools - this limitation is documented in reasoning"
  ]
}
```

**AVAILABLE TOOLS/OPERATIONS:**
You MUST only use operations from this list. Each operation's input_layer must be the output_layer of a previous step.

- `load_osm_data(location, output_layer)`: Loads OpenStreetMap data for a location. Returns all feature types (points, lines, polygons).
- `filter_by_attribute(input_layer, key, value, output_layer)`: Filters features where attribute 'key' equals 'value'. Use value='*' for any value.
- `buffer(input_layer, distance_meters, output_layer)`: Creates buffer zones around features.
- `clip(input_layer, clip_layer, output_layer)`: Clips input_layer to boundaries of clip_layer (spatial intersection).
- `dissolve(input_layer, by, output_layer)`: Merges features sharing the same 'by' attribute value.
- `rename_layer(input_layer, output_layer)`: Renames a layer. Use as final step to create 'final_result'.

**CRITICAL INSTRUCTIONS:**

1. **CHAIN-OF-THOUGHT REASONING:** Start your reasoning with "CHAIN-OF-THOUGHT ANALYSIS:" and break down the problem step-by-step. Consider spatial relationships, data requirements, and operation sequence.

2. **OSM DATA STRUCTURE:** Common OSM tags include:
   - amenity: restaurant, school, hospital, bank, cafe, pub
   - leisure: park, playground, sports_centre, swimming_pool
   - landuse: residential, commercial, industrial, forest, farmland
   - highway: primary, secondary, residential, footway, cycleway
   - building: yes, house, commercial, industrial
   - natural: water, forest, grassland, wetland

3. **SPATIAL RELATIONSHIPS:** 
   - "near" → use buffer + clip operations
   - "within" → use clip operation
   - "far from" → difficult with available tools (note limitations)
   - "intersects" → use clip operation

4. **DATA FLOW:** Ensure each step's input_layer matches a previous step's output_layer. Plan the sequence carefully.

5. **ERROR ANTICIPATION:** Consider common GIS issues like CRS mismatches, empty results, geometry errors.

6. **COMPLEXITY ASSESSMENT:** Evaluate if the task requires advanced operations not available in your toolset.

**ENHANCED GUIDELINES:**
- For flood risk mapping: Focus on elevation, water bodies, and drainage patterns
- For site suitability: Consider multiple criteria and constraint overlays
- For land cover analysis: Use landuse and natural tags effectively
- Always consider the geographic context and scale of analysis

**Knowledge Base Guidance:**
{{ guidance_from_rag }}

**User Request Details:**
{{ parsed_query }}

**TASK:** Generate a complete workflow plan with Chain-of-Thought reasoning that demonstrates expert-level spatial analysis thinking.
"""

    def __init__(self):
        """Initialize the WorkflowGenerator with enhanced configuration."""
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("OLLAMA_MODEL", "mistral")
        self.max_retries = 3
        self.timeout = 180  # Increased timeout for complex reasoning

    def generate_workflow(self, parsed_query: ParsedQuery, guidance_from_rag: str = "") -> Dict[str, Any]:
        """
        Uses an LLM to generate a reasoned workflow plan with Chain-of-Thought reasoning.
        
        Args:
            parsed_query: The parsed user query with spatial constraints
            guidance_from_rag: Additional guidance from knowledge base
            
        Returns:
            Dict containing reasoning, plan, complexity assessment, and error handling
        """
        parsed_query_json = parsed_query.model_dump_json(indent=2)
        
        # Enhanced prompt rendering with Jinja2
        template = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True
        ).from_string(self.SYSTEM_PROMPT)
        
        prompt = template.render(
            parsed_query=parsed_query_json,
            guidance_from_rag=guidance_from_rag if guidance_from_rag else "No specific guidance available."
        )

        # Retry logic for robust LLM interaction
        for attempt in range(self.max_retries):
            try:
                print(f"🔄 Attempting workflow generation (attempt {attempt + 1}/{self.max_retries})...")
                llm_response = self._make_llm_call(prompt)
                validated_response = LLMWorkflowResponse(**llm_response)
                
                # Additional validation of the workflow
                self._validate_workflow_logic(validated_response.plan)
                
                print("✅ Workflow generation successful!")
                return validated_response.model_dump()
                
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    print("❌ All attempts failed. Generating fallback workflow...")
                    return self._generate_fallback_workflow(parsed_query)
                
    def _make_llm_call(self, prompt: str) -> Dict:
        """Enhanced LLM call with better error handling and configuration."""
        if not self.ollama_base_url:
            raise ValueError("OLLAMA_BASE_URL environment variable is not configured.")
        
        full_api_url = f"{self.ollama_base_url}/api/generate"
        
        # Enhanced payload with better generation parameters
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,  # Lower temperature for more consistent outputs
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": 2048  # Increased for detailed reasoning
            }
        }

        try:
            response = requests.post(full_api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            response_data = response.json()
            if 'response' not in response_data:
                raise ValueError("Invalid response format from LLM")
                
            return json.loads(response_data['response'])
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Could not connect to Ollama at {self.ollama_base_url}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"LLM request timed out after {self.timeout} seconds")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from LLM: {e}")

    def _validate_workflow_logic(self, plan: List[Dict]) -> None:
        """Validates the logical flow of the generated workflow."""
        if not plan:
            raise ValueError("Empty workflow plan generated")
            
        # Check if first step loads data
        if plan[0].get("operation") != "load_osm_data":
            raise ValueError("Workflow must start with load_osm_data operation")
            
        # Check if last step renames to final_result
        if plan[-1].get("operation") != "rename_layer" or plan[-1].get("output_layer") != "final_result":
            raise ValueError("Workflow must end with rename_layer to 'final_result'")
            
        # Validate data flow between steps
        available_layers = set()
        for i, step in enumerate(plan):
            operation = step.get("operation")
            
            if operation == "load_osm_data":
                available_layers.add(step.get("output_layer"))
            elif "input_layer" in step:
                input_layer = step.get("input_layer")
                if input_layer not in available_layers:
                    raise ValueError(f"Step {i+1}: input_layer '{input_layer}' not available")
                available_layers.add(step.get("output_layer"))
            else:
                available_layers.add(step.get("output_layer"))

    def _generate_fallback_workflow(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """Generates a basic fallback workflow when LLM fails."""
        return {
            "reasoning": f"FALLBACK WORKFLOW: Generated a basic workflow to find {parsed_query.target} in {parsed_query.location}. This is a simplified approach due to LLM generation failure.",
            "plan": [
                {
                    "operation": "load_osm_data",
                    "location": parsed_query.location,
                    "output_layer": "all_data"
                },
                {
                    "operation": "filter_by_attribute",
                    "input_layer": "all_data",
                    "key": "amenity",
                    "value": parsed_query.target,
                    "output_layer": "filtered_results"
                },
                {
                    "operation": "rename_layer",
                    "input_layer": "filtered_results",
                    "output_layer": "final_result"
                }
            ],
            "complexity_assessment": "Low complexity fallback workflow",
            "error_handling": ["Basic error handling - check data availability"]
        }

    def analyze_query_complexity(self, parsed_query: ParsedQuery) -> str:
        """Analyzes and returns the complexity level of the query."""
        complexity_score = 0
        
        # Base complexity
        complexity_score += len(parsed_query.constraints)
        
        # Check for complex spatial relationships
        complex_relationships = [SpatialRelationship.FAR_FROM, SpatialRelationship.NOT_WITHIN]
        for constraint in parsed_query.constraints:
            if constraint.relationship in complex_relationships:
                complexity_score += 2
            if constraint.distance_meters and constraint.distance_meters > 1000:
                complexity_score += 1
                
        if complexity_score <= 2:
            return "Low"
        elif complexity_score <= 5:
            return "Medium"
        else:
            return "High"


# Enhanced standalone testing
if __name__ == '__main__':
    print("🚀 Running Enhanced AI-GIS Workflow Generator Test")
    print("=" * 70)

    # Test cases covering different complexity levels
    test_cases = [
        {
            "name": "Simple Restaurant Query",
            "query": ParsedQuery(
                target='restaurant',
                location='downtown Mumbai, India',
                constraints=[]
            ),
            "guidance": "Basic amenity search in urban area."
        },
        {
            "name": "Complex Flood Risk Analysis",
            "query": ParsedQuery(
                target='residential',
                location='coastal Kerala, India',
                constraints=[
                    SpatialConstraint(
                        feature_type='water',
                        relationship=SpatialRelationship.NEAR,
                        distance_meters=100
                    ),
                    SpatialConstraint(
                        feature_type='forest',
                        relationship=SpatialRelationship.FAR_FROM,
                        distance_meters=500
                    )
                ]
            ),
            "guidance": "For flood risk analysis, consider proximity to water bodies and elevation. Use buffer operations for distance constraints."
        },
        {
            "name": "Site Suitability Analysis",
            "query": ParsedQuery(
                target='commercial',
                location='Bangalore, Karnataka, India',
                constraints=[
                    SpatialConstraint(
                        feature_type='highway',
                        relationship=SpatialRelationship.NEAR,
                        distance_meters=200
                    ),
                    SpatialConstraint(
                        feature_type='residential',
                        relationship=SpatialRelationship.WITHIN,
                        distance_meters=1000
                    )
                ]
            ),
            "guidance": "Site suitability requires multi-criteria analysis. Consider accessibility, proximity to target demographics, and infrastructure availability."
        }
    ]

    generator = WorkflowGenerator()

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['name']}")
        print("-" * 50)
        
        try:
            # Analyze complexity
            complexity = generator.analyze_query_complexity(test_case['query'])
            print(f"📊 Query Complexity: {complexity}")
            
            # Generate workflow
            result = generator.generate_workflow(
                parsed_query=test_case['query'],
                guidance_from_rag=test_case['guidance']
            )

            print(f"\n🧠 Chain-of-Thought Reasoning:")
            print(result.get("reasoning", "No reasoning provided."))

            print(f"\n📋 Generated Workflow Plan:")
            for j, step in enumerate(result.get("plan", []), 1):
                print(f"  Step {j}: {step}")

            print(f"\n⚖️  Complexity Assessment:")
            print(result.get("complexity_assessment", "No assessment provided."))

            print(f"\n🛡️  Error Handling:")
            for error_strategy in result.get("error_handling", []):
                print(f"  - {error_strategy}")

        except Exception as e:
            print(f"❌ Test failed: {e}")

        print("\n" + "=" * 70)

    print("🎯 Enhanced AI-GIS Workflow Generator testing completed!")
    print("💡 This system demonstrates Chain-of-Thought reasoning for complex geospatial workflows.")