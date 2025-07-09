#!/usr/bin/env python3
"""
AI-GIS Workflow Generator (Data-Grounded Version)
Generates correct, executable spatial analysis workflows by interpreting a factual
DataRealityReport from a DataScout agent, following the "Discover, then Plan" model.
"""

import json
import os
import requests
import jinja2
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field

# Local project imports
from src.core.agents.data_scout import DataScout, DataRealityReport
from src.core.agents.schemas import ClarificationAsk
from src.gis.tools.definitions import TOOL_REGISTRY
from src.core.planners.query_parser import ParsedQuery

class ToolCallIntent(BaseModel):
    """Represents the LLM's intent to use a specific tool."""
    tool_name: str = Field(..., description="The exact name of the tool from the AVAILABLE TOOLS list.")
    parameters: Dict[str, Any] = Field(..., description="A dictionary of conceptual parameters for the tool.")

class LLMStrategyResponse(BaseModel):
    """The LLM's high-level strategic plan as a sequence of tool-use intents."""
    reasoning: str = Field(..., description="High-level explanation of the overall strategy.")
    tool_calls: List[ToolCallIntent] = Field(..., description="The sequence of tools to be called to execute the strategy.")


class WorkflowGenerator:
    """
    Generates a logical sequence of spatial operations by interpreting a factual
    DataRealityReport provided by a DataScout agent. This implements the
    "Discover, then Plan" architectural pattern.
    """

    SYSTEM_PROMPT = """You are a master GIS Strategist. You have been given a factual Data Reality Report from a DataScout agent. Your task is to devise a high-level strategy to answer the user's request by selecting the correct sequence of tools.

**Your Goal:** Decide WHICH tools to use and in WHAT order. You will NOT generate the final code, only the intent to use a tool.

**RULES:**
1.  Your output MUST be a JSON object with two keys: "reasoning" and "tool_calls".
2.  "tool_calls" must be a list of objects, where each object represents one tool you intend to use.
3.  Each tool call object MUST have "tool_name" and "parameters".
4.  For "parameters", use conceptual layer names that are human-readable (e.g., "schools_data", "parks_data"). The executor will handle the real variables. Use the `canonical_name` from the report for locations.
5.  Use intuitive parameter names - the system will translate them to internal formats.

**AVAILABLE TOOLS (Your Palette):**
{{ tool_documentation }}

**Data Reality Report (Ground Truth):**
{{ data_reality_report }}

---
**OUTPUT FORMAT EXAMPLE (This is what you must generate):**

```json
{
  "reasoning": "First, I will load the school and park data because the report shows they are available. Then, to find features 'near' each other, I will create a buffer around the parks. Finally, I will select only the schools that fall within these park buffers to get the final result.",
  "tool_calls": [
    {
      "tool_name": "load_osm_data",
      "parameters": {
        "area_name": "Potsdam, Brandenburg, Deutschland",
        "tags": {"amenity": "school"},
        "conceptual_name": "all_schools_in_area"
      }
    },
    {
      "tool_name": "load_osm_data",
      "parameters": {
        "area_name": "Potsdam, Brandenburg, Deutschland",
        "tags": {"leisure": "park"},
        "conceptual_name": "all_parks_in_area"
      }
    },
    {
      "tool_name": "buffer",
      "parameters": {
        "input_layer": "all_parks_in_area",
        "distance_meters": 500,
        "conceptual_name": "park_buffer_zones"
      }
    },
    {
      "tool_name": "spatial_join",
      "parameters": {
        "left_layer": "all_schools_in_area",
        "right_layer": "park_buffer_zones",
        "predicate": "intersects",
        "conceptual_name": "final_schools_near_parks"
      }
    }
  ]
}
```

TASK: Generate the strategic JSON plan based only on the facts in the Data Reality Report.
"""

    # Parameter alias mapping - centralized for easy maintenance
    PARAMETER_ALIASES = {
        "input_layer": "layer_name",
        "distance_meters": "distance",
        "left_layer": "left_layer_name", 
        "right_layer": "right_layer_name",
        "clip_layer": "clip_layer_name",
        "buffer_layer": "layer_name",
        "target_layer": "layer_name",
        "source_layer": "layer_name",
        "geometry_layer": "layer_name",
        "points_layer": "layer_name",
        "polygons_layer": "layer_name",
        "lines_layer": "layer_name",
        "layer1": "layer1_name",
        "layer2": "layer2_name",
        "first_layer": "layer1_name",
        "second_layer": "layer2_name",
        "overlay_layer": "overlay_layer_name",
        "union_layer": "union_layer_name",
        "intersect_layer": "intersect_layer_name",
        "difference_layer": "difference_layer_name"
    }

    # Layer reference parameters - these should be resolved to actual layer names
    LAYER_REFERENCE_PARAMS = {
        "layer_name", "left_layer_name", "right_layer_name", "clip_layer_name",
        "layer1_name", "layer2_name", "overlay_layer_name", "union_layer_name",
        "intersect_layer_name", "difference_layer_name"
    }

    def __init__(self, data_scout: DataScout):
        """
        Initializes the WorkflowGenerator with its required agents using dependency injection.

        Args:
            data_scout: An initialized instance of the DataScout agent.
        """
        self.data_scout = data_scout
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("OLLAMA_MODEL", "mistral")
        self.timeout = 120

    def generate_workflow(self, parsed_query: ParsedQuery, guidance_from_rag: str = "") -> Dict[str, Any]:
        """
        Orchestrates the new "Discover, then Plan" workflow. The DataScout now handles all validation.
        """
        print("🛰️  Data Scout: Starting discovery and validation phase...")

        entities_to_probe = [parsed_query.target] + [c.feature_type for c in parsed_query.constraints]
        
        try:
            report = self.data_scout.generate_data_reality_report(
                location_string=parsed_query.location,
                entities_to_probe=entities_to_probe
            )
            
            # The scout now returns a failure object or None, so we check for that
            if isinstance(report, ClarificationAsk):
                return self._generate_failure_plan(f"Ambiguity found for entity '{report.original_entity}': {report.message}")
            if not report:
                return self._generate_failure_plan(f"Location '{parsed_query.location}' is invalid or could not be geocoded.")

        except Exception as e:
            return self._generate_failure_plan(f"A critical error occurred during the data scouting phase: {e}")

        # Check if the primary target entity exists and has data in the report.
        primary_target_found = any(
            probe.original_entity == parsed_query.target and probe.count > 0
            for probe in report.probe_results
        )

        if not primary_target_found:
            return self._generate_failure_plan(
                f"The primary target '{parsed_query.target}' has 0 features at the specified location. Cannot generate a plan."
            )

        # --- If we get here, everything is validated ---
        prompt_context = self._format_report_for_prompt(report)
        tool_docs = self._generate_tool_documentation_string()
        print("✍️  Strategist: Devising high-level plan based on Data Reality Report...")
        
        template = jinja2.Environment(loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True).from_string(self.SYSTEM_PROMPT)
        prompt = template.render(data_reality_report=prompt_context, tool_documentation=tool_docs)

        try:
            llm_response = self._make_llm_call(prompt)
            strategy = LLMStrategyResponse(**llm_response)

            print("🔧 Dispatcher: Translating strategy into executable plan...")
            executable_plan = self._translate_strategy_to_plan(strategy)
            
            # Enforce the final step AFTER the plan is fully translated
            final_plan = self._enforce_final_step(executable_plan)

            return {
                "reasoning": strategy.reasoning,
                "plan": final_plan,
                "complexity_assessment": "Assessed by Strategist",
                "error_handling": []
            }
        except ValueError as ve:
            return self._generate_failure_plan(f"The LLM's strategy was flawed or its response was malformed. Error: {ve}")
        except Exception as e:
            return self._generate_failure_plan(f"A critical error occurred during planning. Error: {e}")

    def _translate_parameter_aliases(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translates common LLM parameter names to the system's canonical names.
        Enhanced with comprehensive alias mapping and validation.
        """
        translated_params = {}
        
        for key, value in params.items():
            # Skip conceptual_name as it's handled separately
            if key == "conceptual_name":
                translated_params[key] = value
                continue
                
            # Use the canonical name if the key is an alias, otherwise use the original key
            canonical_key = self.PARAMETER_ALIASES.get(key, key)
            translated_params[canonical_key] = value
            
        return translated_params

    def _translate_strategy_to_plan(self, strategy: LLMStrategyResponse) -> List[Dict]:
        """
        Translates the high-level strategy into an executable plan.
        Enhanced with alias translation and improved validation.
        """
        executable_plan = []
        conceptual_map = {}

        for i, tool_call in enumerate(strategy.tool_calls):
            tool_name = tool_call.tool_name
            
            # --- ALIAS TRANSLATION STEP ---
            # First, translate any known aliases to our internal canonical names.
            params = self._translate_parameter_aliases(tool_call.parameters.copy())
            
            # Check if tool exists before proceeding
            if tool_name not in TOOL_REGISTRY:
                raise ValueError(f"Strategy is flawed. It proposed a non-existent tool: '{tool_name}'")

            # --- ENHANCED PARAMETER VALIDATION ---
            tool_definition = TOOL_REGISTRY[tool_name]
            expected_params = {p.name for p in tool_definition.parameters}
            provided_params = set(params.keys())

            # Account for special dispatcher parameters that aren't in the tool definition
            dispatcher_special_params = {"conceptual_name"}
            valid_llm_params = provided_params - dispatcher_special_params

            # Check for unknown parameters
            if not valid_llm_params.issubset(expected_params):
                unknown_params = valid_llm_params - expected_params
                raise ValueError(
                    f"Strategy is flawed. For tool '{tool_name}' (step {i+1}), "
                    f"it provided unknown parameters: {unknown_params}. "
                    f"Expected parameters: {expected_params}"
                )

            # Check for required parameters (those without default values)
            required_params = {p.name for p in tool_definition.parameters if not hasattr(p, 'default') or p.default is None}
            missing_params = required_params - valid_llm_params

            if missing_params:
                raise ValueError(
                    f"Strategy is flawed. For tool '{tool_name}' (step {i+1}), "
                    f"it's missing required parameters: {missing_params}"
                )

            # Validate parameter types and values
            self._validate_parameter_types(tool_name, params, i+1)

            # --- ENHANCED LAYER RESOLUTION ---
            # Resolve conceptual input layers to real layer names
            for param_key, param_value in params.items():
                if param_key in self.LAYER_REFERENCE_PARAMS:
                    conceptual_name = param_value
                    if conceptual_name not in conceptual_map:
                        raise ValueError(
                            f"Strategy is flawed. Tool call {i+1} ('{tool_name}') tried to use "
                            f"conceptual layer '{conceptual_name}' which has not been created yet. "
                            f"Available layers: {list(conceptual_map.keys())}"
                        )
                    params[param_key] = conceptual_map[conceptual_name]
            
            # Define the output layer name with better naming
            output_conceptual_name = params.pop("conceptual_name", f"{tool_name}_output")
            real_output_layer = self._generate_layer_name(output_conceptual_name, i+1)
            conceptual_map[output_conceptual_name] = real_output_layer
            
            # Build the final, executable step
            executable_step = {
                "operation": tool_name,
                "parameters": {**params, "output_layer": real_output_layer}
            }
            executable_plan.append(executable_step)

        return executable_plan

    def _generate_layer_name(self, conceptual_name: str, step_number: int) -> str:
        """
        Generates a clean, predictable layer name from conceptual name and step number.
        """
        # Clean the conceptual name
        clean_name = conceptual_name.lower().replace(' ', '_').replace('-', '_')
        # Remove any non-alphanumeric characters except underscores
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
        # Ensure it doesn't start with a number
        if clean_name and clean_name[0].isdigit():
            clean_name = f"layer_{clean_name}"
        # Fallback if name is empty
        if not clean_name:
            clean_name = "layer"
        
        return f"{clean_name}_{step_number}"

    def _validate_parameter_types(self, tool_name: str, params: Dict[str, Any], step_num: int) -> None:
        """
        Enhanced parameter type validation with more comprehensive checks.
        """
        # Distance validation for buffer operations
        if tool_name == "buffer" and "distance" in params:
            distance = params["distance"]
            if not isinstance(distance, (int, float)) or distance <= 0:
                raise ValueError(
                    f"Strategy is flawed. For tool '{tool_name}' (step {step_num}), "
                    f"'distance' must be a positive number, got: {distance}"
                )

        # OSM tags validation
        if tool_name == "load_osm_data" and "tags" in params:
            tags = params["tags"]
            if not isinstance(tags, dict):
                raise ValueError(
                    f"Strategy is flawed. For tool '{tool_name}' (step {step_num}), "
                    f"'tags' must be a dictionary, got: {type(tags).__name__}"
                )
            # Validate tag structure
            for key, value in tags.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError(
                        f"Strategy is flawed. For tool '{tool_name}' (step {step_num}), "
                        f"'tags' must contain string key-value pairs, got: {key}={value}"
                    )

        # Spatial predicate validation
        if "predicate" in params:
            valid_predicates = {
                "intersects", "within", "contains", "overlaps", 
                "touches", "crosses", "disjoint", "equals"
            }
            predicate = params["predicate"]
            if predicate not in valid_predicates:
                raise ValueError(
                    f"Strategy is flawed. For tool '{tool_name}' (step {step_num}), "
                    f"'predicate' must be one of {valid_predicates}, got: '{predicate}'"
                )

        # Area name validation
        if "area_name" in params:
            area_name = params["area_name"]
            if not isinstance(area_name, str) or not area_name.strip():
                raise ValueError(
                    f"Strategy is flawed. For tool '{tool_name}' (step {step_num}), "
                    f"'area_name' must be a non-empty string, got: {area_name}"
                )

        # Boolean parameter validation
        boolean_params = {"dissolve_result", "include_geom"}
        for param_name in boolean_params:
            if param_name in params:
                param_value = params[param_name]
                if not isinstance(param_value, bool):
                    raise ValueError(
                        f"Strategy is flawed. For tool '{tool_name}' (step {step_num}), "
                        f"'{param_name}' must be a boolean, got: {param_value}"
                    )

    def _generate_failure_plan(self, reason: str) -> Dict[str, Any]:
        """Creates a standardized failure response when planning is not possible."""
        print(f"❌ PLANNING FAILED: {reason}")
        return {
            "reasoning": f"PLANNING FAILED: {reason}",
            "plan": [],
            "complexity_assessment": "Unplannable",
            "error_handling": ["Process halted due to fatal error in discovery or planning phase."]
        }

    def _format_report_for_prompt(self, report: DataRealityReport) -> str:
        """Formats the rich DataRealityReport into a simple, clear string for the LLM."""
        lines = [
            f"- Verified Location: {report.location.canonical_name}",
            "- Data Availability Report:"
        ]
        for result in report.probe_results:
            lines.append(f"  - For tags `{result.tag}`: Found {result.count} features.")

        if report.recommendations:
            lines.append("- Scout's Recommendations:")
            for rec in report.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)

    def _generate_tool_documentation_string(self) -> str:
        """
        Enhanced tool documentation generation with better formatting and examples.
        """
        doc_lines = []
        for tool in TOOL_REGISTRY.values():
            # Build parameter documentation with enhanced information
            param_docs = []
            for param in tool.parameters:
                param_type = getattr(param, 'type', 'Any')
                is_required = not hasattr(param, 'default') or param.default is None
                required_marker = " (required)" if is_required else " (optional)"
                
                # Add common aliases in documentation
                aliases = [alias for alias, canonical in self.PARAMETER_ALIASES.items() if canonical == param.name]
                alias_info = f" [aliases: {', '.join(aliases)}]" if aliases else ""
                
                param_docs.append(f"{param.name}: {param_type}{required_marker}{alias_info}")
            
            params_str = ", ".join(param_docs)
            doc_lines.append(f"- `{tool.operation_name}({params_str})`: {tool.description}")
        
        return "\n".join(doc_lines)

    def _enforce_final_step(self, plan: List[Dict]) -> List[Dict]:
        """
        Enhanced final step enforcement with better error handling.
        """
        if not plan:
            return []
        
        # Remove any existing rename_layer steps at the end
        while plan and plan[-1].get("operation") == "rename_layer":
            plan.pop()
        
        if not plan:
            return []

        # Get the output layer from the last step's parameters
        last_step = plan[-1]
        last_output_layer = last_step.get("parameters", {}).get("output_layer")
        
        if not last_output_layer:
            print("❌ Warning: Final step of the translated plan has no output layer to rename.")
            return plan

        # Create the final step with the correct structure
        final_step = {
            "operation": "rename_layer",
            "parameters": {
                "input_layer": last_output_layer,  # Use correct parameter name
                "output_layer": "final_result"
            }
        }
        return plan + [final_step]

    def _make_llm_call(self, prompt: str) -> Dict:
        """
        Enhanced LLM call with better error handling and response validation.
        """
        if not self.ollama_base_url:
            raise ValueError("OLLAMA_BASE_URL environment variable is not configured.")
        
        full_api_url = f"{self.ollama_base_url}/api/generate"
        payload = { 
            "model": self.model_name, 
            "prompt": prompt, 
            "stream": False, 
            "format": "json", 
            "options": {
                "temperature": 0.1, 
                "top_p": 0.9,
                "num_predict": 3072,  # Increased for complex responses
                "stop": ["```", "---"]  # Stop tokens to prevent over-generation
            } 
        }

        try:
            response = requests.post(full_api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            response_data = response.json()
            
            if 'response' not in response_data:
                raise ValueError("Invalid response format from LLM: 'response' key missing.")
            
            raw_llm_json_str = response_data['response'].strip()
            
            # Clean up common LLM response artifacts
            if raw_llm_json_str.startswith('```json'):
                raw_llm_json_str = raw_llm_json_str[7:]
            if raw_llm_json_str.endswith('```'):
                raw_llm_json_str = raw_llm_json_str[:-3]
            raw_llm_json_str = raw_llm_json_str.strip()
            
            print("--- RAW LLM JSON RESPONSE ---")
            print(raw_llm_json_str)
            print("---------------------------")
            
            # Parse and validate the JSON structure
            parsed_response = json.loads(raw_llm_json_str)
            
            # Enhanced structure validation
            if not isinstance(parsed_response, dict):
                raise ValueError("LLM response must be a JSON object")
            
            required_keys = {"reasoning", "tool_calls"}
            if not all(key in parsed_response for key in required_keys):
                missing_keys = required_keys - set(parsed_response.keys())
                raise ValueError(f"LLM response missing required keys: {missing_keys}")
            
            if not isinstance(parsed_response["tool_calls"], list):
                raise ValueError("'tool_calls' must be a list")
            
            if not parsed_response["tool_calls"]:
                raise ValueError("'tool_calls' cannot be empty")
            
            # Validate each tool call structure
            for i, tool_call in enumerate(parsed_response["tool_calls"]):
                if not isinstance(tool_call, dict):
                    raise ValueError(f"Tool call {i+1} must be a dictionary")
                if "tool_name" not in tool_call or "parameters" not in tool_call:
                    raise ValueError(f"Tool call {i+1} missing required keys 'tool_name' or 'parameters'")
                if not isinstance(tool_call["parameters"], dict):
                    raise ValueError(f"Tool call {i+1} 'parameters' must be a dictionary")
            
            return parsed_response
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Could not connect to Ollama at {self.ollama_base_url}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"LLM request timed out after {self.timeout} seconds")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"HTTP error from LLM service: {e}")


if __name__ == '__main__':
    import time
    from src.core.planners.query_parser import SpatialConstraint, SpatialRelationship

    print("🚀 Running Enhanced Data-Grounded AI-GIS Workflow Generator Test")
    print("=" * 70)

    # --- Dependency Injection in action ---
    data_scout_agent = DataScout()
    generator = WorkflowGenerator(data_scout=data_scout_agent)

    test_cases = [
        { 
            "name": "Simple School Query (Success)", 
            "query": ParsedQuery(
                target='school', 
                location='Kolkata, West Bengal, India', 
                constraints=[], 
                summary_required=True
            ) 
        },
        { 
            "name": "Complex Query with NEAR constraint (Success)", 
            "query": ParsedQuery( 
                target='hospital', 
                location='Pune, India', 
                constraints=[SpatialConstraint(
                    feature_type='park', 
                    relationship=SpatialRelationship.NEAR, 
                    distance_meters=500
                )], 
                summary_required=True 
            ) 
        },
        { 
            "name": "Query with Ambiguous Term (Expected Failure)", 
            "query": ParsedQuery(
                target='hangout spot', 
                location='Mumbai, India', 
                constraints=[], 
                summary_required=True
            ) 
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['name']}")
        print(f"   Query: Find '{test_case['query'].target}' in '{test_case['query'].location}'")
        print("-" * 50)

        start_time = time.time()
        try:
            result = generator.generate_workflow(parsed_query=test_case['query'])
            end_time = time.time()
            print(f"⏱️  Generation Time: {end_time - start_time:.2f} seconds")

            print(f"\n🧠 Reasoning: {result.get('reasoning', 'N/A')}")
            print("\n📋 Generated Plan:")
            if result.get("plan"):
                for j, step in enumerate(result.get("plan", []), 1):
                    print(f"  Step {j}: {step}")
            else:
                print("  No plan generated, as expected for a failed or unplannable request.")
        
        except Exception as e:
            end_time = time.time()
            print(f"⏱️  Generation Time: {end_time - start_time:.2f} seconds")
            print(f"❌ Test failed with exception: {e}")

        print("\n" + "=" * 70)