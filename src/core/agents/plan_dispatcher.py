#!/usr/bin/env python3
"""
Plan Dispatcher Agent
Responsible for translating high-level LLM strategies into validated, executable workflow plans.
Acts as a Quality Control gate.
"""

from typing import List, Dict, Any, Set
from src.core.agents.strategy_planner import LLMStrategyResponse
from src.gis.tools.definitions import TOOL_REGISTRY


class PlanDispatcher:
    """
    An agent responsible for translating a high-level LLM strategy into a
    validated, executable workflow plan. It acts as a Quality Control gate.
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

    def translate_and_validate(self, strategy: LLMStrategyResponse) -> List[Dict]:
        """
        The main entry point for this agent.
        
        Args:
            strategy: The high-level strategy from the LLM
            
        Returns:
            List[Dict]: Validated, executable workflow plan
        """
        executable_plan = self._translate_strategy_to_plan(strategy)
        final_plan = self._enforce_final_step(executable_plan)
        return final_plan
    
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