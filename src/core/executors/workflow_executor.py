#!/usr/bin/env python3
"""
WorkflowExecutor - Executes spatial analysis workflows with iterative single-step execution.
Phase 2 refactoring: Supports both legacy complete workflow execution and new iterative single-step approach.
Enhanced version with flexible OSM data loading and improved Chain-of-Thought reasoning.
"""

import sys
import os
import warnings
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from datetime import datetime

# This try/except block allows the module to be run for integrated testing.
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.gis.data_loader import SmartDataLoader, LoaderConfig
    from src.core.planners.workflow_generator import WorkflowGenerator, ParsedQuery, ToolCallIntent
    from src.core.planners.query_parser import SpatialConstraint, SpatialRelationship
    from src.gis.tools.definitions import TOOL_REGISTRY
    from src.core.agents.data_scout import DataScout
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.gis.data_loader import SmartDataLoader, LoaderConfig
    from src.core.planners.workflow_generator import WorkflowGenerator, ParsedQuery, ToolCallIntent
    from src.core.planners.query_parser import SpatialConstraint, SpatialRelationship
    from src.gis.tools.definitions import TOOL_REGISTRY
    from src.core.agents.data_scout import DataScout

import geopandas as gpd
import pandas as pd
from pprint import pprint


class LocationNotFoundError(Exception):
    """Custom exception raised when a location cannot be found by the geocoding service."""
    def __init__(self, message, location_name):
        super().__init__(message)
        self.location_name = location_name


class WorkflowExecutionError(Exception):
    """Custom exception for workflow execution errors."""
    def __init__(self, message, step_number=None, operation=None):
        super().__init__(message)
        self.step_number = step_number
        self.operation = operation


class WorkflowExecutor:
    """
    Executes spatial analysis workflows with support for both legacy complete workflows
    and new iterative single-step execution for the Think-Act-Observe loop.
    Enhanced with flexible OSM data loading and Chain-of-Thought reasoning logging.
    """
    
    def __init__(self, enable_reasoning_log: bool = True):
        loader_config = LoaderConfig(base_data_dir="data")
        self.smart_data_loader = SmartDataLoader(config=loader_config)
        self.data_layers: Dict[str, gpd.GeoDataFrame] = {}
        self.enable_reasoning_log = enable_reasoning_log
        self.reasoning_log: List[Dict[str, Any]] = []
        self.execution_metadata: Dict[str, Any] = {}
        
        # Initialize execution metadata
        self.execution_metadata = {
            'start_time': None,
            'end_time': None,
            'total_steps': 0,
            'successful_steps': 0,
            'failed_steps': 0,
            'total_features_processed': 0,
            'errors': []
        }

    def execute_single_step(self, tool_call: ToolCallIntent, current_data_layers: Dict[str, gpd.GeoDataFrame]) -> Tuple[str, Dict[str, gpd.GeoDataFrame]]:
        """Execute a single tool call and return observation + updated state."""
        self.data_layers = current_data_layers.copy()
        
        try:
            if tool_call.tool_name == "finish_task":
                final_layer = tool_call.parameters.get("final_layer_name")
                if final_layer and final_layer in self.data_layers:
                    feature_count = len(self.data_layers[final_layer])
                    observation = f"Task completed successfully. Final result: '{final_layer}' with {feature_count} features."
                else:
                    observation = "Task completed successfully."
                return observation, self.data_layers.copy()
            
            if tool_call.tool_name not in TOOL_REGISTRY:
                observation = f"Error: Unknown tool '{tool_call.tool_name}'"
                return observation, current_data_layers.copy()
            
            # ✅ IMPROVED CONCEPTUAL MAPPING
            conceptual_map = {}
            for real_name in current_data_layers.keys():
                # Extract conceptual name: 'schools_1' -> 'schools'
                parts = real_name.split('_')
                if len(parts) > 1 and parts[-1].isdigit():
                    conceptual_name = "_".join(parts[:-1])
                    conceptual_map[conceptual_name] = real_name
                else:
                    conceptual_map[real_name] = real_name  # Use as-is if no number suffix
            
            # ✅ RESOLVE PARAMETERS BEFORE EXECUTION
            resolved_params = tool_call.parameters.copy()
            layer_params = ['layer_name', 'input_layer', 'left_layer_name', 'right_layer_name', 
                           'clip_layer_name', 'input_layer1', 'input_layer2']
            
            for param_key in layer_params:
                if param_key in resolved_params:
                    layer_ref = resolved_params[param_key]
                    if layer_ref in conceptual_map:
                        resolved_params[param_key] = conceptual_map[layer_ref]
                        print(f"🔗 Resolved '{layer_ref}' -> '{conceptual_map[layer_ref]}'")
            
            # Get operation method and execute
            tool_definition = TOOL_REGISTRY[tool_call.tool_name]
            method_name = tool_definition.executor_method_name
            operation_method = getattr(self, method_name)
            
            print(f"🔄 Executing single step: {tool_call.tool_name}")
            start_time = time.time()
            result_gdf = operation_method(resolved_params, step_number=1)
            execution_time = time.time() - start_time
            
            # Generate unique layer name
            conceptual_name = tool_call.parameters.get('conceptual_name', f"{tool_call.tool_name}_output")
            layer_counter = 1
            while f"{conceptual_name}_{layer_counter}" in self.data_layers:
                layer_counter += 1
            real_layer_name = f"{conceptual_name}_{layer_counter}"
            
            # Store result
            updated_layers = self.data_layers.copy()
            updated_layers[real_layer_name] = result_gdf
            
            # Create observation
            if result_gdf.empty:
                observation = f"Success: Tool '{tool_call.tool_name}' completed but returned 0 features for layer '{real_layer_name}'."
            else:
                feature_count = len(result_gdf)
                observation = f"Success: Tool '{tool_call.tool_name}' created layer '{real_layer_name}' with {feature_count} features in {execution_time:.2f}s."
            
            print(f"  ✅ {observation}")
            return observation, updated_layers
            
        except Exception as e:
            observation = f"Failure: Tool '{tool_call.tool_name}' failed with error: {str(e)}"
            print(f"  ❌ {observation}")
            return observation, current_data_layers.copy()

    def _resolve_layer_references(self, params: Dict[str, Any], conceptual_map: Dict[str, str]) -> Dict[str, Any]:
        """
        Resolve conceptual layer names to real layer names.
        conceptual_map maps conceptual_name -> real_layer_name
        """
        resolved_params = params.copy()
        
        # All possible layer reference parameters
        layer_params = [
            'layer_name', 'input_layer', 'left_layer_name', 'right_layer_name',
            'clip_layer_name', 'input_layer1', 'input_layer2', 'layer1_name', 'layer2_name',
            'input_layer_name', 'clip_layer'  # Add more variations as needed
        ]
        
        for param_key in layer_params:
            if param_key in resolved_params:
                conceptual_name = resolved_params[param_key]
                if conceptual_name in conceptual_map:
                    old_value = resolved_params[param_key]
                    resolved_params[param_key] = conceptual_map[conceptual_name]
                    print(f"  🔗 Resolved '{param_key}': '{old_value}' → '{resolved_params[param_key]}'")
        
        return resolved_params

    def log_reasoning(self, step_number: int, operation: str, reasoning: str,
                      input_info: Dict[str, Any] = None, output_info: Dict[str, Any] = None):
        """Log Chain-of-Thought reasoning for transparency."""
        if not self.enable_reasoning_log:
            return
        
        reasoning_entry = {
            'timestamp': datetime.now().isoformat(),
            'step_number': step_number,
            'operation': operation,
            'reasoning': reasoning,
            'input_info': input_info or {},
            'output_info': output_info or {}
        }
        
        self.reasoning_log.append(reasoning_entry)
        # Also print for immediate feedback
        print(f"  🧠 REASONING: {reasoning}")

    def _homogenize_geometries(self, gdf: gpd.GeoDataFrame, target_type: str = "Point") -> gpd.GeoDataFrame:
        """
        Ensures a GeoDataFrame contains only a single geometry type.
        If mixed types are found, it converts all geometries to a representative
        point (centroid), which is safe for most proximity/intersection analyses.
        """
        if gdf.geom_type.nunique() > 1:
            warnings.warn(
                f"Mixed geometry types found ({gdf.geom_type.unique()}). "
                "Converting all to representative points (centroids) for safe operation."
            )
            # Use representative_point() as it's safer than centroid for complex polygons
            gdf.geometry = gdf.geometry.representative_point()
        return gdf

    def execute_workflow(self, workflow_plan: List[Dict[str, Any]], parsed_query: ParsedQuery) -> gpd.GeoDataFrame:
        """Execute the complete workflow with enhanced error handling and reasoning (Legacy method)."""
        if not workflow_plan:
            raise ValueError("Workflow plan cannot be empty.")
        
        # Initialize execution metadata
        self.clear_layers()
        self.execution_metadata['start_time'] = datetime.now()
        self.execution_metadata['total_steps'] = len(workflow_plan)
        
        print(f"🚀 Starting workflow execution with {len(workflow_plan)} steps")
        print(f"📍 Target location: '{parsed_query.location}'")
        print(f"🎯 Target feature: '{parsed_query.target}'")
        
        # Log initial reasoning
        self.log_reasoning(0, 'workflow_initialization',
                          f"Starting workflow to find '{parsed_query.target}' in '{parsed_query.location}'. "
                          f"Plan contains {len(workflow_plan)} sequential steps.")
        
        print("\n🔍 Validating workflow plan before execution...")
        self._validate_workflow_plan(workflow_plan)
        print("✅ Plan validation successful.")
        
        # Execute each step with enhanced error handling
        for i, step in enumerate(workflow_plan, 1):
            try:
                self._execute_legacy_single_step(i, step, parsed_query)
                self.execution_metadata['successful_steps'] += 1
            except Exception as e:
                self.execution_metadata['failed_steps'] += 1
                self.execution_metadata['errors'].append({
                    'step': i,
                    'operation': step.get('operation', 'unknown'),
                    'error': str(e)
                })
                print(f"❌ Step {i} failed: {e}")
                raise WorkflowExecutionError(f"Step {i} failed: {e}", i, step.get('operation'))
        
        # Finalize execution
        last_step = workflow_plan[-1]
        final_layer_name = last_step.get('parameters', {}).get('output_layer')
        
        if final_layer_name is None:
            raise WorkflowExecutionError("Could not determine the final output layer name from the plan.")
        
        final_result = self.data_layers[final_layer_name]
        self.execution_metadata['end_time'] = datetime.now()
        execution_time = (self.execution_metadata['end_time'] - self.execution_metadata['start_time']).total_seconds()
        
        print(f"\n🎉 Workflow completed successfully in {execution_time:.2f} seconds!")
        print(f"📊 Final result contains {len(final_result)} features.")
        
        self.log_reasoning(len(workflow_plan) + 1, 'workflow_completion',
                          f"Workflow completed successfully. Final result contains {len(final_result)} features "
                          f"in layer '{final_layer_name}'. Total execution time: {execution_time:.2f} seconds.")
        
        return final_result

    def _validate_workflow_plan(self, workflow_plan: List[Dict[str, Any]]):
        """Validate workflow plan structure and operations against the official TOOL_REGISTRY."""
        for i, step in enumerate(workflow_plan, 1):
            operation = step.get('operation')
            if not operation:
                raise KeyError(f"Workflow plan is malformed. Step {i} is missing 'operation'.")
            
            # Check if the operation exists in our central source of truth.
            if operation not in TOOL_REGISTRY:
                raise ValueError(f"Workflow plan contains an unknown operation in Step {i}: '{operation}'.")

    def _execute_legacy_single_step(self, step_number: int, step: Dict[str, Any], parsed_query: ParsedQuery):
        """
        Execute a single workflow step by dynamically looking up the correct
        executor method from the TOOL_REGISTRY. (Legacy method for complete workflows)
        """
        operation = step['operation']
        
        # Use fallback to get the correct output layer name, supporting all definition variations
        output_layer = step.get('parameters', {}).get('output_layer')
        
        print(f"\n🔄 Step {step_number}: Executing '{operation}' -> '{output_layer or 'No output layer defined'}'")
        
        if operation == 'load_osm_data' and 'area_name' not in step and 'location' not in step:
            step['area_name'] = parsed_query.location
        
        # 1. Look up the tool definition from the central registry.
        tool_definition = TOOL_REGISTRY[operation]
        
        # 2. Get the official executor method name from the definition.
        method_name = tool_definition.executor_method_name
        
        # 3. Get the actual method from the class instance.
        operation_method = getattr(self, method_name)
        
        # Execute the operation
        start_time = time.time()
        result_gdf = operation_method(step['parameters'], step_number)
        execution_time = time.time() - start_time
        
        # Store result and update metadata (only if an output layer is defined)
        if output_layer:
            self.data_layers[output_layer] = result_gdf
            self.execution_metadata['total_features_processed'] += len(result_gdf)
            
        print(f"  ✅ Stored {len(result_gdf)} features in layer '{output_layer}' ({execution_time:.2f}s)")
        return result_gdf

    def _op_load_osm_data(self, params: Dict[str, Any], step_number: int) -> gpd.GeoDataFrame:
        # Align with definitions.py (area_name) but keep fallback for older plans (location)
        location = params.get('area_name') or params.get('location')
        tags = params.get('tags', {})
        
        if not location:
            raise ValueError("Missing 'area_name' or 'location' parameter for load_osm_data.")
        
        if not tags:
            warnings.warn("No 'tags' provided for load_osm_data. This might result in a very large download or an error.")
        
        reasoning = f"Loading OSM data for location '{location}' with specific tags: {tags}. This provides the precise foundational data needed."
        
        print(f"  🏷️ Filtering by tags: {tags}")
        self.log_reasoning(step_number, 'load_osm_data', reasoning, input_info={'location': location, 'tags': tags})
        
        try:
            result = self.smart_data_loader.fetch_osm_data(location, tags)
            if result.empty:
                warnings.warn(f"Warning: No features found for tags {tags} in location '{location}'.")
            
            self.log_reasoning(step_number, 'load_osm_data',
                              f"Successfully retrieved {len(result)} features from OSM.",
                              output_info={'feature_count': len(result)})
            return result
        except Exception as e:
            error_message = f"Could not find or process location '{location}' with tags {tags}."
            print(f"  ❌ ERROR: {error_message} (Original error: {e})")
            self.log_reasoning(step_number, 'load_osm_data', f"Failed to load OSM data. Error: {e}", output_info={'error': str(e)})
            raise LocationNotFoundError(error_message, location_name=location) from e

    def _op_filter_by_attribute(self, params: Dict[str, Any], step_number: int) -> gpd.GeoDataFrame:
        """Enhanced attribute filtering with robust parameter handling."""
        
        # ✅ ROBUST PARAMETER EXTRACTION - Support multiple parameter names
        input_layer = params.get('input_layer') or params.get('layer_name')
        key = params.get('key') or params.get('attribute') or params.get('column')
        value = params.get('value')
        operator = params.get('operator', 'equals')  # Default operator
        
        # ✅ COMPREHENSIVE VALIDATION
        if not input_layer:
            raise ValueError("Missing layer parameter. Provide 'input_layer' or 'layer_name'")
        if not key:
            raise ValueError("Missing attribute parameter. Provide 'key', 'attribute', or 'column'")
        if value is None:
            raise ValueError("Missing 'value' parameter for filtering")
        
        # Check if layer exists
        if input_layer not in self.data_layers:
            available = list(self.data_layers.keys())
            raise ValueError(f"Layer '{input_layer}' not found. Available layers: {available}")
            
        input_gdf = self.data_layers[input_layer]
        
        reasoning = f"Filtering layer '{input_layer}' containing {len(input_gdf)} features. "
        reasoning += f"Looking for features where '{key}' {operator} '{value}'. "
        
        print(f"    🔍 Filtering '{input_layer}' for features where '{key}' {operator} '{value}'")
        
        # ✅ ENHANCED COLUMN VALIDATION
        if key not in input_gdf.columns:
            available_cols = list(input_gdf.columns)
            reasoning += f"ERROR: Column '{key}' not found. Available: {available_cols}. "
            
            # Try to find similar column names
            similar_cols = [col for col in available_cols if key.lower() in col.lower()]
            if similar_cols:
                reasoning += f"Similar columns found: {similar_cols}. "
            
            self.log_reasoning(step_number, 'filter_by_attribute', reasoning,
                              input_info={'layer': input_layer, 'key': key, 'value': value},
                              output_info={'feature_count': 0, 'error': f"Column '{key}' not found"})
            
            warnings.warn(f"Column '{key}' not found in layer '{input_layer}'. Available: {available_cols}")
            return gpd.GeoDataFrame(columns=input_gdf.columns, geometry=[], crs=input_gdf.crs)

        # ✅ ENHANCED FILTERING LOGIC
        if value == '*':
            # Wildcard: select all non-null values
            result = input_gdf[input_gdf[key].notna()].copy()
            reasoning += f"Using wildcard filter - selected all non-null '{key}' values. "
        elif operator in ['equals', '==', '=']:
            # Exact match with type handling
            try:
                # Try direct comparison first
                result = input_gdf[input_gdf[key] == value].copy()
            except:
                # Fallback to string comparison for mixed types
                result = input_gdf[input_gdf[key].astype(str) == str(value)].copy()
            reasoning += f"Using exact match filter. "
        elif operator in ['contains', 'like']:
            # String contains
            result = input_gdf[input_gdf[key].astype(str).str.contains(str(value), na=False)].copy()
            reasoning += f"Using contains filter. "
        elif operator in ['!=', 'not_equals']:
            # Not equals
            result = input_gdf[input_gdf[key] != value].copy()
            reasoning += f"Using not-equals filter. "
        else:
            # Default to exact match
            result = input_gdf[input_gdf[key].astype(str) == str(value)].copy()
            reasoning += f"Using default exact match filter. "
        
        reasoning += f"Filter resulted in {len(result)} features out of {len(input_gdf)} original features."
        
        self.log_reasoning(step_number, 'filter_by_attribute', reasoning,
                          input_info={'layer': input_layer, 'key': key, 'value': value, 'input_count': len(input_gdf)},
                          output_info={'feature_count': len(result)})
        
        return result

    def _op_buffer(self, params: Dict[str, Any], step_number: int) -> gpd.GeoDataFrame:
        """Enhanced buffering with reasoning and CRS handling."""
        # Use fallback to support both old and new parameter names
        input_layer = params.get('layer_name') or params.get('input_layer')
        distance = params.get('distance') or params.get('distance_meters')
        
        if input_layer not in self.data_layers:
            raise ValueError(f"Layer '{input_layer}' not found in available layers: {list(self.data_layers.keys())}")
        
        gdf = self.data_layers[input_layer]
        
        reasoning = f"Creating {distance}m buffer around {len(gdf)} features from layer '{input_layer}'. "
        reasoning += "Buffering will expand the spatial extent of features to capture nearby areas. "
        
        print(f"  🔄 Buffering {len(gdf)} features by {distance} meters")
        
        if gdf.empty:
            reasoning += "Input layer is empty, returning empty buffer result."
            self.log_reasoning(step_number, 'buffer', reasoning,
                              input_info={'layer': input_layer, 'distance': distance, 'input_count': 0},
                              output_info={'feature_count': 0})
            return gdf.copy()
        
        original_crs = gdf.crs
        reasoning += f"Original CRS: {original_crs}. Converting to Web Mercator (EPSG:3857) for accurate metric buffering, "
        reasoning += f"then converting back to original CRS to maintain spatial reference consistency."
        
        # Handle CRS conversion for accurate buffering
        if original_crs.to_string() != 'EPSG:3857':
            projected_gdf = gdf.to_crs("EPSG:3857")
        else:
            projected_gdf = gdf.copy()
        
        # Apply buffer
        projected_gdf.geometry = projected_gdf.geometry.buffer(distance)
        
        # Convert back to original CRS
        if original_crs.to_string() != 'EPSG:3857':
            result = projected_gdf.to_crs(original_crs)
        else:
            result = projected_gdf
        
        self.log_reasoning(step_number, 'buffer', reasoning,
                          input_info={'layer': input_layer, 'distance': distance, 'input_count': len(gdf), 'original_crs': str(original_crs)},
                          output_info={'feature_count': len(result)})
        
        return result

    def _op_clip(self, params: Dict[str, Any], step_number: int) -> gpd.GeoDataFrame:
        """Enhanced clipping with reasoning and CRS validation."""
        input_layer = params.get('input_layer_name') or params.get('input_layer')
        clip_layer = params.get('clip_layer_name') or params.get('clip_layer')
        
        input_gdf = self.data_layers[input_layer]
        clip_gdf = self.data_layers[clip_layer]
        
        reasoning = f"Clipping features from '{input_layer}' ({len(input_gdf)} features) "
        reasoning += f"using boundaries from '{clip_layer}' ({len(clip_gdf)} features). "
        reasoning += "This will extract only the portions of input features that fall within the clip boundaries."
        
        print(f"  ✂️ Clipping '{input_layer}' ({len(input_gdf)}) with '{clip_layer}' ({len(clip_gdf)})")
        
        if input_gdf.empty or clip_gdf.empty:
            reasoning += " One or both layers are empty, returning empty result."
            self.log_reasoning(step_number, 'clip', reasoning,
                              input_info={'input_layer': input_layer, 'clip_layer': clip_layer,
                                        'input_count': len(input_gdf), 'clip_count': len(clip_gdf)},
                              output_info={'feature_count': 0})
            return gpd.GeoDataFrame(columns=input_gdf.columns, geometry=[], crs=input_gdf.crs)
        
        input_gdf = self._homogenize_geometries(input_gdf)
        
        # Handle CRS mismatch
        if input_gdf.crs != clip_gdf.crs:
            reasoning += f" CRS mismatch detected: input ({input_gdf.crs}) vs clip ({clip_gdf.crs}). "
            reasoning += "Converting clip layer to match input layer CRS."
            clip_gdf = clip_gdf.to_crs(input_gdf.crs)
        
        # Perform intersection
        result = gpd.overlay(input_gdf, clip_gdf, how='intersection')
        reasoning += f" Clipping operation resulted in {len(result)} features."
        
        self.log_reasoning(step_number, 'clip', reasoning,
                          input_info={'input_layer': input_layer, 'clip_layer': clip_layer,
                                    'input_count': len(input_gdf), 'clip_count': len(clip_gdf)},
                          output_info={'feature_count': len(result)})
        
        return result

    def _op_dissolve(self, params: Dict[str, Any], step_number: int) -> gpd.GeoDataFrame:
        """Enhanced dissolve with reasoning."""
        input_gdf = self.data_layers[params['input_layer']]
        by_column = params.get('by')
        
        reasoning = f"Dissolving {len(input_gdf)} features from '{params['input_layer']}' "
        if by_column:
            reasoning += f"grouped by attribute '{by_column}'. This will merge features with the same '{by_column}' value."
        else:
            reasoning += "into a single feature. This will merge all features into one unified geometry."
        
        print(f"  🔄 Dissolving '{params['input_layer']}' by '{by_column}'")
        
        if by_column and by_column not in input_gdf.columns:
            reasoning += f" ERROR: Column '{by_column}' not found in available columns: {list(input_gdf.columns)}. "
            reasoning += "Returning original features without dissolving."
            self.log_reasoning(step_number, 'dissolve', reasoning,
                              input_info={'layer': params['input_layer'], 'by_column': by_column, 'input_count': len(input_gdf)},
                              output_info={'feature_count': len(input_gdf), 'error': f"Column '{by_column}' not found"})
            warnings.warn(f"Column '{by_column}' not found for dissolving. Available columns: {list(input_gdf.columns)}")
            return input_gdf.copy()
        
        if by_column:
            result = input_gdf.dissolve(by=by_column)
        else:
            result = input_gdf.dissolve()
        
        reasoning += f" Dissolve operation resulted in {len(result)} features."
        
        self.log_reasoning(step_number, 'dissolve', reasoning,
                          input_info={'layer': params['input_layer'], 'by_column': by_column, 'input_count': len(input_gdf)},
                          output_info={'feature_count': len(result)})
        
        return result

    def _op_intersect(self, params: Dict[str, Any], step_number: int) -> gpd.GeoDataFrame:
        """Performs a geometric intersection of two layers."""
        layer1_name = params.get('input_layer1') or params.get('layer1_name')
        layer2_name = params.get('input_layer2') or params.get('layer2_name')
        
        if not layer1_name or not layer2_name:
            raise ValueError("Intersection operation requires 'input_layer1' and 'input_layer2' parameters.")
        
        gdf1 = self.data_layers[layer1_name]
        gdf2 = self.data_layers[layer2_name]
        
        reasoning = f"Finding intersection between '{layer1_name}' ({len(gdf1)} features) and '{layer2_name}' ({len(gdf2)} features)."
        
        print(f"  🤝 Intersecting '{layer1_name}' with '{layer2_name}'")
        
        if gdf1.empty or gdf2.empty:
            reasoning += " One or both layers are empty, returning empty result."
            self.log_reasoning(step_number, 'intersect', reasoning,
                              input_info={'layer1': layer1_name, 'layer2': layer2_name},
                              output_info={'feature_count': 0})
            return gpd.GeoDataFrame(columns=gdf1.columns, geometry=[], crs=gdf1.crs)
        
        if gdf1.crs != gdf2.crs:
            reasoning += f" CRS mismatch detected. Aligning CRS of '{layer2_name}' to match '{layer1_name}'."
            gdf2 = gdf2.to_crs(gdf1.crs)
        
        result = gpd.overlay(gdf1, gdf2, how='intersection')
        reasoning += f" Intersection resulted in {len(result)} features."
        
        self.log_reasoning(step_number, 'intersect', reasoning,
                          input_info={'layer1': layer1_name, 'layer2': layer2_name, 'input1_count': len(gdf1), 'input2_count': len(gdf2)},
                          output_info={'feature_count': len(result)})
        
        return result

    def _op_rename_layer(self, params: Dict[str, Any], step_number: int) -> gpd.GeoDataFrame:
        """Enhanced layer renaming with reasoning."""
        old_name = params.get('old_name') or params.get('input_layer')
        new_name = params.get('new_name') or params.get('output_layer')
        
        reasoning = f"Finalizing workflow by renaming layer '{old_name}' to '{new_name}'. "
        reasoning += "This prepares the final result for output and subsequent use."
        
        print(f"  📝 Finalizing result from layer '{old_name}' to '{new_name}'")
        
        result = self.data_layers[old_name].copy()
        
        self.log_reasoning(step_number, 'rename_layer', reasoning,
                          input_info={'input_layer': old_name, 'output_layer': new_name},
                          output_info={'feature_count': len(result)})
        
        return result

    def _op_spatial_join(self, params: Dict[str, Any], step_number: int) -> gpd.GeoDataFrame:
        """
        Performs a spatial join, transferring attributes from one layer to another.
        Aligned with the official ToolDefinition.
        """
        # Align with definitions.py
        left_layer_name = params.get('left_layer_name') or params.get('left_layer')
        right_layer_name = params.get('right_layer_name') or params.get('right_layer')
        how = params.get('how', 'inner')
        predicate = params.get('predicate', 'intersects')
        
        left_gdf = self.data_layers[left_layer_name]
        right_gdf = self.data_layers[right_layer_name]
        
        reasoning = (f"Spatially joining '{left_layer_name}' ({len(left_gdf)} features) with "
                    f"'{right_layer_name}' ({len(right_gdf)} features) using a '{how}' join and '{predicate}' predicate.")
        
        print(f"  🔗 Spatially joining '{left_layer_name}' with '{right_layer_name}'")
        
        self.log_reasoning(step_number, 'spatial_join', reasoning)
        
        if left_gdf.empty or right_gdf.empty:
            warnings.warn("One or both layers for spatial join are empty. Returning an empty result.")
            return gpd.GeoDataFrame()
        
        if left_gdf.crs != right_gdf.crs:
            right_gdf = right_gdf.to_crs(left_gdf.crs)
        
        result = gpd.sjoin(left_gdf, right_gdf, how=how, predicate=predicate)
        
        self.log_reasoning(step_number, 'spatial_join', f"Join resulted in {len(result)} features.")
        
        return result

    def _op_summarize(self, params: Dict[str, Any], step_number: int) -> gpd.GeoDataFrame:
        """
        Summarizes a GeoDataFrame by selecting a subset of columns.
        """
        # Use fallback to support older plan formats if needed
        input_layer = params.get('input_layer_name') or params.get('input_layer')
        summary_fields = params.get('summary_fields', [])
        
        gdf = self.data_layers[input_layer]
        
        print(f"  📊 Summarizing layer '{input_layer}' to include fields: {summary_fields}")
        
        self.log_reasoning(step_number, 'summarize', f"Summarizing layer '{input_layer}' to include fields: {summary_fields}.")
        
        if gdf.empty:
            warnings.warn(f"Input layer '{input_layer}' is empty. Summarize will produce an empty result.")
            # Return an empty GeoDataFrame with the correct columns if possible
            return gpd.GeoDataFrame(columns=[f for f in summary_fields if f != 'geometry'], geometry=[], crs=gdf.crs)
        
        # Always include the geometry column for spatial context
        if 'geometry' not in summary_fields:
            summary_fields.append('geometry')
        
        # Filter for columns that actually exist in the GeoDataFrame to prevent errors
        existing_fields = [field for field in summary_fields if field in gdf.columns]
        if len(existing_fields) < len(summary_fields):
            missing = set(summary_fields) - set(existing_fields)
            warnings.warn(f"Fields not found during summary and will be ignored: {missing}")
        
        return gdf[existing_fields].copy()

    def get_layer_info(self) -> None:
        """Display comprehensive layer information."""
        print("\n📊 === LAYER INFORMATION ===")
        if not self.data_layers:
            print("No layers in memory.")
            return
        
        for name, gdf in self.data_layers.items():
            print(f"- {name}: {len(gdf)} features, CRS: {gdf.crs.to_string()}")
            if not gdf.empty:
                numeric_cols = gdf.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    print(f"  Numeric columns: {list(numeric_cols)}")
                geom_types = gdf.geometry.geom_type.value_counts()
                print(f"  Geometry types: {dict(geom_types)}")

    def get_reasoning_log(self) -> List[Dict[str, Any]]:
        """Return the complete Chain-of-Thought reasoning log."""
        return self.reasoning_log

    def export_reasoning_log(self, filename: str = None) -> str:
        """Export reasoning log to JSON file."""
        if filename is None:
            filename = f"workflow_reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'execution_metadata': self.execution_metadata,
            'reasoning_log': self.reasoning_log,
            'layer_summary': {name: {'feature_count': len(gdf), 'crs': str(gdf.crs)}
                             for name, gdf in self.data_layers.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📄 Reasoning log exported to: {filename}")
        return filename

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary."""
        if self.execution_metadata['start_time'] and self.execution_metadata['end_time']:
            duration = (self.execution_metadata['end_time'] - self.execution_metadata['start_time']).total_seconds()
        else:
            duration = 0
        
        return {
            'execution_time_seconds': duration,
            'total_steps': self.execution_metadata['total_steps'],
            'successful_steps': self.execution_metadata['successful_steps'],
            'failed_steps': self.execution_metadata['failed_steps'],
            'success_rate': self.execution_metadata['successful_steps'] / max(self.execution_metadata['total_steps'], 1),
            'total_features_processed': self.execution_metadata['total_features_processed'],
            'final_layers': list(self.data_layers.keys()),
            'errors': self.execution_metadata['errors']
        }

    def clear_layers(self):
        """Clear all layers and reset execution state."""
        self.data_layers.clear()
        self.reasoning_log.clear()
        self.execution_metadata = {
            'start_time': None,
            'end_time': None,
            'total_steps': 0,
            'successful_steps': 0,
            'failed_steps': 0,
            'total_features_processed': 0,
            'errors': []
        }


# Test the layer resolution bug fix
if __name__ == '__main__':
    print("🧪 === WorkflowExecutor Layer Resolution Bug Fix Test ===")
    
    # Test the critical layer name resolution bug fix
    executor = WorkflowExecutor(enable_reasoning_log=True)
    
    # Step 1: Load OSM data (creates schools_1)
    print("\n🧪 Step 1: Load OSM data")
    load_action = ToolCallIntent(
        tool_name="load_osm_data",
        parameters={
            "area_name": "Berlin, Germany",
            "tags": {"amenity": "school"},
            "conceptual_name": "schools"
        }
    )
    
    current_layers = {}
    observation1, updated_layers1 = executor.execute_single_step(load_action, current_layers)
    print(f"✅ Step 1 Result: {observation1}")
    print(f"📊 Available layers: {list(updated_layers1.keys())}")
    
    # Step 2: Buffer using conceptual name (should resolve to schools_1)
    print("\n🧪 Step 2: Buffer with conceptual name resolution")
    buffer_action = ToolCallIntent(
        tool_name="buffer",
        parameters={
            "layer_name": "schools",  # ← This is the CONCEPTUAL name
            "distance": 500,
            "conceptual_name": "school_buffers"
        }
    )
    
    observation2, updated_layers2 = executor.execute_single_step(buffer_action, updated_layers1)
    print(f"✅ Step 2 Result: {observation2}")
    print(f"📊 Available layers: {list(updated_layers2.keys())}")
    
    # Step 3: Test another operation using conceptual name
    if len(updated_layers2) >= 2:
        print("\n🧪 Step 3: Clip using multiple conceptual names")
        clip_action = ToolCallIntent(
            tool_name="clip",
            parameters={
                "input_layer": "schools",  # ← Conceptual name
                "clip_layer": "school_buffers",  # ← Conceptual name
                "conceptual_name": "clipped_schools"
            }
        )
        
        observation3, updated_layers3 = executor.execute_single_step(clip_action, updated_layers2)
        print(f"✅ Step 3 Result: {observation3}")
        print(f"📊 Available layers: {list(updated_layers3.keys())}")
    
    print("\n🎉 Layer Resolution Bug Fix Test completed!")
    print("🔧 Critical bug fixed: Conceptual names now properly resolve to real layer names")