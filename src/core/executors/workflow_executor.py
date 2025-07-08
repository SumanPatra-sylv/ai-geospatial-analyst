#!/usr/bin/env python3
"""
WorkflowExecutor - Executes spatial analysis workflows from generated plans.
Enhanced version with flexible OSM data loading and improved Chain-of-Thought reasoning.
"""

import sys
import os
import warnings
from typing import Dict, List, Any, Optional
import json
import time
from datetime import datetime

# This try/except block allows the module to be run for integrated testing.
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.gis.data_loader import SmartDataLoader
    from src.core.planners.workflow_generator import WorkflowGenerator, ParsedQuery
    from src.core.planners.query_parser import SpatialConstraint, SpatialRelationship
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.gis.data_loader import SmartDataLoader
    from src.core.planners.workflow_generator import WorkflowGenerator, ParsedQuery
    from src.core.planners.query_parser import SpatialConstraint, SpatialRelationship

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
    Executes spatial analysis workflows based on structured plans.
    Enhanced with flexible OSM data loading and Chain-of-Thought reasoning logging.
    """
    
    def __init__(self, enable_reasoning_log: bool = True):
        self.smart_data_loader = SmartDataLoader(base_data_dir="data")
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
        print(f"    🧠 REASONING: {reasoning}")
    # In src/core/executors/workflow_executor.py, inside the WorkflowExecutor class

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
        """Execute the complete workflow with enhanced error handling and reasoning."""
        if not workflow_plan:
            raise ValueError("Workflow plan cannot be empty.")
        
        # Initialize execution metadata
        self.execution_metadata['start_time'] = datetime.now()
        self.execution_metadata['total_steps'] = len(workflow_plan)
        
        self.clear_layers()
        self.reasoning_log.clear()
        
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
                self._execute_single_step(i, step, parsed_query)
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
        final_layer_name = workflow_plan[-1]['output_layer']
        final_result = self.data_layers[final_layer_name]

        # *** MINOR FIX: Moved this line down to only run on success ***
        self.execution_metadata['end_time'] = datetime.now()
        execution_time = (self.execution_metadata['end_time'] - self.execution_metadata['start_time']).total_seconds()

        print(f"\n🎉 Workflow completed successfully in {execution_time:.2f} seconds!")
        print(f"📊 Final result contains {len(final_result)} features.")
        
        # Log final reasoning
        self.log_reasoning(len(workflow_plan) + 1, 'workflow_completion',
                          f"Workflow completed successfully. Final result contains {len(final_result)} features "
                          f"in layer '{final_layer_name}'. Total execution time: {execution_time:.2f} seconds.")
        
        return final_result
    
    def _validate_workflow_plan(self, workflow_plan: List[Dict[str, Any]]):
        """Validate workflow plan structure and operations."""
        for i, step in enumerate(workflow_plan, 1):
            operation = step.get('operation')
            output_layer = step.get('output_layer')

            if not operation or not output_layer:
                raise KeyError(f"Workflow plan is malformed. Step {i} is missing 'operation' or 'output_layer'.")

            method_name = f"_op_{operation}"
            if not hasattr(self, method_name):
                raise ValueError(f"Workflow plan contains an unknown operation in Step {i}: '{operation}'.")
    
    def _execute_single_step(self, step_number: int, step: Dict[str, Any], parsed_query: ParsedQuery):
        """Execute a single workflow step with reasoning and metadata tracking."""
        operation = step['operation']
        output_layer = step['output_layer']
        
        print(f"\n🔄 Step {step_number}: Executing '{operation}' -> '{output_layer}'")
        
        # Add location context for OSM operations
        if operation == 'load_osm_data':
            step['location'] = parsed_query.location
        
        # Execute the operation
        start_time = time.time()
        operation_method = getattr(self, f"_op_{operation}")
        result_gdf = operation_method(step, step_number)
        execution_time = time.time() - start_time
        
        # Store result and update metadata
        self.data_layers[output_layer] = result_gdf
        self.execution_metadata['total_features_processed'] += len(result_gdf)
        
        if isinstance(result_gdf, gpd.GeoDataFrame):
            print(f"  ✅ Stored {len(result_gdf)} features in layer '{output_layer}' ({execution_time:.2f}s)")
        
        return result_gdf
    
    # *** CORE FIX: This entire method is replaced to use the planner's tags ***
    def _op_load_osm_data(self, params: Dict[str, Any], step_number: int) -> gpd.GeoDataFrame:
        location = params['location']
        # The planner now generates a 'tags' dictionary in the plan. We must use it.
        tags = params.get('tags') 
        
        if not tags:
            # Fallback for old plans or simple loads without specific tags
            tags = {'landuse': True} 
            print(f"    ⚠️  No specific tags found in plan. Defaulting to loading all landuse for: {location}")

        reasoning = f"Loading OSM data for location '{location}' with specific tags: {tags}. This provides the precise foundational data needed."
        print(f"    🏷️  Filtering by tags: {tags}")
        self.log_reasoning(step_number, 'load_osm_data', reasoning, input_info={'location': location, 'tags': tags})
        
        try:
            # This will now call the powerful method in SmartDataLoader that accepts tags.
            # This fixes the "AttributeError: 'SmartDataLoader' object has no attribute 'fetch_osm_data'"
            result = self.smart_data_loader.fetch_osm_data(location, tags)
            
            if result.empty:
                warnings.warn(f"Warning: No features found for tags {tags} in location '{location}'. The workflow will continue with an empty dataset for this step.")

            self.log_reasoning(step_number, 'load_osm_data', 
                            f"Successfully retrieved {len(result)} features from OSM.",
                            output_info={'feature_count': len(result)})
            return result
            
        except Exception as e:
            error_message = f"Could not find or process location '{location}' with tags {tags}."
            print(f"      ❌ ERROR: {error_message} (Original error: {e})")
            self.log_reasoning(step_number, 'load_osm_data', f"Failed to load OSM data. Error: {e}", output_info={'error': str(e)})
            raise LocationNotFoundError(error_message, location_name=location) from e
    
    def _op_filter_by_attribute(self, params: Dict[str, Any], step_number: int) -> gpd.GeoDataFrame:
        """Enhanced attribute filtering with reasoning."""
        input_gdf = self.data_layers[params['input_layer']]
        key = params['key']
        value = params['value']
        
        reasoning = f"Filtering layer '{params['input_layer']}' containing {len(input_gdf)} features. "
        reasoning += f"Looking for features where attribute '{key}' equals '{value}'. "
        
        print(f"    🔍 Filtering '{params['input_layer']}' for features where '{key}' = '{value}'")
        
        if key not in input_gdf.columns:
            reasoning += f"ERROR: Attribute '{key}' not found in available columns: {list(input_gdf.columns)}. "
            reasoning += "Returning empty result to prevent workflow failure."
            
            self.log_reasoning(step_number, 'filter_by_attribute', reasoning,
                              input_info={'layer': params['input_layer'], 'key': key, 'value': value},
                              output_info={'feature_count': 0, 'error': f"Column '{key}' not found"})
            
            warnings.warn(f"Attribute '{key}' not found in layer '{params['input_layer']}'. Available columns: {list(input_gdf.columns)}")
            return gpd.GeoDataFrame(columns=input_gdf.columns, geometry=[], crs=input_gdf.crs)

        if value == '*':
            result = input_gdf[input_gdf[key].notna()].copy()
            reasoning += f"Using wildcard filter - selecting all features with non-null '{key}' values. "
        else:
            result = input_gdf[input_gdf[key] == value].copy()
            reasoning += f"Using exact match filter. "
        
        reasoning += f"Filter resulted in {len(result)} features out of {len(input_gdf)} original features."
        
        self.log_reasoning(step_number, 'filter_by_attribute', reasoning,
                          input_info={'layer': params['input_layer'], 'key': key, 'value': value, 'input_count': len(input_gdf)},
                          output_info={'feature_count': len(result)})
        
        return result
    
    def _op_buffer(self, params: Dict[str, Any], step_number: int) -> gpd.GeoDataFrame:
        """Enhanced buffering with reasoning and CRS handling."""
        gdf = self.data_layers[params['input_layer']]
        distance = params['distance_meters']
        
        reasoning = f"Creating {distance}m buffer around {len(gdf)} features from layer '{params['input_layer']}'. "
        reasoning += "Buffering will expand the spatial extent of features to capture nearby areas. "
        
        print(f"    🔄 Buffering {len(gdf)} features by {distance} meters")

        if gdf.empty:
            reasoning += "Input layer is empty, returning empty buffer result."
            self.log_reasoning(step_number, 'buffer', reasoning,
                              input_info={'layer': params['input_layer'], 'distance': distance, 'input_count': 0},
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
                          input_info={'layer': params['input_layer'], 'distance': distance, 'input_count': len(gdf), 'original_crs': str(original_crs)},
                          output_info={'feature_count': len(result)})
        
        return result
    
    # In src/core/executors/workflow_executor.py

    def _op_clip(self, params: Dict[str, Any], step_number: int) -> gpd.GeoDataFrame:
        """Enhanced clipping with reasoning and CRS validation."""
        input_gdf = self.data_layers[params['input_layer']]
        clip_gdf = self.data_layers[params['clip_layer']]
        
        reasoning = f"Clipping features from '{params['input_layer']}' ({len(input_gdf)} features) "
        reasoning += f"using boundaries from '{params['clip_layer']}' ({len(clip_gdf)} features). "
        reasoning += "This will extract only the portions of input features that fall within the clip boundaries."
        
        print(f"    ✂️  Clipping '{params['input_layer']}' ({len(input_gdf)}) with '{params['clip_layer']}' ({len(clip_gdf)})")

        if input_gdf.empty or clip_gdf.empty:
            reasoning += " One or both layers are empty, returning empty result."
            self.log_reasoning(step_number, 'clip', reasoning,
                              input_info={'input_layer': params['input_layer'], 'clip_layer': params['clip_layer'], 
                                        'input_count': len(input_gdf), 'clip_count': len(clip_gdf)},
                              output_info={'feature_count': 0})
            return gpd.GeoDataFrame(columns=input_gdf.columns, geometry=[], crs=input_gdf.crs)
        
        # *** THIS IS THE FIX ***
        # Homogenize geometries to prevent mixed-type errors before overlay.
        input_gdf = self._homogenize_geometries(input_gdf)
        # We don't need to homogenize the clip_gdf, as it's always polygons (from buffer).

        # Handle CRS mismatch
        if input_gdf.crs != clip_gdf.crs:
            reasoning += f" CRS mismatch detected: input ({input_gdf.crs}) vs clip ({clip_gdf.crs}). "
            reasoning += "Converting clip layer to match input layer CRS."
            clip_gdf = clip_gdf.to_crs(input_gdf.crs)
        
        # Perform intersection
        result = gpd.overlay(input_gdf, clip_gdf, how='intersection')
        reasoning += f" Clipping operation resulted in {len(result)} features."
        
        self.log_reasoning(step_number, 'clip', reasoning,
                          input_info={'input_layer': params['input_layer'], 'clip_layer': params['clip_layer'], 
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
        
        print(f"    🔄 Dissolving '{params['input_layer']}' by '{by_column}'")

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
        print(f"    🤝 Intersecting '{layer1_name}' with '{layer2_name}'")
        
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
        reasoning = f"Finalizing workflow by renaming layer '{params['input_layer']}' to '{params['output_layer']}'. "
        reasoning += "This prepares the final result for output and subsequent use."
        
        print(f"    📝 Finalizing result from layer '{params['input_layer']}' to '{params['output_layer']}'")
        
        result = self.data_layers[params['input_layer']].copy()
        
        self.log_reasoning(step_number, 'rename_layer', reasoning,
                          input_info={'input_layer': params['input_layer'], 'output_layer': params['output_layer']},
                          output_info={'feature_count': len(result)})
        
        return result
    
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

if __name__ == '__main__':
    # Enhanced integration test
    print("🧪 === WorkflowExecutor Integration Test ===")
    
    # A test case that forces the use of multiple tools
    test_query = ParsedQuery(
        target='school',
        location='Potsdam, Germany',
        constraints=[
            SpatialConstraint(
                feature_type='park', 
                relationship=SpatialRelationship.NEAR, 
                distance_meters=500
            )
        ],
        summary_required=True
    )
    
    rag_guidance = "To find a target 'near' a constraint, a good strategy is to: 1. Load the target features. 2. Load the constraint features. 3. Create a buffer around the constraint features. 4. Find the target features that are within the buffer zone using a spatial join or intersection."

    print(f"\n🎯 Testing query: Find '{test_query.target}' near a '{test_query.constraints[0].feature_type}' in '{test_query.location}'")
    
    try:
        # PHASE 1: Generate the plan
        print("\n[PHASE 1] Generating workflow plan...")
        generator = WorkflowGenerator()
        generation_result = generator.generate_workflow(test_query, rag_guidance)
        plan = generation_result.get("plan", [])
        
        print("\n--- Generated Plan ---")
        pprint(plan)
        
        # Check if the generated plan is logical
        if not plan or len(plan) < 4:
             print("\n❌ VALIDATION FAILED: The generated plan is too simple. It likely didn't use the advanced tools.")
             exit()
        
        # PHASE 2: Execute the plan
        print("\n[PHASE 2] Executing generated workflow plan...")
        executor = WorkflowExecutor(enable_reasoning_log=True)
        final_result = executor.execute_workflow(plan, test_query)
        
        print("\n📊 === FINAL RESULT ===")
        print(f"Result contains {len(final_result)} features.")
        
        if not final_result.empty:
            print("\n📋 Data Summary:")
            final_result.info()
        
        # Display execution summary
        print("\n⏱️  === EXECUTION SUMMARY ===")
        summary = executor.get_execution_summary()
        pprint(summary)
        
        # Export reasoning log
        log_file = executor.export_reasoning_log()
        
        print(f"\n✅ Integration Test SUCCESSFUL!")
        
    except Exception as e:
        print(f"\n❌ Integration Test FAILED: {e}")
        import traceback
        traceback.print_exc()