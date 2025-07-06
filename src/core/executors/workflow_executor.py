#!/usr/bin/env python3
"""
WorkflowExecutor - Executes spatial analysis workflows from generated plans.
"""

import sys
import os
import warnings
from typing import Dict, List, Any

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
from pprint import pprint

class LocationNotFoundError(Exception):
    """Custom exception raised when a location cannot be found by the geocoding service."""
    def __init__(self, message, location_name):
        super().__init__(message)
        self.location_name = location_name

class WorkflowExecutor:
    """
    Executes spatial analysis workflows based on structured plans.
    """
    
    def __init__(self):
        self.smart_data_loader = SmartDataLoader(base_data_dir="data")
        self.data_layers: Dict[str, gpd.GeoDataFrame] = {}
    
    def execute_workflow(self, workflow_plan: List[Dict[str, Any]], parsed_query: ParsedQuery) -> gpd.GeoDataFrame:
        if not workflow_plan:
            raise ValueError("Workflow plan cannot be empty.")
        
        self.clear_layers()
        print(f"--- Starting workflow execution with {len(workflow_plan)} steps for location: '{parsed_query.location}' ---")
        
        print("Validating workflow plan before execution...")
        for i, step in enumerate(workflow_plan, 1):
            operation = step.get('operation')
            output_layer = step.get('output_layer')

            if not operation or not output_layer:
                raise KeyError(f"Workflow plan is malformed. Step {i} is missing 'operation' or 'output_layer'.")

            method_name = f"_op_{operation}"
            if not hasattr(self, method_name):
                raise ValueError(f"Workflow plan contains an unknown operation in Step {i}: '{operation}'.")
        print("Plan validation successful.")
        
        for i, step in enumerate(workflow_plan, 1):
            operation = step['operation']
            output_layer = step['output_layer']
            
            print(f"Step {i}/{len(workflow_plan)}: Executing '{operation}' -> '{output_layer}'")
            
            if operation == 'load_osm_data':
                step['location'] = parsed_query.location
            
            operation_method = getattr(self, f"_op_{operation}")
            result_gdf = operation_method(step)
            
            self.data_layers[output_layer] = result_gdf
            if isinstance(result_gdf, gpd.GeoDataFrame):
                 print(f"  -> Stored {len(result_gdf)} features in layer '{output_layer}'.")
        
        final_layer_name = workflow_plan[-1]['output_layer']
        final_result = self.data_layers[final_layer_name]
        print(f"--- Workflow completed. Final result has {len(final_result)} features. ---")
        
        return final_result
    
    def _op_load_osm_data(self, params: Dict[str, Any]) -> gpd.GeoDataFrame:
        location = params['location']
        print(f"    Loading all OSM polygon data for: {location}")
        try:
            return self.smart_data_loader.fetch_osm_landuse(location)
        except Exception as e:
            error_message = f"Could not find or process location '{location}'. The geocoding service failed."
            print(f"      ERROR: {error_message} (Original error: {e})")
            raise LocationNotFoundError(error_message, location_name=location) from e
    
    def _op_filter_by_attribute(self, params: Dict[str, Any]) -> gpd.GeoDataFrame:
        input_gdf = self.data_layers[params['input_layer']]
        key = params['key']
        value = params['value']
        
        print(f"    Filtering '{params['input_layer']}' for features where '{key}' = '{value}'")
        
        if key not in input_gdf.columns:
            warnings.warn(f"Attribute '{key}' not found in layer '{params['input_layer']}'. Returning empty GeoDataFrame.")
            return gpd.GeoDataFrame(columns=input_gdf.columns, geometry=[], crs=input_gdf.crs)

        if value == '*':
            return input_gdf[input_gdf[key].notna()].copy()
        else:
            return input_gdf[input_gdf[key] == value].copy()
    
    def _op_buffer(self, params: Dict[str, Any]) -> gpd.GeoDataFrame:
        gdf = self.data_layers[params['input_layer']]
        distance = params['distance_meters']
        print(f"    Buffering {len(gdf)} features by {distance} meters")

        if gdf.empty:
            return gdf.copy()

        original_crs = gdf.crs
        projected_gdf = gdf.to_crs("EPSG:3857")
        projected_gdf.geometry = projected_gdf.geometry.buffer(distance)
        return projected_gdf.to_crs(original_crs)
    
    def _op_clip(self, params: Dict[str, Any]) -> gpd.GeoDataFrame:
        input_gdf = self.data_layers[params['input_layer']]
        clip_gdf = self.data_layers[params['clip_layer']]
        print(f"    Clipping '{params['input_layer']}' ({len(input_gdf)}) with '{params['clip_layer']}' ({len(clip_gdf)})")

        if input_gdf.empty or clip_gdf.empty:
            return gpd.GeoDataFrame(columns=input_gdf.columns, geometry=[], crs=input_gdf.crs)
        
        if input_gdf.crs != clip_gdf.crs:
            clip_gdf = clip_gdf.to_crs(input_gdf.crs)
        
        return gpd.overlay(input_gdf, clip_gdf, how='intersection')

    def _op_dissolve(self, params: Dict[str, Any]) -> gpd.GeoDataFrame:
        input_gdf = self.data_layers[params['input_layer']]
        by_column = params.get('by')
        print(f"    Dissolving '{params['input_layer']}' by '{by_column}'")

        if not by_column or by_column not in input_gdf.columns:
            warnings.warn(f"Column '{by_column}' not found for dissolving. Returning original features.")
            return input_gdf.copy()
            
        return input_gdf.dissolve(by=by_column)
    
    def _op_rename_layer(self, params: Dict[str, Any]) -> gpd.GeoDataFrame:
        print(f"    Finalizing result from layer '{params['input_layer']}' to '{params['output_layer']}'")
        return self.data_layers[params['input_layer']].copy()
    
    def get_layer_info(self) -> None:
        print("\n=== LAYER INFORMATION ===")
        if not self.data_layers:
            print("No layers in memory.")
            return
        for name, gdf in self.data_layers.items():
            print(f"- {name}: {len(gdf)} features, CRS: {gdf.crs.to_string()}")

    def clear_layers(self):
        self.data_layers.clear()

if __name__ == '__main__':
    # (The test block remains the same and will now benefit from the validation)
    print("=== WorkflowExecutor Integration Test ===")
    query = ParsedQuery(
        target='park',
        location='Potsdam, Germany',
        constraints=[SpatialConstraint(feature_type='residential', relationship=SpatialRelationship.NEAR, distance_meters=250)]
    )
    rag_guidance = "To find a target 'near' a constraint, first find all features of the constraint type, buffer them, and then clip the target features with that buffer."

    print("\n[PHASE 1] Generating workflow plan...")
    generator = WorkflowGenerator()
    generation_result = generator.generate_workflow(query, rag_guidance)
    plan = generation_result.get("plan", [])
    
    print("\n--- Generated Plan ---")
    pprint(plan)
    
    if not plan:
        print("\n--- ERROR: WorkflowGenerator did not produce a plan. Cannot execute. ---")
    else:
        print("\n[PHASE 2] Executing generated workflow plan...")
        executor = WorkflowExecutor()
        try:
            final_result = executor.execute_workflow(plan, query)
            print("\n=== FINAL RESULT ===")
            print(f"Result contains {len(final_result)} features.")
            if not final_result.empty:
                final_result.info()
            executor.get_layer_info()
            print("\n--- Integration Test SUCCESSFUL ---")
        except Exception as e:
            print(f"\n--- ERROR during execution: {e} ---")