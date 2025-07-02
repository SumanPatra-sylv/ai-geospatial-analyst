# core/tools.py

import os
from typing import Dict
import geopandas as gpd
from langchain_core.tools import tool

# Import all schemas from the schemas file
from .schemas import (
    ListAvailableDataSchema,
    LoadVectorDataSchema,
    SaveVectorDataSchema,
    CreateBufferSchema,
    IntersectLayersSchema
)

# --- Stateful Tool Implementations ---

@tool(args_schema=ListAvailableDataSchema)
def list_available_data(data_store: Dict[str, gpd.GeoDataFrame]) -> Dict:
    """Lists available files in the 'data/' directory and layers loaded in the in-memory data_store."""
    try:
        # Ensure the data directory exists before trying to list files
        if not os.path.isdir('data'):
            os.makedirs('data')
            
        files = [f for f in os.listdir('data') if not f.startswith('.')]
        loaded_layers = list(data_store.keys())
        
        status = (f"✅ Analysis Environment Status:\n"
                  f"📁 Files in 'data/' directory: {files or 'None'}\n"
                  f"📊 Loaded layers in memory: {loaded_layers or 'None'}")
        
        return {"status": status, "data_store": data_store}
    except Exception as e:
        return {"status": f"❌ Error listing data: {e}", "data_store": data_store}

@tool(args_schema=LoadVectorDataSchema)
def load_vector_data(file_path: str, layer_name: str, data_store: Dict[str, gpd.GeoDataFrame]) -> Dict:
    """Loads a vector file into the data_store."""
    try:
        if not os.path.exists(file_path):
            return {"status": f"❌ Error: File not found at '{file_path}'.", "data_store": data_store}
        
        gdf = gpd.read_file(file_path)
        data_store[layer_name] = gdf
        status = f"✅ Successfully loaded '{file_path}' as layer '{layer_name}' with {len(gdf)} features."
        return {"status": status, "data_store": data_store}
    except Exception as e:
        return {"status": f"❌ Error loading file '{file_path}': {e}", "data_store": data_store}

@tool(args_schema=SaveVectorDataSchema)
def save_vector_data(layer_name: str, output_path: str, data_store: Dict[str, gpd.GeoDataFrame]) -> Dict:
    """Saves a layer from the data_store to a file."""
    try:
        if layer_name not in data_store:
            return {"status": f"❌ Error: Layer '{layer_name}' not found in data store.", "data_store": data_store}
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        data_store[layer_name].to_file(output_path)
        status = f"✅ Successfully saved layer '{layer_name}' to '{output_path}'."
        return {"status": status, "data_store": data_store}
    except Exception as e:
        return {"status": f"❌ Error saving layer '{layer_name}': {e}", "data_store": data_store}

@tool(args_schema=CreateBufferSchema)
def create_buffer(input_layer_name: str, distance_meters: float, output_layer_name: str, data_store: Dict[str, gpd.GeoDataFrame]) -> Dict:
    """Creates a buffer around features in a specified layer and updates the data_store."""
    try:
        if input_layer_name not in data_store:
            return {"status": f"❌ Error: Input layer '{input_layer_name}' not found.", "data_store": data_store}
        
        gdf = data_store[input_layer_name]
        
        if gdf.empty:
            return {"status": f"⚠️ Warning: Input layer '{input_layer_name}' is empty. Cannot create buffer.", "data_store": data_store}

        original_crs = gdf.crs
        # Use a projected CRS for accurate distance calculations
        utm_crs = gdf.estimate_utm_crs()
        gdf_proj = gdf.to_crs(utm_crs)
        
        # Perform the buffer operation
        gdf_proj['geometry'] = gdf_proj.geometry.buffer(distance_meters)
        
        # Convert back to the original CRS
        buffered_gdf = gdf_proj.to_crs(original_crs)
        
        data_store[output_layer_name] = buffered_gdf
        status = f"✅ Created {distance_meters}m buffer from '{input_layer_name}' and stored as '{output_layer_name}'."
        return {"status": status, "data_store": data_store}
    except Exception as e:
        return {"status": f"❌ Error creating buffer: {e}", "data_store": data_store}

@tool(args_schema=IntersectLayersSchema)
def intersect_layers(layer_1_name: str, layer_2_name: str, output_layer_name: str, data_store: Dict[str, gpd.GeoDataFrame]) -> Dict:
    """Intersects two layers and updates the data_store."""
    try:
        if layer_1_name not in data_store:
            return {"status": f"❌ Error: Layer '{layer_1_name}' not found.", "data_store": data_store}
        if layer_2_name not in data_store:
            return {"status": f"❌ Error: Layer '{layer_2_name}' not found.", "data_store": data_store}

        gdf1 = data_store[layer_1_name]
        gdf2 = data_store[layer_2_name]

        # Ensure CRS match before intersection
        if gdf1.crs != gdf2.crs:
            gdf2 = gdf2.to_crs(gdf1.crs)

        intersection = gpd.overlay(gdf1, gdf2, how='intersection')
        data_store[output_layer_name] = intersection
        
        if intersection.empty:
            status = f"⚠️ Intersection of '{layer_1_name}' and '{layer_2_name}' is empty. Result stored as '{output_layer_name}'."
        else:
            status = f"✅ Intersected '{layer_1_name}' and '{layer_2_name}', stored as '{output_layer_name}' with {len(intersection)} features."
        
        return {"status": status, "data_store": data_store}
    except Exception as e:
        return {"status": f"❌ Error intersecting layers: {e}", "data_store": data_store}