# tools.py - Simplified Version with Single-String Input Parsing

import os
import re
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import Point, Polygon
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import warnings
import folium

warnings.filterwarnings('ignore', category=UserWarning, module='geopandas')

# --- SETUP ---
CWD = Path.cwd().resolve()
DATA_DIR = CWD / 'data'
RESULTS_DIR = CWD / 'results'
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

def _parse_args(args_str: str, expected_count: int) -> List[str]:
    """Parse comma-separated arguments and validate count."""
    if not args_str or not args_str.strip():
        raise ValueError(f"Expected {expected_count} arguments, got empty string")
    
    args = [arg.strip() for arg in args_str.split(',')]
    if len(args) != expected_count:
        raise ValueError(f"Expected {expected_count} arguments, got {len(args)}")
    
    return args

def _sanitize_name(name: str) -> str:
    """Sanitize layer/file names."""
    if not name or not isinstance(name, str):
        raise ValueError("Name must be a non-empty string")
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name.strip())

# --- GEOSPATIAL CONTEXT ---
class GeoContext:
    """A singleton class to hold the application's geospatial data state."""
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(GeoContext, cls).__new__(cls)
            # Initialize attributes directly in __new__ to ensure they exist
            cls._instance.vector_layers: Dict[str, gpd.GeoDataFrame] = {}
            cls._instance.raster_layers: Dict[str, Any] = {}
            cls._instance.operation_log: List[Dict] = []
            cls._instance._initialized = True
        return cls._instance
    
    def __init__(self):
        # No need to reinitialize if already done in __new__
        pass
    
    @property
    def layers(self) -> Dict[str, List[str]]:
        """A property to safely access all layer names, for UI compatibility."""
        return {
            'vector': list(self.vector_layers.keys()),
            'raster': list(self.raster_layers.keys())
        }
    
    def clear(self):
        """Clear all loaded layers and reset context."""
        # Close any open raster files
        for layer_name, raster_src in self.raster_layers.items():
            try:
                if hasattr(raster_src, 'close') and not raster_src.closed:
                    raster_src.close()
            except Exception:
                pass
        
        self.vector_layers.clear()
        self.raster_layers.clear()
        self.operation_log.clear()
        return "✅ GeoContext cleared successfully."
    
    def log_operation(self, tool: str, params: str, result: str):
        """Log an operation for debugging and history tracking."""
        self.operation_log.append({
            'tool': tool,
            'params': params,
            'result': result[:100] + '...' if len(result) > 100 else result,
            'timestamp': pd.Timestamp.now().isoformat()
        })

# Create the global context instance
geo_context = GeoContext()

# --- TOOL FUNCTIONS ---

def list_available_data(input_arg: str = "") -> str:
    """List all available data files and loaded layers."""
    try:
        files = [f for f in os.listdir(DATA_DIR) if not f.startswith('.') and not os.path.isdir(DATA_DIR / f)]
        vector_layers = list(geo_context.vector_layers.keys())
        raster_layers = list(geo_context.raster_layers.keys())
        
        result = (f"✅ Analysis Environment Status:\n"
                  f"📁 Files in 'data/' directory ({len(files)}): {files or 'None'}\n"
                  f"📊 Loaded vector layers: {vector_layers or 'None'}\n"
                  f"🗺️ Loaded raster layers: {raster_layers or 'None'}")
        
        geo_context.log_operation("list_available_data", "", result)
        return result
        
    except Exception as e:
        return f"❌ Error listing data: {str(e)}"

def create_sample_data(input_arg: str = "") -> str:
    """Create sample geospatial data files for testing."""
    try:
        # Create sample cities
        cities_data = {
            'name': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata'],
            'population': [12442373, 11007835, 8443675, 4646732, 4496694],
            'geometry': [
                Point(72.8777, 19.0760), Point(77.1025, 28.7041), Point(77.5946, 12.9716),
                Point(80.2707, 13.0827), Point(88.3639, 22.5726)
            ]
        }
        cities_gdf = gpd.GeoDataFrame(cities_data, crs='EPSG:4326')
        cities_path = DATA_DIR / 'sample_cities.shp'
        cities_gdf.to_file(cities_path)

        # Create sample districts
        districts_data = {
            'district': ['Mumbai_Dist', 'Delhi_Dist', 'Bangalore_Dist'],
            'area_km2': [1500, 2200, 1800],
            'geometry': [
                Polygon([(72.5, 18.8), (73.2, 18.8), (73.2, 19.4), (72.5, 19.4)]),
                Polygon([(76.8, 28.4), (77.4, 28.4), (77.4, 29.0), (76.8, 29.0)]),
                Polygon([(77.3, 12.7), (77.9, 12.7), (77.9, 13.2), (77.3, 13.2)])
            ]
        }
        districts_gdf = gpd.GeoDataFrame(districts_data, crs='EPSG:4326')
        districts_path = DATA_DIR / 'sample_districts.shp'
        districts_gdf.to_file(districts_path)

        result = f"✅ Created sample datasets: '{cities_path.name}' and '{districts_path.name}'"
        geo_context.log_operation("create_sample_data", "", result)
        return result
        
    except Exception as e:
        return f"❌ Error creating sample data: {str(e)}"

def load_vector_data(args_str: str) -> str:
    """Load vector data from file. Format: 'filepath,layer_name'"""
    try:
        filepath, layer_name = _parse_args(args_str, 2)
        layer_name = _sanitize_name(layer_name)
        
        # Construct path safely within DATA_DIR
        filename = Path(filepath).name
        full_path = DATA_DIR / filename
        
        if not full_path.exists():
            available_files = [f for f in os.listdir(DATA_DIR) if not f.startswith('.')]
            return f"❌ File '{filename}' not found in 'data/' directory. Available: {available_files}"
        
        gdf = gpd.read_file(full_path)
        geo_context.vector_layers[layer_name] = gdf
        
        result = f"✅ Loaded {len(gdf)} features from '{filename}' as layer '{layer_name}'. CRS: {gdf.crs}"
        geo_context.log_operation("load_vector_data", args_str, result)
        return result
        
    except ValueError as e:
        return f"❌ Invalid input format: {str(e)}. Use: 'filepath,layer_name'"
    except Exception as e:
        return f"❌ Error loading vector data: {str(e)}"

def load_raster_data(args_str: str) -> str:
    """Load raster data from file. Format: 'filepath,layer_name'"""
    try:
        filepath, layer_name = _parse_args(args_str, 2)
        layer_name = _sanitize_name(layer_name)
        
        filename = Path(filepath).name
        full_path = DATA_DIR / filename
        
        if not full_path.exists():
            available_files = [f for f in os.listdir(DATA_DIR) if not f.startswith('.')]
            return f"❌ File '{filename}' not found in 'data/' directory. Available: {available_files}"
        
        raster_src = rasterio.open(full_path)
        geo_context.raster_layers[layer_name] = raster_src
        
        result = f"✅ Loaded raster '{filename}' as layer '{layer_name}'. Shape: {raster_src.shape}, CRS: {raster_src.crs}"
        geo_context.log_operation("load_raster_data", args_str, result)
        return result
        
    except ValueError as e:
        return f"❌ Invalid input format: {str(e)}. Use: 'filepath,layer_name'"
    except Exception as e:
        return f"❌ Error loading raster data: {str(e)}"

def get_layer_info(layer_name: str) -> str:
    """Get detailed information about a loaded layer."""
    try:
        layer_name = _sanitize_name(layer_name)
        
        if layer_name in geo_context.vector_layers:
            gdf = geo_context.vector_layers[layer_name]
            info = (f"📊 Layer: '{layer_name}' (Vector)\n"
                    f"  - Features: {len(gdf)}\n"
                    f"  - CRS: {gdf.crs}\n"
                    f"  - Bounding Box: {gdf.total_bounds.tolist()}\n"
                    f"  - Columns: {list(gdf.columns)}\n"
                    f"  - Geometry Types: {gdf.geom_type.value_counts().to_dict()}")
            
        elif layer_name in geo_context.raster_layers:
            raster = geo_context.raster_layers[layer_name]
            info = (f"🗺️ Layer: '{layer_name}' (Raster)\n"
                    f"  - Dimensions: ({raster.count}, {raster.height}, {raster.width})\n"
                    f"  - CRS: {raster.crs}\n"
                    f"  - Bounding Box: {list(raster.bounds)}\n"
                    f"  - Data Types: {raster.dtypes}")
        else:
            available = list(geo_context.vector_layers.keys()) + list(geo_context.raster_layers.keys())
            return f"❌ Layer '{layer_name}' not found. Available: {available}"
        
        geo_context.log_operation("get_layer_info", layer_name, "Retrieved layer info")
        return info
        
    except Exception as e:
        return f"❌ Error getting layer info: {str(e)}"

def create_buffer(args_str: str) -> str:
    """Create buffer around features. Format: 'input_layer,distance_meters,output_layer'"""
    try:
        input_layer, distance_str, output_layer = _parse_args(args_str, 3)
        input_layer = _sanitize_name(input_layer)
        output_layer = _sanitize_name(output_layer)
        distance_meters = float(distance_str)
        
        if input_layer not in geo_context.vector_layers:
            return f"❌ Input layer '{input_layer}' not found. Available: {list(geo_context.vector_layers.keys())}"
        
        gdf = geo_context.vector_layers[input_layer]
        
        # Reproject to UTM for accurate buffering in meters
        original_crs = gdf.crs
        try:
            utm_crs = gdf.estimate_utm_crs()
            gdf_proj = gdf.to_crs(utm_crs)
        except:
            # Fallback to Web Mercator if UTM estimation fails
            gdf_proj = gdf.to_crs(epsg=3857)
        
        # Create buffer and convert back to original CRS
        gdf_proj['geometry'] = gdf_proj.geometry.buffer(distance_meters)
        buffered_gdf = gdf_proj.to_crs(original_crs)
        
        geo_context.vector_layers[output_layer] = buffered_gdf
        
        result = f"✅ Created {distance_meters}m buffer from '{input_layer}' → '{output_layer}' ({len(buffered_gdf)} features)"
        geo_context.log_operation("create_buffer", args_str, result)
        return result
        
    except ValueError as e:
        return f"❌ Invalid input format: {str(e)}. Use: 'input_layer,distance_meters,output_layer'"
    except Exception as e:
        return f"❌ Error creating buffer: {str(e)}"

def intersect_layers(args_str: str) -> str:
    """Intersect two vector layers. Format: 'layer_1,layer_2,output_layer'"""
    try:
        layer_1, layer_2, output_layer = _parse_args(args_str, 3)
        layer_1 = _sanitize_name(layer_1)
        layer_2 = _sanitize_name(layer_2)
        output_layer = _sanitize_name(output_layer)
        
        if layer_1 not in geo_context.vector_layers:
            return f"❌ Layer '{layer_1}' not found"
        if layer_2 not in geo_context.vector_layers:
            return f"❌ Layer '{layer_2}' not found"
        
        gdf1 = geo_context.vector_layers[layer_1]
        gdf2 = geo_context.vector_layers[layer_2]
        
        # Ensure CRS match
        if gdf1.crs != gdf2.crs:
            gdf2 = gdf2.to_crs(gdf1.crs)
        
        intersection = gpd.overlay(gdf1, gdf2, how='intersection')
        
        if intersection.empty:
            result = f"⚠️ Intersection of '{layer_1}' and '{layer_2}' is empty"
        else:
            geo_context.vector_layers[output_layer] = intersection
            result = f"✅ Intersected '{layer_1}' ∩ '{layer_2}' → '{output_layer}' ({len(intersection)} features)"
            
        geo_context.log_operation("intersect_layers", args_str, result)
        return result
        
    except ValueError as e:
        return f"❌ Invalid input format: {str(e)}. Use: 'layer_1,layer_2,output_layer'"
    except Exception as e:
        return f"❌ Error intersecting layers: {str(e)}"

def spatial_join(args_str: str) -> str:
    """Spatial join two layers. Format: 'left_layer,right_layer,output_layer,how,predicate'"""
    try:
        parts = _parse_args(args_str, 5)
        left_layer, right_layer, output_layer, how, predicate = [_sanitize_name(p) if i < 3 else p for i, p in enumerate(parts)]

        if left_layer not in geo_context.vector_layers:
            return f"❌ Left layer '{left_layer}' not found"
        if right_layer not in geo_context.vector_layers:
            return f"❌ Right layer '{right_layer}' not found"

        left_gdf = geo_context.vector_layers[left_layer]
        right_gdf = geo_context.vector_layers[right_layer]

        # Ensure same CRS
        if left_gdf.crs != right_gdf.crs:
            right_gdf = right_gdf.to_crs(left_gdf.crs)

        joined = gpd.sjoin(left_gdf, right_gdf, how=how, predicate=predicate)
        geo_context.vector_layers[output_layer] = joined

        result = f"✅ Spatial join '{left_layer}' + '{right_layer}' → '{output_layer}' ({how}, {predicate}, {len(joined)} features)"
        geo_context.log_operation("spatial_join", args_str, result)
        return result

    except ValueError as e:
        return f"❌ Invalid input format: {str(e)}. Use: 'left_layer,right_layer,output_layer,how,predicate'"
    except Exception as e:
        return f"❌ Error during spatial join: {str(e)}"

def filter_by_attribute(args_str: str) -> str:
    """Filter layer by attribute. Format: 'layer_name,column,operator,value,output_layer'"""
    try:
        layer_name, column, operator, value, output_layer = _parse_args(args_str, 5)
        layer_name = _sanitize_name(layer_name)
        output_layer = _sanitize_name(output_layer)
        
        if layer_name not in geo_context.vector_layers:
            return f"❌ Layer '{layer_name}' not found"
        
        gdf = geo_context.vector_layers[layer_name]
        
        if column not in gdf.columns:
            return f"❌ Column '{column}' not found. Available: {list(gdf.columns)}"
        
        # Try to convert value to appropriate type
        try:
            # Try numeric conversion first
            value = float(value)
            if value.is_integer():
                value = int(value)
        except ValueError:
            # Keep as string if conversion fails
            pass
        
        # Build query string
        if isinstance(value, str):
            query_str = f"`{column}` {operator} '{value}'"
        else:
            query_str = f"`{column}` {operator} {value}"
            
        filtered_gdf = gdf.query(query_str)
        geo_context.vector_layers[output_layer] = filtered_gdf
        
        result = f"✅ Filtered '{layer_name}' where {column} {operator} {value} → '{output_layer}' ({len(filtered_gdf)} features)"
        geo_context.log_operation("filter_by_attribute", args_str, result)
        return result
        
    except ValueError as e:
        return f"❌ Invalid input format: {str(e)}. Use: 'layer_name,column,operator,value,output_layer'"
    except Exception as e:
        return f"❌ Error filtering: {str(e)}"

def calculate_area(args_str: str) -> str:
    """Calculate area of polygons. Format: 'layer_name,output_column' or just 'layer_name'"""
    try:
        parts = [p.strip() for p in args_str.split(',')]
        layer_name = _sanitize_name(parts[0])
        output_column = parts[1] if len(parts) > 1 else "area_sqkm"
        
        if layer_name not in geo_context.vector_layers:
            return f"❌ Layer '{layer_name}' not found"
        
        gdf = geo_context.vector_layers[layer_name]
        
        # Reproject to equal-area projection for accurate calculation
        try:
            # Use a cylindrical equal area projection
            gdf_proj = gdf.to_crs("+proj=cea")
        except:
            # Fallback to Web Mercator
            gdf_proj = gdf.to_crs(epsg=3857)
        
        # Calculate area in square meters and convert to square kilometers
        gdf[output_column] = gdf_proj.geometry.area / 1_000_000
        
        result = f"✅ Calculated area for '{layer_name}' → column '{output_column}' (sq km)"
        geo_context.log_operation("calculate_area", args_str, result)
        return result
        
    except Exception as e:
        return f"❌ Error calculating area: {str(e)}"

def calculate_zonal_statistics(args_str: str) -> str:
    """Calculate zonal statistics. Format: 'vector_layer,raster_layer,stats' (stats like 'mean,max,min')"""
    try:
        parts = _parse_args(args_str, 3)
        vector_layer, raster_layer = [_sanitize_name(p) for p in parts[:2]]
        stats = [s.strip() for s in parts[2].split(',')]
        
        if vector_layer not in geo_context.vector_layers:
            return f"❌ Vector layer '{vector_layer}' not found"
        if raster_layer not in geo_context.raster_layers:
            return f"❌ Raster layer '{raster_layer}' not found"
        
        gdf = geo_context.vector_layers[vector_layer]
        raster_src = geo_context.raster_layers[raster_layer]
        
        # Ensure vector CRS matches raster CRS
        if gdf.crs != raster_src.crs:
            gdf = gdf.to_crs(raster_src.crs)
            
        stats_str = " ".join(stats)
        z_stats = zonal_stats(
            gdf, 
            raster_src.read(1), 
            affine=raster_src.transform, 
            stats=stats_str, 
            nodata=raster_src.nodata
        )
        
        # Add results to the GeoDataFrame
        stats_df = pd.DataFrame(z_stats)
        gdf_with_stats = gdf.join(stats_df)
        
        # Update the layer in the context
        geo_context.vector_layers[vector_layer] = gdf_with_stats
        
        result = f"✅ Calculated zonal stats ({stats_str}) for '{raster_layer}' using '{vector_layer}' zones"
        geo_context.log_operation("calculate_zonal_statistics", args_str, result)
        return result
        
    except ValueError as e:
        return f"❌ Invalid input format: {str(e)}. Use: 'vector_layer,raster_layer,stats'"
    except Exception as e:
        return f"❌ Error calculating zonal statistics: {str(e)}"

def save_layer(args_str: str) -> str:
    """Save layer to file. Format: 'layer_name,filename,format' or 'layer_name,filename'"""
    try:
        parts = [p.strip() for p in args_str.split(',')]
        layer_name = _sanitize_name(parts[0])
        filename = _sanitize_name(parts[1])
        file_format = parts[2] if len(parts) > 2 else "GeoJSON"
        
        if layer_name not in geo_context.vector_layers:
            return f"❌ Layer '{layer_name}' not found"
        
        gdf = geo_context.vector_layers[layer_name]
        output_path = RESULTS_DIR / filename
        
        # Determine driver based on format
        if file_format.upper() == "GEOJSON":
            driver = "GeoJSON"
        elif file_format.upper() in ["SHAPEFILE", "SHP"]:
            driver = "ESRI Shapefile"
        elif file_format.upper() == "GPKG":
            driver = "GPKG"
        else:
            return f"❌ Unsupported format '{file_format}'. Use: GeoJSON, Shapefile, or GPKG"

        gdf.to_file(output_path, driver=driver)

        result = f"✅ Saved layer '{layer_name}' → '{output_path}' as {file_format}"
        geo_context.log_operation("save_layer", args_str, result)
        return result
        
    except Exception as e:
        return f"❌ Error saving layer: {str(e)}"

def visualize_layer(layer_name: str) -> str:
    """Create interactive map of a layer."""
    try:
        layer_name = _sanitize_name(layer_name)

        if layer_name not in geo_context.vector_layers:
            available = list(geo_context.vector_layers.keys())
            return f"❌ Layer '{layer_name}' not found. Available: {available}"

        gdf = geo_context.vector_layers[layer_name]
        
        if gdf.empty:
            return f"⚠️ Layer '{layer_name}' is empty. Cannot create map."

        # Ensure WGS84 for web mapping
        gdf_display = gdf.to_crs("EPSG:4326")
        
        # Create map centered on data
        bounds = gdf_display.total_bounds
        map_center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        
        m = folium.Map(location=map_center, zoom_start=6, tiles="CartoDB positron")
        
        # Fit map to bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        
        # Add layer with tooltips (exclude geometry column for cleaner display)
        display_cols = [col for col in gdf_display.columns if col != 'geometry']
        
        folium.GeoJson(
            gdf_display,
            tooltip=folium.GeoJsonTooltip(
                fields=display_cols, 
                aliases=[f"<b>{col}</b>" for col in display_cols]
            )
        ).add_to(m)
        
        # Save map
        map_filename = f"{layer_name}_map.html"
        map_path = RESULTS_DIR / map_filename
        m.save(str(map_path))
        
        # Return machine-readable result for UI detection
        result = f"MAP_GENERATED|{map_path}|Interactive map of '{layer_name}' created → {map_filename}"
        geo_context.log_operation("visualize_layer", layer_name, result)
        return result
        
    except Exception as e:
        return f"❌ Error creating map: {str(e)}"

def clear_context(input_arg: str = "") -> str:
    """Clear all loaded layers and reset context."""
    try:
        result = geo_context.clear()
        geo_context.log_operation("clear_context", "", result)
        return result
    except Exception as e:
        return f"❌ Error clearing context: {str(e)}"

def show_operation_log(input_arg: str = "") -> str:
    """Show log of recent operations."""
    try:
        if not geo_context.operation_log:
            return "📝 No operations recorded yet."
        
        # Show last 10 operations
        log_entries = []
        for i, entry in enumerate(reversed(geo_context.operation_log[-10:]), 1):
            timestamp = entry['timestamp'][:19].replace('T', ' ')
            log_entries.append(f"{i}. [{timestamp}] {entry['tool']} → {entry['result']}")
        
        result = "📝 Recent Operations (latest first):\n" + "\n".join(log_entries)
        return result
        
    except Exception as e:
        return f"❌ Error showing log: {str(e)}"

# --- FUNCTION REGISTRY ---
# This makes it easy to see all available functions and their expected input formats
AVAILABLE_FUNCTIONS = {
    # No arguments
    'list_available_data': 'No arguments needed',
    'create_sample_data': 'No arguments needed', 
    'clear_context': 'No arguments needed',
    'show_operation_log': 'No arguments needed',
    
    # Single argument
    'get_layer_info': 'layer_name',
    'visualize_layer': 'layer_name',
    
    # Multiple arguments (comma-separated)
    'load_vector_data': 'filepath,layer_name',
    'load_raster_data': 'filepath,layer_name',
    'create_buffer': 'input_layer,distance_meters,output_layer',
    'intersect_layers': 'layer_1,layer_2,output_layer',
    'spatial_join': 'left_layer,right_layer,output_layer,how,predicate',
    'filter_by_attribute': 'layer_name,column,operator,value,output_layer',
    'calculate_area': 'layer_name,output_column (optional)',
    'calculate_zonal_statistics': 'vector_layer,raster_layer,stats',
    'save_layer': 'layer_name,filename,format (optional)'
}

def help_functions(input_arg: str = "") -> str:
    """Show available functions and their formats."""
    help_text = "📋 Available GIS Functions:\n\n"
    for func_name, format_str in AVAILABLE_FUNCTIONS.items():
        help_text += f"• {func_name}: {format_str}\n"
    
    help_text += "\nExample usage:"
    help_text += "\n• load_vector_data('sample_cities.shp,cities')"
    help_text += "\n• create_buffer('cities,1000,cities_buffer')"
    help_text += "\n• visualize_layer('cities')"
    
    return help_text