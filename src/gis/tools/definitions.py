"""
Central Tool Registry for GIS Operations
=========================================

This module defines the structure of GIS tools and maintains a master registry
of all available operations for the LLM-powered geospatial workflow system.

The registry serves as the source of truth for:
- Tool definitions and parameters
- LLM-friendly descriptions
- Executor method mappings
- Parameter validation schemas
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Union, Optional, Any
from enum import Enum


class ParameterType(str, Enum):
    """Enumeration of supported parameter types for better type safety."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DICT = "dict"
    LIST = "list"
    GEOMETRY = "geometry"
    CRS = "crs"
    FILE_PATH = "file_path"


class ToolParameter(BaseModel):
    """
    Defines a parameter for a GIS tool operation.
    
    This model provides comprehensive parameter information for LLM understanding
    and automatic workflow generation.
    """
    name: str = Field(..., description="Parameter name")
    description: str = Field(..., description="LLM-friendly parameter description")
    type: str = Field(..., description="Parameter type (string, integer, boolean, etc.)")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value if not required")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Parameter constraints (min, max, choices, etc.)")
    
    @validator('type')
    def validate_type(cls, v):
        """Validate parameter type against supported types."""
        valid_types = [t.value for t in ParameterType]
        if v not in valid_types:
            raise ValueError(f"Parameter type must be one of: {valid_types}")
        return v


class ToolDefinition(BaseModel):
    """
    Defines a complete GIS tool with all metadata needed for LLM orchestration.
    
    This model encapsulates everything an LLM needs to understand and use a tool:
    - What it does (description)
    - What inputs it needs (parameters)
    - How to execute it (executor_method_name)
    - Usage examples and constraints
    """
    operation_name: str = Field(..., description="Unique operation identifier")
    description: str = Field(..., description="Clear, LLM-friendly description of the tool")
    parameters: List[ToolParameter] = Field(..., description="List of tool parameters")
    executor_method_name: str = Field(..., description="Corresponding WorkflowExecutor method name")
    category: str = Field(..., description="Tool category (data_loading, analysis, processing, etc.)")
    examples: Optional[List[str]] = Field(default=None, description="Usage examples for LLM context")
    prerequisites: Optional[List[str]] = Field(default=None, description="Required conditions before using this tool")
    outputs: Optional[List[str]] = Field(default=None, description="Expected outputs from this tool")
    
    @validator('operation_name')
    def validate_operation_name(cls, v):
        """Ensure operation name follows naming conventions."""
        if not v.replace('_', '').isalnum():
            raise ValueError("Operation name must contain only alphanumeric characters and underscores")
        return v


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

# Data Loading Tools
LOAD_OSM_DATA = ToolDefinition(
    operation_name="load_osm_data",
    description="Load OpenStreetMap data for a specified area using flexible tag filtering. "
                "Supports complex queries with multiple tags and values for comprehensive data extraction.",
    parameters=[
        ToolParameter(
            name="area_name",
            description="Name of the geographic area to load data for (e.g., 'New York City', 'London')",
            type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="tags",
            description="Dictionary of OSM tags to filter data. Keys are tag names, values can be strings or lists of strings. "
                       "Example: {'highway': ['primary', 'secondary'], 'amenity': 'school'}",
            type=ParameterType.DICT,
            required=True,
            constraints={
                "example": {"highway": ["primary", "secondary"], "amenity": "school"},
                "note": "Use specific tags for better data quality"
            }
        ),
        ToolParameter(
            name="geometry_type",
            description="Type of geometry to extract (point, line, polygon, all)",
            type=ParameterType.STRING,
            required=False,
            default="all",
            constraints={"choices": ["point", "line", "polygon", "all"]}
        ),
        ToolParameter(
            name="layer_name",
            description="Name for the resulting layer",
            type=ParameterType.STRING,
            required=False,
            default="osm_data"
        )
    ],
    executor_method_name="_op_load_osm_data",
    category="data_loading",
    examples=[
        "Load schools and hospitals in Mumbai",
        "Extract all highways in a city",
        "Get building footprints for urban analysis"
    ],
    outputs=["GeoDataFrame with OSM features", "Layer added to workspace"]
)

LOAD_BHOONIDHI_DATA = ToolDefinition(
    operation_name="load_bhoonidhi_data",
    description="Load geospatial data from Bhoonidhi (India's national geospatial data repository). "
                "Supports various data types including administrative boundaries, land use, and infrastructure.",
    parameters=[
        ToolParameter(
            name="dataset_id",
            description="Bhoonidhi dataset identifier or name",
            type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="state",
            description="State name for filtering (optional)",
            type=ParameterType.STRING,
            required=False
        ),
        ToolParameter(
            name="district",
            description="District name for filtering (optional)",
            type=ParameterType.STRING,
            required=False
        ),
        ToolParameter(
            name="layer_name",
            description="Name for the resulting layer",
            type=ParameterType.STRING,
            required=False,
            default="bhoonidhi_data"
        )
    ],
    executor_method_name="_op_load_bhoonidhi_data",
    category="data_loading",
    examples=[
        "Load administrative boundaries for West Bengal",
        "Get land use data for Mumbai district"
    ],
    outputs=["GeoDataFrame with Bhoonidhi data", "Layer added to workspace"]
)

# Data Processing Tools
FILTER_BY_ATTRIBUTE = ToolDefinition(
    operation_name="filter_by_attribute",
    description="Filter features in a layer based on attribute values. "
                "Supports various comparison operators and logical conditions for precise data selection.",
    parameters=[
        ToolParameter(
            name="layer_name",
            description="Name of the layer to filter",
            type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="attribute",
            description="Name of the attribute column to filter on",
            type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="operator",
            description="Comparison operator (equals, not_equals, greater_than, less_than, contains, starts_with, ends_with, in_list)",
            type=ParameterType.STRING,
            required=True,
            constraints={"choices": ["equals", "not_equals", "greater_than", "less_than", "contains", "starts_with", "ends_with", "in_list"]}
        ),
        ToolParameter(
            name="value",
            description="Value to compare against (can be string, number, or list for 'in_list' operator)",
            type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="output_layer_name",
            description="Name for the filtered output layer",
            type=ParameterType.STRING,
            required=False,
            default="filtered_layer"
        )
    ],
    executor_method_name="_op_filter_by_attribute",
    category="data_processing",
    examples=[
        "Filter buildings with height > 50 meters",
        "Select roads of type 'highway'",
        "Find amenities that are schools or hospitals"
    ],
    prerequisites=["Layer must exist in workspace"],
    outputs=["Filtered GeoDataFrame", "New layer with filtered features"]
)

BUFFER_ANALYSIS = ToolDefinition(
    operation_name="buffer",
    description="Create buffer zones around features at specified distances. "
                "Useful for proximity analysis, impact assessment, and zone creation.",
    parameters=[
        ToolParameter(
            name="layer_name",
            description="Name of the layer to buffer",
            type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="distance",
            description="Buffer distance in the layer's coordinate system units (usually meters)",
            type=ParameterType.FLOAT,
            required=True,
            constraints={"min": 0.1, "note": "Distance should be positive"}
        ),
        ToolParameter(
            name="unit",
            description="Unit of measurement for buffer distance (meters, kilometers, feet, miles)",
            type=ParameterType.STRING,
            required=False,
            default="meters",
            constraints={"choices": ["meters", "kilometers", "feet", "miles"]}
        ),
        ToolParameter(
            name="dissolve_result",
            description="Whether to dissolve overlapping buffers into single features",
            type=ParameterType.BOOLEAN,
            required=False,
            default=False
        ),
        ToolParameter(
            name="output_layer_name",
            description="Name for the buffer output layer",
            type=ParameterType.STRING,
            required=False,
            default="buffer_layer"
        )
    ],
    executor_method_name="_op_buffer",
    category="spatial_analysis",
    examples=[
        "Create 100m buffer around schools for safety zones",
        "Generate 5km buffer around hospitals for service areas",
        "Buffer roads by 50m for noise impact analysis"
    ],
    prerequisites=["Layer must exist in workspace"],
    outputs=["Buffer polygons as GeoDataFrame", "New buffer layer"]
)

CLIP_ANALYSIS = ToolDefinition(
    operation_name="clip",
    description="Clip features from one layer using the boundaries of another layer. "
                "Extracts only the portions of features that fall within the clipping boundary.",
    parameters=[
        ToolParameter(
            name="input_layer_name",
            description="Name of the layer to be clipped",
            type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="clip_layer_name",
            description="Name of the layer to use as clipping boundary",
            type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="output_layer_name",
            description="Name for the clipped output layer",
            type=ParameterType.STRING,
            required=False,
            default="clipped_layer"
        )
    ],
    executor_method_name="_op_clip",
    category="spatial_analysis",
    examples=[
        "Clip roads to city boundary",
        "Extract buildings within flood zone",
        "Clip land use data to study area"
    ],
    prerequisites=["Both input and clip layers must exist in workspace"],
    outputs=["Clipped features as GeoDataFrame", "New clipped layer"]
)

DISSOLVE_ANALYSIS = ToolDefinition(
    operation_name="dissolve",
    description="Dissolve (merge) features based on common attribute values. "
                "Combines adjacent or overlapping features with the same attribute value into single features.",
    parameters=[
        ToolParameter(
            name="layer_name",
            description="Name of the layer to dissolve",
            type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="dissolve_field",
            description="Attribute field to dissolve on (features with same value will be merged)",
            type=ParameterType.STRING,
            required=False,
            default=None,
            constraints={"note": "If not specified, all features will be dissolved into one"}
        ),
        ToolParameter(
            name="aggregate_fields",
            description="Dictionary of fields to aggregate during dissolve (field_name: aggregation_method)",
            type=ParameterType.DICT,
            required=False,
            default=None,
            constraints={"example": {"population": "sum", "area": "sum", "name": "first"}}
        ),
        ToolParameter(
            name="output_layer_name",
            description="Name for the dissolved output layer",
            type=ParameterType.STRING,
            required=False,
            default="dissolved_layer"
        )
    ],
    executor_method_name="_op_dissolve",
    category="data_processing",
    examples=[
        "Dissolve parcels by land use type",
        "Merge administrative boundaries by state",
        "Combine building footprints by block"
    ],
    prerequisites=["Layer must exist in workspace"],
    outputs=["Dissolved features as GeoDataFrame", "New dissolved layer"]
)

SPATIAL_JOIN = ToolDefinition(
    operation_name="spatial_join",
    description="Join attributes from one layer to another based on spatial relationships. "
                "Supports various spatial predicates like intersects, contains, within, etc.",
    parameters=[
        ToolParameter(
            name="left_layer_name",
            description="Name of the left (target) layer to join attributes to",
            type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="right_layer_name",
            description="Name of the right (source) layer to join attributes from",
            type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="how",
            description="Type of join (left, right, inner)",
            type=ParameterType.STRING,
            required=False,
            default="left",
            constraints={"choices": ["left", "right", "inner"]}
        ),
        ToolParameter(
            name="predicate",
            description="Spatial relationship predicate (intersects, contains, within, touches, crosses, overlaps)",
            type=ParameterType.STRING,
            required=False,
            default="intersects",
            constraints={"choices": ["intersects", "contains", "within", "touches", "crosses", "overlaps"]}
        ),
        ToolParameter(
            name="output_layer_name",
            description="Name for the joined output layer",
            type=ParameterType.STRING,
            required=False,
            default="spatial_join_layer"
        )
    ],
    executor_method_name="_op_spatial_join",
    category="spatial_analysis",
    examples=[
        "Join census data to administrative boundaries",
        "Add district names to point locations",
        "Join land use attributes to building footprints"
    ],
    prerequisites=["Both layers must exist in workspace"],
    outputs=["Joined features as GeoDataFrame", "New joined layer"]
)

INTERSECT_ANALYSIS = ToolDefinition(
    operation_name="intersect",
    description="Find geometric intersections between two layers. "
                "Returns only the overlapping portions along with attributes from both layers.",
    parameters=[
        ToolParameter(
            name="layer1_name",
            description="Name of the first layer",
            type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="layer2_name",
            description="Name of the second layer",
            type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="output_layer_name",
            description="Name for the intersection output layer",
            type=ParameterType.STRING,
            required=False,
            default="intersection_layer"
        )
    ],
    executor_method_name="_op_intersect",
    category="spatial_analysis",
    examples=[
        "Find intersection of flood zones and residential areas",
        "Intersect protected areas with development zones",
        "Find overlap between different land use types"
    ],
    prerequisites=["Both layers must exist in workspace"],
    outputs=["Intersection features as GeoDataFrame", "New intersection layer"]
)

# Data Management Tools
RENAME_LAYER = ToolDefinition(
    operation_name="rename_layer",
    description="Rename an existing layer in the workspace. "
                "Useful for organizing workflow outputs and maintaining clear layer names.",
    parameters=[
        ToolParameter(
            name="old_name",
            description="Current name of the layer",
            type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="new_name",
            description="New name for the layer",
            type=ParameterType.STRING,
            required=True
        )
    ],
    executor_method_name="_op_rename_layer",
    category="data_management",
    examples=[
        "Rename 'temp_layer' to 'flood_risk_zones'",
        "Change 'osm_data' to 'mumbai_roads'"
    ],
    prerequisites=["Layer with old_name must exist in workspace"],
    outputs=["Layer renamed in workspace"]
)

REPROJECT_LAYER = ToolDefinition(
    operation_name="reproject_layer",
    description="Reproject a layer to a different coordinate reference system (CRS). "
                "Essential for ensuring all layers use compatible coordinate systems.",
    parameters=[
        ToolParameter(
            name="layer_name",
            description="Name of the layer to reproject",
            type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="target_crs",
            description="Target CRS as EPSG code (e.g., 'EPSG:4326', 'EPSG:3857') or CRS string",
            type=ParameterType.STRING,
            required=True,
            constraints={"example": "EPSG:4326"}
        ),
        ToolParameter(
            name="output_layer_name",
            description="Name for the reprojected output layer",
            type=ParameterType.STRING,
            required=False,
            default="reprojected_layer"
        )
    ],
    executor_method_name="_op_reproject_layer",
    category="data_management",
    examples=[
        "Reproject data to WGS84 (EPSG:4326)",
        "Convert to Web Mercator (EPSG:3857) for web mapping",
        "Reproject to local UTM zone for accurate area calculations"
    ],
    prerequisites=["Layer must exist in workspace"],
    outputs=["Reprojected layer", "New layer with target CRS"]
)

CALCULATE_AREA = ToolDefinition(
    operation_name="calculate_area",
    description="Calculate area for polygon features and add as a new attribute. "
                "Automatically handles CRS and unit conversions for accurate measurements.",
    parameters=[
        ToolParameter(
            name="layer_name",
            description="Name of the polygon layer to calculate area for",
            type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="area_field_name",
            description="Name of the new field to store area values",
            type=ParameterType.STRING,
            required=False,
            default="area"
        ),
        ToolParameter(
            name="unit",
            description="Unit for area calculation (square_meters, square_kilometers, hectares, acres)",
            type=ParameterType.STRING,
            required=False,
            default="square_meters",
            constraints={"choices": ["square_meters", "square_kilometers", "hectares", "acres"]}
        )
    ],
    executor_method_name="_op_calculate_area",
    category="analysis",
    examples=[
        "Calculate area of land use polygons in hectares",
        "Add area field to administrative boundaries",
        "Calculate building footprint areas"
    ],
    prerequisites=["Layer must exist and contain polygon features"],
    outputs=["Layer with added area field", "Area statistics"]
)

CALCULATE_DISTANCE = ToolDefinition(
    operation_name="calculate_distance",
    description="Calculate distance between features or to nearest features from another layer. "
                "Supports various distance calculations including nearest neighbor analysis.",
    parameters=[
        ToolParameter(
            name="from_layer_name",
            description="Name of the layer to calculate distances from",
            type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="to_layer_name",
            description="Name of the layer to calculate distances to",
            type=ParameterType.STRING,
            required=True
        ),
        ToolParameter(
            name="distance_field_name",
            description="Name of the new field to store distance values",
            type=ParameterType.STRING,
            required=False,
            default="distance"
        ),
        ToolParameter(
            name="unit",
            description="Unit for distance calculation (meters, kilometers, feet, miles)",
            type=ParameterType.STRING,
            required=False,
            default="meters",
            constraints={"choices": ["meters", "kilometers", "feet", "miles"]}
        )
    ],
    executor_method_name="_op_calculate_distance",
    category="analysis",
    examples=[
        "Calculate distance from buildings to nearest hospital",
        "Find distance from points to coastline",
        "Calculate accessibility to public transport"
    ],
    prerequisites=["Both layers must exist in workspace"],
    outputs=["Layer with added distance field", "Distance statistics"]
)

# =============================================================================
# TOOL REGISTRY
# =============================================================================

TOOL_REGISTRY: Dict[str, ToolDefinition] = {
    # Data Loading Tools
    "load_osm_data": LOAD_OSM_DATA,
    "load_bhoonidhi_data": LOAD_BHOONIDHI_DATA,
    
    # Data Processing Tools
    "filter_by_attribute": FILTER_BY_ATTRIBUTE,
    "dissolve": DISSOLVE_ANALYSIS,
    "rename_layer": RENAME_LAYER,
    "reproject_layer": REPROJECT_LAYER,
    
    # Spatial Analysis Tools
    "buffer": BUFFER_ANALYSIS,
    "clip": CLIP_ANALYSIS,
    "spatial_join": SPATIAL_JOIN,
    "intersect": INTERSECT_ANALYSIS,
    
    # Analysis Tools
    "calculate_area": CALCULATE_AREA,
    "calculate_distance": CALCULATE_DISTANCE,
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_tool_by_name(operation_name: str) -> Optional[ToolDefinition]:
    """
    Retrieve a tool definition by its operation name.
    
    Args:
        operation_name: The name of the operation to retrieve
        
    Returns:
        ToolDefinition if found, None otherwise
    """
    return TOOL_REGISTRY.get(operation_name)


def get_tools_by_category(category: str) -> List[ToolDefinition]:
    """
    Get all tools in a specific category.
    
    Args:
        category: The category to filter by
        
    Returns:
        List of ToolDefinition objects in the specified category
    """
    return [tool for tool in TOOL_REGISTRY.values() if tool.category == category]


def get_all_categories() -> List[str]:
    """
    Get all unique categories of tools.
    
    Returns:
        List of unique category names
    """
    return list(set(tool.category for tool in TOOL_REGISTRY.values()))


def validate_tool_parameters(operation_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate parameters for a given tool operation.
    
    Args:
        operation_name: Name of the operation
        parameters: Dictionary of parameters to validate
        
    Returns:
        Dictionary of validation results
        
    Raises:
        ValueError: If operation not found or parameters are invalid
    """
    tool = get_tool_by_name(operation_name)
    if not tool:
        raise ValueError(f"Tool '{operation_name}' not found in registry")
    
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check required parameters
    required_params = [p.name for p in tool.parameters if p.required]
    missing_params = [p for p in required_params if p not in parameters]
    
    if missing_params:
        validation_results["valid"] = False
        validation_results["errors"].append(f"Missing required parameters: {missing_params}")
    
    # Check parameter types and constraints
    for param_def in tool.parameters:
        if param_def.name in parameters:
            value = parameters[param_def.name]
            
            # Basic type validation could be added here
            # For now, we'll just check if constraints exist
            if param_def.constraints:
                if "choices" in param_def.constraints:
                    if value not in param_def.constraints["choices"]:
                        validation_results["valid"] = False
                        validation_results["errors"].append(
                            f"Parameter '{param_def.name}' must be one of: {param_def.constraints['choices']}"
                        )
    
    return validation_results


def get_tool_summary() -> Dict[str, Any]:
    """
    Get a summary of all available tools.
    
    Returns:
        Dictionary containing tool statistics and categories
    """
    categories = get_all_categories()
    category_counts = {cat: len(get_tools_by_category(cat)) for cat in categories}
    
    return {
        "total_tools": len(TOOL_REGISTRY),
        "categories": categories,
        "category_counts": category_counts,
        "tool_names": list(TOOL_REGISTRY.keys())
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ToolParameter",
    "ToolDefinition", 
    "ParameterType",
    "TOOL_REGISTRY",
    "get_tool_by_name",
    "get_tools_by_category",
    "get_all_categories",
    "validate_tool_parameters",
    "get_tool_summary"
]