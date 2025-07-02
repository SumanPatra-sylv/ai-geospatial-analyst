# core/schemas.py

from pydantic import BaseModel, Field

class ListAvailableDataSchema(BaseModel):
    """Input schema for the list_available_data tool. Takes no arguments."""
    pass

class LoadVectorDataSchema(BaseModel):
    """Input schema for the load_vector_data tool."""
    file_path: str = Field(description="The full path to the geospatial file to load (e.g., 'data/cities.shp').")
    layer_name: str = Field(description="The variable name to assign to the loaded layer in the data store.")

class SaveVectorDataSchema(BaseModel):
    """Input schema for the save_vector_data tool."""
    layer_name: str = Field(description="The name of the layer in the data store to save.")
    output_path: str = Field(description="The file path where the data will be saved (e.g., 'results/final_output.geojson').")

class CreateBufferSchema(BaseModel):
    """Input schema for the create_buffer tool."""
    input_layer_name: str = Field(description="The name of the layer in the data store to buffer.")
    distance_meters: float = Field(description="The buffer distance in meters. Must be a positive number.", gt=0)
    output_layer_name: str = Field(description="The variable name to assign to the new buffered layer.")

class IntersectLayersSchema(BaseModel):
    """Input schema for the intersect_layers tool."""
    layer_1_name: str = Field(description="The name of the first layer for the intersection.")
    layer_2_name: str = Field(description="The name of the second layer for the intersection.")
    output_layer_name: str = Field(description="The variable name to assign to the new intersection layer.")