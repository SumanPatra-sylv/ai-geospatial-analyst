#!/usr/bin/env python3
"""
Smart Data Loader for Geospatial Data
=====================================
This module provides the SmartDataLoader class for fetching, caching, and loading
geospatial data from various sources including OpenStreetMap and local files.

Author: Generated for GIS Project (Final Refined Version)
File: src/gis/data_loader.py
"""

import re
import warnings
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import osmnx as ox

# Suppress noisy UserWarnings from osmnx about feature queries.
# This can be commented out during debugging if library warnings are needed.
warnings.filterwarnings('ignore', category=UserWarning, module='osmnx')


class SmartDataLoader:
    """
    A smart data loader for geospatial data with caching capabilities.
    
    This class provides methods to fetch OpenStreetMap data with intelligent
    caching and load pre-downloaded boundary files.
    """
    
    def __init__(self, base_data_dir: str = "data"):
        """
        Initialize the SmartDataLoader.
        
        Args:
            base_data_dir (str): Base directory for data storage. Defaults to "data".
        """
        self.base_data_dir = Path(base_data_dir)
        self.cache_dir = self.base_data_dir / "cache"
        self.boundaries_dir = self.base_data_dir / "boundaries"
        
        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.boundaries_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_osm_landuse(self, city_name: str) -> gpd.GeoDataFrame:
        """
        Dynamically download land use data from OpenStreetMap with caching.
        
        Args:
            city_name (str): Name of the city (e.g., "Pune, India")
            
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing land use polygons with 
                             standardized categories
                             
        Raises:
            Exception: If OSM data cannot be downloaded or processed. The original
                       exception is chained for debugging.
        """
        # Create a robust, filesystem-safe cache file path
        safe_city_name = re.sub(r'[^\w\.-]', '_', city_name.lower())
        cache_file = self.cache_dir / f"{safe_city_name}_landuse.gpkg"
        
        # Check if cache file exists
        if cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            try:
                return gpd.read_file(cache_file)
            except Exception as e:
                print(f"Warning: Cache file corrupted, re-downloading. Error: {e}")
                cache_file.unlink()
        
        print(f"Downloading OSM land use data for: {city_name}")
        
        # Define OSM tags for land use data
        landuse_tags = {
            'landuse': ['residential', 'industrial', 'commercial'],
            'leisure': ['park', 'recreation_ground']
        }
        
        try:
            print("Fetching data from OpenStreetMap...")
            gdf_landuse = ox.features_from_place(city_name, tags=landuse_tags)
            
            if gdf_landuse.empty:
                raise ValueError(f"No land use data found for {city_name} with the specified tags.")
            
            processed_gdf = self._process_landuse_data(gdf_landuse)
            
            print(f"Saving to cache: {cache_file}")
            processed_gdf.to_file(cache_file, driver='GPKG')
            
            print(f"Successfully downloaded and cached {len(processed_gdf)} land use features.")
            return processed_gdf
            
        except Exception as e:
            error_msg = f"Failed to download OSM data for '{city_name}'"
            print(f"Error: {error_msg}: {e}")
            
            suggestions = [
                "Check your internet connection.",
                "Verify the city name spelling (e.g., 'City, Country').",
                "The city might not have sufficient OSM data for the requested tags."
            ]
            print("Suggestions:")
            for suggestion in suggestions:
                print(f"  - {suggestion}")
            
            raise Exception(error_msg) from e
    
    def _process_landuse_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Process and standardize OSM land use data using vectorized operations.
        
        Args:
            gdf (gpd.GeoDataFrame): Raw OSM data from osmnx.
            
        Returns:
            gpd.GeoDataFrame: Processed data with standardized categories.
        """
        processed_gdf = gdf.copy()
        
        # A nested mapping for precise and efficient categorization. This structure
        # ensures we only check for 'residential' in the 'landuse' column, etc.
        category_mapping = {
            'landuse': {
                'residential': 'residential',
                'industrial': 'industrial',
                'commercial': 'commercial',
            },
            'leisure': {
                'park': 'park',
                'recreation_ground': 'park'
            }
        }
        
        conditions = []
        choices = []
        
        # Build conditions and choices for categorization
        for osm_key, tag_map in category_mapping.items():
            if osm_key in processed_gdf.columns:
                for tag_value, category in tag_map.items():
                    conditions.append(processed_gdf[osm_key] == tag_value)
                    choices.append(category)
        
        # Use vectorized np.select for efficient conditional assignment
        processed_gdf['category'] = np.select(conditions, choices, default='other')
        
        # Keep only essential and available optional columns
        desired_columns = [
            'geometry', 'category', 'name', 
            'landuse', 'leisure', 'amenity'
        ]
        cols_to_keep = [col for col in desired_columns if col in processed_gdf.columns]
        processed_gdf = processed_gdf[cols_to_keep]
        
        # Ensure we have valid geometries and reset index
        processed_gdf = processed_gdf[processed_gdf.geometry.is_valid].reset_index(drop=True)
        
        return processed_gdf
    
    def load_manual_boundary(self, boundary_name: str) -> gpd.GeoDataFrame:
        """
        Load high-quality, pre-downloaded boundary files.
        
        Args:
            boundary_name (str): Name of the boundary file (without extension).
            
        Returns:
            gpd.GeoDataFrame: Loaded boundary data.
            
        Raises:
            FileNotFoundError: If the boundary file doesn't exist.
            Exception: If the file fails to load or is invalid.
        """
        extensions = ['.geojson', '.shp', '.gpkg', '.json']
        boundary_file = None
        for ext in extensions:
            potential_file = self.boundaries_dir / f"{boundary_name}{ext}"
            if potential_file.exists():
                boundary_file = potential_file
                break
        
        if boundary_file is None:
            available_files = [f.name for f in self.boundaries_dir.glob('*') if f.is_file()]
            error_msg = (
                f"Boundary file '{boundary_name}' not found in {self.boundaries_dir}\n"
                f"Looked for extensions: {extensions}\n"
                f"Available files: {available_files if available_files else 'None'}"
            )
            raise FileNotFoundError(error_msg)
        
        try:
            print(f"Loading boundary file: {boundary_file}")
            gdf = gpd.read_file(boundary_file)
            
            if gdf.empty:
                raise ValueError(f"Boundary file '{boundary_file}' is empty.")
            if not hasattr(gdf, 'geometry') or gdf.geometry.isna().all():
                raise ValueError(f"Boundary file '{boundary_file}' has no valid geometries.")
            
            print(f"Successfully loaded {len(gdf)} boundary features.")
            return gdf
            
        except Exception as e:
            error_msg = f"Failed to load or validate boundary file '{boundary_file}'"
            print(f"Error: {error_msg}: {e}")
            raise Exception(error_msg) from e
    
    def list_cached_files(self) -> List[str]:
        """List all cached files."""
        return [f.name for f in self.cache_dir.glob('*') if f.is_file()]
    
    def list_boundary_files(self) -> List[str]:
        """List all available boundary files."""
        return [f.name for f in self.boundaries_dir.glob('*') if f.is_file()]
    
    def clear_cache(self, city_name: Optional[str] = None) -> None:
        """
        Clear cache files.
        
        Args:
            city_name (str, optional): Specific city to clear. If None, clears all cache.
        """
        if city_name:
            safe_city_name = re.sub(r'[^\w\.-]', '_', city_name.lower())
            cache_file = self.cache_dir / f"{safe_city_name}_landuse.gpkg"
            if cache_file.exists():
                cache_file.unlink()
                print(f"Cleared cache for: {city_name}")
            else:
                print(f"No cache found for: {city_name}")
        else:
            for cache_file in self.cache_dir.glob('*'):
                if cache_file.is_file():
                    cache_file.unlink()
            print("Cleared all cache files.")


if __name__ == '__main__':
    """
    Example usage of the SmartDataLoader class.
    """
    print("=== SmartDataLoader Example Usage ===\n")
    
    loader = SmartDataLoader()
    
    # Example 1: Fetch OSM land use data
    print("1. Fetching OSM Land Use Data for 'Potsdam, Germany':")
    print("-" * 50)
    try:
        # Using a well-mapped city for a good example
        potsdam_landuse = loader.fetch_osm_landuse("Potsdam, Germany")
        print(f"Loaded {len(potsdam_landuse)} land use features for Potsdam.")
        print(f"CRS: {potsdam_landuse.crs}")
        print(f"Categories found: {potsdam_landuse['category'].unique().tolist()}")
        print("\nSample data:")
        print(potsdam_landuse[['category', 'name']].head())
        print()
    except Exception as e:
        print(f"ERROR during fetch: {e}\n")

    # Example 2: Load a manual boundary (will fail unless you create the file)
    print("2. Loading Manual Boundary 'test_area':")
    print("-" * 50)
    try:
        # To make this example work:
        # 1. Create a directory `data/boundaries`
        # 2. Place a shapefile/geojson named `test_area.shp` inside it.
        boundary = loader.load_manual_boundary("test_area")
        print(f"Loaded boundary with {len(boundary)} features. CRS: {boundary.crs}\n")
    except FileNotFoundError as e:
        print(f"As expected, file not found. Details:\n{e}\n")
    except Exception as e:
        print(f"An unexpected error occurred: {e}\n")
    
    # Example 3: List available files
    print("3. Listing Available Files:")
    print("-" * 50)
    print(f"Cached files: {loader.list_cached_files() or 'None'}")
    print(f"Boundary files: {loader.list_boundary_files() or 'None'}")
    print()

    print("=== Example Complete ===")