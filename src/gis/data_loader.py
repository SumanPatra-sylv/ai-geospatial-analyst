#!/usr/bin/env python3
"""
Smart Data Loader for Geospatial Data
=====================================
This module provides the SmartDataLoader class for fetching, caching, and loading
geospatial data from various sources including OpenStreetMap and local files.

Enhanced version with flexible OSM data fetching capabilities for AI-driven workflows.

Author: Generated for GIS Project (Enhanced Version)
File: src/gis/data_loader.py
"""

import re
import warnings
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd

# Configure OSMnx settings
cache_dir = Path("/app/data/cache")
cache_dir.mkdir(parents=True, exist_ok=True)
ox.settings.cache_folder = str(cache_dir)
ox.settings.use_cache = True

# Suppress noisy UserWarnings from osmnx about feature queries
warnings.filterwarnings('ignore', category=UserWarning, module='osmnx')


class SmartDataLoader:
    """
    A smart data loader for geospatial data with caching capabilities.
    
    This class provides methods to fetch OpenStreetMap data with intelligent
    caching and load pre-downloaded boundary files. Enhanced with flexible
    OSM data fetching for AI-driven geospatial workflows.
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
        
        # Initialize metadata tracking for better workflow management
        self.metadata = {
            'fetch_history': [],
            'error_log': [],
            'cache_stats': {'hits': 0, 'misses': 0}
        }
    
    def fetch_osm_data(self, location: str, tags: dict) -> gpd.GeoDataFrame:
        """
        Dynamically fetches and caches OpenStreetMap data based on a flexible tags dictionary.

        This is the primary method for loading data in response to an AI-generated plan.

        Args:
            location (str): The name of the location (e.g., "Potsdam, Germany").
            tags (dict): The OSM tags to filter by (e.g., {'amenity': 'school'}).

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the requested features.
                              Returns an empty GeoDataFrame if no features are found.
        """
        # Create a robust, unique cache filename based on location and the specific tags
        safe_location_name = re.sub(r'[^\w\.-]', '_', location.lower())
        tags_string = json.dumps(tags, sort_keys=True)
        tags_hash = hashlib.md5(tags_string.encode()).hexdigest()
        cache_file = self.cache_dir / f"{safe_location_name}_{tags_hash}.gpkg"

        # Log the fetch attempt
        self.metadata['fetch_history'].append({
            'location': location,
            'tags': tags,
            'timestamp': pd.Timestamp.now(),
            'cache_file': str(cache_file)
        })

        # 1. Check for cached data first
        if cache_file.exists():
            print(f"✅ Loading from cache: {cache_file}")
            self.metadata['cache_stats']['hits'] += 1
            try:
                return gpd.read_file(cache_file)
            except Exception as e:
                print(f"⚠️  Warning: Cache file corrupted, re-fetching. Error: {e}")
                cache_file.unlink()  # Delete corrupted file
                self.metadata['error_log'].append({
                    'type': 'cache_corruption',
                    'location': location,
                    'tags': tags,
                    'error': str(e),
                    'timestamp': pd.Timestamp.now()
                })

        # 2. If no cache, fetch from OSM
        print(f"⬇️  Fetching from OSM for '{location}' with tags: {tags}")
        self.metadata['cache_stats']['misses'] += 1
        
        try:
            gdf = ox.features_from_place(location, tags)

            # 3. Handle empty results gracefully
            if gdf.empty:
                warnings.warn(
                    f"No features found in '{location}' for tags {tags}. "
                    "An empty GeoDataFrame will be used. This is often normal."
                )
                # Create empty GeoDataFrame with proper structure
                gdf = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')
            else:
                # 4. Basic data validation and cleaning
                gdf = self._validate_and_clean_osm_data(gdf)
            
            # 5. Save to cache before returning
            print(f"💾 Saving to cache: {cache_file}")
            gdf.to_file(cache_file, driver='GPKG')

            return gdf

        except Exception as e:
            # Log the error for debugging
            self.metadata['error_log'].append({
                'type': 'osm_fetch_error',
                'location': location,
                'tags': tags,
                'error': str(e),
                'timestamp': pd.Timestamp.now()
            })
            
            # Re-raise a more informative error for the executor to catch
            error_msg = f"Failed to download or process OSM data for '{location}' with tags {tags}."
            print(f"❌ ERROR: {error_msg} (Original error: {e})")
            raise ConnectionError(error_msg) from e

    def _validate_and_clean_osm_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Validate and clean OSM data to ensure quality.
        
        Args:
            gdf (gpd.GeoDataFrame): Raw OSM data from osmnx.
            
        Returns:
            gpd.GeoDataFrame: Cleaned and validated data.
        """
        # Remove invalid geometries
        if not gdf.empty:
            initial_count = len(gdf)
            gdf = gdf[gdf.geometry.is_valid].copy()
            
            if len(gdf) < initial_count:
                print(f"⚠️  Removed {initial_count - len(gdf)} invalid geometries")
            
            # Reset index
            gdf = gdf.reset_index(drop=True)
            
            # Ensure CRS is set
            if gdf.crs is None:
                gdf.set_crs('EPSG:4326', inplace=True)
                print("📍 Set CRS to EPSG:4326")
        
        return gdf

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
            self.metadata['cache_stats']['hits'] += 1
            try:
                return gpd.read_file(cache_file)
            except Exception as e:
                print(f"Warning: Cache file corrupted, re-downloading. Error: {e}")
                cache_file.unlink()
        
        print(f"Downloading OSM land use data for: {city_name}")
        self.metadata['cache_stats']['misses'] += 1
        
        # Define OSM tags for land use data
        landuse_tags = {
            'landuse': ['residential', 'industrial', 'commercial'],
            'leisure': ['park', 'recreation_ground'],
            'amenity': ['hospital','clinic','pharmacy']
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
        
        # A nested mapping for precise and efficient categorization
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
    
    def batch_fetch_osm_data(self, requests: List[Dict[str, Any]]) -> Dict[str, gpd.GeoDataFrame]:
        """
        Batch fetch multiple OSM data requests efficiently.
        
        Args:
            requests (List[Dict]): List of dicts with 'location' and 'tags' keys.
            
        Returns:
            Dict[str, gpd.GeoDataFrame]: Dictionary mapping request IDs to GeoDataFrames.
        """
        results = {}
        
        for i, request in enumerate(requests):
            request_id = request.get('id', f"request_{i}")
            location = request['location']
            tags = request['tags']
            
            try:
                results[request_id] = self.fetch_osm_data(location, tags)
                print(f"✅ Completed batch request {request_id}")
            except Exception as e:
                print(f"❌ Failed batch request {request_id}: {e}")
                results[request_id] = gpd.GeoDataFrame()  # Return empty GDF
        
        return results
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about data loader operations.
        
        Returns:
            Dict: Metadata including fetch history, errors, and cache stats.
        """
        return self.metadata.copy()
    
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
            # Clear both old and new cache formats
            safe_city_name = re.sub(r'[^\w\.-]', '_', city_name.lower())
            
            # Clear old format cache
            cache_file = self.cache_dir / f"{safe_city_name}_landuse.gpkg"
            if cache_file.exists():
                cache_file.unlink()
                print(f"Cleared landuse cache for: {city_name}")
            
            # Clear new format cache (all files with location prefix)
            cleared_count = 0
            for cache_file in self.cache_dir.glob(f"{safe_city_name}_*.gpkg"):
                cache_file.unlink()
                cleared_count += 1
            
            if cleared_count > 0:
                print(f"Cleared {cleared_count} cache files for: {city_name}")
            else:
                print(f"No cache found for: {city_name}")
        else:
            cache_files = list(self.cache_dir.glob('*'))
            for cache_file in cache_files:
                if cache_file.is_file():
                    cache_file.unlink()
            print(f"Cleared {len(cache_files)} cache files.")
    
    def export_workflow_data(self, output_dir: str = "workflow_outputs") -> Dict[str, str]:
        """
        Export current workflow data for AI system integration.
        
        Args:
            output_dir (str): Directory to export workflow data.
            
        Returns:
            Dict[str, str]: Mapping of data types to file paths.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exports = {}
        
        # Export metadata
        metadata_file = output_path / "loader_metadata.json"
        with open(metadata_file, 'w') as f:
            # Convert timestamps to strings for JSON serialization
            metadata_copy = self.metadata.copy()
            for entry in metadata_copy.get('fetch_history', []):
                if 'timestamp' in entry:
                    entry['timestamp'] = entry['timestamp'].isoformat()
            for entry in metadata_copy.get('error_log', []):
                if 'timestamp' in entry:
                    entry['timestamp'] = entry['timestamp'].isoformat()
            
            json.dump(metadata_copy, f, indent=2)
        
        exports['metadata'] = str(metadata_file)
        
        # Export cache inventory
        cache_inventory = {
            'cached_files': self.list_cached_files(),
            'boundary_files': self.list_boundary_files(),
            'cache_directory': str(self.cache_dir),
            'boundaries_directory': str(self.boundaries_dir)
        }
        
        inventory_file = output_path / "cache_inventory.json"
        with open(inventory_file, 'w') as f:
            json.dump(cache_inventory, f, indent=2)
        
        exports['inventory'] = str(inventory_file)
        
        print(f"📄 Exported workflow data to: {output_path}")
        return exports


if __name__ == '__main__':
    """
    Example usage and testing for the SmartDataLoader class.
    """
    print("=== SmartDataLoader Test & Example Usage ===\n")
    
    loader = SmartDataLoader()
    
    # --- Test Case 1: Fetch specific features using the NEW generic method ---
    print("1. Fetching specific 'school' features using fetch_osm_data:")
    print("-" * 60)
    try:
        school_tags = {'amenity': 'school'}
        schools_gdf = loader.fetch_osm_data("Potsdam, Germany", school_tags)
        print(f"✅ Success! Found {len(schools_gdf)} schools in Potsdam.")
        if not schools_gdf.empty:
            print("Sample data:")
            print(schools_gdf.head())
        print()
    except Exception as e:
        print(f"❌ ERROR during fetch_osm_data test: {e}\n")

    # --- Test Case 2: Fetch another type of feature (parks) to test caching ---
    print("2. Fetching 'park' features to test separate caching:")
    print("-" * 60)
    try:
        park_tags = {'leisure': 'park'}
        parks_gdf = loader.fetch_osm_data("Potsdam, Germany", park_tags)
        print(f"✅ Success! Found {len(parks_gdf)} parks in Potsdam.")
        print()
    except Exception as e:
        print(f"❌ ERROR during fetch_osm_data test: {e}\n")

    # --- Test Case 3: Test batch fetching ---
    print("3. Testing batch fetch functionality:")
    print("-" * 60)
    try:
        batch_requests = [
            {'id': 'hospitals', 'location': 'Potsdam, Germany', 'tags': {'amenity': 'hospital'}},
            {'id': 'restaurants', 'location': 'Potsdam, Germany', 'tags': {'amenity': 'restaurant'}}
        ]
        batch_results = loader.batch_fetch_osm_data(batch_requests)
        for req_id, gdf in batch_results.items():
            print(f"✅ {req_id}: {len(gdf)} features")
        print()
    except Exception as e:
        print(f"❌ ERROR during batch fetch test: {e}\n")

    # --- Test Case 4: Test the legacy landuse function for backward compatibility ---
    print("4. Testing legacy fetch_osm_landuse method:")
    print("-" * 60)
    try:
        landuse_gdf = loader.fetch_osm_landuse("Potsdam, Germany")
        print(f"✅ Success! Found {len(landuse_gdf)} landuse features using legacy method.")
        print(f"Categories found: {landuse_gdf['category'].unique().tolist()}")
        print()
    except Exception as e:
        print(f"❌ ERROR during fetch_osm_landuse test: {e}\n")

    # --- Test Case 5: Display metadata and export workflow data ---
    print("5. Displaying metadata and exporting workflow data:")
    print("-" * 60)
    try:
        metadata = loader.get_metadata()
        print(f"Cache stats: {metadata['cache_stats']}")
        print(f"Fetch history entries: {len(metadata['fetch_history'])}")
        print(f"Error log entries: {len(metadata['error_log'])}")
        
        # Export workflow data
        exports = loader.export_workflow_data()
        print(f"Exported files: {list(exports.keys())}")
        print()
    except Exception as e:
        print(f"❌ ERROR during metadata/export test: {e}\n")

    # --- Test Case 6: Listing files ---
    print("6. Listing available files:")
    print("-" * 60)
    print(f"Cached files: {loader.list_cached_files() or 'None'}")
    print(f"Boundary files: {loader.list_boundary_files() or 'None'}")
    print()

    print("=== Test Complete ===")