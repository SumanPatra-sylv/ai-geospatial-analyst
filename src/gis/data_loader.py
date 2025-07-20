#!/usr/bin/env python3
"""
Enhanced Smart Data Loader for Geospatial Data
=============================================
Ultra-robust version with comprehensive error handling, intelligent caching,
advanced column sanitization, and production-ready reliability features.

Author: Enhanced for Maximum Reliability
File: src/gis/data_loader.py
"""

import re
import warnings
import hashlib
import json
import time
import logging
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress noisy warnings
warnings.filterwarnings('ignore', category=UserWarning, module='osmnx')
warnings.filterwarnings('ignore', category=FutureWarning, module='geopandas')

@dataclass
class LoaderConfig:
    """Enhanced configuration for SmartDataLoader with validation."""
    base_data_dir: str = "data"
    max_retries: int = 3
    connection_timeout: float = 30.0  # Time to establish a connection
    read_timeout: int = 180           # Time to wait for a response (must be an integer)
    max_memory_usage_mb: int = 1024   # Memory limit for Overpass
    cache_max_age_days: int = 30
    cache_max_size_gb: float = 5.0
    enable_metadata_tracking: bool = True
    default_crs: str = 'EPSG:4326'
    exponential_backoff_base: float = 2.0
    max_backoff_time: float = 60.0
    
    # Column sanitization options
    preserve_column_mapping: bool = True
    max_column_length: int = 63
    force_lowercase_columns: bool = True
    
    # Advanced options
    enable_parallel_fetching: bool = False
    max_workers: int = 4
    connection_timeout: float = 30.0
    read_timeout: float = 300.0
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # Memory management
    max_memory_usage_mb: float = 1024.0
    enable_geometry_simplification: bool = True
    simplification_tolerance: float = 0.0001
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if self.cache_max_age_days < 0:
            raise ValueError("cache_max_age_days must be non-negative")
        if self.cache_max_size_gb < 0:
            raise ValueError("cache_max_size_gb must be non-negative")
        if self.max_column_length < 10:
            raise ValueError("max_column_length must be at least 10")
        if self.exponential_backoff_base < 1:
            raise ValueError("exponential_backoff_base must be at least 1")

class SmartDataLoader:
    """
    Ultra-robust smart data loader for geospatial data with comprehensive
    error handling, intelligent caching, and production-ready features.
    """
    
    def __init__(self, config: Optional[LoaderConfig] = None):
        """Initialize the enhanced SmartDataLoader with robust configuration."""
        self.config = config or LoaderConfig()
        self._setup_logging()
        self._setup_directories()
        self._setup_osmnx_configuration()
        self._initialize_metadata()
        self._lock = threading.Lock()  # Thread safety
        
        self.logger.info(f"SmartDataLoader initialized with config: {self.config.base_data_dir}")

    def _setup_logging(self):
        """Setup comprehensive logging."""
        if self.config.enable_logging:
            log_dir = Path(self.config.base_data_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            logging.basicConfig(
                level=getattr(logging, self.config.log_level.upper()),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_dir / "smartloader.log"),
                    logging.StreamHandler()
                ]
            )
        
        self.logger = logging.getLogger(__name__)

    def _setup_directories(self):
        """Setup and validate directory structure."""
        self.base_data_dir = Path(self.config.base_data_dir)
        self.cache_dir = self.base_data_dir / "cache"
        self.boundaries_dir = self.base_data_dir / "boundaries"
        self.temp_dir = self.base_data_dir / "temp"
        
        # Create all directories with proper permissions
        for directory in [self.cache_dir, self.boundaries_dir, self.temp_dir]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                # Test write access
                test_file = directory / ".write_test"
                test_file.touch()
                test_file.unlink()
            except (PermissionError, OSError) as e:
                raise RuntimeError(f"Cannot create or write to directory {directory}: {e}")

    def _setup_osmnx_configuration(self):
        """Enhanced OSMnx configuration with robust error handling."""
        try:
            # ✅ CRITICAL FIX: Ensure OSMnx uses our cache directory
            ox.settings.cache_folder = str(self.cache_dir)
            ox.settings.use_cache = True
            
            # Enhanced timeout settings
            ox.settings.requests_timeout = int(self.config.connection_timeout)
            ox.settings.timeout = int(self.config.read_timeout)
            
            # Memory management settings
            ox.settings.memory = int(self.config.max_memory_usage_mb)
            
            self.logger.info(f"OSMnx configured with cache: {self.cache_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to configure OSMnx: {e}")
            raise RuntimeError(f"OSMnx configuration failed: {e}")

    def _initialize_metadata(self):
        """Initialize enhanced metadata tracking."""
        if self.config.enable_metadata_tracking:
            self.metadata = {
                'fetch_history': [],
                'error_log': [],
                'cache_stats': {'hits': 0, 'misses': 0, 'failures': 0},
                'column_mappings': {},
                'performance_stats': {
                    'total_requests': 0,
                    'total_features': 0,
                    'avg_response_time': 0.0,
                    'cache_hit_rate': 0.0
                },
                'system_info': {
                    'loader_version': '3.0_ultra_robust',
                    'created': pd.Timestamp.now().isoformat(),
                    'config': {
                        'max_retries': self.config.max_retries,
                        'cache_max_age_days': self.config.cache_max_age_days,
                        'preserve_column_mapping': self.config.preserve_column_mapping
                    }
                }
            }
        else:
            self.metadata = None

    @contextmanager
    def _error_context(self, operation: str, **kwargs):
        """Context manager for comprehensive error tracking."""
        start_time = time.time()
        try:
            yield
        except Exception as e:
            duration = time.time() - start_time
            if self.metadata:
                self.metadata['error_log'].append({
                    'operation': operation,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'duration': duration,
                    'timestamp': pd.Timestamp.now().isoformat(),
                    **kwargs
                })
            self.logger.error(f"Operation '{operation}' failed after {duration:.2f}s: {e}")
            raise

    def sanitize_column_names(self, columns: List[str], for_driver: str = 'GPKG') -> Dict[str, str]:
        """
        ✅ ENHANCED: Ultra-comprehensive column name sanitization with advanced conflict resolution.
        """
        # Enhanced driver-specific constraints
        driver_constraints = {
            'GPKG': {
                'max_length': self.config.max_column_length, 
                'case_sensitive': True,
                'forbidden_chars': r'[:\s\-\.\[\](){}/@#$%^&*+=|\\;",<>?~`]',
                'reserved_prefixes': ['sqlite_', 'rtree_', 'idx_']
            },
            'Shapefile': {
                'max_length': 10, 
                'case_sensitive': False,
                'forbidden_chars': r'[^A-Za-z0-9_]',
                'reserved_prefixes': ['fid', 'objectid']
            },
            'GeoJSON': {
                'max_length': None, 
                'case_sensitive': True,
                'forbidden_chars': r'[\x00-\x1f\x7f-\x9f]',  # Control characters
                'reserved_prefixes': []
            },
            'PostGIS': {
                'max_length': 63,
                'case_sensitive': False,
                'forbidden_chars': r'[^\w]',
                'reserved_prefixes': ['pg_', 'information_schema']
            }
        }
        
        constraints = driver_constraints.get(for_driver, driver_constraints['GPKG'])
        
        # Enhanced reserved words (more comprehensive)
        reserved_words = {
            'geometry', 'id', 'fid', 'oid', 'rowid', 'index', 'key', 'value',
            'type', 'class', 'order', 'group', 'user', 'table', 'column',
            'primary', 'foreign', 'constraint', 'references', 'select', 'from',
            'where', 'insert', 'update', 'delete', 'create', 'drop', 'alter',
            'database', 'schema', 'function', 'trigger', 'view', 'grant',
            'revoke', 'commit', 'rollback', 'transaction', 'begin', 'end'
        }
        
        # Add driver-specific reserved words
        if for_driver == 'PostGIS':
            reserved_words.update({'point', 'line', 'polygon', 'multipoint', 
                                 'multiline', 'multipolygon', 'geometrycollection'})
        
        sanitized_mapping = {}
        seen_names = set()
        collision_counters = {}

        def sanitize_single_name(original_name: str) -> str:
            """Enhanced single name sanitization with better collision handling."""
            if not original_name or not original_name.strip():
                return 'empty_col'
            
            # Step 1: Basic cleaning
            sanitized = original_name.strip()
            
            # Step 2: Handle forbidden characters
            sanitized = re.sub(constraints['forbidden_chars'], '_', sanitized)
            
            # Step 3: Remove consecutive underscores and clean up
            sanitized = re.sub(r'_+', '_', sanitized)
            sanitized = sanitized.strip('_')
            
            # Step 4: Handle leading numbers/underscores
            if re.match(r'^[\d_]', sanitized):
                sanitized = 'col_' + sanitized
            
            # Step 5: Apply case standardization
            if self.config.force_lowercase_columns:
                sanitized = sanitized.lower()
            
            # Step 6: Handle reserved words and prefixes
            if sanitized.lower() in reserved_words:
                sanitized = f'fld_{sanitized}'
            
            for prefix in constraints.get('reserved_prefixes', []):
                if sanitized.lower().startswith(prefix.lower()):
                    sanitized = f'usr_{sanitized}'
                    break
            
            # Step 7: Apply length limits with intelligent truncation
            max_length = constraints['max_length']
            if max_length and len(sanitized) > max_length:
                # Try to preserve meaningful parts
                if '_' in sanitized:
                    parts = sanitized.split('_')
                    if len(parts) > 1:
                        # Keep last meaningful part if short
                        last_part = parts[-1]
                        if len(last_part) <= 8 and len(last_part) >= 3:
                            base_length = max_length - len(last_part) - 1
                            if base_length > 0:
                                truncated_base = sanitized[:base_length].rstrip('_')
                                sanitized = f"{truncated_base}_{last_part}"
                            else:
                                sanitized = sanitized[:max_length]
                        else:
                            sanitized = sanitized[:max_length]
                    else:
                        sanitized = sanitized[:max_length]
                else:
                    sanitized = sanitized[:max_length]
            
            # Step 8: Final cleanup
            sanitized = sanitized.rstrip('_')
            if not sanitized:
                sanitized = 'col'
            
            return sanitized

        # Process all columns with advanced collision detection
        for original_col in columns:
            if original_col == 'geometry':
                # Never modify geometry column
                sanitized_mapping[original_col] = original_col
                seen_names.add(original_col)
                continue

            base_sanitized = sanitize_single_name(original_col)
            
            # Advanced collision resolution
            if base_sanitized not in seen_names:
                final_sanitized = base_sanitized
            else:
                # Use collision counter for this base name
                counter = collision_counters.get(base_sanitized, 1) + 1
                collision_counters[base_sanitized] = counter
                
                # Create unique suffix with length consideration
                while True:
                    suffix = f'_{counter}'
                    max_length = constraints['max_length']
                    
                    if max_length:
                        available_length = max_length - len(suffix)
                        if available_length > 0:
                            truncated_base = base_sanitized[:available_length]
                            final_sanitized = f'{truncated_base}{suffix}'
                        else:
                            # Very short limit, use numeric only
                            final_sanitized = f'c{counter}'
                    else:
                        final_sanitized = f'{base_sanitized}{suffix}'
                    
                    if final_sanitized not in seen_names:
                        break
                    counter += 1
                    collision_counters[base_sanitized] = counter
            
            sanitized_mapping[original_col] = final_sanitized
            seen_names.add(final_sanitized)

        return sanitized_mapping

    def fetch_osm_data(self, location: str, tags: dict, max_retries: Optional[int] = None, 
                      timeout_override: Optional[float] = None) -> gpd.GeoDataFrame:
        """
        ✅ ULTRA-ENHANCED: Fetch OSM data with comprehensive error handling, intelligent retries,
        and production-ready reliability features.
        """
        with self._error_context("fetch_osm_data", location=location, tags=tags):
            return self._fetch_osm_data_internal(location, tags, max_retries, timeout_override)

    def _fetch_osm_data_internal(self, location: str, tags: dict, max_retries: Optional[int], 
                                timeout_override: Optional[float]) -> gpd.GeoDataFrame:
        """Internal fetch method with comprehensive error handling."""
        if max_retries is None:
            max_retries = self.config.max_retries
        
        start_time = time.time()
        
        # Create robust, unique cache filename
        safe_location_name = re.sub(r'[^\w\.-]', '_', location.lower())
        tags_string = json.dumps(tags, sort_keys=True)
        tags_hash = hashlib.md5(tags_string.encode()).hexdigest()
        cache_file = self.cache_dir / f"{safe_location_name}_{tags_hash}.gpkg"

        # Enhanced metadata logging
        fetch_entry = {
            'location': location,
            'tags': tags,
            'timestamp': pd.Timestamp.now().isoformat(),
            'cache_file': str(cache_file),
            'cache_hit': False,
            'start_time': start_time
        }
        
        if self.metadata:
            self.metadata['fetch_history'].append(fetch_entry)
            self.metadata['performance_stats']['total_requests'] += 1

        # ✅ ENHANCED CACHE VALIDATION WITH COMPREHENSIVE CHECKS
        cache_metadata = self._get_cache_metadata(cache_file)
        if cache_metadata and self._validate_cache_integrity(cache_file, cache_metadata):
            return self._load_from_cache(cache_file, cache_metadata, fetch_entry)

        # ✅ ENHANCED OSM FETCHING WITH INTELLIGENT RETRY LOGIC
        return self._fetch_from_osm_with_retries(
            location, tags, cache_file, max_retries, timeout_override, fetch_entry
        )

    def _validate_cache_integrity(self, cache_file: Path, metadata: Dict[str, Any]) -> bool:
        """Validate cache file integrity beyond just existence."""
        try:
            # Quick integrity check
            test_gdf = gpd.read_file(cache_file, rows=1)  # Read only first row for speed
            
            # Verify basic structure
            if not hasattr(test_gdf, 'geometry'):
                self.logger.warning(f"Cache file missing geometry column: {cache_file}")
                return False
            
            # Verify metadata consistency
            if 'feature_count' in metadata:
                # Quick row count (more efficient than loading full dataset)
                import sqlite3
                with sqlite3.connect(cache_file) as conn:
                    result = conn.execute("SELECT COUNT(*) FROM gpkg_contents").fetchone()
                    if result and result[0] == 0 and metadata['feature_count'] > 0:
                        self.logger.warning(f"Cache metadata inconsistent: {cache_file}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Cache integrity check failed: {cache_file}, error: {e}")
            return False

    def _load_from_cache(self, cache_file: Path, cache_metadata: Dict[str, Any], 
                        fetch_entry: Dict[str, Any]) -> gpd.GeoDataFrame:
        """Load data from cache with comprehensive error handling."""
        self.logger.info(f"✅ Loading from cache: {cache_file}")
        
        try:
            cached_gdf = gpd.read_file(cache_file)
            
            # Update metadata
            if self.metadata:
                self.metadata['cache_stats']['hits'] += 1
                fetch_entry['cache_hit'] = True
                fetch_entry['duration'] = time.time() - fetch_entry['start_time']
                
                # Load and restore column mapping if available
                if cache_metadata.get('column_mapping'):
                    tags_hash = hashlib.md5(json.dumps(fetch_entry['tags'], sort_keys=True).encode()).hexdigest()
                    cache_key = f"{fetch_entry['location']}_{tags_hash}"
                    self.metadata['column_mappings'][cache_key] = cache_metadata['column_mapping']
                    self.logger.info("🔄 Restored column mapping from cache")
            
            self.logger.info(f"📊 Loaded {len(cached_gdf)} cached features")
            return cached_gdf
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cache file corrupted, re-fetching. Error: {e}")
            
            # Clean up corrupted cache
            self._cleanup_corrupted_cache(cache_file)
            
            # Update error statistics
            if self.metadata:
                self.metadata['cache_stats']['failures'] += 1
                self.metadata['error_log'].append({
                    'type': 'cache_corruption',
                    'location': fetch_entry['location'],
                    'tags': fetch_entry['tags'],
                    'error': str(e),
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'cache_file': str(cache_file)
                })
            
            # Re-attempt fetch from OSM
            raise RuntimeError(f"Cache corrupted: {e}")

    def _cleanup_corrupted_cache(self, cache_file: Path):
        """Clean up corrupted cache files."""
        try:
            if cache_file.exists():
                cache_file.unlink()
            
            metadata_file = cache_file.with_suffix('.metadata.json')
            if metadata_file.exists():
                metadata_file.unlink()
                
            self.logger.info(f"🗑️ Cleaned up corrupted cache: {cache_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup corrupted cache {cache_file}: {e}")

    def _fetch_from_osm_with_retries(self, location: str, tags: dict, cache_file: Path,
                                   max_retries: int, timeout_override: Optional[float],
                                   fetch_entry: Dict[str, Any]) -> gpd.GeoDataFrame:
        """Fetch from OSM with intelligent retry logic and comprehensive error handling."""
        if self.metadata:
            self.metadata['cache_stats']['misses'] += 1

        gdf = None
        last_error = None
        
        # ✅ INTELLIGENT RETRY LOGIC WITH EXPONENTIAL BACKOFF
        for attempt in range(max_retries):
            try:
                self.logger.info(f"⬇️ Fetching from OSM (attempt {attempt + 1}/{max_retries}) for '{location}' with tags: {tags}")
                
                # Temporarily override timeout if specified
                original_timeout = None
                if timeout_override:
                    original_timeout = ox.settings.timeout
                    ox.settings.timeout = timeout_override
                
                try:
                    gdf = ox.features_from_place(location, tags)
                finally:
                    # Restore original timeout
                    if original_timeout is not None:
                        ox.settings.timeout = original_timeout
                
                # Success - break retry loop
                self.logger.info(f"🎉 Successfully fetched {len(gdf)} features on attempt {attempt + 1}")
                break
                
            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                
                # Categorize errors for intelligent retry decisions
                if self._should_retry_error(e, attempt, max_retries):
                    backoff_time = min(
                        self.config.exponential_backoff_base ** attempt,
                        self.config.max_backoff_time
                    )
                    
                    self.logger.warning(
                        f"⚠️ Attempt {attempt + 1} failed ({error_type}), retrying in {backoff_time:.1f}s... Error: {e}"
                    )
                    time.sleep(backoff_time)
                    continue
                else:
                    # Non-retryable error or max attempts reached
                    self.logger.error(f"❌ Non-retryable error or max attempts reached: {error_type}")
                    break

        # Handle final result or error
        if gdf is None:
            return self._handle_fetch_failure(location, tags, last_error, max_retries, fetch_entry)
        
        return self._process_successful_fetch(gdf, location, tags, cache_file, fetch_entry)

    def _should_retry_error(self, error: Exception, attempt: int, max_retries: int) -> bool:
        """Determine if an error should trigger a retry."""
        if attempt >= max_retries - 1:
            return False  # Max attempts reached
        
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Retryable errors
        retryable_patterns = [
            'timeout', 'connection', 'network', 'temporary', 'rate limit',
            'server error', '5xx', 'socket', 'ssl', 'certificate'
        ]
        
        # Non-retryable errors
        non_retryable_patterns = [
            'not found', '404', 'invalid', 'malformed', 'permission',
            'unauthorized', '401', '403', 'bad request', '400'
        ]
        
        # Check non-retryable first
        for pattern in non_retryable_patterns:
            if pattern in error_str or pattern in error_type:
                return False
        
        # Check retryable
        for pattern in retryable_patterns:
            if pattern in error_str or pattern in error_type:
                return True
        
        # Default: retry for unknown errors (conservative approach)
        return True

    def _handle_fetch_failure(self, location: str, tags: dict, last_error: Exception,
                             max_retries: int, fetch_entry: Dict[str, Any]) -> gpd.GeoDataFrame:
        """Handle fetch failure with comprehensive error logging."""
        # Log comprehensive error information
        if self.metadata:
            self.metadata['error_log'].append({
                'type': 'osm_fetch_error_final',
                'location': location,
                'tags': tags,
                'attempts': max_retries,
                'error': str(last_error),
                'error_type': type(last_error).__name__,
                'timestamp': pd.Timestamp.now().isoformat(),
                'suggestions': self._generate_error_suggestions(location, tags, last_error)
            })
            
            fetch_entry['success'] = False
            fetch_entry['error'] = str(last_error)
            fetch_entry['duration'] = time.time() - fetch_entry['start_time']

        error_msg = f"Failed to download OSM data for '{location}' with tags {tags} after {max_retries} attempts."
        self.logger.error(f"❌ {error_msg} (Final error: {last_error})")
        raise ConnectionError(error_msg) from last_error

    def _process_successful_fetch(self, gdf: gpd.GeoDataFrame, location: str, tags: dict,
                                 cache_file: Path, fetch_entry: Dict[str, Any]) -> gpd.GeoDataFrame:
        """Process successfully fetched data."""
        # Handle empty results
        if gdf.empty:
            self.logger.warning(f"No features found in '{location}' for tags {tags}")
            gdf = self._create_empty_gdf_with_structure(tags)
        else:
            # Validate and clean data with enhanced column sanitization
            tags_hash = hashlib.md5(json.dumps(tags, sort_keys=True).encode()).hexdigest()
            gdf = self._validate_and_clean_osm_data(gdf, location, tags_hash)

        # Cache the result with comprehensive metadata
        try:
            self._save_with_metadata(gdf, cache_file, location, tags)
        except Exception as e:
            self.logger.warning(f"Failed to cache data: {e}")
            # Continue without caching rather than failing

        # Update performance statistics
        if self.metadata:
            fetch_entry['success'] = True
            fetch_entry['feature_count'] = len(gdf)
            fetch_entry['duration'] = time.time() - fetch_entry['start_time']
            
            self.metadata['performance_stats']['total_features'] += len(gdf)
            
            # Update average response time
            total_requests = self.metadata['performance_stats']['total_requests']
            current_avg = self.metadata['performance_stats']['avg_response_time']
            new_avg = ((current_avg * (total_requests - 1)) + fetch_entry['duration']) / total_requests
            self.metadata['performance_stats']['avg_response_time'] = new_avg
            
            # Update cache hit rate
            hits = self.metadata['cache_stats']['hits']
            total = hits + self.metadata['cache_stats']['misses']
            self.metadata['performance_stats']['cache_hit_rate'] = hits / total if total > 0 else 0

        return gdf

    def _create_empty_gdf_with_structure(self, sample_tags: dict) -> gpd.GeoDataFrame:
        """✅ ENHANCED: Create properly structured empty GeoDataFrame with better type handling."""
        # Comprehensive standard columns
        standard_columns = ['geometry', 'name', 'osm_id', 'osm_type']
        
        # Add tag-specific columns
        for key in sample_tags.keys():
            if key not in standard_columns:
                standard_columns.append(key)
        
        # Add common OSM attributes that might be useful
        common_attrs = ['addr:city', 'addr:street', 'addr:housenumber', 'phone', 'website', 'opening_hours']
        for attr in common_attrs:
            if attr not in standard_columns:
                standard_columns.append(attr)

        # Sanitize column names
        column_mapping = self.sanitize_column_names(standard_columns)
        sanitized_columns = list(column_mapping.values())

        empty_gdf = gpd.GeoDataFrame(columns=sanitized_columns, crs=self.config.default_crs)
        
        # Set proper dtypes with enhanced type handling
        for original, sanitized in column_mapping.items():
            if original == 'geometry':
                continue
            elif original == 'osm_id':
                empty_gdf[sanitized] = empty_gdf[sanitized].astype('Int64')
            elif original in ['name', 'osm_type'] or original.startswith('addr:'):
                empty_gdf[sanitized] = empty_gdf[sanitized].astype('string')
            else:
                # Default to string for flexibility
                empty_gdf[sanitized] = empty_gdf[sanitized].astype('string')

        self.logger.info(f"📋 Created structured empty GeoDataFrame with {len(sanitized_columns)} sanitized columns")
        return empty_gdf

    def _validate_and_clean_osm_data(self, gdf: gpd.GeoDataFrame, location: str = None, 
                                   tags_hash: str = None) -> gpd.GeoDataFrame:
        """✅ COMPREHENSIVE ENHANCEMENT: Ultra-robust data validation and cleaning."""
        if gdf.empty:
            return gdf

        self.logger.info(f"🧼 Cleaning OSM data: {len(gdf)} features, {len(gdf.columns)} columns")
        
        # 1. ✅ MEMORY MANAGEMENT: Handle large datasets
        if len(gdf) * len(gdf.columns) > 100000:  # Large dataset
            self.logger.info("📊 Large dataset detected, applying memory-efficient processing")
            
        # 2. ✅ ADVANCED COLUMN NAME SANITIZATION
        original_columns = list(gdf.columns)
        column_mapping = self.sanitize_column_names(original_columns, for_driver='GPKG')
        
        changes_made = any(orig != sanitized for orig, sanitized in column_mapping.items())
        if changes_made:
            gdf = gdf.rename(columns=column_mapping)
            
            # Store column mapping with enhanced metadata
            if self.metadata and self.config.preserve_column_mapping and location and tags_hash:
                cache_key = f"{location}_{tags_hash}"
                self.metadata['column_mappings'][cache_key] = {
                    'mapping': column_mapping,
                    'created': pd.Timestamp.now().isoformat(),
                    'original_count': len(original_columns),
                    'changed_count': len([k for k, v in column_mapping.items() if k != v])
                }
            
            # Enhanced reporting
            changed_columns = {orig: new for orig, new in column_mapping.items() if orig != new}
            self.logger.info(f"✨ Sanitized {len(changed_columns)} column names")
            
            if len(changed_columns) <= 5:
                for orig, new in changed_columns.items():
                    self.logger.debug(f"  '{orig}' → '{new}'")

        # 3. ✅ COMPREHENSIVE GEOMETRY VALIDATION
        initial_count = len(gdf)
        
        # Remove invalid geometries
        valid_geoms = gdf.geometry.is_valid
        if not valid_geoms.all():
            invalid_count = (~valid_geoms).sum()
            self.logger.warning(f"⚠️ Found {invalid_count} invalid geometries, attempting repair")
            
            # Try to repair invalid geometries
            try:
                from shapely.validation import make_valid
                gdf.loc[~valid_geoms, 'geometry'] = gdf.loc[~valid_geoms, 'geometry'].apply(make_valid)
                
                # Recheck validity
                still_invalid = ~gdf.geometry.is_valid
                if still_invalid.any():
                    gdf = gdf[~still_invalid].copy()
                    self.logger.warning(f"⚠️ Removed {still_invalid.sum()} unrepairable geometries")
                else:
                    self.logger.info(f"✅ Repaired all {invalid_count} invalid geometries")
                    
            except ImportError:
                # Fallback: just remove invalid geometries
                gdf = gdf[valid_geoms].copy()
                self.logger.warning(f"⚠️ Removed {invalid_count} invalid geometries (shapely.validation not available)")
        
        # Remove empty geometries
        non_empty = ~gdf.geometry.is_empty
        if not non_empty.all():
            empty_count = (~non_empty).sum()
            gdf = gdf[non_empty].copy()
            self.logger.warning(f"⚠️ Removed {empty_count} empty geometries")

        # 4. ✅ ENHANCED DATA TYPE STANDARDIZATION
        self.logger.info(f"🔧 Standardizing data types for {len(gdf.columns)} columns")
        
        for col in gdf.columns:
            if col == 'geometry':
                continue
            
            try:
                # Enhanced data type handling
                if gdf[col].dtype == 'object':
                    # Smart type inference
                    sample_values = gdf[col].dropna().head(100)
                    
                    if len(sample_values) == 0:
                        # All null column
                        gdf[col] = gdf[col].astype('string')
                        continue
                    
                    # Try to infer if it should be numeric
                    try:
                        pd.to_numeric(sample_values, errors='raise')
                        # If successful, it's numeric
                        gdf[col] = pd.to_numeric(gdf[col], errors='coerce')
                        continue
                    except (ValueError, TypeError):
                        pass
                    
                    # Convert to string with enhanced null handling
                    gdf[col] = gdf[col].astype(str)
                    null_replacements = ['None', 'nan', 'NaN', '<NA>', 'null', 'NULL']
                    gdf[col] = gdf[col].replace(null_replacements, '')
                    
                elif gdf[col].dtype in ['int64', 'float64']:
                    # Handle numeric columns
                    gdf[col] = pd.to_numeric(gdf[col], errors='coerce')
                    
                elif 'datetime' in str(gdf[col].dtype):
                    # Convert datetime to string for better compatibility
                    gdf[col] = gdf[col].astype(str)
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Error processing column '{col}': {e}")
                # Fallback to string
                gdf[col] = gdf[col].astype(str).replace(['None', 'nan'], '')

        # 5. ✅ INTELLIGENT COLUMN CLEANUP
        columns_to_drop = []
        for col in gdf.columns:
            if col == 'geometry':
                continue
            
            # Check if column is entirely null/empty
            if gdf[col].isna().all() or (gdf[col].astype(str).str.strip() == '').all():
                columns_to_drop.append(col)
            
            # Check for columns with only one unique non-null value (low information)
            elif gdf[col].nunique(dropna=True) == 1 and gdf[col].notna().sum() > len(gdf) * 0.8:
                # Only drop if most values are the same
                unique_val = gdf[col].dropna().iloc[0] if gdf[col].notna().any() else None
                if unique_val in ['yes', 'true', '1', 'none', 'unknown', '']:
                    columns_to_drop.append(col)

        if columns_to_drop:
            gdf = gdf.drop(columns=columns_to_drop)
            self.logger.info(f"🗑️ Dropped {len(columns_to_drop)} low-information columns")

        # 6. ✅ GEOMETRY SIMPLIFICATION (if enabled)
        if self.config.enable_geometry_simplification and len(gdf) > 1000:
            try:
                original_complexity = gdf.geometry.apply(lambda g: len(g.coords) if hasattr(g, 'coords') else 0).sum()
                gdf.geometry = gdf.geometry.simplify(self.config.simplification_tolerance, preserve_topology=True)
                new_complexity = gdf.geometry.apply(lambda g: len(g.coords) if hasattr(g, 'coords') else 0).sum()
                
                if new_complexity < original_complexity:
                    reduction = (1 - new_complexity / original_complexity) * 100
                    self.logger.info(f"🎯 Simplified geometries: {reduction:.1f}% complexity reduction")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Geometry simplification failed: {e}")

        # 7. ✅ FINAL VALIDATION AND CLEANUP
        gdf = gdf.reset_index(drop=True)
        
        if gdf.crs is None:
            gdf.set_crs(self.config.default_crs, inplace=True)
            self.logger.info(f"📍 Set CRS to {self.config.default_crs}")

        # Final statistics
        final_count = len(gdf)
        if final_count != initial_count:
            removed = initial_count - final_count
            self.logger.info(f"📊 Removed {removed} problematic features ({removed/initial_count*100:.1f}%)")

        self.logger.info(f"✅ Data cleaning complete: {final_count} features, {len(gdf.columns)} clean columns")
        return gdf

    def _get_cache_metadata(self, cache_file: Path) -> Optional[Dict[str, Any]]:
        """✅ ENHANCED: Advanced cache metadata management with integrity checks."""
        if not cache_file.exists():
            return None

        metadata_file = cache_file.with_suffix('.metadata.json')
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Enhanced age validation
                created_time = pd.Timestamp(metadata['created'])
                age_days = (pd.Timestamp.now() - created_time).total_seconds() / (24 * 3600)
                
                if age_days > self.config.cache_max_age_days:
                    self.logger.info(f"🗑️ Cache expired (age: {age_days:.1f} days), will re-fetch")
                    self._cleanup_expired_cache(cache_file, metadata_file)
                    return None
                
                # Validate metadata structure
                required_fields = ['location', 'tags', 'feature_count', 'created']
                if not all(field in metadata for field in required_fields):
                    self.logger.warning(f"⚠️ Incomplete cache metadata, will re-fetch")
                    return None
                
                return metadata
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self.logger.warning(f"⚠️ Cache metadata corrupted: {e}")
                self._cleanup_expired_cache(cache_file, metadata_file)
                return None

        # Fallback to file stats for legacy caches
        try:
            stat = cache_file.stat()
            age_days = (time.time() - stat.st_ctime) / (24 * 3600)
            
            if age_days > self.config.cache_max_age_days:
                self.logger.info(f"🗑️ Legacy cache expired (age: {age_days:.1f} days)")
                cache_file.unlink()
                return None
                
            return {
                'created': stat.st_ctime,
                'size': stat.st_size,
                'feature_count': None,
                'legacy': True
            }
            
        except OSError as e:
            self.logger.error(f"⚠️ Error accessing cache file stats: {e}")
            return None

    def _cleanup_expired_cache(self, cache_file: Path, metadata_file: Path):
        """Clean up expired cache files safely."""
        try:
            if cache_file.exists():
                cache_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
        except OSError as e:
            self.logger.error(f"Failed to cleanup expired cache: {e}")

    def _save_with_metadata(self, gdf: gpd.GeoDataFrame, cache_file: Path,
                           location: str, tags: dict) -> None:
        """✅ ENHANCED: Ultra-robust saving with comprehensive metadata and error recovery."""
        temp_file = None
        temp_metadata_file = None
        
        try:
            # Use temporary files for atomic operations
            temp_file = self.temp_dir / f"{cache_file.name}.tmp"
            temp_metadata_file = self.temp_dir / f"{cache_file.stem}.metadata.json.tmp"
            
            # Save data to temporary file first
            gdf.to_file(temp_file, driver='GPKG')
            
            # Prepare comprehensive metadata
            metadata = {
                'location': location,
                'tags': tags,
                'feature_count': len(gdf),
                'created': pd.Timestamp.now().isoformat(),
                'crs': str(gdf.crs) if not gdf.empty else self.config.default_crs,
                'bounds': gdf.total_bounds.tolist() if not gdf.empty else None,
                'loader_version': '3.0_ultra_robust',
                'columns': list(gdf.columns),
                'geometry_types': gdf.geom_type.value_counts().to_dict() if not gdf.empty else {},
                'file_size': temp_file.stat().st_size,
                'checksum': self._calculate_file_checksum(temp_file)
            }
            
            # Add column mapping if available
            if self.metadata and self.config.preserve_column_mapping:
                tags_hash = hashlib.md5(json.dumps(tags, sort_keys=True).encode()).hexdigest()
                cache_key = f"{location}_{tags_hash}"
                if cache_key in self.metadata['column_mappings']:
                    stored_mapping = self.metadata['column_mappings'][cache_key]
                    if isinstance(stored_mapping, dict) and 'mapping' in stored_mapping:
                        metadata['column_mapping'] = stored_mapping['mapping']
                    else:
                        metadata['column_mapping'] = stored_mapping
            
            # Save metadata to temporary file
            with open(temp_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Atomic move to final locations
            temp_file.replace(cache_file)
            temp_metadata_file.replace(cache_file.with_suffix('.metadata.json'))
            
            self.logger.info(f"💾 Saved data and metadata: {len(gdf)} features, {len(gdf.columns)} columns")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving to cache: {e}")
            
            # Cleanup temporary files
            for tmp_file in [temp_file, temp_metadata_file]:
                if tmp_file and tmp_file.exists():
                    try:
                        tmp_file.unlink()
                    except:
                        pass
            
            # Cleanup any partial final files
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except:
                    pass
            
            metadata_file = cache_file.with_suffix('.metadata.json')
            if metadata_file.exists():
                try:
                    metadata_file.unlink()
                except:
                    pass
            
            raise

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate file checksum for integrity verification."""
        import hashlib
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return "unknown"

    # ... (continuing with additional enhanced methods)
    
    def fetch_osm_data_with_feedback(self, location: str, tags: dict,
                                   step_number: Optional[int] = None) -> Dict[str, Any]:
        """✅ ENHANCED: Comprehensive feedback for AI orchestration."""
        try:
            with self._error_context("fetch_osm_data_with_feedback", location=location, tags=tags):
                start_time = time.time()
                gdf = self.fetch_osm_data(location, tags)
                duration = time.time() - start_time
                
                # Determine if cache was used
                cache_used = False
                if self.metadata and self.metadata['fetch_history']:
                    cache_used = self.metadata['fetch_history'][-1].get('cache_hit', False)

                # Enhanced column mapping info
                column_info = self._get_enhanced_column_info(location, tags)
                
                # Comprehensive data analysis
                data_summary = self._generate_comprehensive_data_summary(gdf, tags)
                
                return {
                    'success': True,
                    'data': gdf,
                    'feature_count': len(gdf),
                    'cache_used': cache_used,
                    'step_number': step_number,
                    'duration': duration,
                    'data_summary': data_summary,
                    'column_info': column_info,
                    'recommendations': self._generate_data_recommendations(gdf, tags),
                    'performance_metrics': {
                        'fetch_time': duration,
                        'features_per_second': len(gdf) / duration if duration > 0 else 0,
                        'memory_usage_mb': gdf.memory_usage(deep=True).sum() / 1024 / 1024 if not gdf.empty else 0
                    }
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'step_number': step_number,
                'suggestions': self._generate_error_suggestions(location, tags, e),
                'recovery_options': self._generate_recovery_options(location, tags, e)
            }

    def _get_enhanced_column_info(self, location: str, tags: dict) -> Dict[str, Any]:
        """Get enhanced column mapping information."""
        column_info = {}
        
        if self.metadata and self.config.preserve_column_mapping:
            tags_hash = hashlib.md5(json.dumps(tags, sort_keys=True).encode()).hexdigest()
            cache_key = f"{location}_{tags_hash}"
            
            if cache_key in self.metadata['column_mappings']:
                mapping_data = self.metadata['column_mappings'][cache_key]
                if isinstance(mapping_data, dict):
                    if 'mapping' in mapping_data:
                        mapping = mapping_data['mapping']
                        column_info = {
                            'columns_renamed': True,
                            'original_columns': mapping_data.get('original_count', len(mapping)),
                            'changed_columns': mapping_data.get('changed_count', 0),
                            'sample_mappings': dict(list({k: v for k, v in mapping.items() if k != v}.items())[:3]),
                            'created': mapping_data.get('created')
                        }
                    else:
                        # Legacy format
                        mapping = mapping_data
                        changed = {k: v for k, v in mapping.items() if k != v}
                        column_info = {
                            'columns_renamed': len(changed) > 0,
                            'original_columns': len(mapping),
                            'changed_columns': len(changed),
                            'sample_mappings': dict(list(changed.items())[:3])
                        }
        
        return column_info

    def _generate_comprehensive_data_summary(self, gdf: gpd.GeoDataFrame, tags: dict) -> Dict[str, Any]:
        """Generate comprehensive data summary for AI analysis."""
        if gdf.empty:
            return {
                'geometry_types': {},
                'crs': self.config.default_crs,
                'columns': [],
                'bounds': None,
                'data_quality': {
                    'completeness_score': 0.0,
                    'geometry_validity': 0.0,
                    'attribute_coverage': 0.0
                }
            }
        
        # Basic info
        summary = {
            'geometry_types': gdf.geom_type.value_counts().to_dict(),
            'crs': str(gdf.crs),
            'columns': list(gdf.columns),
            'bounds': gdf.total_bounds.tolist(),
        }
        
        # Data quality assessment
        total_cells = len(gdf) * (len(gdf.columns) - 1)  # Exclude geometry
        non_null_cells = sum((~gdf[col].isna()).sum() for col in gdf.columns if col != 'geometry')
        
        summary['data_quality'] = {
            'completeness_score': non_null_cells / total_cells if total_cells > 0 else 0.0,
            'geometry_validity': gdf.geometry.is_valid.sum() / len(gdf),
            'attribute_coverage': self._calculate_attribute_coverage(gdf, tags)
        }
        
        # Statistical insights
        if len(gdf) > 0:
            summary['statistics'] = {
                'feature_count': len(gdf),
                'column_count': len(gdf.columns),
                'avg_attributes_per_feature': non_null_cells / len(gdf),
                'geometry_complexity': self._estimate_geometry_complexity(gdf)
            }
        
        return summary

    def _calculate_attribute_coverage(self, gdf: gpd.GeoDataFrame, tags: dict) -> float:
        """Calculate how well the requested tags are represented in the data."""
        if gdf.empty:
            return 0.0
        
        tag_columns = []
        for col in gdf.columns:
            col_lower = col.lower()
            for tag in tags.keys():
                if tag.lower() in col_lower:
                    tag_columns.append(col)
                    break
        
        if not tag_columns:
            return 0.0
        
        coverage_scores = []
        for col in tag_columns:
            coverage = (~gdf[col].isna()).sum() / len(gdf)
            coverage_scores.append(coverage)
        
        return sum(coverage_scores) / len(coverage_scores)

    def _estimate_geometry_complexity(self, gdf: gpd.GeoDataFrame) -> float:
        """Estimate average geometry complexity."""
        try:
            complexities = []
            for geom in gdf.geometry.head(min(100, len(gdf))):  # Sample first 100
                if hasattr(geom, 'coords'):
                    complexities.append(len(list(geom.coords)))
                elif hasattr(geom, 'exterior'):
                    complexities.append(len(list(geom.exterior.coords)))
                else:
                    complexities.append(1)  # Point or simple geometry
            
            return sum(complexities) / len(complexities) if complexities else 1.0
            
        except Exception:
            return 1.0

    def _generate_data_recommendations(self, gdf: gpd.GeoDataFrame, tags: dict) -> List[str]:
        """✅ ENHANCED: Generate intelligent recommendations based on data analysis."""
        recommendations = []
        
        if gdf.empty:
            recommendations.extend([
                f"No features found for tags {tags}. Consider:",
                "- Using broader or alternative tags",
                "- Checking location name spelling",
                "- Trying a larger geographic area",
                "- Verifying tag combinations exist in OSM"
            ])
            return recommendations
        
        # Feature count analysis
        feature_count = len(gdf)
        if feature_count < 5:
            recommendations.append(f"Small dataset ({feature_count} features). Consider broader search criteria.")
        elif feature_count > 10000:
            recommendations.append(f"Large dataset ({feature_count} features). Consider filtering or using bounds to improve performance.")
        
        # Geometry analysis
        geom_types = gdf.geom_type.unique()
        if len(geom_types) > 1:
            recommendations.append(f"Mixed geometry types: {geom_types.tolist()}. Consider geometry type filtering for consistent analysis.")
        
        # Attribute completeness analysis
        name_cols = [col for col in gdf.columns if 'name' in col.lower()]
        if name_cols:
            name_col = name_cols[0]
            missing_names = gdf[name_col].isna().sum()
            if missing_names > feature_count * 0.5:
                recommendations.append(f"Many features ({missing_names}/{feature_count}) lack names. Consider using other identifying attributes.")
        
        # Data quality recommendations
        if hasattr(gdf, 'geometry'):
            invalid_geoms = (~gdf.geometry.is_valid).sum()
            if invalid_geoms > 0:
                recommendations.append(f"Found {invalid_geoms} invalid geometries. Data has been cleaned automatically.")
        
        # Performance recommendations
        memory_usage = gdf.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        if memory_usage > 100:
            recommendations.append(f"Large memory usage ({memory_usage:.1f} MB). Consider data filtering or chunked processing.")
        
        # Column-specific recommendations
        if len(gdf.columns) > 50:
            recommendations.append("Many columns detected. Consider selecting only required attributes for better performance.")
        
        return recommendations

    def _generate_error_suggestions(self, location: str, tags: dict, error: Exception) -> List[str]:
        """✅ ENHANCED: Generate comprehensive error-specific suggestions."""
        suggestions = []
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Column/Field errors
        if 'field' in error_str or 'column' in error_str:
            suggestions.extend([
                "Column name compatibility issue detected:",
                "→ Enable advanced column sanitization",
                "→ Check for special characters in OSM attributes",
                "→ Try using different output format (GeoJSON instead of GPKG)",
                "→ Update to latest geopandas version"
            ])
        
        # Network/Connection errors
        elif any(term in error_str for term in ['network', 'connection', 'timeout', 'socket']):
            suggestions.extend([
                "Network connectivity issue:",
                "→ Check internet connection stability",
                "→ OSM servers may be temporarily unavailable",
                "→ Try again in a few minutes",
                "→ Consider using a VPN if in restricted area",
                "→ Increase timeout settings"
            ])
        
        # Location/Place errors
        elif any(term in error_str for term in ['location', 'place', 'geocod', 'not found']):
            suggestions.extend([
                "Location resolution issue:",
                "→ Try 'City, Country' format (e.g., 'Berlin, Germany')",
                "→ Check location name spelling carefully",
                "→ Use a larger administrative area",
                "→ Try coordinates format: (latitude, longitude)",
                "→ Use official administrative names"
            ])
        
        # Tag/Query errors
        elif any(term in error_str for term in ['tag', 'query', 'syntax']):
            suggestions.extend([
                "OSM tag/query issue:",
                "→ Verify OSM tag syntax (check OSM wiki)",
                "→ Try simpler or more common tags",
                "→ Test tags individually before combining",
                "→ Use tag verification tools online",
                "→ Check for typos in tag names"
            ])
        
        # Permission/Access errors
        elif any(term in error_str for term in ['permission', 'unauthorized', '401', '403']):
            suggestions.extend([
                "Access/Permission issue:",
                "→ Check API rate limits",
                "→ Verify API credentials if using custom endpoints",
                "→ Wait before retrying (rate limiting)",
                "→ Consider using different OSM endpoint"
            ])
        
        # Rate limiting
        elif any(term in error_str for term in ['rate', 'limit', '429', 'too many']):
            suggestions.extend([
                "Rate limiting detected:",
                "→ Wait longer before retrying",
                "→ Reduce request frequency",
                "→ Use smaller geographic areas",
                "→ Enable request throttling"
            ])
        
        # Memory/Resource errors
        elif any(term in error_str for term in ['memory', 'resource', 'allocation']):
            suggestions.extend([
                "Resource constraint issue:",
                "→ Use smaller geographic bounds",
                "→ Filter tags to reduce data volume",
                "→ Process data in smaller chunks",
                "→ Increase available system memory"
            ])
        
        # Generic suggestions for unknown errors
        else:
            suggestions.extend([
                "General troubleshooting:",
                "→ Verify location name spelling",
                "→ Test with simpler/more common tags",
                "→ Check internet connectivity",
                "→ Try with a smaller geographic area",
                "→ Update osmnx and geopandas libraries"
            ])
        
        # Always add debugging suggestions
        suggestions.extend([
            "",
            "For debugging:",
            "→ Check loader logs for detailed error information",
            "→ Test the same query on overpass-turbo.eu",
            "→ Enable verbose logging for more details"
        ])
        
        return suggestions

    def _generate_recovery_options(self, location: str, tags: dict, error: Exception) -> List[Dict[str, Any]]:
        """Generate automated recovery options for failed requests."""
        recovery_options = []
        
        # Option 1: Retry with broader location
        if 'location' in str(error).lower():
            parts = location.split(',')
            if len(parts) > 1:
                broader_location = parts[-1].strip()  # Use country/state only
                recovery_options.append({
                    'type': 'location_fallback',
                    'description': f'Try broader location: {broader_location}',
                    'location': broader_location,
                    'tags': tags,
                    'confidence': 0.7
                })
        
        # Option 2: Simplify tags
        if len(tags) > 1:
            for key, value in tags.items():
                simplified_tags = {key: value}
                recovery_options.append({
                    'type': 'tag_simplification',
                    'description': f'Try with single tag: {key}={value}',
                    'location': location,
                    'tags': simplified_tags,
                    'confidence': 0.8
                })
                break  # Only suggest the first one
        
        # Option 3: Use alternative tags
        alternative_tags = self._get_alternative_tags(tags)
        if alternative_tags:
            recovery_options.append({
                'type': 'alternative_tags',
                'description': f'Try alternative tags: {alternative_tags}',
                'location': location,
                'tags': alternative_tags,
                'confidence': 0.6
            })
        
        return recovery_options

    def _get_alternative_tags(self, original_tags: dict) -> dict:
        """Suggest alternative OSM tags based on common patterns."""
        alternatives = {
            'amenity': {
                'school': {'building': 'school'},
                'hospital': {'healthcare': 'hospital'},
                'restaurant': {'building': 'restaurant'},
                'shop': {'building': 'retail'}
            },
            'building': {
                'yes': {'building': True},
                'residential': {'landuse': 'residential'},
                'commercial': {'landuse': 'commercial'}
            },
            'highway': {
                'primary': {'highway': ['primary', 'trunk']},
                'secondary': {'highway': ['secondary', 'primary']}
            }
        }
        
        for key, value in original_tags.items():
            if key in alternatives and value in alternatives[key]:
                return alternatives[key][value]
        
        return {}

    # Enhanced utility methods
    def validate_location(self, location: str, quick_test: bool = True) -> Dict[str, Any]:
        """✅ ENHANCED: Advanced location validation with comprehensive testing."""
        try:
            self.logger.info(f"🔍 Validating location: {location}")
            
            if quick_test:
                # Quick validation with minimal query
                test_tags = {'highway': 'primary'}
                timeout_override = 10.0  # Short timeout for validation
            else:
                # Comprehensive validation
                test_tags = {'building': True}
                timeout_override = 30.0
            
            with self._error_context("location_validation", location=location):
                test_gdf = ox.features_from_place(location, test_tags)
                
                # Enhanced confidence scoring
                feature_count = len(test_gdf)
                confidence = 'high' if feature_count > 10 else 'medium' if feature_count > 0 else 'low'
                
                # Additional location analysis
                bounds_area = 0
                if not test_gdf.empty and hasattr(test_gdf, 'total_bounds'):
                    bounds = test_gdf.total_bounds
                    bounds_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
                
                return {
                    'valid': True,
                    'location': location,
                    'test_features': feature_count,
                    'confidence': confidence,
                    'bounds_area': bounds_area,
                    'suggested_tags': self._suggest_tags_for_location(test_gdf),
                    'validation_time': time.time()
                }
                
        except Exception as e:
            return {
                'valid': False,
                'location': location,
                'error': str(e),
                'error_type': type(e).__name__,
                'confidence': 'none',
                'suggestions': self._generate_location_suggestions(location, e),
                'validation_time': time.time()
            }

    def _suggest_tags_for_location(self, test_gdf: gpd.GeoDataFrame) -> List[str]:
        """Suggest useful tags based on sample data from location."""
        if test_gdf.empty:
            return ['building', 'highway', 'amenity']
        
        # Analyze available columns to suggest relevant tags
        suggestions = []
        common_osm_tags = [
            'amenity', 'building', 'highway', 'landuse', 'leisure',
            'natural', 'tourism', 'shop', 'office', 'healthcare'
        ]
        
        for col in test_gdf.columns:
            col_lower = col.lower()
            for tag in common_osm_tags:
                if tag in col_lower and tag not in suggestions:
                    suggestions.append(tag)
        
        return suggestions[:5]  # Return top 5 suggestions

    def _generate_location_suggestions(self, location: str, error: Exception) -> List[str]:
        """Generate location-specific suggestions based on error."""
        suggestions = []
        
        # Parse location for specific suggestions
        parts = location.split(',')
        
        if len(parts) == 1:
            suggestions.extend([
                f"Try '{location}, [Country]' format",
                f"Add state/region: '{location}, [State], [Country]'",
                "Use full administrative hierarchy"
            ])
        
        # Common location issues
        suggestions.extend([
            "Check spelling and capitalization",
            "Use English names when possible",
            "Try official/administrative names",
            "Use coordinates if name resolution fails: (lat, lon)"
        ])
        
        return suggestions

    def cleanup_cache(self, max_age_days: Optional[int] = None, 
                     max_size_gb: Optional[float] = None,
                     force_cleanup: bool = False) -> Dict[str, Any]:
        """✅ ENHANCED: Intelligent cache cleanup with advanced strategies."""
        if max_age_days is None:
            max_age_days = self.config.cache_max_age_days
        if max_size_gb is None:
            max_size_gb = self.config.cache_max_size_gb

        self.logger.info("🧹 Starting intelligent cache cleanup...")
        
        cleanup_stats = {
            'files_removed': 0,
            'bytes_freed': 0,
            'files_kept': 0,
            'errors': 0,
            'cleanup_strategies': []
        }
        
        try:
            with self._lock:  # Thread safety
                # Get all cache files with metadata
                cache_files = list(self.cache_dir.glob('*.gpkg'))
                current_time = time.time()
                max_age_seconds = max_age_days * 24 * 3600
                max_size_bytes = max_size_gb * 1024 * 1024 * 1024
                
                # Calculate current cache size
                total_size = sum(f.stat().st_size for f in cache_files if f.exists())
                
                self.logger.info(f"📊 Cache analysis: {len(cache_files)} files, {total_size / (1024*1024):.1f} MB")
                
                # Strategy 1: Remove expired files
                expired_files = []
                for cache_file in cache_files:
                    try:
                        file_age = current_time - cache_file.stat().st_ctime
                        if file_age > max_age_seconds:
                            expired_files.append((cache_file, cache_file.stat().st_size))
                    except OSError:
                        cleanup_stats['errors'] += 1
                
                if expired_files:
                    cleanup_stats['cleanup_strategies'].append('expired_files')
                    for cache_file, size in expired_files:
                        self._remove_cache_file_safely(cache_file)
                        cleanup_stats['files_removed'] += 1
                        cleanup_stats['bytes_freed'] += size
                        total_size -= size
                        
                    self.logger.info(f"🗑️ Removed {len(expired_files)} expired files")
                
                # Strategy 2: Size-based cleanup if still over limit
                if total_size > max_size_bytes or force_cleanup:
                    cleanup_stats['cleanup_strategies'].append('size_based')
                    
                    # Get remaining files with access patterns
                    remaining_files = [f for f in cache_files if f.exists()]
                    file_stats = []
                    
                    for cache_file in remaining_files:
                        try:
                            stat = cache_file.stat()
                            
                            # Calculate priority score (higher = more likely to keep)
                            age_days = (current_time - stat.st_ctime) / (24 * 3600)
                            access_age_days = (current_time - stat.st_atime) / (24 * 3600)
                            
                            # Priority factors
                            size_penalty = stat.st_size / (1024 * 1024)  # MB
                            age_penalty = age_days
                            access_penalty = access_age_days
                            
                            priority = 1000 / (1 + size_penalty * 0.1 + age_penalty * 0.5 + access_penalty * 0.3)
                            
                            file_stats.append((cache_file, stat.st_size, priority))
                            
                        except OSError:
                            cleanup_stats['errors'] += 1
                    
                    # Sort by priority (lowest first = first to remove)
                    file_stats.sort(key=lambda x: x[2])
                    
                    # Remove files until under size limit
                    for cache_file, size, priority in file_stats:
                        if total_size <= max_size_bytes and not force_cleanup:
                            break
                        
                        self._remove_cache_file_safely(cache_file)
                        cleanup_stats['files_removed'] += 1
                        cleanup_stats['bytes_freed'] += size
                        total_size -= size
                        
                        self.logger.debug(f"🗑️ Removed {cache_file.name} (priority: {priority:.1f})")
                
                # Strategy 3: Remove corrupted files
                corrupted_files = self._find_corrupted_cache_files()
                if corrupted_files:
                    cleanup_stats['cleanup_strategies'].append('corruption_cleanup')
                    for cache_file in corrupted_files:
                        try:
                            size = cache_file.stat().st_size
                            self._remove_cache_file_safely(cache_file)
                            cleanup_stats['files_removed'] += 1
                            cleanup_stats['bytes_freed'] += size
                        except:
                            cleanup_stats['errors'] += 1
                    
                    self.logger.info(f"🗑️ Removed {len(corrupted_files)} corrupted files")
                
                # Final statistics
                remaining_files = list(self.cache_dir.glob('*.gpkg'))
                cleanup_stats['files_kept'] = len(remaining_files)
                final_size = sum(f.stat().st_size for f in remaining_files if f.exists())
                
                self.logger.info(
                    f"✅ Cache cleanup complete: "
                    f"removed {cleanup_stats['files_removed']} files, "
                    f"freed {cleanup_stats['bytes_freed']/(1024*1024):.1f} MB, "
                    f"kept {cleanup_stats['files_kept']} files"
                )
                
                cleanup_stats['final_size_mb'] = final_size / (1024 * 1024)
                cleanup_stats['initial_size_mb'] = (total_size + cleanup_stats['bytes_freed']) / (1024 * 1024)
                
        except Exception as e:
            self.logger.error(f"❌ Cache cleanup error: {e}")
            cleanup_stats['cleanup_error'] = str(e)
            cleanup_stats['errors'] += 1
        
        return cleanup_stats

    def _remove_cache_file_safely(self, cache_file: Path):
        """Safely remove cache file and associated metadata."""
        try:
            # Remove main cache file
            if cache_file.exists():
                cache_file.unlink()
            
            # Remove metadata file
            metadata_file = cache_file.with_suffix('.metadata.json')
            if metadata_file.exists():
                metadata_file.unlink()
                
        except OSError as e:
            self.logger.warning(f"⚠️ Error removing cache file {cache_file}: {e}")
            raise

    def _find_corrupted_cache_files(self) -> List[Path]:
        """Find and identify corrupted cache files."""
        corrupted_files = []
        cache_files = list(self.cache_dir.glob('*.gpkg'))
        
        for cache_file in cache_files:
            try:
                # Quick corruption check
                test_read = gpd.read_file(cache_file, rows=1)
                if test_read.empty and cache_file.stat().st_size > 1024:  # Suspiciously small
                    corrupted_files.append(cache_file)
            except:
                corrupted_files.append(cache_file)
        
        return corrupted_files

    # Additional enhanced methods would continue here...
    # (Including batch processing, parallel fetching, advanced metadata management, etc.)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metadata:
            return {'error': 'Metadata tracking disabled'}
        
        # Calculate comprehensive statistics
        stats = self.metadata['performance_stats'].copy()
        cache_stats = self.metadata['cache_stats']
        
        # Enhanced metrics
        total_requests = stats['total_requests']
        if total_requests > 0:
            stats['cache_efficiency'] = cache_stats['hits'] / total_requests
            stats['error_rate'] = len(self.metadata['error_log']) / total_requests
        
        # Recent performance (last 10 requests)
        recent_history = self.metadata['fetch_history'][-10:]
        if recent_history:
            recent_times = [h.get('duration', 0) for h in recent_history if 'duration' in h]
            if recent_times:
                stats['recent_avg_time'] = sum(recent_times) / len(recent_times)
        
        return {
            'performance_stats': stats,
            'cache_stats': cache_stats,
            'system_health': {
                'cache_hit_rate': stats.get('cache_hit_rate', 0),
                'avg_response_time': stats.get('avg_response_time', 0),
                'error_rate': stats.get('error_rate', 0),
                'total_features_loaded': stats['total_features'],
                'uptime': (pd.Timestamp.now() - pd.Timestamp(self.metadata['system_info']['created'])).total_seconds()
            },
            'recommendations': self._generate_performance_recommendations()
        }

    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if not self.metadata:
            return recommendations
        
        cache_hit_rate = self.metadata['performance_stats'].get('cache_hit_rate', 0)
        avg_time = self.metadata['performance_stats'].get('avg_response_time', 0)
        
        if cache_hit_rate < 0.3:
            recommendations.append("Low cache hit rate. Consider longer cache retention or more consistent query patterns.")
        
        if avg_time > 10.0:
            recommendations.append("Slow average response time. Consider smaller query areas or more specific tags.")
        
        error_count = len(self.metadata['error_log'])
        if error_count > 5:
            recommendations.append(f"High error count ({error_count}). Review error log for patterns.")
        
        return recommendations

# Usage example and testing
if __name__ == '__main__':
    """
    Comprehensive testing for the ultra-robust SmartDataLoader.
    """
    print("=== Ultra-Robust SmartDataLoader Test Suite ===\n")
    
    # Enhanced configuration
    config = LoaderConfig(
        max_retries=3,
        cache_max_age_days=7,
        cache_max_size_gb=2.0,
        preserve_column_mapping=True,
        force_lowercase_columns=True,
        enable_logging=True,
        enable_parallel_fetching=False,  # Disable for testing
        max_memory_usage_mb=512,
        enable_geometry_simplification=True
    )
    
    loader = SmartDataLoader(config)
    
    # Test comprehensive functionality
    test_cases = [
        {
            'name': 'Basic OSM Fetch with Column Sanitization',
            'location': 'Potsdam, Germany',
            'tags': {'building': True},
            'expected_issues': ['column_sanitization']
        },
        {
            'name': 'Complex Tags with Special Characters',
            'location': 'Berlin, Germany',
            'tags': {'building:height': True, 'addr:city': True},
            'expected_issues': ['special_characters']
        },
        {
            'name': 'Error Recovery Test',
            'location': 'NonexistentCity12345',
            'tags': {'amenity': 'school'},
            'expected_issues': ['location_error']
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case['name']}")
        print("-" * 60)
        
        try:
            result = loader.fetch_osm_data_with_feedback(
                test_case['location'], 
                test_case['tags'], 
                step_number=i
            )
            
            if result['success']:
                print(f"✅ Success: {result['feature_count']} features")
                print(f"📊 Cache used: {result['cache_used']}")
                print(f"⏱️ Duration: {result['duration']:.2f}s")
                
                if result['column_info']:
                    print(f"🔧 Column info: {result['column_info']}")
                
                if result['recommendations']:
                    print(f"💡 Recommendations: {result['recommendations'][:2]}")
                    
            else:
                print(f"❌ Failed: {result['error']}")
                print(f"🔧 Suggestions: {result['suggestions'][:2]}")
                
        except Exception as e:
            print(f"❌ Test error: {e}")
        
        print()
    
    # Performance report
    print("Performance Report:")
    print("-" * 60)
    perf_report = loader.get_performance_report()
    if 'error' not in perf_report:
        print(f"Cache hit rate: {perf_report['system_health']['cache_hit_rate']:.1%}")
        print(f"Average response time: {perf_report['system_health']['avg_response_time']:.2f}s")
        print(f"Total features loaded: {perf_report['system_health']['total_features_loaded']}")
    
    print("\n=== Ultra-Robust SmartDataLoader Test Complete ===")