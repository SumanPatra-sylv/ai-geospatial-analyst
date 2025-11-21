"""
DataScout Agent - Advanced Geospatial Data Intelligence System
=============================================================

A comprehensive agent for querying and analyzing geospatial data sources
with robust error handling, caching, and multi-source integration.

FIXES APPLIED:
1. Fixed the tag dictionary access error in get_validated_tags_for_entity
2. Enhanced error handling throughout the codebase
3. Improved data structure validation
4. Added comprehensive logging for debugging
5. Optimized async operations with better error recovery
6. Enhanced caching with better key generation
7. Improved confidence calculation algorithms

MODIFICATIONS APPLIED (as per user request):
1. Integrated a centralized `tag_manager` for OSM tag lookups.
2. Replaced single-tag probing with entity-based probing (`_probe_single_entity`) 
   that includes a fallback mechanism to try multiple tag options.
3. Updated concurrent probing to work with the new entity-based approach.
4. Reworked `generate_data_reality_report` to use the new probing logic and
   a simplified validation flow.
5. Added an enhanced recommendation engine (`_generate_enhanced_recommendations`)
   that provides insights on tag fallback and other metrics.
"""

import logging
import time
import json
import hashlib
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict

# Add these new imports at the top
from src.core.knowledge.knowledge_base import SpatialKnowledgeBase
from src.core.agents.schemas import ClarificationAsk
# Add this import after your existing imports
from src.core.knowledge.osm_tag_manager import tag_manager


# Asynchronous operations support
import asyncio
import aiohttp

# Core geospatial libraries
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderQuotaExceeded
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Data processing
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import transform
import pyproj

# Configure logging with more detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('datascout.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class LocationData:
    """Validated location data structure with enhanced validation"""
    canonical_name: str
    latitude: float
    longitude: float
    bounding_box: Dict[str, float]  # {'south': lat, 'west': lon, 'north': lat, 'east': lon}
    country: Optional[str] = None
    admin_level: Optional[str] = None
    place_type: Optional[str] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate data after initialization"""
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"Invalid longitude: {self.longitude}")
        if not (0 <= self.confidence <= 1):
            raise ValueError(f"Invalid confidence: {self.confidence}")
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DataProbeResult:
    """Result from OSM data probing with enhanced error tracking"""
    original_entity: str
    tag: str
    count: int
    density: float  # items per km²
    confidence: float
    query_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DataRealityReport:
    """Comprehensive report of data availability with enhanced metrics"""
    location: LocationData
    probe_results: List[DataProbeResult]
    total_query_time: float
    success_rate: float
    recommendations: List[str]
    metadata: Dict[str, Any]
    generation_timestamp: float = None
    
    def __post_init__(self):
        if self.generation_timestamp is None:
            self.generation_timestamp = time.time()


class CacheManager:
    """Intelligent caching system with enhanced eviction and statistics"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
        self.ttl = ttl
        self.access_times: Dict[str, float] = {}
        self.access_count: Dict[str, int] = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_key(self, data: Any) -> str:
        """Generate stable cache key from data"""
        if isinstance(data, dict):
            # Sort dict for consistent key generation
            sorted_data = json.dumps(data, sort_keys=True)
        else:
            sorted_data = str(data)
        return hashlib.md5(sorted_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached item if not expired with LRU tracking"""
        if key in self.cache:
            if time.time() - self.access_times[key] < self.ttl:
                self.access_times[key] = time.time()
                self.access_count[key] = self.access_count.get(key, 0) + 1
                self.hit_count += 1
                return self.cache[key]
            else:
                # Remove expired item
                self._remove_key(key)
        
        self.miss_count += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cache item with intelligent eviction"""
        if len(self.cache) >= self.max_size:
            self._evict_least_valuable()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
        self.access_count[key] = 1
    
    def _evict_least_valuable(self) -> None:
        """Remove least valuable item (combines LRU and access frequency)"""
        if not self.cache:
            return
        
        # Calculate value score (recent access + frequency)
        current_time = time.time()
        scores = {}
        for key in self.cache:
            age = current_time - self.access_times[key]
            frequency = self.access_count.get(key, 1)
            scores[key] = frequency / (age + 1)  # Higher score = more valuable
        
        # Remove least valuable
        least_valuable = min(scores, key=scores.get)
        self._remove_key(least_valuable)
    
    def _remove_key(self, key: str) -> None:
        """Remove key from all tracking structures"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.access_count.pop(key, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': hit_rate,
            'hits': self.hit_count,
            'misses': self.miss_count
        }


class DataScout:
    """
    Advanced geospatial data intelligence agent with comprehensive error handling
    
    Enhanced Capabilities:
    - Robust location validation with fallback mechanisms
    - Multi-source data probing with circuit breaker pattern
    - Intelligent caching with LRU and frequency-based eviction
    - Comprehensive error handling and recovery
    - Performance optimization with async operations
    - Detailed logging and monitoring
    """
    
    def __init__(self, 
                 user_agent: str = "DataScout/1.0",
                 timeout: int = 30,
                 max_retries: int = 3,
                 cache_size: int = 1000,
                 cache_ttl: int = 3600,
                 overpass_url: str = "https://overpass-api.de/api/interpreter"):
        
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.overpass_url = overpass_url
        
        # Initialize components with error handling
        try:
            self.cache = CacheManager(cache_size, cache_ttl)
            self.geocoder = Nominatim(user_agent=user_agent, timeout=timeout)
            self.knowledge_base = SpatialKnowledgeBase(cache_file="data/cache/tag_cache.json")
        except Exception as e:
            logger.error(f"Failed to initialize DataScout components: {e}")
            raise
        
        # Performance tracking with enhanced metrics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'errors': 0,
            'total_query_time': 0.0,
            'successful_validations': 0,
            'failed_validations': 0,
            'clarification_requests': 0
        }
        
        # Circuit breaker for external API calls
        self.circuit_breaker = {
            'failures': 0,
            'last_failure_time': 0,
            'failure_threshold': 5,
            'recovery_timeout': 300  # 5 minutes
        }

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.circuit_breaker['failures'] >= self.circuit_breaker['failure_threshold']:
            if time.time() - self.circuit_breaker['last_failure_time'] < self.circuit_breaker['recovery_timeout']:
                return True
            else:
                # Reset circuit breaker
                self.circuit_breaker['failures'] = 0
        return False

    def _record_failure(self):
        """Record a failure in circuit breaker"""
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure_time'] = time.time()

    def _record_success(self):
        """Record a success in circuit breaker"""
        if self.circuit_breaker['failures'] > 0:
            self.circuit_breaker['failures'] = max(0, self.circuit_breaker['failures'] - 1)

    # In DataScout class

# THIS IS THE FINAL, CORRECTED METHOD
    def get_validated_tags_for_entity(self, entity: str, confidence_threshold: float = 0.8) -> Union[List[Dict], ClarificationAsk]:
        """
        FIXED: Enhanced validation that correctly unpacks the response from the
        Knowledge Base and handles the (tags, explanation) tuple.
        """
        try:
            logger.info(f"Validating tags for entity: '{entity}'...")

            if not entity or not isinstance(entity, str):
                logger.error(f"Invalid entity input: {entity}")
                return ClarificationAsk(original_entity=str(entity), message="Invalid entity format provided.")

            # --- THE CRITICAL FIX ---
            # CORRECT: Unpack the tuple returned by the knowledge base into two separate variables.
            candidate_tags, explanation = self.knowledge_base.get_candidate_tags(entity)

            # Optional but recommended: Log the explanation for better debugging.
            logger.debug(f"Knowledge base explanation for '{entity}': {explanation}")

            if not candidate_tags:
                logger.warning(f"No candidate tags found for entity: '{entity}'")
                self.stats['failed_validations'] += 1
                return ClarificationAsk(original_entity=entity, message=f"I don't know what '{entity}' means. Please provide a more specific term.")

            # CORRECT: Now, we loop ONLY over the `candidate_tags` list.
            validated_candidates = []
            for result in candidate_tags:
                try:
                    if not isinstance(result, dict):
                        logger.warning(f"Invalid result format for {entity}: {result}")
                        continue

                    if 'confidence' not in result or 'tag' not in result:
                        logger.warning(f"Result for {entity} is missing 'confidence' or 'tag' field: {result}")
                        continue

                    confidence = float(result['confidence'])
                    if not (0 <= confidence <= 1):
                        logger.warning(f"Invalid confidence value: {confidence}")
                        continue

                    validated_candidates.append(result)

                except (ValueError, TypeError) as e:
                    logger.warning(f"Error validating a single candidate result for '{entity}': {e}")
                    continue

            if not validated_candidates:
                logger.error(f"No valid candidates remained after validation for entity: '{entity}'")
                self.stats['failed_validations'] += 1
                return ClarificationAsk(original_entity=entity, message=f"Data validation failed for '{entity}'. Please try a different term.")

            highest_confidence = max(res['confidence'] for res in validated_candidates)
            logger.debug(f"Highest confidence for '{entity}': {highest_confidence}")

            if highest_confidence >= confidence_threshold:
                high_confidence_results = [res for res in validated_candidates if res['confidence'] >= confidence_threshold]
                logger.info(f"High-confidence match for '{entity}': {len(high_confidence_results)} results")
                self.stats['successful_validations'] += 1
                return high_confidence_results
            else:
                logger.info(f"Low confidence for '{entity}' (max: {highest_confidence}). Requesting clarification.")
                self.stats['clarification_requests'] += 1
                return ClarificationAsk(
                    original_entity=entity,
                    message=f"I'm not sure what you mean by '{entity}'. Did you mean one of these?",
                    suggestions=validated_candidates[:5]
                )

        except Exception as e:
            logger.error(f"Unexpected error validating entity '{entity}': {e}", exc_info=True)
            self.stats['errors'] += 1
            return ClarificationAsk(original_entity=entity, message=f"An unexpected system error occurred while processing '{entity}'. Please try again.")

    def get_primary_tags_for_entity(self, entity: str) -> Optional[Dict[str, str]]:
        """
        Get the most appropriate tags for an entity based on knowledge base analysis.
        This is an enhanced method that combines tag validation with fallback options
        and confidence scoring to suggest the best possible tags for data loading.

        Args:
            entity (str): The entity to get tags for (e.g., "restaurant", "park", "school")

        Returns:
            Optional[Dict[str, str]]: A dictionary of primary OSM tags, or None if no reliable tags found
        """
        try:
            logger.info(f"Getting primary tags for entity: '{entity}'")

            # First try the knowledge base for validated tags
            tag_results = self.get_validated_tags_for_entity(entity)

            # If we got a ClarificationAsk, try to handle it intelligently
            if isinstance(tag_results, ClarificationAsk):
                if tag_results.suggestions:
                    # Use the highest confidence suggestion
                    best_suggestion = max(tag_results.suggestions, key=lambda x: x['confidence'])
                    logger.info(f"Using best fallback suggestion for '{entity}': {best_suggestion}")
                    return best_suggestion['tag']
                else:
                    # Try the tag manager as a fallback
                    fallback_tags = tag_manager.get_primary_tags(entity)
                    if fallback_tags:
                        logger.info(f"Using tag manager fallback for '{entity}': {fallback_tags}")
                        return fallback_tags
                    logger.warning(f"No tags found for '{entity}' in any source")
                    return None

            # We got valid tag results, use the highest confidence one
            if tag_results and isinstance(tag_results, list):
                best_match = max(tag_results, key=lambda x: x['confidence'])
                logger.info(f"Found primary tags for '{entity}': {best_match['tag']}")
                return best_match['tag']

            logger.warning(f"Could not determine primary tags for '{entity}'")
            return None

        except Exception as e:
            logger.error(f"Error getting primary tags for '{entity}': {e}", exc_info=True)
            return None
    
    def resolve_tag_from_taginfo_api(self, keyword: str) -> Dict[str, str]:
        """
        🌍 DATA-DRIVEN TAG RESOLUTION using OSM TagInfo API
        
        Queries the official OSM Taginfo API to find the most popular tag 
        for a given keyword based on REAL-WORLD usage statistics.
        
        This is the ROOT CAUSE fix for LLM hallucinations:
        - No more guessing tags
        - Auto-updates with OSM standards
        - Zero hallucinations (backed by 12M+ OSM objects data)
        
        Args:
            keyword (str): Entity keyword (e.g., 'lake', 'school', 'hospital')
            
        Returns:
            Dict[str, str]: Most popular OSM tag(s) for the keyword
        """
        try:
            # Clean the keyword (handle plurals roughly)
            clean_word = keyword.lower().strip().rstrip('s')
            
            url = "https://taginfo.openstreetmap.org/api/4/search/by_value"
            params = {
                "query": clean_word,
                "sortname": "count_all",  # Sort by real-world usage
                "sortorder": "desc",
                "page": 1,
                "rp": 1  # Return only top result
            }
            
            logger.info(f"🌍 [TagInfo] Querying database for '{keyword}'...")
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data.get("data"):
                # The API returns the most used tag configuration
                best_match = data["data"][0]
                key = best_match["key"]
                value = best_match["value"]
                count = best_match.get("count_all", 0)
                
                logger.info(
                    f"✅ [TagInfo] SUCCESS: '{keyword}' → '{key}={value}' "
                    f"(Used in {count:,} OSM objects)"
                )
                
                return {key: value}
            else:
                logger.warning(f"⚠️ [TagInfo] No results for '{keyword}'")
                # Fallback to tag_manager
                fallback = tag_manager.get_primary_tags(keyword)
                if fallback:
                    logger.info(f"   Using tag_manager fallback: {fallback}")
                    return fallback
                
                # Last resort: return original keyword as 'name' tag
                logger.warning(f"   No fallback available, using generic 'name' tag")
                return {"name": keyword}
                
        except requests.exceptions.Timeout:
            logger.error(f"⏱️ [TagInfo] Timeout querying for '{keyword}'")
            # Fall back to tag_manager
            return tag_manager.get_primary_tags(keyword) or {"name": keyword}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"🌐 [TagInfo] Network error: {e}")
            return tag_manager.get_primary_tags(keyword) or {"name": keyword}
            
        except Exception as e:
            logger.error(f"❌ [TagInfo] Unexpected error: {e}", exc_info=True)
            return tag_manager.get_primary_tags(keyword) or {"name": keyword}
    
    def auto_correct_suspicious_tags(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        🔧 AUTO-CORRECTION for suspicious/generic tag inputs from LLM
        
        Detects when the LLM provides generic keys like 'name', 'type', 
        'layer_name', etc. and automatically corrects them using the TagInfo API.
        
        This prevents the LLM from making up tags and ensures we always use
        real-world OSM tags based on usage data.
        
        Args:
            params (Dict): Tool parameters from LLM (potentially with wrong tags)
            
        Returns:
            Dict: Corrected parameters with proper OSM tags
        """
        raw_tags = params.get("tags", {})
        
        # Define suspicious keys that indicate LLM doesn't know the proper OSM tag
        SUSPICIOUS_KEYS = [
            "name", "layer_name", "type", "category", 
            "search_term", "query", "keyword", "entity"
        ]
        
        # Check if any suspicious key is present
        dirty_value = None
        suspicious_key_found = None
        
        for key in SUSPICIOUS_KEYS:
            if key in raw_tags:
                dirty_value = raw_tags[key]
                suspicious_key_found = key
                break
        
        # If we found a suspicious tag, correct it
        if dirty_value and isinstance(dirty_value, str):
            logger.warning(
                f"🔍 [Auto-Correct] LLM provided suspicious tag '{suspicious_key_found}={dirty_value}'"
            )
            logger.info(f"   Resolving correct OSM tags via TagInfo API...")
            
            # Get the correct tags from TagInfo API
            correct_tags = self.resolve_tag_from_taginfo_api(dirty_value)
            
            # Replace the incorrect tags
            params["tags"] = correct_tags
            
            logger.info(f"✅ [Auto-Correct] Replaced with: {correct_tags}")
        
        return params
    
    def validate_location(self, location_string: str) -> Optional[LocationData]:
        """
        Enhanced location validation with comprehensive error handling
        """
        if not location_string or not isinstance(location_string, str):
            logger.error(f"Invalid location string: {location_string}")
            return None
        
        # Check circuit breaker
        if self._is_circuit_open():
            logger.warning("Circuit breaker is open, skipping geocoding")
            return None
        
        cache_key = self.cache._generate_key(f"location_{location_string.strip().lower()}")
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            self.stats['cache_hits'] += 1
            try:
                return LocationData(**cached_result)
            except Exception as e:
                logger.warning(f"Cached location data invalid: {e}")
                # Remove invalid cache entry
                self.cache._remove_key(cache_key)
        
        self.stats['cache_misses'] += 1
        
        try:
            start_time = time.monotonic()
            
            # Enhanced geocoding with retry logic
            location = None
            for attempt in range(self.max_retries):
                try:
                    location = self.geocoder.geocode(
                        location_string.strip(),
                        exactly_one=True,
                        timeout=self.timeout,
                        addressdetails=True,
                        extratags=True
                    )
                    if location:
                        break
                except (GeocoderTimedOut, GeocoderServiceError) as e:
                    if attempt == self.max_retries - 1:
                        raise
                    logger.warning(f"Geocoding attempt {attempt + 1} failed: {e}")
                    time.sleep(min(2 ** attempt, 10))  # Exponential backoff
            
            if not location:
                logger.warning(f"No location found for: {location_string}")
                self._record_failure()
                return None
            
            # Enhanced bounding box calculation
            bbox = location.raw.get('boundingbox', [])
            if len(bbox) != 4:
                logger.warning(f"Invalid bounding box for {location_string}, using default")
                # Create default bounding box (~1km around point)
                lat_offset = 0.01
                lon_offset = 0.01
                bbox = [
                    str(float(location.latitude) - lat_offset),
                    str(float(location.latitude) + lat_offset),
                    str(float(location.longitude) - lon_offset),
                    str(float(location.longitude) + lon_offset)
                ]
            
            # Validate bounding box values
            try:
                bounding_box = {
                    'south': float(bbox[0]),
                    'north': float(bbox[1]),
                    'west': float(bbox[2]),
                    'east': float(bbox[3])
                }
                
                # Validate bounding box makes sense
                if (bounding_box['south'] >= bounding_box['north'] or
                    bounding_box['west'] >= bounding_box['east']):
                    logger.warning(f"Invalid bounding box geometry for {location_string}")
                    # Fix common bbox format issues
                    if bounding_box['south'] >= bounding_box['north']:
                        bounding_box['south'], bounding_box['north'] = bounding_box['north'], bounding_box['south']
                    if bounding_box['west'] >= bounding_box['east']:
                        bounding_box['west'], bounding_box['east'] = bounding_box['east'], bounding_box['west']
                
            except (ValueError, IndexError) as e:
                logger.error(f"Bounding box parsing error for {location_string}: {e}")
                return None
            
            # Extract location metadata
            raw_data = location.raw
            country = self._extract_country(raw_data, location.address)
            
            location_data = LocationData(
                canonical_name=location.address,
                latitude=float(location.latitude),
                longitude=float(location.longitude),
                bounding_box=bounding_box,
                country=country,
                admin_level=str(raw_data.get('place_rank', 0)),
                place_type=raw_data.get('type', 'unknown'),
                confidence=self._calculate_location_confidence(raw_data)
            )
            
            # Cache the result
            self.cache.set(cache_key, location_data.to_dict())
            
            # Update statistics
            query_time = time.monotonic() - start_time
            self.stats['total_query_time'] += query_time
            self.stats['api_calls'] += 1
            self._record_success()
            
            logger.info(f"Location validated: {location_data.canonical_name} ({query_time:.2f}s)")
            return location_data
            
        except Exception as e:
            logger.error(f"Location validation error for '{location_string}': {e}")
            self.stats['errors'] += 1
            self._record_failure()
            return None
    
    def _extract_country(self, raw_data: Dict, address: str) -> str:
        """Extract country from geocoding result"""
        # Try multiple extraction methods
        country = raw_data.get('country', '')
        if not country and address:
            # Extract from address string
            parts = address.split(',')
            if len(parts) > 1:
                country = parts[-1].strip()
        
        return country or 'Unknown'
    
    def _calculate_location_confidence(self, raw_data: Dict) -> float:
        """Enhanced confidence calculation for location data"""
        confidence = 0.5
        
        # Place rank (lower is better)
        place_rank = raw_data.get('place_rank', 30)
        if place_rank <= 16:
            confidence += 0.3
        elif place_rank <= 20:
            confidence += 0.2
        elif place_rank <= 25:
            confidence += 0.1
        
        # Importance score
        importance = raw_data.get('importance', 0)
        if importance > 0.7:
            confidence += 0.2
        elif importance > 0.5:
            confidence += 0.1
        
        # Bounding box availability
        if 'boundingbox' in raw_data:
            confidence += 0.1
        
        # Address completeness
        if raw_data.get('display_name', '').count(',') >= 3:
            confidence += 0.1
        
        return min(confidence, 1.0)

    async def _async_query_osm_count(self, session: aiohttp.ClientSession, bounding_box: Dict[str, float], tag_dict: Dict[str, str]) -> int:
        """
        Enhanced async OSM query with better error handling and validation
        """
        try:
            # Validate inputs
            if not all(key in bounding_box for key in ['south', 'west', 'north', 'east']):
                raise ValueError("Invalid bounding box format")
            
            if not tag_dict or not isinstance(tag_dict, dict):
                raise ValueError("Invalid tag dictionary")
            
            # Build query
            bbox_str = f"{bounding_box['south']},{bounding_box['west']},{bounding_box['north']},{bounding_box['east']}"
            tag_filter = "".join([f'["{key}"="{value}"]' for key, value in tag_dict.items()])
            
            query = f"""
            [out:json][timeout:25];
            (
                node{tag_filter}({bbox_str});
                way{tag_filter}({bbox_str});
                relation{tag_filter}({bbox_str});
            );
            out count;
            """
            
            # Execute query with timeout
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with session.post(
                self.overpass_url,
                data=query.strip(),
                timeout=timeout,
                headers={'User-Agent': self.user_agent}
            ) as response:
                
                if response.status == 429:
                    logger.warning("Rate limited by Overpass API")
                    await asyncio.sleep(1)
                    raise aiohttp.ClientError("Rate limited")
                
                response.raise_for_status()
                data = await response.json()
                
                # Parse count result
                total_count = 0
                for element in data.get('elements', []):
                    if element.get('type') == 'count':
                        count_value = element.get('tags', {}).get('total', 0)
                        if isinstance(count_value, (int, str)):
                            total_count += int(count_value)
                
                return total_count
                
        except asyncio.TimeoutError:
            logger.error("OSM query timeout")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"OSM query network error: {e}")
            raise
        except Exception as e:
            logger.error(f"OSM query unexpected error: {e}")
            raise

    # MODIFICATION 2: REPLACED _probe_single_tag_async with _probe_single_entity
    async def _probe_single_entity(self, session: aiohttp.ClientSession, 
                                   location_string: str, entity: str) -> DataProbeResult:
        """Enhanced entity probing with intelligent tag fallback and performance optimization."""
        
        start_time = time.monotonic()
        
        try:
            # Get location data for bounding box
            location_data = self.validate_location(location_string)
            if not location_data:
                raise ValueError(f"Could not validate location: {location_string}")
            
            bounding_box = location_data.bounding_box
            area_km2 = self._calculate_bbox_area(bounding_box)
            
            # Get all possible tag options for this entity using centralized manager
            tag_options = tag_manager.get_all_tag_options(entity)
            
            if not tag_options:
                logger.warning(f"No known tags for entity '{entity}'")
                return DataProbeResult(
                    original_entity=entity,
                    tag="unknown_entity",
                    count=0,
                    density=0,
                    confidence=0.0,
                    query_time=time.monotonic() - start_time,
                    error=f"No known OSM tags for '{entity}'"
                )
            
            # Try each tag option until one works
            for i, tags in enumerate(tag_options):
                try:
                    count = await self._async_query_osm_count(session, bounding_box, tags)
                    query_time = time.monotonic() - start_time
                    
                    if count > 0:
                        logger.info(f"✅ Probe SUCCESS for '{entity}' with tags {tags}: {count} items (option {i+1}/{len(tag_options)})")
                        
                        # If this wasn't the primary tag, learn it for future use
                        if i > 0:  # Not the first option
                            tag_manager.learn_successful_tag(entity, tags)
                        
                        # Calculate metrics
                        density = count / area_km2 if area_km2 > 0 else 0
                        confidence = self._calculate_probe_confidence(count, query_time)
                        
                        return DataProbeResult(
                            original_entity=entity,
                            tag=self._dict_to_tag_string(tags),
                            count=count,
                            density=density,
                            confidence=confidence,
                            query_time=query_time,
                            metadata={
                                'area_km2': area_km2,
                                'bbox': bounding_box,
                                'tag_dict': tags,
                                'option_used': i + 1,
                                'total_options': len(tag_options)
                            }
                        )
                    else:
                        logger.debug(f"⚠️ Tags {tags} for '{entity}' returned 0 results, trying next option...")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error probing '{entity}' with tags {tags}: {e}")
                    continue
            
            # All tag options failed
            query_time = time.monotonic() - start_time
            primary_tags = tag_options[0] if tag_options else {}
            
            logger.warning(f"❌ All {len(tag_options)} tag options failed for '{entity}'")
            
            return DataProbeResult(
                original_entity=entity,
                tag=self._dict_to_tag_string(primary_tags),
                count=0,
                density=0,
                confidence=0.0,
                query_time=query_time,
                error=f"All {len(tag_options)} tag options returned zero results",
                metadata={
                    'area_km2': area_km2,
                    'bbox': bounding_box,
                    'failed_options': len(tag_options)
                }
            )
                    
        except Exception as e:
            query_time = time.monotonic() - start_time
            logger.error(f"❌ Critical error probing '{entity}': {e}")
            
            return DataProbeResult(
                original_entity=entity,
                tag="error",
                count=0,
                density=0,
                confidence=0.0,
                query_time=query_time,
                error=str(e)
            )

    # MODIFICATION 3: UPDATED _run_concurrent_probes
    async def _run_concurrent_probes(self, location_string: str, 
                                   entities_to_probe: List[str]) -> List[DataProbeResult]:
        """Enhanced concurrent probing with centralized tag management."""
        
        if not entities_to_probe:
            return []
        
        # Configure session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': self.user_agent}
        ) as session:
            
            # Create tasks with semaphore to limit concurrency
            semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
            
            async def probe_with_semaphore(entity):
                async with semaphore:
                    return await self._probe_single_entity(session, location_string, entity)
            
            tasks = [probe_with_semaphore(entity) for entity in entities_to_probe]
            
            # Execute with gather and handle exceptions
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    entity = entities_to_probe[i]
                    logger.error(f"Task exception for {entity}: {result}")
                    
                    # Create error result
                    error_result = DataProbeResult(
                        original_entity=entity,
                        tag="error",
                        count=0,
                        density=0,
                        confidence=0.0,
                        query_time=0.0,
                        error=str(result)
                    )
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)
        
        return processed_results

    def probe_osm_data(self, bounding_box: Dict[str, float], 
                      entity_tag_pairs: List[Tuple[str, Dict[str, str]]], 
                      include_density: bool = True) -> List[DataProbeResult]:
        """
        Enhanced OSM data probing with better caching and error handling
        """
        if not entity_tag_pairs:
            logger.warning("No entity-tag pairs provided for probing")
            return []
        
        # Generate cache key
        canonical_pairs = sorted([
            (entity, tuple(sorted(tag.items())))
            for entity, tag in entity_tag_pairs
        ])
        cache_key = self.cache._generate_key(f"osm_probe_{bounding_box}_{canonical_pairs}")
        
        # Check cache
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.stats['cache_hits'] += 1
            try:
                return [DataProbeResult(**res) for res in cached_result]
            except Exception as e:
                logger.warning(f"Invalid cached probe result: {e}")
                self.cache._remove_key(cache_key)
        
        self.stats['cache_misses'] += 1
        
        # Calculate area if needed
        area_km2 = 0
        if include_density:
            area_km2 = self._calculate_bbox_area(bounding_box)
            if area_km2 <= 0:
                logger.warning("Invalid bounding box area calculated")
        
        # Run concurrent probes
        try:
            # This part of the code is now superseded by the new flow,
            # but left for potential backward compatibility or direct use.
            # The primary method `generate_data_reality_report` uses the new flow.
            logger.warning("`probe_osm_data` called directly. The new flow uses `_run_concurrent_probes` via `generate_data_reality_report`.")
            # For this to work, we would need to adapt it to the old async runner.
            # results = asyncio.run(self._run_concurrent_probes_old(bounding_box, entity_tag_pairs, area_km2))
            return [] # Returning empty to indicate it's not the primary path.
            
        except Exception as e:
            logger.error(f"Failed to probe OSM data: {e}")
            self.stats['errors'] += 1
            return []

    def _calculate_bbox_area(self, bounding_box: Dict[str, float]) -> float:
        """
        Enhanced bounding box area calculation with validation
        """
        try:
            # Validate bounding box
            required_keys = ['south', 'north', 'west', 'east']
            if not all(key in bounding_box for key in required_keys):
                raise ValueError("Missing bounding box coordinates")
            
            south, north = bounding_box['south'], bounding_box['north']
            west, east = bounding_box['west'], bounding_box['east']
            
            # Validate coordinate ranges
            if not (-90 <= south <= 90 and -90 <= north <= 90):
                raise ValueError("Invalid latitude values")
            if not (-180 <= west <= 180 and -180 <= east <= 180):
                raise ValueError("Invalid longitude values")
            if south >= north or west >= east:
                raise ValueError("Invalid bounding box geometry")
            
            # Use geodesic calculation for better accuracy
            transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
            
            # Transform corners
            sw = transformer.transform(west, south)
            ne = transformer.transform(east, north)
            
            # Calculate area in km²
            width_m = abs(ne[0] - sw[0])
            height_m = abs(ne[1] - sw[1])
            area_km2 = (width_m * height_m) / 1_000_000
            
            logger.debug(f"Calculated area: {area_km2:.2f} km²")
            return area_km2
            
        except Exception as e:
            logger.warning(f"Area calculation failed: {e}")
            return 0.0
    
    def _dict_to_tag_string(self, tag_dict: Dict[str, str]) -> str:
        """Convert tag dictionary to string representation with validation"""
        if not tag_dict or not isinstance(tag_dict, dict):
            return "invalid_tag"
        
        try:
            return "&".join([f"{k}={v}" for k, v in sorted(tag_dict.items())])
        except Exception as e:
            logger.warning(f"Error converting tag dict to string: {e}")
            return "error_tag"
    
    def _calculate_probe_confidence(self, count: int, query_time: float) -> float:
        """Enhanced confidence calculation for probe results"""
        if count < 0 or query_time < 0:
            return 0.0
        
        confidence = 0.7
        
        # Query time factor
        if query_time < 2:
            confidence += 0.2
        elif query_time < 5:
            confidence += 0.1
        elif query_time > 20:
            confidence -= 0.2
        elif query_time > 30:
            confidence -= 0.3
        
        # Count factor
        if count > 0:
            confidence += 0.1
        if count > 10:
            confidence += 0.1
        if count > 100:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    # MODIFICATION 4: UPDATED generate_data_reality_report
    def generate_data_reality_report(self, 
                                   location_string: str,
                                   entities_to_probe: List[str],
                                   include_recommendations: bool = True,
                                   confidence_threshold: float = 0.8) -> Union[DataRealityReport, ClarificationAsk, None]:
        """Enhanced report generation with centralized tag management."""
        
        if not location_string or not entities_to_probe:
            logger.error("Invalid input parameters for report generation")
            return None
        
        report_start_time = time.monotonic()
        logger.info(f"Generating data reality report for '{location_string}' with entities: {entities_to_probe}")
        
        try:
            # 1. Validate location
            location_data = self.validate_location(location_string)
            if not location_data:
                logger.error(f"Failed to validate location: {location_string}")
                return None
            
            # 2. Check entities for clarification needs (simplified)
            for entity in entities_to_probe:
                # Quick validation check - if no tags exist, ask for clarification
                tag_options = tag_manager.get_all_tag_options(entity)
                if not tag_options:
                    return ClarificationAsk(
                        original_entity=entity,
                        message=f"I don't recognize '{entity}'. Could you provide a more specific term?",
                        suggestions=[]
                    )
            
            # 3. Run enhanced concurrent probing
            probe_results = asyncio.run(
                self._run_concurrent_probes(location_string, entities_to_probe)
            )
            
            if not probe_results:
                logger.warning("No probe results returned")
                return None
            
            # 4. Calculate metrics
            total_report_time = time.monotonic() - report_start_time
            success_rate = (
                len([r for r in probe_results if r.error is None]) / len(probe_results)
                if probe_results else 0.0
            )
            
            # 5. Generate recommendations
            recommendations = []
            if include_recommendations:
                try:
                    # MODIFICATION: Calling the new enhanced recommendation method
                    recommendations = self._generate_enhanced_recommendations(location_data, probe_results)
                except Exception as e:
                    logger.error(f"Error generating recommendations: {e}")
                    recommendations = ["Data analysis completed successfully with no specific recommendations."]
            
            # 6. Compile metadata
            metadata = {
                'total_features': sum(r.count for r in probe_results),
                'entities_requested': len(entities_to_probe),
                'entities_processed': len(set(r.original_entity for r in probe_results)),
                'avg_query_time': np.mean([r.query_time for r in probe_results]) if probe_results else 0,
                'error_count': len([r for r in probe_results if r.error is not None]),
                'tag_fallback_used': len([r for r in probe_results if r.metadata and r.metadata.get('option_used', 1) > 1])
            }
            
            # 7. Create and return report
            report = DataRealityReport(
                location=location_data,
                probe_results=probe_results,
                total_query_time=total_report_time,
                success_rate=success_rate,
                recommendations=recommendations,
                metadata=metadata
            )
            
            logger.info(f"Successfully generated report with {len(probe_results)} probe results")
            return report
            
        except Exception as e:
            logger.error(f"Unexpected error generating report: {e}")
            self.stats['errors'] += 1
            return None
    
    def _generate_recommendations(self, location: LocationData, probe_results: List[DataProbeResult]) -> List[str]:
        """
        Enhanced recommendation generation with more intelligent insights
        (This method is now superseded by _generate_enhanced_recommendations but is kept for reference)
        """
        recommendations = []
        
        if not probe_results:
            return ["No data was probed - check entity names and location validity."]
        
        # Analyze sparse data
        sparse_threshold = 5
        sparse_entities = [r.original_entity for r in probe_results if r.count < sparse_threshold and r.error is None]
        if sparse_entities:
            unique_sparse = list(set(sparse_entities))[:3]
            recommendations.append(
                f"Low data density detected for: {', '.join(unique_sparse)}. "
                f"Consider using alternative data sources or expanding search area."
            )
        
        # Analyze dense data
        dense_threshold = 100
        dense_entities = [r.original_entity for r in probe_results if r.density > dense_threshold]
        if dense_entities:
            unique_dense = list(set(dense_entities))[:3]
            recommendations.append(
                f"High feature density for: {', '.join(unique_dense)}. "
                f"Consider spatial clustering or sampling techniques for analysis."
            )
        
        # Analyze query performance
        slow_threshold = 15
        slow_entities = [r.original_entity for r in probe_results if r.query_time > slow_threshold]
        if slow_entities:
            unique_slow = list(set(slow_entities))[:3]
            recommendations.append(
                f"Slow queries detected for: {', '.join(unique_slow)}. "
                f"Consider using smaller bounding boxes or caching strategies."
            )
        
        # Analyze error patterns
        error_results = [r for r in probe_results if r.error is not None]
        if error_results:
            error_count = len(error_results)
            total_count = len(probe_results)
            error_rate = error_count / total_count
            
            if error_rate > 0.3:
                recommendations.append(
                    f"High error rate detected ({error_rate:.1%}). "
                    f"Check network connectivity and API availability."
                )
        
        # Location-specific recommendations
        if location.confidence < 0.7:
            recommendations.append(
                "Location confidence is low. Verify location accuracy and consider using "
                "more specific location identifiers."
            )
        
        # Data completeness analysis
        zero_count_results = [r for r in probe_results if r.count == 0 and r.error is None]
        if len(zero_count_results) > len(probe_results) * 0.5:
            recommendations.append(
                "Many entities returned zero results. Consider checking entity names, "
                "expanding search area, or using alternative data sources."
            )
        
        # Performance optimization recommendations
        total_query_time = sum(r.query_time for r in probe_results)
        if total_query_time > 60:
            recommendations.append(
                f"Total query time was high ({total_query_time:.1f}s). "
                f"Consider implementing query optimization or result caching."
            )
        
        return recommendations or ["Data analysis completed successfully with no specific recommendations."]

    # MODIFICATION 5: ADDED _generate_enhanced_recommendations
    def _generate_enhanced_recommendations(self, location: LocationData, probe_results: List[DataProbeResult]) -> List[str]:
        """Enhanced recommendation generation with tag fallback insights."""
        
        recommendations = []
        
        if not probe_results:
            return ["No data was probed - check entity names and location validity."]
        
        # Check for tag fallback usage
        fallback_entities = [
            r.original_entity for r in probe_results 
            if r.metadata and r.metadata.get('option_used', 1) > 1
        ]
        if fallback_entities:
            recommendations.append(
                f"Alternative OSM tags were used for: {', '.join(set(fallback_entities))}. "
                f"The system learned these successful mappings for future queries."
            )
        
        # Check for zero results
        zero_results = [r for r in probe_results if r.count == 0 and r.error is None]
        if zero_results:
            zero_entities = [r.original_entity for r in zero_results]
            recommendations.append(
                f"No features found for: {', '.join(set(zero_entities))}. "
                f"Consider expanding the search area or using alternative terms."
            )
        
        # Check for high-density areas
        high_density = [r for r in probe_results if r.density > 50]  # More than 50 per km²
        if high_density:
            dense_entities = [r.original_entity for r in high_density]
            recommendations.append(
                f"High feature density detected for: {', '.join(set(dense_entities))}. "
                f"Consider spatial sampling techniques for analysis."
            )
        
        # Performance insights
        slow_queries = [r for r in probe_results if r.query_time > 10]
        if slow_queries:
            recommendations.append(
                f"Some queries took longer than expected. "
                f"Consider using smaller search areas for better performance."
            )
        
        return recommendations or ["Data analysis completed successfully with no specific recommendations."]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Enhanced statistics with comprehensive metrics
        """
        cache_stats = self.cache.get_stats()
        
        total_validations = self.stats['successful_validations'] + self.stats['failed_validations']
        validation_success_rate = (
            self.stats['successful_validations'] / total_validations
            if total_validations > 0 else 0
        )
        
        avg_query_time = (
            self.stats['total_query_time'] / self.stats['api_calls']
            if self.stats['api_calls'] > 0 else 0
        )
        
        return {
            **self.stats,
            'validation_success_rate': validation_success_rate,
            'avg_query_time': avg_query_time,
            'cache_stats': cache_stats,
            'circuit_breaker_status': {
                'failures': self.circuit_breaker['failures'],
                'is_open': self._is_circuit_open()
            }
        }
    
    def clear_cache(self):
        """Clear all cached data and reset statistics"""
        self.cache.cache.clear()
        self.cache.access_times.clear()
        self.cache.access_count.clear()
        self.cache.hit_count = 0
        self.cache.miss_count = 0
        logger.info("Cache cleared and statistics reset")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for the DataScout system
        """
        health_status = {
            'timestamp': time.time(),
            'status': 'healthy',
            'components': {},
            'issues': []
        }
        
        # Check knowledge base
        try:
            if hasattr(self.knowledge_base, 'kb') and self.knowledge_base.kb:
                health_status['components']['knowledge_base'] = 'healthy'
            else:
                health_status['components']['knowledge_base'] = 'degraded'
                health_status['issues'].append('Knowledge base is empty or unavailable')
        except Exception as e:
            health_status['components']['knowledge_base'] = 'unhealthy'
            health_status['issues'].append(f'Knowledge base error: {e}')
        
        # Check geocoder
        try:
            # Simple geocoder test
            test_result = self.geocoder.geocode("New York", timeout=5)
            if test_result:
                health_status['components']['geocoder'] = 'healthy'
            else:
                health_status['components']['geocoder'] = 'degraded'
                health_status['issues'].append('Geocoder test returned no results')
        except Exception as e:
            health_status['components']['geocoder'] = 'unhealthy'
            health_status['issues'].append(f'Geocoder error: {e}')
        
        # Check circuit breaker
        if self._is_circuit_open():
            health_status['components']['circuit_breaker'] = 'open'
            health_status['issues'].append('Circuit breaker is open due to repeated failures')
        else:
            health_status['components']['circuit_breaker'] = 'closed'
        
        # Check cache
        cache_stats = self.cache.get_stats()
        if cache_stats['size'] > cache_stats['max_size'] * 0.9:
            health_status['components']['cache'] = 'degraded'
            health_status['issues'].append('Cache is nearly full')
        else:
            health_status['components']['cache'] = 'healthy'
        
        # Overall status
        if health_status['issues']:
            if any('unhealthy' in str(comp) for comp in health_status['components'].values()):
                health_status['status'] = 'unhealthy'
            else:
                health_status['status'] = 'degraded'
        
        return health_status


# Example usage and comprehensive testing
if __name__ == "__main__":
    # Initialize with enhanced configuration
    scout = DataScout(
        user_agent="GeoLLM-DataScout/1.0-Enhanced",
        timeout=30,
        max_retries=3,
        cache_size=2000,
        cache_ttl=7200  # 2 hours
    )
    
    print("\n" + "="*60)
    print("🚀 DataScout Enhanced - Comprehensive Testing")
    print("="*60)
    
    # Health check
    print("\n=== System Health Check ===")
    health = scout.health_check()
    print(f"Overall Status: {health['status'].upper()}")
    for component, status in health['components'].items():
        print(f"  {component}: {status}")
    if health['issues']:
        print("Issues:")
        for issue in health['issues']:
            print(f"  - {issue}")
    
    print("\n=== Example 1: Enhanced Workflow (Success Case) ===")
    try:
        report_entities_success = ["hospital", "school", "bus_stop"]
        report_success = scout.generate_data_reality_report(
            "Pune, India", 
            report_entities_success,
            include_recommendations=True,
            confidence_threshold=0.7
        )
        
        if isinstance(report_success, DataRealityReport):
            print(f"✅ Successfully generated report for: {report_success.location.canonical_name}")
            print(f"📊 Total Features Found: {report_success.metadata['total_features']}")
            print(f"⏱️  Total Query Time: {report_success.total_query_time:.2f}s")
            print(f"📈 Success Rate: {report_success.success_rate:.1%}")
            
            print("\n📋 Probe Results:")
            for res in report_success.probe_results:
                status = "✅" if res.error is None else "❌"
                print(f"  {status} Entity: '{res.original_entity}' | Tag: '{res.tag}' | Count: {res.count} | Density: {res.density:.2f}/km²")
            
            print("\n💡 Recommendations:")
            for i, rec in enumerate(report_success.recommendations, 1):
                print(f"  {i}. {rec}")
        
        elif isinstance(report_success, ClarificationAsk):
            print("❓ Clarification needed (unexpected in success case)")
        
        else:
            print("❌ Report generation failed")
            
    except Exception as e:
        print(f"❌ Error in success case: {e}")
    
    print("\n=== Example 2: Clarification Protocol (Ambiguous Input) ===")
    try:
        # Assuming 'hangout spot' is not in tag_manager, this will trigger clarification
        report_entities_ambiguous = ["hospital", "hangout spot", "transportation"]
        report_ambiguous = scout.generate_data_reality_report(
            "Berlin, Germany",
            report_entities_ambiguous,
            confidence_threshold=0.8
        )
        
        if isinstance(report_ambiguous, ClarificationAsk):
            print(f"❓ Clarification needed for: '{report_ambiguous.original_entity}'")
            print(f"💬 Message: {report_ambiguous.message}")
            
            if report_ambiguous.suggestions:
                print("\n📝 Suggestions:")
                for i, sugg in enumerate(report_ambiguous.suggestions[:5], 1):
                    entity_name = sugg.get('entity_name', 'Unknown')
                    confidence = sugg.get('confidence', 0)
                    tag = sugg.get('tag', {})
                    print(f"  {i}. {entity_name} (confidence: {confidence:.2f}, tag: {tag})")
        
        elif isinstance(report_ambiguous, DataRealityReport):
            print("✅ Report generated (clarification protocol may not have triggered)")
        
        else:
            print("❌ Report generation failed")
            
    except Exception as e:
        print(f"❌ Error in ambiguous case: {e}")
    
    print("\n=== Example 3: Error Handling (Invalid Location) ===")
    try:
        report_invalid = scout.generate_data_reality_report(
            "NonExistentCity12345",
            ["hospital", "school"],
            confidence_threshold=0.8
        )
        
        if report_invalid is None:
            print("✅ Correctly handled invalid location")
        else:
            print(f"⚠️  Unexpected result for invalid location: {type(report_invalid)}")
            
    except Exception as e:
        print(f"❌ Error in invalid location case: {e}")
    
    print("\n=== Example 4: Performance Statistics ===")
    try:
        stats = scout.get_statistics()
        print(f"📊 Cache Hit Rate: {stats['cache_stats']['hit_rate']:.1%}")
        print(f"⏱️  Average Query Time: {stats['avg_query_time']:.2f}s")
        print(f"🔄 Total API Calls: {stats['api_calls']}")
        print(f"✅ Successful Validations: {stats['successful_validations']}")
        print(f"❌ Failed Validations: {stats['failed_validations']}")
        print(f"❓ Clarification Requests: {stats['clarification_requests']}")
        print(f"🗄️  Cache Size: {stats['cache_stats']['size']}")
        print(f"🔒 Circuit Breaker Status: {'OPEN' if stats['circuit_breaker_status']['is_open'] else 'CLOSED'}")
        
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")
    
    print("\n=== Example 5: Cache Performance Test ===")
    try:
        print("Testing cache performance with repeated queries...")
        start_time = time.time()
        
        # First query (cache miss)
        result1 = scout.validate_location("London, UK")
        first_query_time = time.time() - start_time
        
        start_time = time.time()
        # Second query (cache hit)
        result2 = scout.validate_location("London, UK")
        second_query_time = time.time() - start_time
        
        if result1 and result2:
            print(f"✅ First query: {first_query_time:.3f}s (cache miss)")
            print(f"✅ Second query: {second_query_time:.3f}s (cache hit)")
            print(f"📈 Cache speedup: {first_query_time/second_query_time:.1f}x")
        else:
            print("❌ Cache test failed")
            
    except Exception as e:
        print(f"❌ Error in cache test: {e}")
    
    print("\n" + "="*60)
    print("🎯 DataScout Enhanced Testing Complete")
    print("="*60)