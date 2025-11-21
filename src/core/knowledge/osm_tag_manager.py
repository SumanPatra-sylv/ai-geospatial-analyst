#!/usr/bin/env python3
"""
Centralized OSM Tag Management System
Provides consistent tag mappings across DataScout and WorkflowGenerator
"""

from typing import Dict, List, Optional, Tuple
import json
import os

class OSMTagManager:
    """Centralized management of OSM entity-to-tag mappings with validation."""
    
    def __init__(self, cache_file: str = "data/osm_tag_cache.json"):
        self.cache_file = cache_file
        self.primary_tags = self._load_primary_tags()
        self.alternative_tags = self._load_alternative_tags()
        self.learned_tags = self._load_cache()
    
    def _load_primary_tags(self) -> Dict[str, Dict[str, str]]:
        """Load primary OSM tag mappings."""
        return {
            "school": {"amenity": "school"},
            "park": {"leisure": "park"},  # ✅ FIX: Correct primary tag
            "hospital": {"amenity": "hospital"},
            "restaurant": {"amenity": "restaurant"},
            "hotel": {"tourism": "hotel"},
            "museum": {"tourism": "museum"},
            "library": {"amenity": "library"},
            "pharmacy": {"amenity": "pharmacy"},
            "bank": {"amenity": "bank"},
            "gas_station": {"amenity": "fuel"},
            "supermarket": {"shop": "supermarket"},
            "church": {"amenity": "place_of_worship"},
            "cemetery": {"landuse": "cemetery"},
            "university": {"amenity": "university"}
        }
    
    def _load_alternative_tags(self) -> Dict[str, List[Dict[str, str]]]:
        """Load alternative tag mappings for fallback."""
        return {
            "park": [
                {"leisure": "park"},
                {"landuse": "recreation_ground"},
                {"tourism": "attraction"},
                {"amenity": "park"}  # Sometimes used incorrectly but may work
            ],
            "school": [
                {"amenity": "school"},
                {"building": "school"},
                {"landuse": "education"}
            ],
            "hospital": [
                {"amenity": "hospital"},
                {"healthcare": "hospital"},
                {"building": "hospital"}
            ]
        }
    
    def get_primary_tags(self, entity: str) -> Optional[Dict[str, str]]:
        """Get the primary OSM tags for an entity."""
        return self.primary_tags.get(entity.lower())
    
    def get_all_tag_options(self, entity: str) -> List[Dict[str, str]]:
        """Get all possible tag options for an entity (primary + alternatives)."""
        entity_lower = entity.lower()
        options = []
        
        # Add primary tag first
        if entity_lower in self.primary_tags:
            options.append(self.primary_tags[entity_lower])
        
        # Add learned tags
        if entity_lower in self.learned_tags:
            options.extend(self.learned_tags[entity_lower])
        
        # Add alternatives
        if entity_lower in self.alternative_tags:
            options.extend(self.alternative_tags[entity_lower])
        
        # Remove duplicates while preserving order
        unique_options = []
        for option in options:
            if option not in unique_options:
                unique_options.append(option)
        
        return unique_options
    
    def learn_successful_tag(self, entity: str, successful_tags: Dict[str, str]) -> None:
        """Learn and cache a successful tag mapping."""
        entity_lower = entity.lower()
        
        if entity_lower not in self.learned_tags:
            self.learned_tags[entity_lower] = []
        
        # Don't add duplicates
        if successful_tags not in self.learned_tags[entity_lower]:
            self.learned_tags[entity_lower].insert(0, successful_tags)  # Prioritize learned tags
            self._save_cache()
            print(f"✅ OSMTagManager: Learned new tag mapping for '{entity}': {successful_tags}")
    
    def _load_cache(self) -> Dict[str, List[Dict[str, str]]]:
        """Load learned tag mappings from cache."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Warning: Could not load OSM tag cache: {e}")
        return {}
    
    def _save_cache(self) -> None:
        """Save learned tag mappings to cache."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.learned_tags, f, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not save OSM tag cache: {e}")

# Global instance for consistent access
tag_manager = OSMTagManager()