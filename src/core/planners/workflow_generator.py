#!/usr/bin/env python3
"""
AI-GIS Workflow Generator (Enhanced LLM-Powered Version)
Generates correct, executable spatial analysis workflows from parsed queries using a large language model
with Chain-of-Thought reasoning for complex geospatial tasks.

MAJOR FIXES IMPLEMENTED:
1. Intelligent LLM Integration with Auto-Correction
2. Smart Location Scoping with Multi-Layer Defense
3. Advanced Error Recovery and Fallback Mechanisms
4. Performance Optimizations and Caching
"""

import json
import os
import requests
import jinja2
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field
from pprint import pprint
import hashlib
import time
from functools import lru_cache

# This try/except block is preserved for standalone testing.
try:
    from src.core.planners.query_parser import ParsedQuery, SpatialConstraint, SpatialRelationship
except ImportError:
    # Fallback for standalone testing
    from enum import Enum

    class SpatialRelationship(Enum):
        WITHIN = "within"
        NOT_WITHIN = "not within"
        NEAR = "near"
        FAR_FROM = "far from"
        INTERSECTS = "intersects"
        CONTAINS = "contains"

    class SpatialConstraint(BaseModel):
        feature_type: str
        relationship: SpatialRelationship
        distance_meters: Optional[int] = None

    class ParsedQuery(BaseModel):
        target: str
        location: str
        constraints: List[SpatialConstraint]


class LLMWorkflowResponse(BaseModel):
    """Defines the expected structured response from the LLM for workflow generation."""
    reasoning: str = Field(..., description="Chain-of-Thought explanation of the spatial analysis approach.")
    plan: List[Dict] = Field(..., description="Sequential workflow steps with proper data flow.")
    complexity_assessment: str = Field(..., description="Assessment of task complexity and potential challenges.")
    error_handling: List[str] = Field(..., description="Anticipated error scenarios and mitigation strategies.")


class LocationScopeValidator:
    """Validates and corrects location scope to prevent catastrophic data fetching."""
    
    # Multi-tier banned locations (from largest to smallest scope)
    BANNED_COUNTRIES = {
        "india", "china", "usa", "united states", "russia", "brazil", "germany", 
        "france", "united kingdom", "uk", "canada", "australia", "japan"
    }
    
    BANNED_STATES = {
        "west bengal", "maharashtra", "uttar pradesh", "rajasthan", "karnataka",
        "california", "texas", "new york", "florida", "bavaria", "ontario"
    }
    
    BANNED_LARGE_REGIONS = {
        "north india", "south india", "eastern europe", "western europe",
        "southeast asia", "middle east", "central asia", "sub-saharan africa"
    }
    
    # Known city alternatives for common large queries
    LOCATION_ALTERNATIVES = {
        "west bengal": ["Kolkata", "Asansol", "Siliguri", "Durgapur"],
        "germany": ["Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt"],
        "maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik"],
        "karnataka": ["Bangalore", "Mysore", "Hubli", "Mangalore"],
        "california": ["Los Angeles", "San Francisco", "San Diego", "Sacramento"],
        "texas": ["Houston", "Dallas", "Austin", "San Antonio"]
    }
    
    @classmethod
    def is_location_too_broad(cls, location: str) -> Tuple[bool, str, List[str]]:
        """
        Checks if a location is too broad for efficient data fetching.
        Returns: (is_too_broad, reason, suggested_alternatives)
        """
        location_lower = location.lower().strip()
        
        # Remove common suffixes that don't change the scope
        location_clean = re.sub(r',?\s*(india|germany|usa|china)$', '', location_lower)
        
        if location_clean in cls.BANNED_COUNTRIES:
            return True, "entire country", cls.LOCATION_ALTERNATIVES.get(location_clean, [])
        
        if location_clean in cls.BANNED_STATES:
            return True, "entire state/province", cls.LOCATION_ALTERNATIVES.get(location_clean, [])
        
        if location_clean in cls.BANNED_LARGE_REGIONS:
            return True, "large geographic region", cls.LOCATION_ALTERNATIVES.get(location_clean, [])
        
        return False, "", []
    
    @classmethod
    def suggest_scoped_location(cls, broad_location: str) -> str:
        """Suggests a more specific location for broad queries."""
        location_lower = broad_location.lower().strip()
        location_clean = re.sub(r',?\s*(india|germany|usa|china)$', '', location_lower)
        
        alternatives = cls.LOCATION_ALTERNATIVES.get(location_clean, [])
        if alternatives:
            # Return the most prominent city (first in list)
            return f"{alternatives[0]}, {broad_location}"
        
        # Generic fallback
        return f"downtown {broad_location}"


class WorkflowGenerator:
    """
    Generates a logical sequence of spatial operations from a ParsedQuery using an LLM
    with enhanced Chain-of-Thought reasoning and robust error handling.
    """
    
    # Enhanced SYSTEM_PROMPT with strict location scoping rules
    SYSTEM_PROMPT = """You are an expert AI Geospatial Workflow Planner with deep knowledge of GIS operations and spatial analysis. Your task is to convert natural language queries into precise, executable geospatial workflows using Chain-of-Thought reasoning.

**CRITICAL LOCATION SCOPING RULES (MUST FOLLOW):**
- NEVER use entire countries (e.g., "India", "Germany", "USA")
- NEVER use entire states/provinces (e.g., "West Bengal", "California", "Bavaria")
- NEVER use large regions (e.g., "North India", "Eastern Europe")
- ALWAYS use specific cities or districts (e.g., "Kolkata, West Bengal", "Berlin, Germany")
- If given a broad location, automatically scope it to the largest city in that region
- Maximum geographic scope: metropolitan area or large city boundaries

**RESPONSE FORMAT:**
You MUST respond with a single JSON object containing four keys: "reasoning", "plan", "complexity_assessment", and "error_handling".

**COMPLETE RESPONSE EXAMPLE:**
```json
{
  "reasoning": "CHAIN-OF-THOUGHT ANALYSIS: The user wants to find restaurants near parks in Kolkata (scoped from 'West Bengal' to avoid massive data download). Breaking this down: 1) Load OSM data for Kolkata metropolitan area only. 2) Filter for restaurants (amenity=restaurant). 3) Filter for parks (leisure=park). 4) Create 500m buffers around parks. 5) Find restaurants within park buffers using spatial intersection.",
  "plan": [
    {
      "operation": "load_osm_data",
      "location": "Kolkata metropolitan area, West Bengal, India",
      "output_layer": "kolkata_osm_data"
    },
    {
      "operation": "filter_by_attribute",
      "input_layer": "kolkata_osm_data",
      "key": "amenity",
      "value": "restaurant",
      "output_layer": "restaurants"
    },
    {
      "operation": "filter_by_attribute",
      "input_layer": "kolkata_osm_data",
      "key": "leisure",
      "value": "park",
      "output_layer": "parks"
    },
    {
      "operation": "buffer",
      "input_layer": "parks",
      "distance_meters": 500,
      "output_layer": "park_buffers"
    },
    {
      "operation": "clip",
      "input_layer": "restaurants",
      "clip_layer": "park_buffers",
      "output_layer": "restaurants_near_parks"
    },
    {
      "operation": "rename_layer",
      "input_layer": "restaurants_near_parks",
      "output_layer": "final_result"
    }
  ],
  "complexity_assessment": "Medium complexity - involves spatial buffers and intersection operations in urban area.",
  "error_handling": [
    "Validate OSM data availability for Kolkata",
    "Check for empty restaurant or park datasets",
    "Ensure buffer operations don't exceed memory limits",
    "Verify CRS compatibility between all layers"
  ]
}
```

**AVAILABLE TOOLS/OPERATIONS:**
- `load_osm_data(location, output_layer)`: Loads OpenStreetMap data for a SPECIFIC location (city/district level only)
- `filter_by_attribute(input_layer, key, value, output_layer)`: Filters features where attribute 'key' equals 'value'
- `buffer(input_layer, distance_meters, output_layer)`: Creates buffer zones around features
- `clip(input_layer, clip_layer, output_layer)`: Clips input_layer to boundaries of clip_layer
- `dissolve(input_layer, by, output_layer)`: Merges features sharing the same 'by' attribute value
- `rename_layer(input_layer, output_layer)`: Renames a layer

**MANDATORY WORKFLOW STRUCTURE:**
1. Start with load_osm_data using a SCOPED location
2. Perform analysis operations
3. END with rename_layer to create "final_result"

**OSM DATA TAGS:**
- amenity: restaurant, school, hospital, bank, cafe, pub, pharmacy
- leisure: park, playground, sports_centre, swimming_pool, garden
- landuse: residential, commercial, industrial, forest, farmland
- highway: primary, secondary, residential, trunk, motorway
- building: yes, house, commercial, industrial, apartments
- natural: water, forest, grassland, wetland, beach

**Knowledge Base Guidance:**
{{ guidance_from_rag }}

**User Request Details:**
{{ parsed_query }}

**TASK:** Generate a complete workflow plan with Chain-of-Thought reasoning that uses appropriately scoped locations."""

    def __init__(self):
        """Initialize the WorkflowGenerator with enhanced configuration."""
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("OLLAMA_MODEL", "mistral")
        self.max_retries = 3
        self.timeout = 120  # Reduced timeout since we're preventing large queries
        self.location_validator = LocationScopeValidator()
        
        # Simple in-memory cache for workflow results
        self._workflow_cache = {}
        self._max_cache_size = 100

    def generate_workflow(self, parsed_query: ParsedQuery, guidance_from_rag: str = "") -> Dict[str, Any]:
        """
        Uses an LLM to generate a reasoned workflow plan with enhanced error handling.
        
        Args:
            parsed_query: The parsed user query with spatial constraints
            guidance_from_rag: Additional guidance from knowledge base
            
        Returns:
            Dict containing reasoning, plan, complexity assessment, and error handling
        """
        # Create cache key
        cache_key = self._create_cache_key(parsed_query, guidance_from_rag)
        if cache_key in self._workflow_cache:
            print("📋 Using cached workflow result")
            return self._workflow_cache[cache_key]
        
        # Pre-validate location scope
        is_too_broad, reason, alternatives = self.location_validator.is_location_too_broad(parsed_query.location)
        if is_too_broad:
            print(f"⚠️  Location scope too broad ({reason}): {parsed_query.location}")
            if alternatives:
                print(f"🔄 Auto-scoping to: {alternatives[0]}")
                # Create a new query with scoped location
                scoped_query = ParsedQuery(
                    target=parsed_query.target,
                    location=f"{alternatives[0]}, {parsed_query.location}",
                    constraints=parsed_query.constraints
                )
                parsed_query = scoped_query
        
        parsed_query_json = parsed_query.model_dump_json(indent=2)
        
        # Enhanced prompt rendering
        template = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True
        ).from_string(self.SYSTEM_PROMPT)
        
        prompt = template.render(
            parsed_query=parsed_query_json,
            guidance_from_rag=guidance_from_rag if guidance_from_rag else "No specific guidance available."
        )

        # Enhanced retry logic with different strategies
        for attempt in range(self.max_retries):
            try:
                print(f"🔄 Attempting workflow generation (attempt {attempt + 1}/{self.max_retries})...")
                
                # Use different strategies for different attempts
                if attempt == 0:
                    # First attempt: normal generation
                    llm_response = self._make_llm_call(prompt)
                elif attempt == 1:
                    # Second attempt: simplified prompt
                    llm_response = self._make_llm_call(self._create_simplified_prompt(parsed_query))
                else:
                    # Third attempt: template-based with AI enhancement
                    llm_response = self._generate_template_enhanced_workflow(parsed_query)
                
                validated_response = LLMWorkflowResponse(**llm_response)
                
                # Auto-correct the plan if needed
                corrected_plan = self._enforce_final_step(validated_response.plan)
                validated_response.plan = corrected_plan
                
                # Enhanced workflow validation
                self._validate_workflow_logic(validated_response.plan)
                
                # Cache the result
                result = validated_response.model_dump()
                self._cache_workflow(cache_key, result)
                
                print("✅ Workflow generation successful!")
                return result
                
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    print("❌ All attempts failed. Generating intelligent fallback...")
                    return self._generate_intelligent_fallback(parsed_query, str(e))
    
    def _enforce_final_step(self, plan: List[Dict]) -> List[Dict]:
        """
        Intelligently ensures the plan ends with rename_layer to 'final_result'.
        This is the key fix for the "Broken LLM Integration" problem.
        """
        if not plan:
            return plan
        
        last_step = plan[-1]
        
        # Check if already properly terminated
        if (last_step.get("operation") == "rename_layer" and 
            last_step.get("output_layer") == "final_result"):
            return plan
        
        # Find the output layer of the actual last step
        last_output_layer = last_step.get("output_layer", "unknown_layer")
        
        # Create the corrective final step
        final_step = {
            "operation": "rename_layer",
            "input_layer": last_output_layer,
            "output_layer": "final_result"
        }
        
        # Append the corrective step
        corrected_plan = plan + [final_step]
        
        print(f"🔧 Auto-corrected plan: added final rename_layer step using '{last_output_layer}'")
        return corrected_plan
    
    def _create_simplified_prompt(self, parsed_query: ParsedQuery) -> str:
        """Creates a simplified prompt for retry attempts."""
        return f"""Generate a JSON workflow to find {parsed_query.target} in {parsed_query.location}.

REQUIRED FORMAT:
{{
  "reasoning": "Brief explanation",
  "plan": [
    {{"operation": "load_osm_data", "location": "{parsed_query.location}", "output_layer": "data"}},
    {{"operation": "filter_by_attribute", "input_layer": "data", "key": "amenity", "value": "{parsed_query.target}", "output_layer": "results"}},
    {{"operation": "rename_layer", "input_layer": "results", "output_layer": "final_result"}}
  ],
  "complexity_assessment": "Low complexity",
  "error_handling": ["Basic validation"]
}}

CRITICAL: The plan MUST end with rename_layer to "final_result"."""

    def _generate_template_enhanced_workflow(self, parsed_query: ParsedQuery) -> Dict:
        """Generates a template-based workflow enhanced with query-specific details."""
        # Determine the best OSM tag for the target
        osm_tag, osm_value = self._infer_osm_tags(parsed_query.target)
        
        # Create base template
        plan = [
            {
                "operation": "load_osm_data",
                "location": parsed_query.location,
                "output_layer": "osm_data"
            },
            {
                "operation": "filter_by_attribute",
                "input_layer": "osm_data",
                "key": osm_tag,
                "value": osm_value,
                "output_layer": "filtered_results"
            }
        ]
        
        # Add constraint handling
        current_layer = "filtered_results"
        for i, constraint in enumerate(parsed_query.constraints):
            constraint_tag, constraint_value = self._infer_osm_tags(constraint.feature_type)
            
            # Filter constraint features
            constraint_layer = f"constraint_{i}_features"
            plan.append({
                "operation": "filter_by_attribute",
                "input_layer": "osm_data",
                "key": constraint_tag,
                "value": constraint_value,
                "output_layer": constraint_layer
            })
            
            # Apply spatial relationship
            if constraint.relationship == SpatialRelationship.NEAR:
                buffer_layer = f"constraint_{i}_buffer"
                plan.append({
                    "operation": "buffer",
                    "input_layer": constraint_layer,
                    "distance_meters": constraint.distance_meters or 500,
                    "output_layer": buffer_layer
                })
                
                result_layer = f"step_{i}_result"
                plan.append({
                    "operation": "clip",
                    "input_layer": current_layer,
                    "clip_layer": buffer_layer,
                    "output_layer": result_layer
                })
                current_layer = result_layer
        
        # Add final rename step
        plan.append({
            "operation": "rename_layer",
            "input_layer": current_layer,
            "output_layer": "final_result"
        })
        
        return {
            "reasoning": f"TEMPLATE-ENHANCED WORKFLOW: Generated reliable workflow for {parsed_query.target} in {parsed_query.location} with {len(parsed_query.constraints)} constraints.",
            "plan": plan,
            "complexity_assessment": "Template-based with constraint handling",
            "error_handling": ["Template validation", "OSM tag inference", "Constraint processing"]
        }

    @lru_cache(maxsize=50)
    def _infer_osm_tags(self, feature_type: str) -> Tuple[str, str]:
        """Infers appropriate OSM tags for a feature type."""
        feature_lower = feature_type.lower()
        
        # Common amenities
        amenity_mapping = {
            "restaurant": ("amenity", "restaurant"),
            "school": ("amenity", "school"),
            "hospital": ("amenity", "hospital"),
            "bank": ("amenity", "bank"),
            "cafe": ("amenity", "cafe"),
            "pharmacy": ("amenity", "pharmacy"),
            "shop": ("shop", "*"),
            "store": ("shop", "*"),
        }
        
        # Leisure facilities
        leisure_mapping = {
            "park": ("leisure", "park"),
            "playground": ("leisure", "playground"),
            "sports": ("leisure", "sports_centre"),
            "gym": ("leisure", "fitness_centre"),
            "garden": ("leisure", "garden"),
        }
        
        # Land use
        landuse_mapping = {
            "residential": ("landuse", "residential"),
            "commercial": ("landuse", "commercial"),
            "industrial": ("landuse", "industrial"),
            "forest": ("landuse", "forest"),
        }
        
        # Transportation
        highway_mapping = {
            "road": ("highway", "*"),
            "highway": ("highway", "*"),
            "street": ("highway", "*"),
        }
        
        # Natural features
        natural_mapping = {
            "water": ("natural", "water"),
            "river": ("natural", "water"),
            "lake": ("natural", "water"),
            "mountain": ("natural", "peak"),
        }
        
        # Check all mappings
        for mapping in [amenity_mapping, leisure_mapping, landuse_mapping, 
                       highway_mapping, natural_mapping]:
            if feature_lower in mapping:
                return mapping[feature_lower]
        
        # Default fallback
        return ("amenity", feature_type)

    def _make_llm_call(self, prompt: str) -> Dict:
        """Enhanced LLM call with better error handling."""
        if not self.ollama_base_url:
            raise ValueError("OLLAMA_BASE_URL environment variable is not configured.")
        
        full_api_url = f"{self.ollama_base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": 1500  # Optimized for faster responses
            }
        }

        try:
            response = requests.post(full_api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            response_data = response.json()
            if 'response' not in response_data:
                raise ValueError("Invalid response format from LLM")
                
            return json.loads(response_data['response'])
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Could not connect to Ollama at {self.ollama_base_url}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"LLM request timed out after {self.timeout} seconds")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from LLM: {e}")

    def _validate_workflow_logic(self, plan: List[Dict]) -> None:
        """Enhanced workflow validation with location scope checking."""
        if not plan:
            raise ValueError("Empty workflow plan generated")
            
        # Check if first step loads data
        if plan[0].get("operation") != "load_osm_data":
            raise ValueError("Workflow must start with load_osm_data operation")
        
        # CRITICAL: Validate location scope (Fix for "Catastrophic Data Fetching")
        load_step = plan[0]
        location = load_step.get("location", "")
        is_too_broad, reason, alternatives = self.location_validator.is_location_too_broad(location)
        
        if is_too_broad:
            error_msg = f"Geographic scope is too broad ({reason}): '{location}'"
            if alternatives:
                error_msg += f". Consider using: {', '.join(alternatives[:3])}"
            raise ValueError(error_msg)
            
        # Check if last step renames to final_result
        if plan[-1].get("operation") != "rename_layer" or plan[-1].get("output_layer") != "final_result":
            raise ValueError("Workflow must end with rename_layer to 'final_result'")
            
        # Validate data flow between steps
        available_layers = set()
        for i, step in enumerate(plan):
            operation = step.get("operation")
            
            if operation == "load_osm_data":
                available_layers.add(step.get("output_layer"))
            elif "input_layer" in step:
                input_layer = step.get("input_layer")
                if input_layer not in available_layers:
                    raise ValueError(f"Step {i+1}: input_layer '{input_layer}' not available")
                if "output_layer" in step:
                    available_layers.add(step.get("output_layer"))
            elif "output_layer" in step:
                available_layers.add(step.get("output_layer"))

    def _generate_intelligent_fallback(self, parsed_query: ParsedQuery, error_msg: str) -> Dict[str, Any]:
        """Generates an intelligent fallback workflow based on query analysis."""
        # Analyze the query to determine the best fallback strategy
        osm_tag, osm_value = self._infer_osm_tags(parsed_query.target)
        
        # Ensure location is properly scoped
        location = parsed_query.location
        is_too_broad, _, alternatives = self.location_validator.is_location_too_broad(location)
        if is_too_broad and alternatives:
            location = f"{alternatives[0]}, {location}"
        
        # Create a working fallback plan
        fallback_plan = [
            {
                "operation": "load_osm_data",
                "location": location,
                "output_layer": "osm_data"
            },
            {
                "operation": "filter_by_attribute",
                "input_layer": "osm_data",
                "key": osm_tag,
                "value": osm_value,
                "output_layer": "target_features"
            },
            {
                "operation": "rename_layer",
                "input_layer": "target_features",
                "output_layer": "final_result"
            }
        ]
        
        return {
            "reasoning": f"INTELLIGENT FALLBACK: Generated reliable workflow for {parsed_query.target} in {location}. Original error: {error_msg[:100]}...",
            "plan": fallback_plan,
            "complexity_assessment": "Low complexity fallback with error recovery",
            "error_handling": [
                "Fallback strategy activated due to LLM failures",
                "Location scope validated and corrected",
                "OSM tag inference applied",
                f"Original error: {error_msg[:50]}..."
            ]
        }

    def _create_cache_key(self, parsed_query: ParsedQuery, guidance: str) -> str:
        """Creates a hash-based cache key for workflow results."""
        content = f"{parsed_query.model_dump_json()}{guidance}"
        return hashlib.md5(content.encode()).hexdigest()

    def _cache_workflow(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Caches workflow results with size management."""
        if len(self._workflow_cache) >= self._max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._workflow_cache))
            del self._workflow_cache[oldest_key]
        
        self._workflow_cache[cache_key] = result

    def analyze_query_complexity(self, parsed_query: ParsedQuery) -> str:
        """Enhanced complexity analysis with location scope consideration."""
        complexity_score = 0
        
        # Base complexity from constraints
        complexity_score += len(parsed_query.constraints)
        
        # Location scope impact
        is_too_broad, _, _ = self.location_validator.is_location_too_broad(parsed_query.location)
        if is_too_broad:
            complexity_score += 3  # High penalty for broad locations
        
        # Spatial relationship complexity
        complex_relationships = [SpatialRelationship.FAR_FROM, SpatialRelationship.NOT_WITHIN]
        for constraint in parsed_query.constraints:
            if constraint.relationship in complex_relationships:
                complexity_score += 2
            if constraint.distance_meters and constraint.distance_meters > 1000:
                complexity_score += 1
        
        # Target complexity
        complex_targets = ["flood risk", "suitability", "accessibility", "network"]
        if any(target in parsed_query.target.lower() for target in complex_targets):
            complexity_score += 2
                
        if complexity_score <= 2:
            return "Low"
        elif complexity_score <= 6:
            return "Medium"
        else:
            return "High"

    def get_performance_stats(self) -> Dict[str, Any]:
        """Returns performance statistics for monitoring."""
        return {
            "cache_size": len(self._workflow_cache),
            "cache_hit_rate": "Not implemented",  # Could be enhanced
            "avg_generation_time": "Not implemented",  # Could be enhanced
            "total_queries_processed": "Not implemented"  # Could be enhanced
        }


# Enhanced standalone testing with comprehensive test cases
if __name__ == '__main__':
    print("🚀 Running Enhanced AI-GIS Workflow Generator Test")
    print("=" * 70)

    # Test cases including problematic scenarios
    test_cases = [
        {
            "name": "Simple Restaurant Query (Should Work)",
            "query": ParsedQuery(
                target='restaurant',
                location='Kolkata, West Bengal, India',
                constraints=[]
            ),
            "guidance": "Basic amenity search in urban area."
        },
        {
            "name": "Broad Location Test (Should Auto-Scope)",
            "query": ParsedQuery(
                target='hospital',
                location='West Bengal, India',  # This should be auto-scoped
                constraints=[]
            ),
            "guidance": "Testing automatic location scoping."
        },
        {
            "name": "Complex Multi-Constraint Query",
            "query": ParsedQuery(
                target='school',
                location='Berlin, Germany',
                constraints=[
                    SpatialConstraint(
                        feature_type='park',
                        relationship=SpatialRelationship.NEAR,
                        distance_meters=300
                    ),
                    SpatialConstraint(
                        feature_type='highway',
                        relationship=SpatialRelationship.FAR_FROM,
                        distance_meters=500
                    )
                ]
            ),
            "guidance": "Multi-criteria analysis with spatial constraints."
        },
        {
            "name": "Extremely Broad Location (Should Fail Safely)",
            "query": ParsedQuery(
                target='restaurant',
                location='Germany',  # This should be caught and handled
                constraints=[]
            ),
            "guidance": "Testing location scope validation."
        }
    ]

    generator = WorkflowGenerator()

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['name']}")
        print("-" * 50)
        
        try:
            start_time = time.time()
            
            # Analyze complexity
            complexity = generator.analyze_query_complexity(test_case['query'])
            print(f"📊 Query Complexity: {complexity}")
            
            # Generate workflow
            result = generator.generate_workflow(
                parsed_query=test_case['query'],
                guidance_from_rag=test_case['guidance']
            )
            
            end_time = time.time()
            print(f"⏱️  Generation Time: {end_time - start_time:.2f} seconds")

            print(f"\n🧠 Chain-of-Thought Reasoning:")
            print(result.get("reasoning", "No reasoning provided.")[:200] + "...")

            print(f"\n📋 Generated Workflow Plan:")
            for j, step in enumerate(result.get("plan", []), 1):
                print(f"  Step {j}: {step}")

            print(f"\n⚖️  Complexity Assessment:")
            print(result.get("complexity_assessment", "No assessment provided."))

            print(f"\n🛡️  Error Handling:")
            for error_strategy in result.get("error_handling", []):
                print(f"  - {error_strategy}")

        except Exception as e:
            print(f"❌ Test failed: {e}")

        print("\n" + "=" * 70)

    # Performance statistics
    print("\n📊 Performance Statistics:")
    stats = generator.get_performance_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n🎯 Enhanced AI-GIS Workflow Generator testing completed!")
    print("💡 Key improvements implemented:")
    print("   ✅ Intelligent LLM auto-correction (solves rename_layer failures)")
    print("   ✅ Multi-layer location scope validation (prevents data timeouts)")
    print("   ✅ Enhanced fallback mechanisms with OSM tag inference")
    print("   ✅ Workflow caching for improved performance")
    print("   ✅ Comprehensive error handling and recovery strategies")