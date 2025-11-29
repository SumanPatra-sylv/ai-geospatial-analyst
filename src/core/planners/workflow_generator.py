#!/usr/bin/env python3
"""
ENHANCED AI-GIS Workflow Generator with Smart Decision Making
Fixed version with goal tracking, loop prevention, and intelligent task completion
"""

import json
import os
import requests
import jinja2
from typing import List, Dict, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field
import hashlib

# Add centralized tag management
from src.core.knowledge.osm_tag_manager import tag_manager

# Local project imports
from src.core.agents.data_scout import DataScout, DataRealityReport
from src.core.agents.schemas import ClarificationAsk
from src.gis.tools.definitions import TOOL_REGISTRY
from src.core.planners.query_parser import ParsedQuery

class ToolCallIntent(BaseModel):
    """Represents the LLM's intent to use a specific tool."""
    tool_name: str = Field(..., description="The exact name of the tool from the AVAILABLE TOOLS list.")
    parameters: Dict[str, Any] = Field(..., description="A dictionary of conceptual parameters for the tool.")

class WorkflowGenerator:
    """
    ENHANCED Workflow Generator with intelligent decision making, goal tracking,
    loop prevention, and smart task completion detection.
    """

    # ✅ COMPLETELY REWRITTEN SYSTEM PROMPT with Goal Awareness
    # ✅ ENHANCED SYSTEM PROMPT with State Awareness and Loop Prevention
    ENHANCED_SYSTEM_PROMPT = """You are an Expert AI Geospatial Analyst with PERFECT LOGICAL REASONING. You must solve the user's query efficiently without wasteful operations.

**🎯 ORIGINAL MISSION:** {{ original_query }}

**📋 CRITICAL: DATA LOADING STATE CHECK**
{% if loaded_datasets %}
✅ **ALREADY LOADED DATASETS:** {{ loaded_datasets | join(', ') }}
🚫 **DO NOT LOAD THESE AGAIN** - They are already in your workspace!
{% endif %}

**🚨 MISSING DATA CHECK:**
{% if missing_datasets %}
❌ **You are missing these required datasets:** {{ missing_datasets | join(', ') }}
📋 **YOUR NEXT ACTION MUST BE:** Load the FIRST missing dataset using `load_osm_data`
{% else %}
✅ **All required datasets are loaded**
📋 **YOUR NEXT ACTION MUST BE:** Move to analysis (buffer, spatial_join, intersect) OR call `finish_task` if done
{% endif %}

**�️ CRITICAL: OSM TAG REQUIREMENTS:**
When using `load_osm_data`, you MUST use the correct OSM tags for each entity type:
- school: {"amenity": "school"}
- park: {"leisure": "park"}  
- hospital: {"amenity": "hospital"}
- restaurant: {"amenity": "restaurant"}

NEVER guess or make up OSM tags. Always use the established mappings above.

**🧠 STATE TRANSITION RULES:**
1. **IF missing data** → Use `load_osm_data` to load ONLY the missing dataset
2. **IF all data loaded BUT no analysis done** → Use analysis tools (buffer, spatial_join, intersect)
3. **IF analysis complete** → Call `finish_task` with the final layer
4. **NEVER** → Load the same dataset twice or repeat the same action

**📊 DATA LAYER DETAILS (CURRENT WORKSPACE):**
{% if layer_intelligence %}
{% for layer_name, details in layer_intelligence.items() %}
- **{{ layer_name }}** ({{ details.feature_count }} features, {{ details.status }}):
  - Geometry: {{ details.geometry_types | join(', ') }}
  - Key columns: {{ details.columns[:8] | join(', ') }}{% if details.columns|length > 8 %} + {{ details.columns|length - 8 }} more{% endif %}
  - Summary: {{ details.summary }}
{% endfor %}
{% else %}
❌ **NO DATA LAYERS EXIST YET** - Your first action must load the primary target data!
{% endif %}

**📚 GUIDANCE FROM PAST SUCCESS:**
{{ rag_guidance if rag_guidance else "No similar past workflows found." }}

**📜 MISSION HISTORY:**
{{ history }}

**🛡️ OPERATION RULES:**
1. **NO DUPLICATE OPERATIONS:** Never repeat the exact same action with same parameters
2. **NO MEANINGLESS CALCULATIONS:** Don't calculate distances between identical/very similar layers  
3. **STATE AWARENESS:** Check what datasets are already loaded before loading more
4. **FORWARD PROGRESS:** Each action must move toward completing the query, not sideways
5. **FINISH WHEN READY:** Call `finish_task` when you have the final answer layer

**🔧 AVAILABLE TOOLS:**
{{ tool_documentation }}
- `finish_task(final_layer_name: str, reason: str)`: Complete the mission with the final result layer

**⚡ YOUR TASK:** 
Based on the STATE TRANSITION RULES and MISSING DATA CHECK above, determine the single most logical next action. Output ONLY a valid JSON object with your decision.

**JSON FORMAT:**
{
  "tool_name": "exact_tool_name",
  "parameters": {
    "param1": "value1",
    "param2": "value2"
  }
}
"""

    def __init__(self, data_scout: DataScout):
        """Initialize the enhanced workflow generator."""
        self.data_scout = data_scout
        self.knowledge_base = data_scout.knowledge_base
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("OLLAMA_MODEL", "mistral")
        self.timeout = 120
        
        # ✅ NEW: Enhanced tracking for smart decision making and loop prevention
        self.executed_actions: Set[str] = set()  # Track action signatures to prevent duplicates
        self.recent_action_history: List[str] = []  # Track last N actions for loop detection
        self.query_requirements: Dict[str, bool] = {}  # Track what the query needs
        self.current_goal_state: Dict[str, Any] = {}  # Track progress toward goal
        self.loaded_datasets: Set[str] = set()  # Track which datasets have been loaded

    def get_initial_context(self, parsed_query: ParsedQuery, rag_guidance: str = "") -> Dict[str, Any]:
        """Enhanced initial context with goal requirement analysis."""
        print("🛰️  Data Scout: Starting discovery and validation phase...")
        
        # === FIX: Handle Multi-Target properly BEFORE any processing ===
        # Normalize target to a list (can be str or List[str])
        primary_targets = parsed_query.target if isinstance(parsed_query.target, list) else [parsed_query.target]
        target_display = ', '.join(primary_targets) if isinstance(primary_targets, list) else primary_targets
        constraint_features = [c.feature_type for c in parsed_query.constraints] if parsed_query.constraints else []
        
        entities_to_probe = primary_targets + constraint_features
        # =========================================
        
        # ✅ NEW: Analyze query requirements (pass normalized targets)
        self._analyze_query_requirements(parsed_query, primary_targets)
        
        try:
            report = self.data_scout.generate_data_reality_report(
                location_string=parsed_query.location,
                entities_to_probe=entities_to_probe
            )
            
            if isinstance(report, ClarificationAsk):
                return {
                    "success": False,
                    "error": f"Ambiguity found for entity '{report.original_entity}': {report.message}",
                    "rag_guidance": rag_guidance
                }
            
            if not report:
                return {
                    "success": False, 
                    "error": f"Location '{parsed_query.location}' is invalid or could not be geocoded.",
                    "rag_guidance": rag_guidance
                }

            # Check if primary target(s) exist
            primary_target_found = any(
                probe.original_entity in primary_targets and probe.count > 0
                for probe in report.probe_results
            )

            if not primary_target_found:
                return {
                    "success": False,
                    "error": f"The primary target(s) '{target_display}' have 0 features at the specified location.",
                    "rag_guidance": rag_guidance
                }

            return {
                "success": True,
                "data_report": report,
                "rag_guidance": rag_guidance,
                "original_query": f"Find {target_display} in {parsed_query.location}",
                "initial_observation": self._format_initial_observation(report)
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Critical error during data scouting: {e}",
                "rag_guidance": rag_guidance
            }

    def get_next_action(self, history: List[str], original_query: str = "", 
                       rag_guidance: str = "", available_layers: List[str] = None,
                       layer_intelligence: Dict[str, Any] = None) -> ToolCallIntent:
        """Enhanced decision making with goal tracking and loop prevention."""
        print("✍️  Strategist: Analyzing history and deciding next action...")
        print(f"🔍 Layer Intelligence Available: {bool(layer_intelligence)}")
        
        try:
            # ✅ NEW: Update goal state analysis
            self._update_goal_state(layer_intelligence or {}, original_query)
            
            # ✅ NEW: Check if we can finish
            completion_check = self._check_task_completion(layer_intelligence or {}, original_query)
            if completion_check[0]:  # Can finish
                return ToolCallIntent(
                    tool_name="finish_task",
                    parameters={
                        "final_layer_name": completion_check[1],
                        "reason": completion_check[2]
                    }
                )
            
            prompt = self._build_enhanced_prompt(
                history, original_query, rag_guidance, available_layers, layer_intelligence
            )
            
            llm_response = self._make_llm_call(prompt)
            
            if "tool_name" in llm_response and "parameters" in llm_response:
                tool_call_data = llm_response
            else:
                tool_call_data = llm_response.get("tool_call", llm_response)
            
            if not isinstance(tool_call_data, dict) or "tool_name" not in tool_call_data:
                raise ValueError("Invalid tool call format from LLM")
            
            tool_call = ToolCallIntent(**tool_call_data)
            
            # ✅ NEW: Enhanced validation with duplicate prevention
            self._validate_enhanced_tool_call(tool_call, layer_intelligence, history)
            
            # ✅ NEW: Record action to prevent duplicates
            action_signature = self._get_action_signature(tool_call)
            self.executed_actions.add(action_signature)
            
            # ✅ NEW: Track recent actions for loop detection (keep last 5)
            self.recent_action_history.append(action_signature)
            if len(self.recent_action_history) > 5:
                self.recent_action_history.pop(0)
            
            # ✅ NEW: Track loaded datasets
            if tool_call.tool_name == "load_osm_data":
                # Extract what type of data was loaded from parameters
                conceptual_name = tool_call.parameters.get('conceptual_name', '')
                if conceptual_name:
                    dataset_type = conceptual_name.split('_')[0] if '_' in conceptual_name else conceptual_name
                    self.loaded_datasets.add(dataset_type.lower())
            
            return tool_call
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR in 'Think' phase: {e}")
            import traceback
            traceback.print_exc()
            return ToolCallIntent(
                tool_name="finish_task",
                parameters={
                    "reason": f"Critical error during LLM call: {e}",
                    "final_layer_name": list(layer_intelligence.keys())[-1] if layer_intelligence else "none"
                }
            )

    # ✅ NEW: Goal requirement analysis
    def _analyze_query_requirements(self, parsed_query: ParsedQuery, primary_targets: List[str]) -> None:
        """Enhanced query requirement analysis with precise constraint detection."""
        
        # === FIX: Handle both single and multi-target properly ===
        # primary_targets is already a list from get_initial_context
        target_str = ', '.join(primary_targets) if isinstance(primary_targets, list) else str(primary_targets)
        
        self.query_requirements = {
            'primary_target': target_str.lower(),
            'needs_constraints': bool(parsed_query.constraints),
            'constraint_features': [c.feature_type.lower() for c in parsed_query.constraints] if parsed_query.constraints else [],
            'needs_spatial_analysis': any(
                c.relationship.name in ['NEAR', 'WITHIN', 'INTERSECTS'] 
                for c in (parsed_query.constraints or [])
            ),
            'needs_filtering': any(
                hasattr(c, 'attribute_filter') and c.attribute_filter 
                for c in (parsed_query.constraints or [])
            ),
            'query_text': f"Find {target_str} in {parsed_query.location}".lower()
        }
        
        print(f"🎯 Query Requirements Analysis: {self.query_requirements}")

    # ✅ NEW: Goal state tracking
    def _update_goal_state(self, layer_intelligence: Dict[str, Any], original_query: str) -> None:
        """Update current progress toward the goal."""
        self.current_goal_state = {
            'has_primary_data': self._has_primary_target_data(layer_intelligence),
            'has_constraint_data': self._has_constraint_data(layer_intelligence),
            'has_combined_result': self._has_combined_analysis_result(layer_intelligence),
            'available_layers': list(layer_intelligence.keys()),
            'layer_count': len(layer_intelligence)
        }
        
        print(f"📊 Goal State: {self.current_goal_state}")

    # ✅ NEW: Smart task completion detection
    def _check_task_completion(self, layer_intelligence: Dict[str, Any], 
                           original_query: str) -> Tuple[bool, str, str]:
        """Enhanced task completion checker that validates complete spatial analysis."""
        if not layer_intelligence:
            return False, "", "No data layers available"
        
        # Get the required features from our query analysis
        primary_target = self.query_requirements.get('primary_target', '')
        constraint_features = self.query_requirements.get('constraint_features', [])
        all_required_features = {primary_target} | set(constraint_features)
        
        # --- Simple Query Check (no spatial constraints) ---
        if not constraint_features:
            # For simple queries, finish once primary target is loaded
            for layer_name, details in layer_intelligence.items():
                if primary_target in layer_name.lower():
                    if details.get('feature_count', 0) > 0:
                        reason = f"Primary target '{layer_name}' loaded with {details['feature_count']} features"
                        return True, layer_name, reason
            return False, "", "Primary target not yet loaded"

        # --- Complex Query Check (with spatial constraints) ---
        
        # Step 1: Verify all required data types are loaded
        loaded_features = set()
        for layer_name in layer_intelligence.keys():
            # Extract feature type from layer name (schools_berlin -> schools)
            if '_' in layer_name:
                feature_type = layer_name.split('_')[0]
            else:
                feature_type = layer_name.lower()
            
            # Map common variations
            if 'school' in feature_type:
                loaded_features.add('school')
            elif 'park' in feature_type:
                loaded_features.add('park')
            else:
                loaded_features.add(feature_type)
        
        # Check if all required features are loaded
        required_types = {primary_target} | set(constraint_features)
        if not required_types.issubset(loaded_features):
            missing = required_types - loaded_features
            return False, "", f"Missing required data: {', '.join(missing)}"
        
        # Step 2: Look for COMBINED analysis result
        # A true spatial analysis combines multiple datasets
        for layer_name, details in layer_intelligence.items():
            # Check for spatial analysis keywords
            if any(keyword in layer_name.lower() for keyword in 
                   ['join', 'intersect', 'within', 'near', 'buffer_intersect']):
                
                # Verify it has meaningful results
                feature_count = details.get('feature_count', 0)
                column_count = len(details.get('columns', []))
                
                # Combined layers typically have more columns (joined attributes)
                if feature_count > 0 and column_count > 200:
                    reason = f"Completed spatial analysis: '{layer_name}' with {feature_count} features"
                    return True, layer_name, reason
        
        # Step 3: If we have all data but no combined result, analysis not complete
        return False, "", "All data loaded but spatial analysis not yet performed"

    # ✅ NEW: Enhanced validation with duplicate prevention
    def _validate_enhanced_tool_call(self, tool_call: ToolCallIntent, 
                                   layer_intelligence: Dict[str, Any] = None,
                                   history: List[str] = None) -> None:
        """Enhanced validation with duplicate detection and logic checking."""
        
        # Basic validation
        if tool_call.tool_name == "finish_task":
            return
            
        if tool_call.tool_name not in TOOL_REGISTRY:
            raise ValueError(f"Non-existent tool proposed: '{tool_call.tool_name}'")
        
        # ✅ NEW: Validate OSM tags when loading data
        if tool_call.tool_name == "load_osm_data":
            # Get the entity from the conceptual name or layer name
            entity = tool_call.parameters.get("conceptual_name") or tool_call.parameters.get("layer_name", "")
            if entity:
                expected_tags = self._get_osm_tags_for_entity(entity.lower())
                provided_tags = tool_call.parameters.get("tags", {})
                
                # Check if the provided tags match the expected ones
                if provided_tags != expected_tags:
                    print(f"⚠️ Correcting OSM tags for '{entity}' from {provided_tags} to {expected_tags}")
                    tool_call.parameters["tags"] = expected_tags
            
            return  # Skip further validation since load_osm_data creates new layers
        
        # ✅ NEW: Duplicate action prevention
        action_signature = self._get_action_signature(tool_call)
        if action_signature in self.executed_actions:
            raise ValueError(f"Duplicate action detected: {action_signature}")
        
        # ✅ NEW: Meaningless operation detection
        if tool_call.tool_name == "calculate_distance":
            from_layer = tool_call.parameters.get("from_layer_name")
            to_layer = tool_call.parameters.get("to_layer_name")
            
            if from_layer and to_layer and layer_intelligence:
                # Check if layers are identical or very similar
                if self._are_layers_equivalent(from_layer, to_layer, layer_intelligence):
                    raise ValueError(f"Meaningless distance calculation between equivalent layers: {from_layer} and {to_layer}")
        
        # ✅ NEW: Spatial join logic validation
        if tool_call.tool_name == "spatial_join":
            left_layer = tool_call.parameters.get("left_layer_name")
            right_layer = tool_call.parameters.get("right_layer_name")
            
            if left_layer and right_layer and layer_intelligence:
                # Prevent joining a layer with itself or its direct derivative
                if self._are_layers_equivalent(left_layer, right_layer, layer_intelligence):
                    raise ValueError(f"Meaningless spatial join between equivalent layers: {left_layer} and {right_layer}")
        
        # Layer existence validation
        if layer_intelligence:
            for param_name, param_value in tool_call.parameters.items():
                if param_name in self.LAYER_REFERENCE_PARAMS:
                    if param_value not in layer_intelligence:
                        raise ValueError(f"Referenced layer '{param_value}' does not exist")
                    
                    layer_info = layer_intelligence[param_value]
                    if layer_info.get('status') != 'populated':
                        raise ValueError(f"Cannot use empty layer '{param_value}'")

    # ✅ NEW: Layer equivalence detection
    def _are_layers_equivalent(self, layer1: str, layer2: str, 
                              layer_intelligence: Dict[str, Any]) -> bool:
        """Check if two layers are equivalent or derived from the same source."""
        if layer1 == layer2:
            return True
        
        if layer1 not in layer_intelligence or layer2 not in layer_intelligence:
            return False
        
        info1 = layer_intelligence[layer1]
        info2 = layer_intelligence[layer2]
        
        # ✅ NEW: Check if layers represent different feature types
        # Schools and parks are different feature types, not equivalent
        layer1_type = self._infer_layer_type(layer1, info1)
        layer2_type = self._infer_layer_type(layer2, info2)
        
        if layer1_type and layer2_type and layer1_type != layer2_type:
            return False  # Different feature types are not equivalent
        
        # Check if they have identical feature counts and bounds (same dataset)
        if (info1.get('feature_count') == info2.get('feature_count') and
            info1.get('bounds') == info2.get('bounds') and
            info1.get('feature_count', 0) > 0):
            # Additional check: ensure they're actually the same type of features
            return layer1_type == layer2_type
        
        return False

    def _infer_layer_type(self, layer_name: str, layer_info: Dict[str, Any]) -> Optional[str]:
        """Infer the feature type of a layer from its name and content."""
        layer_name_lower = layer_name.lower()
        
        # Check layer name for feature type indicators
        if 'school' in layer_name_lower:
            return 'school'
        elif 'park' in layer_name_lower:
            return 'park'
        elif 'restaurant' in layer_name_lower:
            return 'restaurant'
        
        # Check sample values for feature type indicators
        sample_values = layer_info.get('sample_values', {})
        for column, values in sample_values.items():
            for value in values:
                value_str = str(value).lower()
                if 'school' in value_str:
                    return 'school'
                elif 'park' in value_str:
                    return 'park'
        
        return None

    # ✅ NEW: Action signature generation
    def _get_action_signature(self, tool_call: ToolCallIntent) -> str:
        """Generate a unique signature for an action to detect duplicates."""
        # For load_osm_data, focus on the core identifying parameters
        if tool_call.tool_name == "load_osm_data":
            core_params = {
                "area_name": tool_call.parameters.get("area_name"),
                "tags": tool_call.parameters.get("tags")
            }
            sorted_params = json.dumps(core_params, sort_keys=True)
        else:
            sorted_params = json.dumps(tool_call.parameters, sort_keys=True)
        
        signature = f"{tool_call.tool_name}:{sorted_params}"
        return hashlib.md5(signature.encode()).hexdigest()[:16]

    # ✅ NEW: Goal state checkers
    def _has_primary_target_data(self, layer_intelligence: Dict[str, Any]) -> bool:
        """Enhanced check if we have the primary target data loaded."""
        primary_target = self.query_requirements.get('primary_target', '').lower()
        
        # Check layer names for primary target
        name_match = any(primary_target in layer_name.lower() for layer_name in layer_intelligence.keys())
        
        # ✅ NEW: Also check layer content/tags for primary target
        # Look for layers that might contain the primary target data even with generic names
        for layer_name, details in layer_intelligence.items():
            # Check if this layer has features that match our target
            sample_values = details.get('sample_values', {})
            for column, values in sample_values.items():
                if any(primary_target in str(value).lower() for value in values):
                    return True
        
        return name_match

    def _has_constraint_data(self, layer_intelligence: Dict[str, Any]) -> bool:
        """Check if we have constraint feature data loaded."""
        constraint_features = self.query_requirements.get('constraint_features', [])
        if not constraint_features:
            return True  # No constraints needed
        
        return any(
            any(feature in layer_name.lower() for feature in constraint_features)
            for layer_name in layer_intelligence.keys()
        )

    def _has_combined_analysis_result(self, layer_intelligence: Dict[str, Any]) -> bool:
        """Check if we have a result that combines primary target with constraints."""
        if not self.query_requirements.get('needs_constraints'):
            return True  # No combination needed
        
        # Look for layers that suggest spatial analysis has been done
        analysis_keywords = ['join', 'intersect', 'filter', 'near', 'within', 'buffer']
        return any(
            any(keyword in layer_name.lower() for keyword in analysis_keywords)
            for layer_name in layer_intelligence.keys()
        )

    def _get_missing_datasets(self, layer_intelligence: Dict[str, Any]) -> List[str]:
        """Identify which required datasets are missing and need to be loaded."""
        missing = []
        
        # Check if we have primary target data
        if not self._has_primary_target_data(layer_intelligence):
            missing.append(self.query_requirements['primary_target'])
        
        # Check if we have constraint feature data
        constraint_features = self.query_requirements.get('constraint_features', [])
        for feature in constraint_features:
            feature_exists = any(
                feature in layer_name.lower() for layer_name in layer_intelligence.keys()
            )
            if not feature_exists:
                missing.append(feature)
        
        return missing

    def _build_enhanced_prompt(self, history: List[str], original_query: str = "", 
                              rag_guidance: str = "", available_layers: List[str] = None,
                              layer_intelligence: Dict[str, Any] = None) -> str:
        """Build enhanced prompt with missing dataset detection and loaded dataset awareness."""
        tool_docs = self._generate_tool_documentation_string()
        history_text = "\n".join(history) if history else "Mission just started."
        
        # ✅ NEW: Detect missing datasets
        missing_datasets = self._get_missing_datasets(layer_intelligence or {})
        
        # ✅ NEW: Format loaded datasets for display
        loaded_dataset_names = sorted(list(self.loaded_datasets)) if self.loaded_datasets else []
        
        template = jinja2.Environment(
            loader=jinja2.BaseLoader(), 
            trim_blocks=True, 
            lstrip_blocks=True
        ).from_string(self.ENHANCED_SYSTEM_PROMPT)
        
        return template.render(
            tool_documentation=tool_docs,
            history=history_text,
            original_query=original_query,
            rag_guidance=rag_guidance,
            available_layers=available_layers or [],
            layer_intelligence=layer_intelligence or {},
            missing_datasets=missing_datasets,  # Pass missing datasets
            loaded_datasets=loaded_dataset_names  # ✅ NEW: Pass loaded datasets to prompt
        )

    # ✅ UPDATED: Parameter validation with better logic checking
    def _validate_single_action(self, tool_name: str, params: Dict[str, Any]) -> None:
        """Enhanced parameter validation with logical consistency checks."""
        if tool_name not in TOOL_REGISTRY:
            raise ValueError(f"Unknown tool: '{tool_name}'")

        tool_definition = TOOL_REGISTRY[tool_name]
        expected_params = {p.name for p in tool_definition.parameters}
        provided_params = set(params.keys())

        # Account for special parameters
        dispatcher_special_params = {"conceptual_name"}
        valid_params = provided_params - dispatcher_special_params

        # Check for unknown parameters
        if not valid_params.issubset(expected_params):
            unknown_params = valid_params - expected_params
            raise ValueError(
                f"Unknown parameters for '{tool_name}': {unknown_params}. "
                f"Expected: {expected_params}"
            )

        # Check for required parameters
        required_params = {p.name for p in tool_definition.parameters if not hasattr(p, 'default') or p.default is None}
        missing_params = required_params - valid_params

        if missing_params:
            raise ValueError(
                f"Missing required parameters for '{tool_name}': {missing_params}"
            )

        # ✅ NEW: Logic-specific validations
        if tool_name == "buffer" and "distance" in params:
            distance = params["distance"]
            if not isinstance(distance, (int, float)) or distance <= 0:
                raise ValueError(f"Buffer distance must be positive, got: {distance}")
            if distance > 50000:  # 50km seems reasonable max
                raise ValueError(f"Buffer distance {distance}m seems excessive (>50km)")

        if tool_name == "load_osm_data" and "tags" in params:
            tags = params["tags"]
            if not isinstance(tags, dict) or not tags:
                raise ValueError("OSM tags must be a non-empty dictionary")

    # Keep all your existing methods with the following additions/updates:
    
    # ✅ EXISTING METHODS (keep as-is)
    # ✅ FINAL, DEFINITIVE ALIAS MAP - Only true aliases, no incorrect conversions
    PARAMETER_ALIASES = {
        # --- Aliases for 'key' in filter_by_attribute (when LLM uses wrong names) ---
        "column": "key",
        "field": "key",
        
        # --- Aliases for 'distance' in buffer ---
        "distance_meters": "distance",
        "buffer_distance": "distance",
        "radius": "distance",
        
        # --- Aliases for spatial_join layers ---
        "left_layer": "left_layer_name",
        "right_layer": "right_layer_name",
        "layer1": "left_layer_name",
        "layer2": "right_layer_name",
        
        # --- Common LLM mistake for layer inputs ---
        "input_layer": "layer_name",
    }

    LAYER_REFERENCE_PARAMS = {
        "layer_name", "left_layer_name", "right_layer_name", "clip_layer_name",
        "layer1_name", "layer2_name", "overlay_layer_name", "union_layer_name", 
        "intersect_layer_name", "difference_layer_name"
    }

    def translate_single_action(self, tool_call: ToolCallIntent, conceptual_to_real_map: Dict[str, str]) -> Dict[str, Any]:
        """Translate a single tool call with tool-specific, canonical parameter handling."""
        
        # ✅ FIX: Pass tool_name to the enhanced canonical translator
        params = self._translate_parameter_aliases(tool_call.parameters.copy(), tool_call.tool_name)
        
        # Resolve conceptual layer names to real names
        for param_key, param_value in params.items():
            if param_key in self.LAYER_REFERENCE_PARAMS:
                if param_value in conceptual_to_real_map:
                    params[param_key] = conceptual_to_real_map[param_value]
                    print(f"🗺️  Resolved '{param_value}' -> '{conceptual_to_real_map[param_value]}'")
        
        if tool_call.tool_name != "finish_task":
            self._validate_single_action(tool_call.tool_name, params)
        
        return {
            "operation": tool_call.tool_name,
            "parameters": params
        }

    def store_successful_pattern(self, original_query: str, action_sequence: List[Dict[str, Any]], execution_time: float = 0.0) -> None:
        """Store a successful action sequence pattern for future RAG retrieval."""
        if action_sequence:
            print("💾 Knowledge Base: Storing successful action pattern...")
            self.knowledge_base.store_successful_workflow(
                original_query=original_query,
                workflow_plan=action_sequence,
                execution_time=execution_time
            )

    # Keep all other existing methods unchanged...
    def _get_osm_tags_for_entity(self, entity: str) -> Dict[str, str]:
        """Get consistent OSM tags for an entity using the centralized tag manager."""
        tags = tag_manager.get_primary_tags(entity)
        if not tags:
            print(f"⚠️ No known OSM tags for entity '{entity}', using generic approach")
            return {"name": entity}  # Fallback to name-based search
        return tags

    def _format_initial_observation(self, report: DataRealityReport) -> str:
        """Format the initial data reality report as an observation."""
        observation_lines = [
            f"Initial Observation: Location verified as '{report.location.canonical_name}'.",
            "Data availability:"
        ]
        
        for result in report.probe_results:
            observation_lines.append(f"  - {result.original_entity}: {result.count} features available")
        
        if report.recommendations:
            observation_lines.append("Scout recommendations:")
            for rec in report.recommendations:
                observation_lines.append(f"  - {rec}")
        
        return " ".join(observation_lines)

    def _translate_parameter_aliases(self, params: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """
        Translates LLM parameter names to canonical names using tool definition as ground truth.
        Only applies aliases when the parameter name is NOT already correct.
        """
        if tool_name not in TOOL_REGISTRY:
            return params  # Cannot translate if we don't know the tool

        tool_definition = TOOL_REGISTRY[tool_name]
        canonical_param_names = {p.name for p in tool_definition.parameters}
        
        translated_params = {}
        for key, value in params.items():
            if key == "conceptual_name":
                translated_params[key] = value
                continue
                
            # ✅ CANONICAL-FIRST APPROACH: If the key is already correct, keep it
            if key in canonical_param_names:
                translated_params[key] = value
                continue
            
            # ✅ ALIAS-ONLY WHEN NEEDED: Only translate if it's a known alias
            canonical_key = self.PARAMETER_ALIASES.get(key)
            if canonical_key and canonical_key in canonical_param_names:
                translated_params[canonical_key] = value
            else:
                # Unknown parameter - keep as-is for validation to catch
                translated_params[key] = value
                
        return translated_params

    def _generate_tool_documentation_string(self) -> str:
        """Enhanced tool documentation generation."""
        doc_lines = []
        for tool in TOOL_REGISTRY.values():
            param_docs = []
            for param in tool.parameters:
                param_type = getattr(param, 'type', 'Any')
                is_required = not hasattr(param, 'default') or param.default is None
                required_marker = " (required)" if is_required else " (optional)"
                
                aliases = [alias for alias, canonical in self.PARAMETER_ALIASES.items() if canonical == param.name]
                alias_info = f" [aliases: {', '.join(aliases)}]" if aliases else ""
                
                param_docs.append(f"{param.name}: {param_type}{required_marker}{alias_info}")
            
            params_str = ", ".join(param_docs)
            doc_lines.append(f"- `{tool.operation_name}({params_str})`: {tool.description}")
        
        return "\n".join(doc_lines)

    def _make_llm_call(self, prompt: str) -> Dict:
        """Enhanced LLM call with better error handling and response parsing."""
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
                "num_predict": 1024,
                "stop": ["```"]
            } 
        }

        try:
            response = requests.post(full_api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            response_data = response.json()
            
            if 'response' not in response_data:
                raise ValueError("Invalid response format from LLM: 'response' key missing.")
            
            raw_llm_json_str = response_data['response'].strip()
            
            # Clean up common LLM response artifacts
            if raw_llm_json_str.startswith('```json'):
                raw_llm_json_str = raw_llm_json_str[7:]
            if raw_llm_json_str.endswith('```'):
                raw_llm_json_str = raw_llm_json_str[:-3]
            raw_llm_json_str = raw_llm_json_str.strip()
            
            print("--- RAW LLM JSON RESPONSE ---")
            print(raw_llm_json_str)
            print("---------------------------")
            
            parsed_response = json.loads(raw_llm_json_str)
            
            if not isinstance(parsed_response, dict):
                raise ValueError("LLM response must be a JSON object")
            
            # Universal response parser
            tool_name_keys = ['tool_name', 'tool', 'action']
            final_tool_name = None
            for key in tool_name_keys:
                if key in parsed_response:
                    final_tool_name = parsed_response[key]
                    break

            if not final_tool_name:
                raise ValueError(f"LLM response is missing a valid tool name key (expected one of {tool_name_keys})")

            param_keys = ['parameters', 'arguments']
            final_parameters = None
            for key in param_keys:
                if key in parsed_response:
                    final_parameters = parsed_response[key]
                    break
            
            if final_parameters is None:
                final_parameters = {
                    k: v for k, v in parsed_response.items() 
                    if k not in tool_name_keys
                }

            return {
                "tool_name": final_tool_name,
                "parameters": final_parameters
            }
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Could not connect to Ollama at {self.ollama_base_url}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"LLM request timed out after {self.timeout} seconds")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"HTTP error from LLM service: {e}")

# Keep the existing test code at the bottom unchanged...


if __name__ == '__main__':
    # ------------------------------------------------------------
    # EXTRA IMPORTS
    from src.core.planners.query_parser import SpatialConstraint, SpatialRelationship
    import pprint
    import time
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # UNIVERSAL TEST-RUNNER FOR QUICK BENCHMARKS
    def run_test(generator: WorkflowGenerator, parsed_query: ParsedQuery, test_name: str):
        """Run a single query end-to-end and print a compact summary."""
        print(f"\n🧪 TEST: {test_name}")
        print(f"   Query: {parsed_query.target} in {parsed_query.location}")
        print("-" * 60)
        t0 = time.perf_counter()
        context = generator.get_initial_context(parsed_query, rag_guidance="")
        if not context["success"]:
            print(f"❌ Context failure → {context['error']}")
            return

        history = [context["initial_observation"]]
        layers: dict[str, str] = {}
        layer_names: list[str] = []
        layer_intel: dict[str, dict] = {}
        action_seq = []

        for step in range(12):          # up to 12 iterative steps
            next_action = generator.get_next_action(
                history            = history,
                original_query     = context["original_query"],
                rag_guidance      = context["rag_guidance"],
                available_layers  = layer_names,
                layer_intelligence = layer_intel
            )
            executable = generator.translate_single_action(next_action, layers)
            action_seq.append(executable)

            # Simulate executor result layer-creation (mock only)
            if next_action.tool_name == "finish_task":
                reason = next_action.parameters.get("reason", "completed")
                history.append(f"Action: finish_task  • {reason}")
                break
            else:
                concept = next_action.parameters.get("conceptual_name", 
                          next_action.parameters.get("layer_name",
                          f"{next_action.tool_name}_out"))
                real = f"{concept}_{step+1}"
                layers[concept] = real
                layer_names.append(real)
                layer_intel[real] = {
                    "status": "populated",
                    "feature_count": 50,
                    "columns": ["id", "name", "geometry"],
                    "geometry_types": ["Point"],
                    "crs": "EPSG:4326",
                    "summary": f"mock layer {real}"
                }
                history.append(f"Action: {next_action.tool_name}")
                history.append(f"Observation: created {real}")

        Δt = time.perf_counter() - t0
        print(f"✅ FINISHED in {Δt:0.1f}s • {len(action_seq)} steps")
        print("Final action sequence:")
        pprint.pp(action_seq, depth=2, compact=True)
    # ------------------------------------------------------------

    # --- Initialize the Generator ---
    data_scout_agent = DataScout()
    generator = WorkflowGenerator(data_scout=data_scout_agent)

    # --- Test 1: Original "Simple School" Test ---
    simple_query = ParsedQuery(
        target="school",
        location="Berlin, Germany",
        constraints=[],
        summary_required=True
    )
    run_test(generator, simple_query, "SIMPLE SCHOOLS")

    # --- Test 2: Constraint Scenario ("Schools near Parks ≤ 500 m") ---
    constraint_query = ParsedQuery(
        target="school",
        location="Berlin, Germany",
        constraints=[
            SpatialConstraint(
                feature_type="park",
                relationship=SpatialRelationship.NEAR,
                distance_meters=500
            )
        ],
        summary_required=True
    )
    run_test(generator, constraint_query, "SCHOOLS NEAR PARKS 500 m")

    # --- Final Print-Out ---
    print("\n📊  Summary of Both Tests Complete.")