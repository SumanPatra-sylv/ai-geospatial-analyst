#!/usr/bin/env python3
"""
AI-GIS Workflow Generator (Iterative Version with RAG and Layer Intelligence)
Generates single-step spatial analysis decisions by using "Think-Act-Observe" loop,
enhanced with RAG retrieval capabilities and full layer visibility for smarter decisions.
"""

import json
import os
import requests
import jinja2
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field

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
    Generates single-step spatial operation decisions using "Think-Act-Observe" pattern.
    Enhanced with RAG retrieval and full layer intelligence visibility for optimal decision-making.
    """

    # ✅ UPGRADED SYSTEM PROMPT - Now uses layer_intelligence for perfect column awareness
    # src/core/planners/workflow_generator.py
    ITERATIVE_SYSTEM_PROMPT = """
    You are a master GIS Strategist.  Your job is to solve the user's spatial query
    by emitting ONE tool call per turn.

    ────────────────────────────────────────────────────────
    1. MISSION GOAL (the full request)
    {{ original_query }}

    2. SUCCESS CRITERIA  – ALL must be true before you may call `finish_task`
    • Every thematic entity mentioned in the mission goal
    (e.g. “schools”, “parks”, “roads”) exists as a populated layer.
    • Every spatial / attribute relationship requested in the goal
    (e.g. “near”, “within 500 m”, “intersect”) has been computed.
    • A final layer exists that directly answers the user’s question.

    3. STRATEGIC CHECKLIST  – Perform internally each turn
    a. Extract the list of thematic entities from the MISSION GOAL.  
    b. For each entity, check DATA LAYER DETAILS.  
    • If a populated layer is missing → next action MUST be `load_osm_data`
        (or another loader) to obtain it.  
    c. If all entities are present, ask: “Has the required spatial /
    attribute relationship been built?”  
    • If not, choose the tool that creates the largest chunk of the
        missing logic (buffer, spatial_join, distance_filter, etc.).  
    d. If SUCCESS CRITERIA are all satisfied, call `finish_task`.

    4. OPERATION RULES
    • Use the EXACT column names shown in DATA LAYER DETAILS.  
    • Operate only on layers whose status is “populated”.  
    • Return exactly ONE JSON object: {"tool_name": ..., "parameters": ...}.  
    • Never output your chain of thought.  
    • `finish_task(final_layer_name: str)` is the last step.

    ────────────────────────────────────────────────────────
    DATA LAYER DETAILS (YOUR WORKSPACE):
    {% if layer_intelligence %}
    {% for layer_name, details in layer_intelligence.items() %}
    - **{{ layer_name }}** (status: {{ details.status }})
    • Features: {{ details.feature_count }}
    • Columns: {{ details.columns | join(', ') }}
    • Geometry: {{ details.geometry_types | join(', ') }}
    • CRS: {{ details.crs }}{% endfor %}
    {% else %}
    No data layers exist yet.  Your first action must be `load_osm_data`.
    {% endif %}
    ────────────────────────────────────────────────────────
    AVAILABLE TOOLS:
    {{ tool_documentation }}
    • finish_task(final_layer_name: str)

    MISSION HISTORY:
    {{ history }}
    **DECISION EXAMPLES:**
{"tool_name": "load_osm_data", "parameters": {"area_name": "Berlin, Germany", "tags": {"amenity": "school"}, "conceptual_name": "schools"}}


{"tool_name": "filter_by_attribute", "parameters": {"layer_name": "schools_1", "key": "name", "value": "Elementary"}}


{"tool_name": "finish_task", "parameters": {"final_layer_name": "filtered_schools_2"}}
EXAMPLE OUTPUT
{"tool_name": "load_osm_data",
 "parameters": {"area_name": "Berlin, Germany",
                "tags": {"amenity": "school"},
                "conceptual_name": "schools"}}

    ────────────────────────────────────────────────────────
    TASK → Decide the SINGLE next tool call that best advances the
    MISSION GOAL according to the STRATEGIC CHECKLIST, and output it as JSON.
    """


    # ✅ ENHANCED PARAMETER MAPPINGS - More comprehensive coverage
    PARAMETER_ALIASES = {
        "layer_name": "input_layer",
        "attribute": "key", 
        "column": "key",
        "field": "key",
        "input_layer": "layer_name",  # Bidirectional mapping
        "distance_meters": "distance",
        "buffer_distance": "distance",
        "radius": "distance",
        "left_layer": "left_layer_name",
        "right_layer": "right_layer_name",
        "clip_layer": "clip_layer_name",
        "buffer_layer": "layer_name",
        "target_layer": "layer_name",
        "source_layer": "layer_name",
        "geometry_layer": "layer_name",
        "points_layer": "layer_name",
        "polygons_layer": "layer_name",
        "lines_layer": "layer_name",
        "layer1": "layer1_name",
        "layer2": "layer2_name",
        "first_layer": "layer1_name",
        "second_layer": "layer2_name",
        "overlay_layer": "overlay_layer_name",
        "union_layer": "union_layer_name",
        "intersect_layer": "intersect_layer_name",
        "difference_layer": "difference_layer_name",
        "filter_value": "value",
        "search_value": "value",
        "attribute_value": "value"
    }

    # Layer reference parameters - these should be resolved to actual layer names
    LAYER_REFERENCE_PARAMS = {
        "layer_name", "left_layer_name", "right_layer_name", "clip_layer_name",
        "layer1_name", "layer2_name", "overlay_layer_name", "union_layer_name", 
        "intersect_layer_name", "difference_layer_name"
    }

    def __init__(self, data_scout: DataScout):
        """
        Initializes the WorkflowGenerator with its required agents using dependency injection.
        
        Args:
            data_scout: An initialized instance of the DataScout agent.
        """
        self.data_scout = data_scout
        self.knowledge_base = data_scout.knowledge_base
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("OLLAMA_MODEL", "mistral")
        self.timeout = 120

    def get_initial_context(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """
        Performs the initial data discovery phase to establish starting context.
        
        Returns:
            Dictionary with initial context including data reality report and RAG guidance
        """
        print("🧠 Knowledge Base: Searching for similar, previously solved problems...")
        
        # Build query context for RAG
        query_context = f"{parsed_query.location} {parsed_query.target}"
        if parsed_query.constraints:
            constraint_desc = " ".join([f"{c.feature_type} {c.relationship.value}" for c in parsed_query.constraints])
            query_context += f" {constraint_desc}"
        
        # Get RAG guidance
        rag_guidance = self._get_rag_guidance(query_context)
        
        # Perform data discovery
        print("🛰️  Data Scout: Starting discovery and validation phase...")
        
        entities_to_probe = [parsed_query.target] + [c.feature_type for c in parsed_query.constraints]
        
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

            # Check if primary target exists
            primary_target_found = any(
                probe.original_entity == parsed_query.target and probe.count > 0
                for probe in report.probe_results
            )

            if not primary_target_found:
                return {
                    "success": False,
                    "error": f"The primary target '{parsed_query.target}' has 0 features at the specified location.",
                    "rag_guidance": rag_guidance
                }

            return {
                "success": True,
                "data_report": report,
                "rag_guidance": rag_guidance,
                "original_query": f"Find {parsed_query.target} in {parsed_query.location}",
                "initial_observation": self._format_initial_observation(report)
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Critical error during data scouting: {e}",
                "rag_guidance": rag_guidance
            }

    # ✅ FIXED: Added layer_intelligence parameter
    def get_next_action(self, history: List[str], original_query: str = "", 
                       rag_guidance: str = "", available_layers: List[str] = None,
                       layer_intelligence: Dict[str, Any] = None) -> ToolCallIntent:
        """
        Decide the single next best action with full context and layer intelligence.
        
        Args:
            history: List of previous thoughts, actions, and observations
            original_query: The original user query for context
            rag_guidance: Guidance from similar past workflows
            available_layers: List of currently available layer names
            layer_intelligence: Full layer details including columns, geometry, etc.
            
        Returns:
            Single ToolCallIntent for the next step
        """
        print("✍️  Strategist: Analyzing history and deciding next action...")
        print(f"🔍 Layer Intelligence Available: {bool(layer_intelligence)}")
        
        try:
            # ✅ FIXED: Pass layer_intelligence to prompt builder
            prompt = self._build_iterative_prompt(
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
            
            # Enhanced validation with layer intelligence
            self._validate_tool_call_with_intelligence(tool_call, layer_intelligence)
            
            return tool_call
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR in 'Think' phase: {e}")
            import traceback
            traceback.print_exc()
            return ToolCallIntent(
                tool_name="finish_task",
                parameters={"reason": f"Critical error during LLM call: {e}"}
            )

    # ✅ FIXED: Added layer_intelligence parameter
    def _build_iterative_prompt(self, history: List[str], original_query: str = "", 
                               rag_guidance: str = "", available_layers: List[str] = None,
                               layer_intelligence: Dict[str, Any] = None) -> str:
        """
        Build comprehensive prompt with all context including layer intelligence.
        
        Args:
            history: List of conversation history entries
            original_query: The original user query
            rag_guidance: Guidance from similar past workflows
            available_layers: List of currently available layer names
            layer_intelligence: Full layer details for smart decision making
            
        Returns:
            Formatted prompt for the LLM
        """
        tool_docs = self._generate_tool_documentation_string()
        history_text = "\n".join(history) if history else "Mission just started."
        
        template = jinja2.Environment(
            loader=jinja2.BaseLoader(), 
            trim_blocks=True, 
            lstrip_blocks=True
        ).from_string(self.ITERATIVE_SYSTEM_PROMPT)
        
        # ✅ FIXED: Pass layer_intelligence to template
        return template.render(
            tool_documentation=tool_docs,
            history=history_text,
            original_query=original_query,
            rag_guidance=rag_guidance,
            available_layers=available_layers or [],
            layer_intelligence=layer_intelligence or {}
        )

    def translate_single_action(self, tool_call: ToolCallIntent, available_layers: Dict[str, str]) -> Dict[str, Any]:
        """
        Translate a single tool call with current layer context.
        
        Args:
            tool_call: The tool call intent from the LLM
            available_layers: Dictionary mapping conceptual names to actual layer names
            
        Returns:
            Dictionary representing the executable action
        """
        params = self._translate_parameter_aliases(tool_call.parameters.copy())
        
        # Validate parameters if not finish_task
        if tool_call.tool_name != "finish_task":
            self._validate_single_action(tool_call.tool_name, params)
        
        # Resolve layer references using available layers
        for param_key, param_value in params.items():
            if param_key in self.LAYER_REFERENCE_PARAMS:
                if param_value in available_layers:
                    params[param_key] = available_layers[param_value]
                # If layer doesn't exist, leave as-is - executor will handle the error
        
        return {
            "operation": tool_call.tool_name,
            "parameters": params
        }

    def store_successful_pattern(self, original_query: str, action_sequence: list[Dict[str, Any]], execution_time: float = 0.0) -> None:
        """
        Store a successful action sequence pattern for future RAG retrieval.

        Args:
            original_query: The original user query
            action_sequence: list of successful actions taken
            execution_time: Total time taken for the successful workflow
        """
        if action_sequence:
            print("💾 Knowledge Base: Storing successful action pattern...")
            
            # ✅ FIX: Call the knowledge base with the correct 'execution_time' argument
            self.knowledge_base.store_successful_workflow(
                original_query=original_query,
                workflow_plan=action_sequence,
                execution_time=execution_time
            )

    # ✅ NEW: Enhanced validation using layer intelligence
    def _validate_tool_call_with_intelligence(self, tool_call: ToolCallIntent, layer_intelligence: Dict[str, Any] = None) -> None:
        """
        Validate tool call using layer intelligence for smarter error detection.
        
        Args:
            tool_call: The tool call to validate
            layer_intelligence: Full layer details for validation
        """
        if tool_call.tool_name == "finish_task" or not layer_intelligence:
            # Basic validation only
            if tool_call.tool_name != "finish_task" and tool_call.tool_name not in TOOL_REGISTRY:
                raise ValueError(f"Non-existent tool proposed: '{tool_call.tool_name}'")
            return
        
        # Advanced validation with layer intelligence
        params = tool_call.parameters
        
        # Check if referenced layers exist and have correct status
        for param_name, param_value in params.items():
            if param_name in self.LAYER_REFERENCE_PARAMS:
                if param_value not in layer_intelligence:
                    raise ValueError(f"Referenced layer '{param_value}' does not exist")
                
                layer_info = layer_intelligence[param_value]
                if layer_info.get('status') != 'populated':
                    raise ValueError(f"Cannot use layer '{param_value}' with status '{layer_info.get('status')}'")
        
        # Validate column references for filtering operations
        if tool_call.tool_name in ["filter_by_attribute", "sort_by_attribute"] and "layer_name" in params and "key" in params:
            layer_name = params["layer_name"]
            column_name = params["key"]
            
            if layer_name in layer_intelligence:
                available_columns = layer_intelligence[layer_name].get('columns', [])
                if column_name not in available_columns:
                    raise ValueError(
                        f"Column '{column_name}' not found in layer '{layer_name}'. "
                        f"Available columns: {', '.join(available_columns)}"
                    )

    def _get_rag_guidance(self, query_context: str) -> str:
        """
        Get RAG guidance from similar past workflows.
        
        Args:
            query_context: Context string for similarity search
            
        Returns:
            Guidance string from past successful workflows
        """
        similar_workflows = self.knowledge_base.retrieve_similar_workflows(query_context)
        
        if similar_workflows:
            best_past_example = similar_workflows[0]
            action_count = len(best_past_example.get('workflow_plan', []))
            actions_used = [step.get('operation', 'unknown') for step in best_past_example.get('workflow_plan', [])]
            
            guidance = (
                f"SIMILAR PAST SUCCESS: Query '{best_past_example['original_query']}' "
                f"(Similarity: {best_past_example['similarity_score']:.2f}) was solved using "
                f"{action_count} actions: {' → '.join(actions_used)}. "
                f"Consider this successful pattern for guidance."
            )
            print(f"✅ Found similar successful workflow (Similarity: {best_past_example['similarity_score']:.2f})")
            return guidance
        else:
            print("ℹ️  No similar workflows found in knowledge base.")
            return ""

    def _format_initial_observation(self, report: DataRealityReport) -> str:
        """
        Format the initial data reality report as an observation.
        
        Args:
            report: DataRealityReport from the data scout
            
        Returns:
            Formatted observation string
        """
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

    def _translate_parameter_aliases(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translates common LLM parameter names to the system's canonical names.
        Enhanced with bidirectional mapping support.
        """
        translated_params = {}
        
        for key, value in params.items():
            if key == "conceptual_name":
                translated_params[key] = value
                continue
                
            # Check for direct mapping first, then reverse mapping
            canonical_key = self.PARAMETER_ALIASES.get(key, key)
            
            # If no direct mapping found, check if it's already a canonical name
            if canonical_key == key:
                # Check if this key is a target of any alias (reverse lookup)
                reverse_mappings = {v: k for k, v in self.PARAMETER_ALIASES.items() if v != k}
                if key in reverse_mappings:
                    canonical_key = key  # It's already canonical
            
            translated_params[canonical_key] = value
            
        return translated_params

    def _validate_single_action(self, tool_name: str, params: Dict[str, Any]) -> None:
        """
        Enhanced parameter validation for a single action.
        
        Args:
            tool_name: Name of the tool to validate
            params: Parameters for the tool
            
        Raises:
            ValueError: If validation fails
        """
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

        # Validate parameter types and values
        self._validate_parameter_types(tool_name, params)

    def _validate_parameter_types(self, tool_name: str, params: Dict[str, Any]) -> None:
        """
        Enhanced parameter type validation.
        """
        # Distance validation for buffer operations
        if tool_name == "buffer" and "distance" in params:
            distance = params["distance"]
            if not isinstance(distance, (int, float)) or distance <= 0:
                raise ValueError(f"Buffer distance must be a positive number, got: {distance}")

        # Coordinate validation for geometric operations
        if "coordinates" in params:
            coords = params["coordinates"]
            if not isinstance(coords, (list, tuple)) or len(coords) != 2:
                raise ValueError(f"Coordinates must be a list/tuple of [longitude, latitude], got: {coords}")
            
            try:
                lon, lat = float(coords[0]), float(coords[1])
                if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                    raise ValueError(f"Invalid coordinate values: longitude={lon}, latitude={lat}")
            except (ValueError, TypeError):
                raise ValueError(f"Coordinates must be numeric values, got: {coords}")

        # Tag validation for OSM operations
        if tool_name == "load_osm_data" and "tags" in params:
            tags = params["tags"]
            if not isinstance(tags, dict):
                raise ValueError(f"OSM tags must be a dictionary, got: {type(tags)}")
            
            if not tags:
                raise ValueError("OSM tags dictionary cannot be empty")

    def _generate_tool_documentation_string(self) -> str:
        """
        Enhanced tool documentation generation with better formatting and alias information.
        """
        doc_lines = []
        for tool in TOOL_REGISTRY.values():
            param_docs = []
            for param in tool.parameters:
                param_type = getattr(param, 'type', 'Any')
                is_required = not hasattr(param, 'default') or param.default is None
                required_marker = " (required)" if is_required else " (optional)"
                
                # Add common aliases in documentation
                aliases = [alias for alias, canonical in self.PARAMETER_ALIASES.items() if canonical == param.name]
                alias_info = f" [aliases: {', '.join(aliases)}]" if aliases else ""
                
                param_docs.append(f"{param.name}: {param_type}{required_marker}{alias_info}")
            
            params_str = ", ".join(param_docs)
            doc_lines.append(f"- `{tool.operation_name}({params_str})`: {tool.description}")
        
        return "\n".join(doc_lines)

    def _make_llm_call(self, prompt: str) -> Dict:
        """
        Enhanced LLM call with better error handling and response validation.
        """
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
                "num_predict": 1024,  # Smaller for single actions
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
            
            # Parse and validate the JSON structure
            parsed_response = json.loads(raw_llm_json_str)
            
            if not isinstance(parsed_response, dict):
                raise ValueError("LLM response must be a JSON object")
            
            # Validate tool call structure
            if "tool_name" not in parsed_response:
                raise ValueError("LLM response missing 'tool_name'")
            
            if "parameters" not in parsed_response:
                parsed_response["parameters"] = {}
            
            if not isinstance(parsed_response["parameters"], dict):
                raise ValueError("'parameters' must be a dictionary")
            
            return parsed_response
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Could not connect to Ollama at {self.ollama_base_url}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"LLM request timed out after {self.timeout} seconds")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"HTTP error from LLM service: {e}")


if __name__ == '__main__':
    import time
    from src.core.planners.query_parser import SpatialConstraint, SpatialRelationship

    print("🚀 Running Enhanced Iterative AI-GIS Workflow Generator Test")
    print("=" * 70)

    # Test the refactored iterative approach
    data_scout_agent = DataScout()
    generator = WorkflowGenerator(data_scout=data_scout_agent)

    # Test case: Simple school query
    test_query = ParsedQuery(
        target='school', 
        location='Berlin, Germany', 
        constraints=[], 
        summary_required=True
    )

    print(f"\n🧪 Testing Enhanced Iterative Approach")
    print(f"   Query: Find '{test_query.target}' in '{test_query.location}'")
    print("-" * 50)

    # Phase 1: Get initial context
    start_time = time.time()
    context = generator.get_initial_context(test_query)
    
    if not context["success"]:
        print(f"❌ Initial context failed: {context['error']}")
    else:
        print(f"✅ Initial context established")
        print(f"📊 {context['initial_observation']}")
        
        # Simulate a few iterative steps with enhanced layer tracking
        history = [context['initial_observation']]
        available_layers = {}
        available_layer_names = []
        action_sequence = []
        
        # Simulate layer intelligence data structure
        layer_intelligence = {}
        
        for step in range(3):  # Test 3 iterative steps
            print(f"\n--- Step {step + 1} ---")
            print(f"📋 Available layers: {available_layer_names}")
            print(f"🔍 Layer Intelligence: {list(layer_intelligence.keys())}")
            
            # Get next action with enhanced context including layer intelligence
            next_action = generator.get_next_action(
                history=history,
                original_query=context.get('original_query', ''),
                rag_guidance=context.get('rag_guidance', ''),
                available_layers=available_layer_names,
                layer_intelligence=layer_intelligence  # ✅ NOW COMPATIBLE!
            )
            
            print(f"🎯 Next Action: {next_action.tool_name}")
            print(f"📝 Parameters: {next_action.parameters}")
            
            # Translate action
            try:
                executable_action = generator.translate_single_action(next_action, available_layers)
                action_sequence.append(executable_action)
                print(f"✅ Action translated successfully")
                
                # Simulate observation (in real system, executor would provide this)
                if next_action.tool_name == "finish_task":
                    observation = "Task completed successfully."
                    history.append(f"Action: {next_action.tool_name}")
                    history.append(f"Observation: {observation}")
                    break
                else:
                    # Simulate layer creation with enhanced tracking
                    conceptual_name = next_action.parameters.get('conceptual_name', f'{next_action.tool_name}_output')
                    real_layer_name = f"{conceptual_name}_{step + 1}"
                    available_layers[conceptual_name] = real_layer_name
                    available_layer_names.append(real_layer_name)
                    
                    # Simulate layer intelligence
                    layer_intelligence[real_layer_name] = {
                        'status': 'populated',
                        'feature_count': 50,
                        'columns': ['id', 'name', 'amenity', 'geometry'],
                        'geometry_types': ['Point'],
                        'crs': 'EPSG:4326'
                    }
                    
                    observation = f"Created layer '{real_layer_name}' with 50 features."
                    history.append(f"Action: {next_action.tool_name}")
                    history.append(f"Observation: {observation}")
                    
            except Exception as e:
                print(f"❌ Action translation failed: {e}")
                break
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n⏱️  Total Time: {total_time:.2f} seconds")
        print(f"🔄 Actions Taken: {len(action_sequence)}")
        
        # Store successful pattern
        if action_sequence:
            generator.store_successful_pattern(
                original_query=context.get('original_query', ''),
                action_sequence=action_sequence,
                execution_time=total_time
            )
    print("\n" + "=" * 70)
