#!/usr/bin/env python3
"""
MasterOrchestrator - The Think-Act-Observe loop controller for iterative GIS workflows.
Enhanced with intelligent failure tracking, recovery mechanisms, robust error handling,
comprehensive layer intelligence, dual-channel RAG guidance, and a validation quality gate.
"""

import json
import time
import traceback
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from functools import lru_cache
import geopandas as gpd
from dataclasses import dataclass
from enum import Enum

# Initialize logger
logger = logging.getLogger(__name__)

from src.core.planners.workflow_generator import WorkflowGenerator, ToolCallIntent
from src.core.executors.workflow_executor import WorkflowExecutor
from src.core.agents.data_scout import DataScout
from src.core.planners.query_parser import ParsedQuery, SpatialConstraint, SpatialRelationship
from src.core.planners.execution_planner import ExecutionPlanner, TaskQueue
from src.core.executors.task_executor import TaskExecutor
from src.gis.tools.definitions import TOOL_REGISTRY


class FailureType(Enum):
    """Types of failures that can occur during execution."""
    TOOL_ERROR = "tool_error"
    DATA_ERROR = "data_error"
    TIMEOUT = "timeout"
    INFINITE_LOOP = "infinite_loop"
    RESOURCE_ERROR = "resource_error"
    STRATEGY_ERROR = "strategy_error"
    PARAMETER_ERROR = "parameter_error"
    OSM_TAG_ERROR = "osm_tag_error"
    VALIDATION_ERROR = "validation_error"


@dataclass
class FailureRecord:
    """Record of a specific failure occurrence."""
    action_signature: str
    tool_name: str
    parameters: Dict[str, Any]
    failure_type: FailureType
    error_message: str
    timestamp: float
    loop_iteration: int
    recovery_attempted: bool = False
    recovery_successful: bool = False


@dataclass
class ExecutionMetrics:
    """Comprehensive metrics for execution analysis."""
    total_loops: int
    successful_actions: int
    failed_actions: int
    unique_failures: int
    consecutive_failures: int
    max_consecutive_failures: int
    recovery_attempts: int
    successful_recoveries: int
    osm_tag_recoveries: int
    validation_failures: int
    execution_time: float
    avg_loop_time: float


class MasterOrchestrator:
    """
    Enhanced central controller for iterative Think-Act-Observe GIS workflows.
    The central controller for the Think-Act-Observe loop. This class is responsible for
    managing state, validating plans, and orchestrating the agent components.
    """

    def __init__(self, max_loops: int = 15, max_consecutive_failures: int = 3,
                 enable_adaptive_recovery: bool = True, enable_performance_metrics: bool = True,
                 enable_osm_tag_recovery: bool = True, use_task_queue: bool = True):
        """Initialize the enhanced orchestrator with optional task queue architecture."""
        # Initialize core agents
        self.data_scout = DataScout()
        self.strategist = WorkflowGenerator(data_scout=self.data_scout)
        self.executor = WorkflowExecutor(enable_reasoning_log=True)
        
        # NEW: Task Queue Architecture components
        self.use_task_queue = use_task_queue
        if use_task_queue:
            self.task_planner = ExecutionPlanner()
            self.task_executor = TaskExecutor()
            logger.info("🎯 Task Queue Architecture ENABLED - loops eliminated by design")

        # State management
        self.conversation_history: List[str] = []
        self.data_layers: Dict[str, gpd.GeoDataFrame] = {}
        self.action_sequence: List[Dict[str, Any]] = []

        # Enhanced failure tracking
        self.failed_actions: Set[str] = set()
        self.failed_tools: Dict[str, int] = {}
        self.failure_history: List[FailureRecord] = []
        self.recovery_strategies: Dict[str, List[str]] = {}
        self.validation_failures: int = 0
        
        # OSM tag recovery system
        self.tag_recovery_suggestions: Dict[str, Dict[str, str]] = {}
        self.osm_recovery_attempts: Dict[str, int] = {}
        self.successful_osm_recoveries: int = 0

        # Performance tracking
        self.loop_times: List[float] = []
        self.action_success_rate: Dict[str, Tuple[int, int]] = {}

        # Configuration
        self.max_loops = max_loops
        self.max_consecutive_failures = max_consecutive_failures
        self.enable_adaptive_recovery = enable_adaptive_recovery
        self.enable_performance_metrics = enable_performance_metrics
        self.enable_osm_tag_recovery = enable_osm_tag_recovery
        
        # Conceptual map to maintain layer name translations
        self.conceptual_to_real_map: Dict[str, str] = {}

    @lru_cache(maxsize=128)
    def _get_cached_osm_tags(self, entity: str) -> Tuple[Tuple[str, str], ...]:
        """Cached version of OSM tag retrieval for performance optimization."""
        try:
            from src.core.knowledge.osm_tag_manager import tag_manager
            tag_options = tag_manager.get_cached_tags(entity)
            # Convert to tuples for caching compatibility
            return tuple(tuple(tags.items()) for tags in tag_options)
        except ImportError:
            print("⚠️ OSM tag manager not available, using fallback tags")
            return self._get_fallback_tags(entity)
        except Exception as e:
            print(f"⚠️ Error getting cached OSM tags: {e}")
            return self._get_fallback_tags(entity)

    def _get_fallback_tags(self, entity: str) -> Tuple[Tuple[str, str], ...]:
        """Fallback tag suggestions when OSM tag manager is unavailable."""
        fallback_tags = {
            'school': (
                (('amenity', 'school'),),
                (('building', 'school'),),
                (('landuse', 'education'),)
            ),
            'hospital': (
                (('amenity', 'hospital'),),
                (('healthcare', 'hospital'),),
                (('building', 'hospital'),)
            ),
            'park': (
                (('leisure', 'park'),),
                (('landuse', 'recreation_ground'),),
                (('leisure', 'garden'),)
            ),
            'restaurant': (
                (('amenity', 'restaurant'),),
                (('amenity', 'fast_food'),),
                (('shop', 'food'),)
            )
        }
        return fallback_tags.get(entity.lower(), ((('amenity', entity),),))

    def _handle_osm_tag_failure(self, failed_entity: str, original_tags: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Handle OSM tag failures with intelligent fallback."""
        
        print(f"🔄 Attempting OSM tag recovery for '{failed_entity}'")
        
        # Track recovery attempts
        self.osm_recovery_attempts[failed_entity] = self.osm_recovery_attempts.get(failed_entity, 0) + 1
        
        # Limit recovery attempts per entity to avoid infinite loops
        if self.osm_recovery_attempts[failed_entity] > 3:
            print(f"⚠️ Maximum recovery attempts reached for '{failed_entity}'")
            return None
        
        try:
            # Get alternative tags from the cached manager
            alternative_options = self._get_cached_osm_tags(failed_entity)
            
            # Convert back to dictionaries and find alternatives
            for tag_tuple in alternative_options:
                tags_dict = dict(tag_tuple)
                if tags_dict != original_tags:  # Don't retry the same tags
                    print(f"💡 Suggesting alternative tags for '{failed_entity}': {tags_dict}")
                    return tags_dict
            
            # If no alternatives found, try semantic alternatives
            return self._get_semantic_alternatives(failed_entity, original_tags)
            
        except Exception as e:
            print(f"⚠️ Error during OSM tag recovery: {e}")
            return None

    def _get_semantic_alternatives(self, entity: str, original_tags: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Get semantically similar alternatives when direct alternatives fail."""
        semantic_alternatives = {
            'school': [
                {'amenity': 'university'},
                {'amenity': 'college'},
                {'building': 'education'}
            ],
            'hospital': [
                {'amenity': 'clinic'},
                {'amenity': 'doctors'},
                {'healthcare': 'clinic'}
            ],
            'park': [
                {'leisure': 'playground'},
                {'natural': 'wood'},
                {'landuse': 'forest'}
            ],
            'restaurant': [
                {'amenity': 'cafe'},
                {'amenity': 'bar'},
                {'shop': 'convenience'}
            ]
        }
        
        alternatives = semantic_alternatives.get(entity.lower(), [])
        for alt_tags in alternatives:
            if alt_tags != original_tags:
                print(f"🔍 Trying semantic alternative for '{entity}': {alt_tags}")
                return alt_tags
        
        return None

    def _generate_dual_channel_rag_guidance(self, original_query: str) -> str:
        """
        Generates RAG guidance by consulting the knowledge base via the DataScout.
        FIXED: Now correctly accesses knowledge_base through data_scout.
        """
        print("🧠 Generating dual-channel RAG guidance...")
        rag_sections = []
        
        # Channel 1: Search for similar successful workflows
        try:
            # FIXED: Access knowledge_base through data_scout
            similar_workflows = self.data_scout.knowledge_base.search_similar_workflows(original_query, limit=1)
            if similar_workflows:
                workflow = similar_workflows[0]
                pattern = f"A similar query '{workflow['original_query']}' was solved successfully. Consider its pattern."
                rag_sections.append(f"**GUIDANCE FROM PAST SUCCESS:**\n- {pattern}")
                print("   ✅ Found a similar workflow pattern.")
        except Exception as e:
            print(f"   ⚠️ Error searching workflow patterns: {e}")

        # Channel 2: Search expert documentation
        try:
            # FIXED: Access knowledge_base through data_scout
            expert_docs = self.data_scout.knowledge_base.search_expert_docs(original_query, k=2)
            if expert_docs:
                docs_guidance = "\n".join([f"- {doc}" for doc in expert_docs])
                rag_sections.append(f"**GUIDANCE FROM EXPERT DOCUMENTATION:**\n{docs_guidance}")
                print(f"   ✅ Found {len(expert_docs)} relevant expert docs.")
        except Exception as e:
            print(f"   ⚠️ Error searching expert documentation: {e}")
        
        return "\n\n".join(rag_sections) if rag_sections else ""

    def _validate_action(self, action: ToolCallIntent) -> Tuple[bool, str]:
        """
        FINAL VERSION: Defense Layer 2 - The "Quality Gate" that validates the LLM's
        proposed action, with special handling for output-creating tools.
        """
        tool_name = action.tool_name
        params = action.parameters

        if tool_name == "finish_task":
            return True, ""

        if tool_name not in TOOL_REGISTRY:
            available_tools = list(TOOL_REGISTRY.keys())
            return False, f"Tool '{tool_name}' does not exist. Please choose from these available tools: {available_tools}"

        # --- THIS IS THE CRITICAL FIX ---
        
        # Define parameters that are for INPUT layers and need to exist beforehand.
        INPUT_LAYER_PARAMS = ["input_layer", "left_layer_name", "right_layer_name", "clip_layer_name"]
        
        # Tools like 'load_osm_data' create NEW layers. Their 'layer_name' parameter
        # should NOT be validated for existence beforehand.
        if tool_name == "load_osm_data":
            tags = params.get("tags")
            if not isinstance(tags, dict):
                return False, f"The 'tags' parameter for '{tool_name}' must be a dictionary (e.g., {{'amenity': 'school'}})."
            if not tags:
                return False, f"The 'tags' dictionary for '{tool_name}' cannot be empty."
            # We explicitly DO NOT validate the 'layer_name' for this tool.
            return True, "" # This plan is valid from the validator's perspective.

        # For all OTHER tools, validate their input layers.
        for param_key in INPUT_LAYER_PARAMS:
            if param_key in params:
                layer_name = params[param_key]
                # Check against both conceptual and real names
                if layer_name not in self.conceptual_to_real_map and layer_name not in self.data_layers:
                    available_layers = list(self.conceptual_to_real_map.keys())
                    return False, (f"The plan references an input layer '{layer_name}' that does not exist. "
                                  f"Please use one of the available layers: {available_layers}")

        return True, ""  # If all checks pass, the plan is valid

    def _build_layer_intelligence(self) -> Dict[str, Dict[str, Any]]:
        """
        Builds a comprehensive 'World Model' for the LLM using CONCEPTUAL names.
        This gives the orchestrator "eyes" to see the actual structure of data layers,
        but presents them with simplified, conceptual names for better understanding.
        """
        layer_intelligence = {}
        
        # Create a reverse map from real names back to conceptual names
        real_to_conceptual_map = {v: k for k, v in self.conceptual_to_real_map.items()}

        for real_name, gdf in self.data_layers.items():
            # Use the conceptual name if it exists, otherwise use the real name
            display_name = real_to_conceptual_map.get(real_name, real_name)
            
            if gdf.empty:
                layer_intelligence[display_name] = {
                    "status": "empty",
                    "columns": [],
                    "feature_count": 0,
                    "geometry_types": [],
                    "crs": str(gdf.crs) if gdf.crs else "unknown",
                    "summary": f"Empty layer with no features",
                    "real_name": real_name
                }
            else:
                # Get column info with smart truncation
                columns = list(gdf.columns)
                if len(columns) > 20:
                    display_columns = columns[:10] + [f"...({len(columns) - 20} more)..."] + columns[-10:]
                else:
                    display_columns = columns

                # Get sample values for key columns (excluding geometry)
                sample_values = {}
                data_columns = [col for col in gdf.columns[:5] if col != 'geometry']
                for col in data_columns:
                    try:
                        sample_vals = gdf[col].dropna().head(3).tolist()
                        sample_values[col] = [str(v) for v in sample_vals]
                    except:
                        sample_values[col] = ["<error_reading>"]

                # Get geometry information
                geometry_types = []
                try:
                    geometry_types = gdf.geom_type.unique().tolist()
                except:
                    geometry_types = ["<unknown>"]

                # Get bounds safely
                bounds = []
                try:
                    bounds = gdf.total_bounds.tolist()
                except:
                    bounds = [0, 0, 0, 0]

                layer_intelligence[display_name] = {
                    "status": "populated",
                    "columns": display_columns,
                    "feature_count": len(gdf),
                    "geometry_types": geometry_types,
                    "crs": str(gdf.crs),
                    "sample_values": sample_values,
                    "bounds": bounds,
                    "real_name": real_name,
                    "summary": f"{len(gdf)} features with {len(columns)} columns ({', '.join(geometry_types)})"
                }

        return layer_intelligence

    def run(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """
        Executes the workflow using either Task Queue or legacy loop architecture.
        """
        if self.use_task_queue:
            return self._run_with_task_queue(parsed_query)
        else:
            return self._run_with_legacy_loop(parsed_query)
    
    def _run_with_task_queue(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """
        NEW: Execute workflow using Task Queue architecture.
        
        Benefits:
        - Zero infinite loops (deterministic task queue)
        - Single-tool isolation (LLM can't hallucinate other tools)
        - Handles complex dependencies (interleaved LOAD/ANALYZE)
        """
        print("\n🎯 === TASK QUEUE ORCHESTRATOR ===" )
        print(f"Query: Find '{parsed_query.target}' in '{parsed_query.location}'")
        print("Architecture: Planner-Executor with Single-Tool Isolation\n")
        
        self._reset_execution_state()
        start_time = time.time()
        
        # Phase 1: Generate initial context
        print("📊 Phase 1: Data Discovery...")
        original_query = f"Find {parsed_query.target} in {parsed_query.location}"
        rag_guidance = self._generate_dual_channel_rag_guidance(original_query)
        
        try:
            context = self.strategist.get_initial_context(parsed_query, rag_guidance=rag_guidance)
            
            # === FIX 1: STOP THE CRASH (Defensive Null Check) ===
            if context is None:
                return self._handle_critical_failure(
                    f"Data Scout failed to validate location '{parsed_query.location}'. "
                    "The location might be invalid, too large (e.g., a continent), or the server is down."
                )
            
            # Check for explicit failure flag safely using .get()
            if not context.get("success", False):
                return self._handle_critical_failure(context.get('error', 'Unknown context error'))
                
        except Exception as e:
            # Catch unexpected crashes during initialization
            return self._handle_critical_failure(f"Context initialization crash: {e}")
        
        print(f"✅ Location validated: {context['data_report'].location.canonical_name}")
        
        # Phase 2: Generate Task Queue
        print("\n🗺️  Phase 2: Generating Execution Plan...")
        try:
            task_queue = self.task_planner.generate_task_queue(
                parsed_query=parsed_query,
                data_report=context['data_report']
            )
            print(f"✅ Task queue generated: {len(task_queue.tasks)} tasks")
            print(f"   Flow: {' → '.join([t.tool_name for t in task_queue.tasks])}\n")
        except Exception as e:
            return self._handle_critical_failure(f"Task planning failed: {e}")
        
        # Phase 3: Execute Task Queue
        print("⚙️  Phase 3: Executing Task Queue...")
        print("   (Each task sees ONLY its assigned tool - no hallucination possible)\n")
        
        try:
            result = self.task_executor.execute_task_queue(
                task_queue=task_queue,
                initial_layers=self.data_layers
            )
            
            execution_time = time.time() - start_time
            completed_tasks = result.get("completed_tasks", 0)
            total_tasks = result.get("total_tasks", len(task_queue.tasks))
            
            if not result["success"]:
                return {
                    "success": False,
                    "error": result.get("error", "Task execution failed"),
                    "loop_count": completed_tasks,  # Map tasks to loop count
                    "completed_tasks": completed_tasks,
                    "total_tasks": total_tasks,
                    "execution_time": execution_time,
                    "metrics": self._calculate_execution_metrics(completed_tasks, execution_time),
                    "failure_analysis": {"error": result.get("error")},
                    "performance_insights": {},
                    "osm_recovery_stats": {},
                    "validation_stats": {}
                }
            
            # Update orchestrator state
            self.data_layers = result["final_layers"]
            
            # Extract finish_task's parameters_hint and probe_results for UI consumption
            finish_task = next((t for t in task_queue.tasks if t.tool_name == "finish_task"), None)
            finish_task_parameters_hint = finish_task.parameters_hint if finish_task else {}
            probe_results_data = context.get('data_report', {}).probe_results if context else []
            
        except Exception as e:
            print(f"❌ Task queue execution failed: {e}")
            traceback.print_exc()
            return self._handle_critical_failure(f"Task execution failed: {e}")
        
        # Phase 4: Finalization
        print(f"\n✅ Execution completed in {execution_time:.2f}s")
        print(f"   Tasks completed: {result['completed_tasks']}/{result['total_tasks']}")
        
       # Store successful pattern
        if result["success"]:
            print("💾 Storing successful workflow pattern...")
            try:
                self.data_scout.knowledge_base.store_successful_workflow(
                    original_query=original_query,
                    workflow_plan=result.get("execution_log", []),
                    execution_time=execution_time
                )
            except Exception as e:
                print(f"⚠️  Could not store pattern: {e}")
            
            # === ENHANCED: Tag Learning Summary (Metadata-Based) ===
            print("\n📚 Tag Learning Summary:")
            if context and 'data_report' in context:
                for res in context['data_report'].probe_results:
                    if res.error:
                        continue
                    
                    # Extract metadata safely
                    source_type = res.metadata.get('tag_source', 'unknown') if res.metadata else 'unknown'
                    tag_dict = res.metadata.get('tag_dict', {}) if res.metadata else {}
                    
                    if source_type == 'verified_primary':
                        print(f"   ✅ {res.original_entity}: Primary tag {tag_dict} (Verified)")
                    elif source_type == 'learned_fallback':
                        option_used = res.metadata.get('option_used', '?') if res.metadata else '?'
                        total_options = res.metadata.get('total_options', '?') if res.metadata else '?'
                        print(f"   🔄 {res.original_entity}: Fallback tag {tag_dict} (Learned, option {option_used}/{total_options})")
                    elif res.metadata and res.metadata.get('used_smart_fallback'):
                        print(f"   🧠 {res.original_entity}: Smart Fallback {tag_dict} (AI Derived)")
                    else:
                        print(f"   ℹ️  {res.original_entity}: {tag_dict}")
            # ================================================================
        
        # === COMPATIBILITY FIX: Return full dictionary expected by test code ===
        return {
            "success": True,
            "final_layer_name": result["final_layer_name"],
            "final_result": self.data_layers.get(result["final_layer_name"]),
            "data_layers": self.data_layers,
            "execution_time": execution_time,
            
            # === KEY FIX: Map completed_tasks to loop_count for backward compatibility ===
            "loop_count": result['completed_tasks'],  # Test code expects this!
            
            "execution_log": result.get("execution_log", []),
            "reasoning_log": result.get("reasoning_log", []),
            
            # Conversation history for compatibility
            "conversation_history": [
                f"Task {i+1}: {entry['tool_name']} - {entry['observation'][:100]}"
                for i, entry in enumerate(result.get("execution_log", []))
            ],
            
            # Metrics expected by test code
            "metrics": self._calculate_execution_metrics(result['completed_tasks'], execution_time),
            "failure_analysis": self._generate_failure_analysis(),
            "performance_insights": self._generate_performance_insights(),
            
            # OSM recovery stats
            "osm_recovery_stats": {
                "total_attempts": sum(self.osm_recovery_attempts.values()),
                "successful_recoveries": self.successful_osm_recoveries,
                "entities_recovered": list(self.osm_recovery_attempts.keys())
            },
            
            # Validation stats
            "validation_stats": {
                "validation_failures": self.validation_failures,
                "validation_enabled": True
            },
            
            # === STREAMLIT UI FIX: Pass finish_task parameters_hint for count extraction ===
            "parameters_hint": finish_task_parameters_hint,
            
            # === STREAMLIT UI FIX: Pass probe_results for massive dataset count extraction ===
            "probe_results": probe_results_data,
            
            # Architecture metadata
            "architecture": "task_queue",
            "loop_prevention": "deterministic_plan",
            "tool_isolation": "single_tool_per_task"
        }
    
    def _run_with_legacy_loop(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """
        LEGACY: Executes the complete Think-Act-Observe loop with the validation quality gate.
        
        This is the original flat loop architecture - kept for backward compatibility.
        """
        print("🎭 === ENHANCED MASTER ORCHESTRATOR: Starting Think-Act-Observe Loop ===")
        print(f"🎯 Query: Find '{parsed_query.target}' in '{parsed_query.location}'")

        self._reset_execution_state()

        # Phase 1: Initialize context
        print("\n🔍 Phase 1: Establishing initial context...")
        original_query = f"Find {parsed_query.target} in {parsed_query.location}"
        
        # RAG generation is now the Orchestrator's responsibility
        rag_guidance = self._generate_dual_channel_rag_guidance(original_query)

        try:
            context = self.strategist.get_initial_context(parsed_query, rag_guidance=rag_guidance)
            if not context["success"]:
                return self._handle_critical_failure(context['error'])
        except Exception as e:
            return self._handle_critical_failure(f"Context initialization failed: {e}")

        self.conversation_history = [context["initial_observation"]]

        print("✅ Initial context established successfully")
        print(f"📊 {context['initial_observation']}")

        # Phase 2: Enhanced Think-Act-Observe Loop with Intelligence and Validation
        print("\n🔄 Phase 2: Starting intelligent Think-Act-Observe loop...")

        start_time = time.time()
        loop_count = 0
        finished = False
        consecutive_failures = 0
        next_action = None

        while not finished and loop_count < self.max_loops:
            loop_start = time.time()
            loop_count += 1
            print(f"\n--- Loop {loop_count} ---")

            # Emergency stop for too many consecutive failures
            if consecutive_failures >= self.max_consecutive_failures:
                print(f"⚠️  Emergency stop: {consecutive_failures} consecutive failures")
                self.conversation_history.append("Observation: Emergency stop due to repeated failures")
                break

            # 🧠 THINK: Get next action with comprehensive layer intelligence
            print("🧠 THINK: Deciding next action with failure awareness and layer intelligence...")

            # BUILD THE WORLD MODEL - Give the agent "eyes"
            layer_intelligence = self._build_layer_intelligence()
            current_layer_names = list(self.data_layers.keys())

            try:
                # Check for tag recovery suggestions
                enhanced_rag_guidance = rag_guidance
                if self.tag_recovery_suggestions:
                    recovery_info = "\n**OSM TAG RECOVERY SUGGESTIONS:**\n"
                    for entity, alt_tags in self.tag_recovery_suggestions.items():
                        recovery_info += f"- For '{entity}': try tags {alt_tags}\n"
                    enhanced_rag_guidance += "\n" + recovery_info

                # PASS THE WORLD MODEL AND ORIGINAL QUERY INTENT TO THE STRATEGIST
                next_action = self.strategist.get_next_action(
                    history=self.conversation_history,
                    original_query=original_query,
                    rag_guidance=enhanced_rag_guidance,
                    available_layers=current_layer_names,
                    layer_intelligence=layer_intelligence
                )

                # Validate the action
                if not isinstance(next_action, ToolCallIntent):
                    raise ValueError(f"Invalid action type returned: {type(next_action)}")

                if not next_action.tool_name:
                    raise ValueError("Empty tool name returned from strategist")

                print(f"💡 Strategist proposed: {next_action.tool_name}({next_action.parameters})")

                # Display layer intelligence for debugging
                if layer_intelligence:
                    print("👁️  Current layer state:")
                    for name, info in layer_intelligence.items():
                        print(f"   - {name}: {info['summary']}")

            except Exception as e:
                # Comprehensive error handling for the Think phase
                error_msg = f"The 'Think' phase failed: {str(e)}"
                print(f"❌ CRITICAL ERROR: {error_msg}")

                # Log the full traceback for debugging
                print("📍 Full error traceback:")
                traceback.print_exc()

                # Record this as a strategy failure
                self._record_failure(
                    "think_phase", "strategic_planning", {},
                    FailureType.STRATEGY_ERROR, error_msg, loop_count
                )

                consecutive_failures += 1
                self.conversation_history.append(f"Observation: Think phase failed - {error_msg}")

                # Try to recover or exit
                if consecutive_failures >= self.max_consecutive_failures:
                    print("💀 Cannot recover from Think phase failures. Terminating execution.")
                    break
                else:
                    print(f"🔄 Attempting to continue (attempt {consecutive_failures}/{self.max_consecutive_failures})")
                    continue

            # Handle completion first
            if next_action.tool_name == "finish_task":
                final_layer = next_action.parameters.get("final_layer_name", "unknown")
                self.conversation_history.append("Action: finish_task")
                self.conversation_history.append(f"Observation: Task completed. Final result: '{final_layer}'")
                finished = True
                break

            # 🛡️ ENHANCEMENT: DEFENSE LAYER 2 - Validate the proposed action BEFORE execution
            is_valid, validation_error = self._validate_action(next_action)
            if not is_valid:
                print(f"🛡️ Plan REJECTED by validator: {validation_error}")
                # Feed the specific error back into the loop for the LLM to learn from
                observation = f"Failure: The proposed plan was invalid. Reason: {validation_error}"
                
                # Record validation failure
                self._record_failure(
                    next_action.tool_name, "validation", next_action.parameters,
                    FailureType.VALIDATION_ERROR, validation_error, loop_count
                )
                self.validation_failures += 1
                consecutive_failures += 1
                
                self.conversation_history.append(f"Action: {next_action.tool_name}({json.dumps(next_action.parameters)})")
                self.conversation_history.append(f"Observation: {observation}")
                
                continue  # Skip the ACT step and go directly to the next THINK loop

            print("✅ Plan accepted by validator. Proceeding to execution.")

            # Apply tag recovery suggestions if available
            if (self.enable_osm_tag_recovery and 
                next_action.tool_name == "load_osm_data" and
                "tags" in next_action.parameters):
                
                entity_hint = self._extract_entity_from_action(next_action)
                if entity_hint in self.tag_recovery_suggestions:
                    suggested_tags = self.tag_recovery_suggestions[entity_hint]
                    print(f"🔄 Applying OSM tag recovery for '{entity_hint}': {suggested_tags}")
                    next_action.parameters["tags"] = suggested_tags
                    # Clear the suggestion after using it
                    del self.tag_recovery_suggestions[entity_hint]

            # Intelligent failure avoidance
            action_signature = self._get_action_signature(next_action)
            tool_name = next_action.tool_name

            # ✅ NEW: STRONG LOOP DETECTION - Prevents repeating successful actions
            if not hasattr(self, 'recent_action_history'):
                self.recent_action_history = []
            
            # Check if this exact action was done in the last 3 loops
            if action_signature in self.recent_action_history[-3:]:
                print(f"🛑 LOOP DETECTED: Repeated '{tool_name}' from recent history")
                error_msg = (
                    f"SYSTEM ALERT: Action '{tool_name}' was already completed successfully. "
                    "Data is loaded. Move to NEXT PHASE: analysis (buffer/spatial_join/intersect) or 'finish_task'."
                )
                self.conversation_history.append(f"Observation: {error_msg}")
                consecutive_failures += 1
                print("  💬 Forcing LLM to pivot with explicit state feedback")
                continue

            # Skip previously failed exact actions
            if action_signature in self.failed_actions:
                print(f"⚠️  Skipping previously failed action: {tool_name}")
                self.conversation_history.append("Observation: Skipping previously failed action to avoid infinite loop")
                consecutive_failures += 1

                # Try recovery strategy if available
                if self.enable_adaptive_recovery:
                    recovery_action = self._attempt_recovery(tool_name, next_action)
                    if recovery_action:
                        next_action = recovery_action
                        action_signature = self._get_action_signature(next_action)
                        print(f"🔄 Applying recovery strategy: {recovery_action.tool_name}")
                    else:
                        continue
                else:
                    continue

            # Warn about problematic tools
            if tool_name in self.failed_tools and self.failed_tools[tool_name] >= 2:
                print(f"⚠️  Warning: Tool '{tool_name}' has failed {self.failed_tools[tool_name]} times")

            # 🔍 Translate conceptual names to real names
            print("🔍 Translating conceptual names to real names...")
            
            # Translate conceptual names in parameters back to real names
            for param_key, param_value in next_action.parameters.items():
                if isinstance(param_value, str) and param_value in self.conceptual_to_real_map:
                    old_value = param_value
                    next_action.parameters[param_key] = self.conceptual_to_real_map[param_value]
                    print(f"   🔄 Translated '{old_value}' -> '{next_action.parameters[param_key]}'")
            
            print(f"🔧 Translated action: {tool_name}({next_action.parameters})")
            
            # ============================================================
            # 🛡️ DEFENSE LAYER: AGGRESSIVE PARAMETER OVERRIDE
            # ============================================================
            if tool_name == "load_osm_data":
                print(f"🔍 [Orchestrator] Intercepting 'load_osm_data' for aggressive sanitation...")
                
                # 1. FIX TAGS - Create NEW dict to break reference chains
                original_params = next_action.parameters.copy()
                corrected_params = self.data_scout.auto_correct_suspicious_tags(original_params)
                
                # FORCE update the action object with new dict
                next_action.parameters = corrected_params
                print(f"   ✅ Tags corrected: {corrected_params.get('tags')}")
                
                # 2. FIX LOCATION - Strip to simple form for better geocoding
                if "area_name" in next_action.parameters:
                    raw_loc = next_action.parameters["area_name"]
                    # Take first part before comma: "Berlin, Deutschland" → "Berlin"
                    clean_loc = raw_loc.split(',')[0].strip()
                    
                    if clean_loc != raw_loc:
                        print(f"   🧹 Sanitizing location: '{raw_loc}' → '{clean_loc}'")
                        next_action.parameters["area_name"] = clean_loc
                
                # 3. REMOVE CONFLICTING PARAMETERS - Clean sweep
                keys_to_remove = ["conceptual_name", "layer_name", "confidence", "reasoning"]
                for key in keys_to_remove:
                    if key in next_action.parameters:
                        removed = next_action.parameters.pop(key)
                        print(f"   🗑️ Removed conflicting param: '{key}'")
                
                print(f"✅ [Orchestrator] FINAL PARAMS for Executor: {next_action.parameters}")
            # ============================================================
            
            # 🎬 ACT: Execute the now fully translated action (Defense Layer 3)
            print("🎬 ACT: Executing action with failure monitoring...")
            try:
                observation, updated_layers = self.executor.execute_single_step(
                    tool_call=next_action,
                    current_data_layers=self.data_layers
                )

                # Validate execution results
                if updated_layers is None:
                    updated_layers = self.data_layers.copy()

            except Exception as e:
                error_msg = f"Execution error: {str(e)}"
                print(f"❌ Execution failed: {error_msg}")
                observation = f"Failure: {error_msg}"
                updated_layers = self.data_layers.copy()

                # Log execution error details
                print("📍 Execution error traceback:")
                traceback.print_exc()

            # Handle action success with recent history tracking
            if observation.startswith("Failure"):
                self._handle_action_failure(next_action, observation, loop_count)
                consecutive_failures += 1
                print(f"❌ Action failed (consecutive failures: {consecutive_failures})")
            else:
                self._handle_action_success(next_action, observation)
                consecutive_failures = 0  # Reset on success
                
                # ✅ NEW: Track successful actions in recent history
                if not hasattr(self, 'recent_action_history'):
                    self.recent_action_history = []
                self.recent_action_history.append(action_signature)
                if len(self.recent_action_history) > 5:
                    self.recent_action_history.pop(0)
                
                print("✅ Action succeeded")

            # 👀 OBSERVE: Update state and history
            print("👀 OBSERVE: Processing results and updating state...")
            
            # The Universal Map Updater
            if observation.startswith("Success") and next_action.tool_name != "finish_task":
                # Robustly find the conceptual name for the NEW layer
                params = next_action.parameters
                conceptual_name = (params.get('conceptual_name') or 
                                   params.get('output_layer_name') or 
                                   params.get('layer_name'))
                
                # Find the newest layer that was just created by the executor
                newest_layers = [name for name in updated_layers if name not in self.data_layers]
                
                if conceptual_name and newest_layers:
                    self.conceptual_to_real_map[conceptual_name] = newest_layers[0]
                    print(f"🗺️  Mapped conceptual name '{conceptual_name}' to real name '{newest_layers[0]}'")
            
            self.conversation_history.append(f"Action: {next_action.tool_name}({json.dumps(next_action.parameters)})")
            self.conversation_history.append(f"Observation: {observation}")
            self.data_layers = updated_layers

            # Performance tracking
            loop_time = time.time() - loop_start
            self.loop_times.append(loop_time)

            print(f"📈 Current layers: {list(self.data_layers.keys())}")
            print(f"⏱️  Loop time: {loop_time:.2f}s")

        # Phase 3: Finalization and comprehensive analysis
        end_time = time.time()
        execution_time = end_time - start_time

        print(f"\n🎉 Execution completed in {execution_time:.2f}s after {loop_count} loops")
        print(f"📊 Final status: {'SUCCESS' if finished else 'STOPPED'}")

        # Store successful pattern for future learning
        if finished and self.action_sequence:
            print("💾 Storing successful workflow pattern for future learning...")
            try:
                self.data_scout.knowledge_base.store_successful_workflow(
                    original_query=original_query,
                    workflow_plan=self.action_sequence,
                    execution_time=execution_time # Pass the execution time
                )
            except Exception as e:
                print(f"⚠️  Warning: Could not store successful pattern: {e}")

        # Calculate comprehensive metrics
        metrics = self._calculate_execution_metrics(loop_count, execution_time)

        # Prepare enhanced result
        final_layer_name = None
        final_result = None

        if self.data_layers:
            # Attempt to get the layer name from the last "finish_task" action
            if next_action and next_action.tool_name == "finish_task":
                final_layer_name = next_action.parameters.get("final_layer_name")
            
            # Fallback to the last known layer if not specified or available
            if final_layer_name and final_layer_name in self.data_layers:
                 final_result = self.data_layers[final_layer_name]
            else:
                 final_layer_name = list(self.data_layers.keys())[-1]
                 final_result = self.data_layers[final_layer_name]

        return {
            "success": finished,
            "final_layer_name": final_layer_name,
            "final_result": final_result,
            "conversation_history": self.conversation_history,
            "action_sequence": self.action_sequence,
            "execution_time": execution_time,
            "loop_count": loop_count,
            "data_layers": self.data_layers,
            "reasoning_log": self.executor.get_reasoning_log() if hasattr(self.executor, 'get_reasoning_log') else [],
            "metrics": metrics,
            "failure_analysis": self._generate_failure_analysis(),
            "performance_insights": self._generate_performance_insights(),
            "osm_recovery_stats": {
                "total_attempts": sum(self.osm_recovery_attempts.values()),
                "successful_recoveries": self.successful_osm_recoveries,
                "entities_recovered": list(self.osm_recovery_attempts.keys())
            },
            "validation_stats": {
                "validation_failures": self.validation_failures,
                "validation_enabled": True
            }
        }

    def _extract_entity_from_action(self, action: ToolCallIntent) -> Optional[str]:
        """Extract entity hint from action parameters for OSM tag recovery."""
        params = action.parameters
        
        # Try common parameter names that might contain entity information
        entity_hints = [
            params.get('entity_type'),
            params.get('feature_type'),
            params.get('target'),
            params.get('layer_name')
        ]
        
        # Also check tag values
        tags = params.get('tags', {})
        for value in tags.values():
            if isinstance(value, str):
                entity_hints.append(value)
        
        # Return the first non-None hint
        for hint in entity_hints:
            if hint:
                return str(hint).lower()
        
        return None

    def _reset_execution_state(self):
        """Reset all execution state for a new run."""
        self.conversation_history.clear()
        self.data_layers.clear()
        self.action_sequence.clear()
        self.failed_actions.clear()
        self.failed_tools.clear()
        self.failure_history.clear()
        self.loop_times.clear()
        self.tag_recovery_suggestions.clear()
        self.osm_recovery_attempts.clear()
        self.successful_osm_recoveries = 0
        self.validation_failures = 0

    def _get_action_signature(self, action: ToolCallIntent) -> str:
        """Generate a unique signature for an action."""
        return f"{action.tool_name}:{json.dumps(action.parameters, sort_keys=True)}"

    def _record_failure(self, tool_name: str, action_type: str, parameters: Dict[str, Any],
                       failure_type: FailureType, error_message: str, loop_iteration: int,
                       recovery_attempted: bool = False, recovery_successful: bool = False):
        """Record a failure for analysis and learning."""
        action_signature = f"{tool_name}:{json.dumps(parameters, sort_keys=True)}"

        failure_record = FailureRecord(
            action_signature=action_signature,
            tool_name=tool_name,
            parameters=parameters,
            failure_type=failure_type,
            error_message=error_message,
            timestamp=time.time(),
            loop_iteration=loop_iteration,
            recovery_attempted=recovery_attempted,
            recovery_successful=recovery_successful
        )

        self.failure_history.append(failure_record)
        self.failed_actions.add(action_signature)
        self.failed_tools[tool_name] = self.failed_tools.get(tool_name, 0) + 1

    def _handle_action_failure(self, action: ToolCallIntent, observation: str, loop_count: int):
        """Enhanced action failure handling with OSM tag recovery."""
        
        # Determine failure type from observation
        failure_type = FailureType.TOOL_ERROR
        recovery_attempted = False
        
        if "timeout" in observation.lower():
            failure_type = FailureType.TIMEOUT
        elif "data" in observation.lower() or "no data" in observation.lower():
            failure_type = FailureType.DATA_ERROR
        elif "resource" in observation.lower() or "memory" in observation.lower():
            failure_type = FailureType.RESOURCE_ERROR
        elif "parameter" in observation.lower() or "invalid" in observation.lower():
            failure_type = FailureType.PARAMETER_ERROR

        # Check if this is an OSM tag failure and attempt recovery
        if (self.enable_osm_tag_recovery and 
            action.tool_name == "load_osm_data" and 
            ("No matching features" in observation or "no data found" in observation.lower())):
            
            failure_type = FailureType.OSM_TAG_ERROR
            entity_hint = self._extract_entity_from_action(action)
            original_tags = action.parameters.get("tags", {})
            
            if entity_hint:
                alternative_tags = self._handle_osm_tag_failure(entity_hint, original_tags)
                if alternative_tags:
                    # Store the alternative for the next attempt
                    self.tag_recovery_suggestions[entity_hint] = alternative_tags
                    recovery_attempted = True
                    print(f"🔄 OSM tag recovery prepared for '{entity_hint}'")

        # Record the failure with recovery information
        self._record_failure(
            action.tool_name, "execution", action.parameters,
            failure_type, observation, loop_count, recovery_attempted
        )

        # Update success rate tracking
        if action.tool_name in self.action_success_rate:
            success, total = self.action_success_rate[action.tool_name]
            self.action_success_rate[action.tool_name] = (success, total + 1)
        else:
            self.action_success_rate[action.tool_name] = (0, 1)

    def _handle_action_success(self, action: ToolCallIntent, observation: str):
        """Handle and record an action success."""
        # Check if this was a successful OSM tag recovery
        if (action.tool_name == "load_osm_data" and 
            observation.startswith("Success") and
            self._extract_entity_from_action(action) in self.osm_recovery_attempts):
            
            self.successful_osm_recoveries += 1
            entity = self._extract_entity_from_action(action)
            print(f"🎉 OSM tag recovery successful for '{entity}'!")

        # Store the ORIGINAL, untranslated action for RAG learning
        if observation.startswith("Success"):
            self.action_sequence.append({
                "tool_name": action.tool_name,
                "parameters": action.parameters
            })

        # Update success rate tracking
        if action.tool_name in self.action_success_rate:
            success, total = self.action_success_rate[action.tool_name]
            self.action_success_rate[action.tool_name] = (success + 1, total + 1)
        else:
            self.action_success_rate[action.tool_name] = (1, 1)

    def _extract_entity_from_action(self, action: ToolCallIntent) -> Optional[str]:
        """
        Helper method to extract the entity name from OSM tags.
        Intelligently identifies the primary entity being searched for based on OSM tag patterns.
        
        Args:
            action: The ToolCallIntent object containing OSM query parameters
            
        Returns:
            str | None: The identified entity name, or None if no clear entity can be determined
        """
        tags = action.parameters.get("tags", {})
        if not isinstance(tags, dict):
            return None
            
        # Prioritized OSM tag keys that typically indicate entity type
        priority_tags = [
            "amenity", "leisure", "building", "shop", "tourism",
            "historic", "natural", "landuse", "highway"
        ]
        
        # Try to find the first matching priority tag
        for tag_key in priority_tags:
            if tag_key in tags:
                return tags[tag_key]
                
        # Fallback to first available tag if no priority tags found
        if tags:
            return next(iter(tags.values()))
            
        return None

    def _attempt_recovery(self, failed_tool: str, original_action: ToolCallIntent) -> Optional[ToolCallIntent]:
        """
        ENHANCED: A context-aware recovery system with smart tag recovery for OSM data loading failures.
        Attempts to recover from a failed action using intelligent strategies and knowledge-based tag correction.
        """
        if not self.enable_adaptive_recovery:
            return None

        # --- NEW: INTELLIGENT TAG RECOVERY FOR load_osm_data ---
        if failed_tool == "load_osm_data" and self.enable_osm_tag_recovery:
            entity = self._extract_entity_from_action(original_action)
            if entity:
                try:
                    # FIX: Call the method on data_scout, not knowledge_base
                    best_tags = self.data_scout.get_primary_tags_for_entity(entity)
                    original_tags = original_action.parameters.get("tags", {})
                    
                    if best_tags and best_tags != original_tags:
                        # Track recovery attempt
                        self.osm_recovery_attempts[entity] = self.osm_recovery_attempts.get(entity, 0) + 1
                        print(f"💡 Recovery: DataScout suggests better tags for '{entity}'. Retrying with {best_tags}.")
                        
                        new_params = original_action.parameters.copy()
                        new_params["tags"] = best_tags
                        return ToolCallIntent(tool_name="load_osm_data", parameters=new_params)
                except Exception as e:
                    print(f"⚠️ Could not get recovery tags: {e}")

        # --- ENHANCED FALLBACK: Structured Recovery Logic ---
        # FIX: Removed all references to 'search_osm_data' - it's not in TOOL_REGISTRY
        # Only include tools that actually exist in the registry
        recovery_strategies = {
            "load_osm_data": ["buffer", "spatial_join"],
            "buffer": ["spatial_join", "clip"],
            "spatial_join": ["intersect", "buffer"]
        }

        if failed_tool not in recovery_strategies:
            return None

        # Try alternative tools with improved tracking
        for alternative_tool in recovery_strategies[failed_tool]:
            if alternative_tool not in self.failed_tools or self.failed_tools[alternative_tool] < 2:
                try:
                    return ToolCallIntent(
                        tool_name=alternative_tool,
                        parameters=self._adapt_parameters_for_recovery(
                            original_action.parameters, alternative_tool
                        )
                    )
                except Exception as e:
                    print(f"⚠️  Recovery strategy failed for {alternative_tool}: {e}")
                    continue

        return None

    def _adapt_parameters_for_recovery(self, original_params: Dict[str, Any],
                                     new_tool: str) -> Dict[str, Any]:
        """Adapt parameters for a recovery action."""
        # Basic parameter adaptation logic
        adapted_params = original_params.copy()

        # Tool-specific adaptations
        if new_tool == "buffer" and "distance" not in adapted_params:
            adapted_params["distance"] = 1000  # Default buffer in meters
        elif new_tool == "load_osm_data":
            # Ensure OSM-specific parameters are preserved
            if "tags" in adapted_params:
                adapted_params["tags"] = adapted_params["tags"]

        return adapted_params

    def _calculate_execution_metrics(self, loop_count: int, execution_time: float) -> ExecutionMetrics:
        """Calculate comprehensive execution metrics including OSM recovery and validation stats."""
        successful_actions = sum(1 for h in self.conversation_history
                               if h.startswith("Observation: Success"))
        failed_actions = len(self.failure_history)
        recovery_attempts = sum(1 for f in self.failure_history if f.recovery_attempted)

        return ExecutionMetrics(
            total_loops=loop_count,
            successful_actions=successful_actions,
            failed_actions=failed_actions,
            unique_failures=len(self.failed_actions),
            consecutive_failures=self._get_max_consecutive_failures(),
            max_consecutive_failures=self.max_consecutive_failures,
            recovery_attempts=recovery_attempts,
            successful_recoveries=sum(1 for f in self.failure_history if f.recovery_successful),
            osm_tag_recoveries=self.successful_osm_recoveries,
            validation_failures=self.validation_failures,
            execution_time=execution_time,
            avg_loop_time=sum(self.loop_times) / max(len(self.loop_times), 1)
        )

    def _get_max_consecutive_failures(self) -> int:
        """Calculate the maximum consecutive failures that occurred."""
        if not self.conversation_history:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for entry in self.conversation_history:
            if entry.startswith("Observation: Failure") or entry.startswith("Observation: Think phase failed"):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            elif entry.startswith("Observation: Success"):
                current_consecutive = 0

        return max_consecutive

    def _generate_failure_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive failure analysis including OSM tag and validation failures."""
        if not self.failure_history:
            return {"total_failures": 0, "analysis": "No failures recorded"}

        failure_by_tool = {}
        failure_by_type = {}
        osm_tag_failures = 0
        validation_failures = 0

        for failure in self.failure_history:
            failure_by_tool[failure.tool_name] = failure_by_tool.get(failure.tool_name, 0) + 1
            failure_by_type[failure.failure_type.value] = failure_by_type.get(failure.failure_type.value, 0) + 1
            
            if failure.failure_type == FailureType.OSM_TAG_ERROR:
                osm_tag_failures += 1
            elif failure.failure_type == FailureType.VALIDATION_ERROR:
                validation_failures += 1

        return {
            "total_failures": len(self.failure_history),
            "unique_failed_actions": len(self.failed_actions),
            "failure_by_tool": failure_by_tool,
            "failure_by_type": failure_by_type,
            "osm_tag_failures": osm_tag_failures,
            "validation_failures": validation_failures,
            "osm_recovery_success_rate": (
                self.successful_osm_recoveries / max(sum(self.osm_recovery_attempts.values()), 1)
            ) if self.osm_recovery_attempts else 0,
            "most_problematic_tool": max(failure_by_tool.items(), key=lambda x: x[1])[0] if failure_by_tool else None,
            "average_failures_per_loop": len(self.failure_history) / max(len(self.loop_times), 1)
        }

    def _generate_performance_insights(self) -> Dict[str, Any]:
        """Generate performance insights and recommendations including OSM tag recovery and validation."""
        insights = {
            "total_execution_time": sum(self.loop_times),
            "average_loop_time": sum(self.loop_times) / max(len(self.loop_times), 1),
            "slowest_loop_time": max(self.loop_times) if self.loop_times else 0,
            "fastest_loop_time": min(self.loop_times) if self.loop_times else 0,
            "tool_success_rates": {},
            "osm_tag_recovery_enabled": self.enable_osm_tag_recovery,
            "validation_enabled": True,
            "osm_recovery_stats": {
                "total_attempts": sum(self.osm_recovery_attempts.values()),
                "successful_recoveries": self.successful_osm_recoveries,
                "success_rate": (
                    self.successful_osm_recoveries / max(sum(self.osm_recovery_attempts.values()), 1)
                ) if self.osm_recovery_attempts else 0
            },
            "validation_stats": {
                "total_validation_failures": self.validation_failures
            }
        }

        # Calculate tool success rates
        for tool, (successes, total) in self.action_success_rate.items():
            insights["tool_success_rates"][tool] = {
                "success_rate": successes / max(total, 1),
                "total_attempts": total,
                "successes": successes
            }

        # Generate recommendations
        recommendations = []
        if insights["average_loop_time"] > 5.0:
            recommendations.append("Consider optimizing slow operations or increasing timeout limits")

        low_success_tools = [tool for tool, rates in insights["tool_success_rates"].items()
                           if rates["success_rate"] < 0.5 and rates["total_attempts"] > 2]
        if low_success_tools:
            recommendations.append(f"Tools with low success rates need attention: {low_success_tools}")

        if self.enable_osm_tag_recovery and insights["osm_recovery_stats"]["success_rate"] < 0.5:
            recommendations.append("OSM tag recovery has low success rate - consider expanding tag alternatives")

        if self.validation_failures > 0:
            recommendations.append(f"Validation caught {self.validation_failures} invalid plans - consider improving LLM prompting")

        insights["recommendations"] = recommendations
        return insights

    def _handle_critical_failure(self, error_message: str) -> Dict[str, Any]:
        """Handle critical failures that prevent execution."""
        return {
            "success": False,
            "error": f"Critical failure: {error_message}",
            "conversation_history": self.conversation_history,
            "execution_time": 0,
            "metrics": ExecutionMetrics(0, 0, 1, 1, 1, self.max_consecutive_failures, 0, 0, 0, 0, 0, 0),
            "failure_analysis": {"critical_failure": error_message},
            "performance_insights": {"status": "execution_aborted"},
            "osm_recovery_stats": {
                "total_attempts": 0,
                "successful_recoveries": 0,
                "entities_recovered": []
            },
            "validation_stats": {
                "validation_failures": 0,
                "validation_enabled": True
            }
        }

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary with enhanced metrics."""
        return {
            "total_loops": len([h for h in self.conversation_history if h.startswith("Action:")]),
            "final_layers": list(self.data_layers.keys()),
            "conversation_length": len(self.conversation_history),
            "actions_taken": len(self.action_sequence),
            "executor_summary": self.executor.get_execution_summary() if hasattr(self.executor, 'get_execution_summary') else {},
            "failure_summary": self._generate_failure_analysis(),
            "performance_summary": self._generate_performance_insights(),
            "intelligence_features": {
                "adaptive_recovery": self.enable_adaptive_recovery,
                "failure_tracking": True,
                "performance_monitoring": self.enable_performance_metrics,
                "layer_intelligence": True,
                "dual_channel_rag": True,
                "osm_tag_recovery": self.enable_osm_tag_recovery,
                "validation_layer": True,
                "max_consecutive_failures": self.max_consecutive_failures
            },
            "osm_recovery_summary": {
                "enabled": self.enable_osm_tag_recovery,
                "total_attempts": sum(self.osm_recovery_attempts.values()),
                "successful_recoveries": self.successful_osm_recoveries,
                "entities_attempted": list(self.osm_recovery_attempts.keys())
            },
            "validation_summary": {
                "enabled": True,
                "validation_failures": self.validation_failures
            }
        }


# Enhanced integration test
if __name__ == '__main__':
    print("🧪 === ENHANCED MASTER ORCHESTRATOR WITH VALIDATION TEST ===")
    
    # Test query: Find schools near parks in Berlin
    test_query = ParsedQuery(
        target='school',
        location='Berlin, Germany',
        constraints=[
            SpatialConstraint(feature_type='park', relationship=SpatialRelationship.NEAR, distance_meters=500)
        ],
        summary_required=True
    )

    # Test with all enhanced features including validation
    orchestrator = MasterOrchestrator(
        max_loops=10,
        max_consecutive_failures=3,
        enable_adaptive_recovery=True,
        enable_performance_metrics=True,
        enable_osm_tag_recovery=True
    )

    try:
        result = orchestrator.run(test_query)

        print("\n📊 === ENHANCED RESULTS ANALYSIS WITH VALIDATION ===")
        print(f"Success: {result['success']}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        print(f"Total Loops: {result['loop_count']}")
        
        if result.get('final_result') is not None:
            print(f"Final Layer: '{result.get('final_layer_name')}'")
            print(f"Final Result: {len(result['final_result'])} features found.")
        else:
            print("Final Result: No result layer was produced.")

        # Enhanced metrics display
        metrics = result['metrics']
        print(f"\n📈 Performance Metrics:")
        print(f"  - Successful Actions: {metrics.successful_actions}")
        print(f"  - Failed Actions: {metrics.failed_actions}")
        print(f"  - Validation Failures: {metrics.validation_failures}")
        print(f"  - OSM Tag Recoveries: {metrics.osm_tag_recoveries}")
        print(f"  - Average Loop Time: {metrics.avg_loop_time:.2f}s")

        # Validation Statistics
        validation_stats = result['validation_stats']
        print(f"\n🛡️ Validation Statistics:")
        print(f"  - Validation Failures Prevented: {validation_stats['validation_failures']}")
        print(f"  - Validation Enabled: {validation_stats['validation_enabled']}")

        # OSM Recovery Statistics
        osm_stats = result['osm_recovery_stats']
        print(f"\n🔄 OSM Tag Recovery Statistics:")
        print(f"  - Total Recovery Attempts: {osm_stats['total_attempts']}")
        print(f"  - Successful Recoveries: {osm_stats['successful_recoveries']}")
        print(f"  - Entities Recovered: {osm_stats['entities_recovered']}")

        # Failure analysis
        failure_analysis = result['failure_analysis']
        print(f"\n❌ Failure Analysis:")
        print(f"  - Total Failures: {failure_analysis.get('total_failures', 0)}")
        print(f"  - OSM Tag Failures: {failure_analysis.get('osm_tag_failures', 0)}")
        print(f"  - Validation Failures: {failure_analysis.get('validation_failures', 0)}")
        if failure_analysis.get('most_problematic_tool'):
            print(f"  - Most Problematic Tool: {failure_analysis['most_problematic_tool']}")
        
        # Performance insights
        insights = result['performance_insights']
        print(f"\n💡 Performance Insights:")
        for tool, rates in insights.get('tool_success_rates', {}).items():
            rate_percent = rates['success_rate'] * 100
            print(f"  - {tool}: {rate_percent:.1f}% success rate ({rates['successes']}/{rates['total_attempts']})")
        
        osm_recovery_rate = insights['osm_recovery_stats']['success_rate'] * 100
        print(f"  - OSM Recovery Success Rate: {osm_recovery_rate:.1f}%")
        
        if insights.get('recommendations'):
            print(f"\n🎯 Recommendations:")
            for rec in insights['recommendations']:
                print(f"  - {rec}")

        print("\n📝 Conversation History:")
        for i, entry in enumerate(result['conversation_history'], 1):
            print(f"{i}. {entry}")

    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        traceback.print_exc()

    print("\n✅ Enhanced MasterOrchestrator with validation layer test completed!")
