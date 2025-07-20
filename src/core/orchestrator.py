#!/usr/bin/env python3
"""
MasterOrchestrator - The Think-Act-Observe loop controller for iterative GIS workflows.
Enhanced with intelligent failure tracking, recovery mechanisms, robust error handling,
and comprehensive layer intelligence for better LLM decision making.
"""

import json
import time
import traceback
from typing import Dict, List, Any, Optional, Set, Tuple
import geopandas as gpd
from dataclasses import dataclass
from enum import Enum

# Mock imports for standalone execution. Replace with your actual project structure.
from src.core.planners.workflow_generator import WorkflowGenerator, ToolCallIntent
from src.core.executors.workflow_executor import WorkflowExecutor
from src.core.agents.data_scout import DataScout
from src.core.planners.query_parser import ParsedQuery, SpatialConstraint, SpatialRelationship


class FailureType(Enum):
    """Types of failures that can occur during execution."""
    TOOL_ERROR = "tool_error"
    DATA_ERROR = "data_error"
    TIMEOUT = "timeout"
    INFINITE_LOOP = "infinite_loop"
    RESOURCE_ERROR = "resource_error"
    STRATEGY_ERROR = "strategy_error"
    PARAMETER_ERROR = "parameter_error"


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
    execution_time: float
    avg_loop_time: float


class MasterOrchestrator:
    """
    Enhanced central controller for iterative Think-Act-Observe GIS workflows.
    Features intelligent failure tracking, recovery mechanisms, comprehensive error handling,
    and layer intelligence for better LLM decision making.
    """

    def __init__(self, max_loops: int = 15, max_consecutive_failures: int = 3,
                 enable_adaptive_recovery: bool = True, enable_performance_metrics: bool = True):
        """
        Initialize the enhanced orchestrator.

        Args:
            max_loops: Maximum number of Think-Act-Observe iterations
            max_consecutive_failures: Emergency stop threshold for consecutive failures
            enable_adaptive_recovery: Enable intelligent recovery strategies
            enable_performance_metrics: Enable detailed performance tracking
        """
        # Initialize core agents
        self.data_scout = DataScout()
        self.strategist = WorkflowGenerator(data_scout=self.data_scout)
        self.executor = WorkflowExecutor(enable_reasoning_log=True)

        # State management
        self.conversation_history: List[str] = []
        self.data_layers: Dict[str, gpd.GeoDataFrame] = {}
        self.action_sequence: List[Dict[str, Any]] = []

        # Enhanced failure tracking
        self.failed_actions: Set[str] = set()
        self.failed_tools: Dict[str, int] = {}
        self.failure_history: List[FailureRecord] = []
        self.recovery_strategies: Dict[str, List[str]] = {}

        # Performance tracking
        self.loop_times: List[float] = []
        self.action_success_rate: Dict[str, Tuple[int, int]] = {}  # tool_name -> (success, total)

        # Configuration
        self.max_loops = max_loops
        self.max_consecutive_failures = max_consecutive_failures
        self.enable_adaptive_recovery = enable_adaptive_recovery
        self.enable_performance_metrics = enable_performance_metrics

    def _build_layer_intelligence(self) -> Dict[str, Dict[str, Any]]:
        """
        Build comprehensive layer intelligence for LLM decision making.
        This gives the orchestrator "eyes" to see the actual structure of data layers.
        """
        layer_intelligence = {}

        for layer_name, gdf in self.data_layers.items():
            if gdf.empty:
                layer_intelligence[layer_name] = {
                    "status": "empty",
                    "columns": [],
                    "feature_count": 0,
                    "geometry_types": [],
                    "crs": str(gdf.crs) if gdf.crs else "unknown",
                    "summary": f"Empty layer with no features"
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
                        sample_values[col] = [str(v) for v in sample_vals] # Ensure serializable
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

                layer_intelligence[layer_name] = {
                    "status": "populated",
                    "columns": display_columns,
                    "feature_count": len(gdf),
                    "geometry_types": geometry_types,
                    "crs": str(gdf.crs),
                    "sample_values": sample_values,
                    "bounds": bounds,
                    "summary": f"{len(gdf)} features with {len(columns)} columns ({', '.join(geometry_types)})"
                }

        return layer_intelligence

    def _build_layer_details(self) -> Dict[str, List[str]]:
        """
        Builds a simpler dictionary of layer names and their column schemas.
        Alternative to the comprehensive layer intelligence method.
        """
        layer_details = {}
        for name, gdf in self.data_layers.items():
            if not gdf.empty:
                # Keep it concise for the prompt
                columns = list(gdf.columns)
                if len(columns) > 20:
                    sample_columns = columns[:10] + [f"...({len(columns) - 20} more)..."] + columns[-10:]
                else:
                    sample_columns = columns
                layer_details[name] = sample_columns
            else:
                layer_details[name] = ["<empty_layer>"]
        return layer_details

    def run(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """
        Execute the complete Think-Act-Observe loop with comprehensive error handling.

        Args:
            parsed_query: The structured query from the query parser

        Returns:
            Dictionary containing execution results and comprehensive metadata
        """
        print("🎭 === ENHANCED MASTER ORCHESTRATOR: Starting Think-Act-Observe Loop ===")
        print(f"🎯 Query: Find '{parsed_query.target}' in '{parsed_query.location}'")

        # Reset state for new execution
        self._reset_execution_state()

        # Phase 1: Initialize context with enhanced error handling
        print("\n🔍 Phase 1: Establishing initial context...")
        try:
            context = self.strategist.get_initial_context(parsed_query)
        except Exception as e:
            error_msg = f"Context initialization failed: {str(e)}"
            print(f"❌ CRITICAL ERROR: {error_msg}")
            traceback.print_exc()
            return self._handle_critical_failure(error_msg)

        if not context["success"]:
            return {
                "success": False,
                "error": context["error"],
                "conversation_history": self.conversation_history,
                "execution_time": 0,
                "metrics": self._calculate_execution_metrics(0, 0)
            }

        # Initialize execution state
        self.conversation_history = [context["initial_observation"]]
        original_query = context.get("original_query", f"Find {parsed_query.target} in {parsed_query.location}")
        rag_guidance = context.get("rag_guidance", "")

        print("✅ Initial context established successfully")
        print(f"📊 {context['initial_observation']}")

        # Phase 2: Enhanced Think-Act-Observe Loop with Intelligence
        print("\n🔄 Phase 2: Starting intelligent Think-Act-Observe loop...")

        start_time = time.time()
        loop_count = 0
        finished = False
        consecutive_failures = 0

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

            # ✅ BUILD THE WORLD MODEL - Give the agent "eyes"
            layer_intelligence = self._build_layer_intelligence()
            current_layer_names = list(self.data_layers.keys())

            try:
                # ✅ PASS THE WORLD MODEL TO THE STRATEGIST
                next_action = self.strategist.get_next_action(
                    history=self.conversation_history,
                    original_query=original_query,
                    rag_guidance=rag_guidance,
                    available_layers=current_layer_names,
                    layer_intelligence=layer_intelligence  # ✅ NEW: Full layer visibility
                )

                # Validate the action
                if not isinstance(next_action, ToolCallIntent):
                    raise ValueError(f"Invalid action type returned: {type(next_action)}")

                if not next_action.tool_name:
                    raise ValueError("Empty tool name returned from strategist")

                print(f"💡 Decision: {next_action.tool_name}({next_action.parameters})")

                # Display layer intelligence for debugging
                if layer_intelligence:
                    print("👁️  Current layer state:")
                    for name, info in layer_intelligence.items():
                        print(f"   - {name}: {info['summary']}")

            except Exception as e:
                # ✅ FIX: Comprehensive error handling for the Think phase
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

            # Intelligent failure avoidance
            action_signature = self._get_action_signature(next_action)
            tool_name = next_action.tool_name

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

            # Handle completion
            if next_action.tool_name == "finish_task":
                final_layer = next_action.parameters.get("final_layer_name", "unknown")
                self.conversation_history.append("Action: finish_task")
                self.conversation_history.append(f"Observation: Task completed. Final result: '{final_layer}'")
                finished = True
                break

            # 🎬 ACT: Execute single step with enhanced error handling
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

            # Enhanced failure and success tracking
            if observation.startswith("Failure"):
                self._handle_action_failure(next_action, observation, loop_count)
                consecutive_failures += 1
                print(f"❌ Action failed (consecutive failures: {consecutive_failures})")
            else:
                self._handle_action_success(next_action, observation)
                consecutive_failures = 0  # Reset on success
                print("✅ Action succeeded")

            # 👀 OBSERVE: Update state and history
            print("👀 OBSERVE: Processing results and updating state...")
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
                self.strategist.store_successful_pattern(original_query, self.action_sequence)
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
            "performance_insights": self._generate_performance_insights()
        }

    def _reset_execution_state(self):
        """Reset all execution state for a new run."""
        self.conversation_history.clear()
        self.data_layers.clear()
        self.action_sequence.clear()
        self.failed_actions.clear()
        self.failed_tools.clear()
        self.failure_history.clear()
        self.loop_times.clear()

    def _get_action_signature(self, action: ToolCallIntent) -> str:
        """Generate a unique signature for an action."""
        return f"{action.tool_name}:{json.dumps(action.parameters, sort_keys=True)}"

    def _record_failure(self, tool_name: str, action_type: str, parameters: Dict[str, Any],
                       failure_type: FailureType, error_message: str, loop_iteration: int):
        """Record a failure for analysis and learning."""
        action_signature = f"{tool_name}:{json.dumps(parameters, sort_keys=True)}"

        failure_record = FailureRecord(
            action_signature=action_signature,
            tool_name=tool_name,
            parameters=parameters,
            failure_type=failure_type,
            error_message=error_message,
            timestamp=time.time(),
            loop_iteration=loop_iteration
        )

        self.failure_history.append(failure_record)
        self.failed_actions.add(action_signature)
        self.failed_tools[tool_name] = self.failed_tools.get(tool_name, 0) + 1

    def _handle_action_failure(self, action: ToolCallIntent, observation: str, loop_count: int):
        """Handle and record an action failure."""
        # Determine failure type from observation
        failure_type = FailureType.TOOL_ERROR
        if "timeout" in observation.lower():
            failure_type = FailureType.TIMEOUT
        elif "data" in observation.lower() or "no data" in observation.lower():
            failure_type = FailureType.DATA_ERROR
        elif "resource" in observation.lower() or "memory" in observation.lower():
            failure_type = FailureType.RESOURCE_ERROR
        elif "parameter" in observation.lower() or "invalid" in observation.lower():
            failure_type = FailureType.PARAMETER_ERROR

        self._record_failure(
            action.tool_name,
            "execution",
            action.parameters,
            failure_type,
            observation,
            loop_count
        )

        # Update success rate tracking
        if action.tool_name in self.action_success_rate:
            success, total = self.action_success_rate[action.tool_name]
            self.action_success_rate[action.tool_name] = (success, total + 1)
        else:
            self.action_success_rate[action.tool_name] = (0, 1)

    def _handle_action_success(self, action: ToolCallIntent, observation: str):
        """Handle and record an action success."""
        # Store successful action for RAG learning
        if observation.startswith("Success"):
            try:
                executable_action = self.strategist.translate_single_action(
                    action, self._build_conceptual_map()
                )
                self.action_sequence.append(executable_action)
            except Exception as e:
                print(f"⚠️  Warning: Could not translate successful action: {e}")

        # Update success rate tracking
        if action.tool_name in self.action_success_rate:
            success, total = self.action_success_rate[action.tool_name]
            self.action_success_rate[action.tool_name] = (success + 1, total + 1)
        else:
            self.action_success_rate[action.tool_name] = (1, 1)

    def _attempt_recovery(self, failed_tool: str, original_action: ToolCallIntent) -> Optional[ToolCallIntent]:
        """Attempt to recover from a failed action using intelligent strategies."""
        if not self.enable_adaptive_recovery:
            return None

        # Define recovery strategies
        recovery_strategies = {
            "load_data": ["search_osm_data", "buffer_features"],
            "search_osm_data": ["load_data", "spatial_filter"],
            "spatial_filter": ["buffer_features", "spatial_intersection"],
            "buffer_features": ["spatial_filter", "spatial_intersection"]
        }

        if failed_tool not in recovery_strategies:
            return None

        # Try alternative tools
        for alternative_tool in recovery_strategies[failed_tool]:
            if alternative_tool not in self.failed_tools or self.failed_tools[alternative_tool] < 2:
                # Create alternative action with adapted parameters
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
        if new_tool == "buffer_features" and "distance_meters" not in adapted_params:
            adapted_params["distance_meters"] = 1000  # Default buffer
        elif new_tool == "spatial_filter" and "geometry_column" not in adapted_params:
            adapted_params["geometry_column"] = "geometry"

        return adapted_params

    def _calculate_execution_metrics(self, loop_count: int, execution_time: float) -> ExecutionMetrics:
        """Calculate comprehensive execution metrics."""
        successful_actions = sum(1 for h in self.conversation_history
                               if h.startswith("Observation: Success"))
        failed_actions = len(self.failure_history)

        return ExecutionMetrics(
            total_loops=loop_count,
            successful_actions=successful_actions,
            failed_actions=failed_actions,
            unique_failures=len(self.failed_actions),
            consecutive_failures=self._get_max_consecutive_failures(),
            max_consecutive_failures=self.max_consecutive_failures,
            recovery_attempts=sum(1 for f in self.failure_history
                                if "recovery" in f.error_message.lower()),
            successful_recoveries=0,  # Would need more sophisticated tracking
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
        """Generate comprehensive failure analysis."""
        if not self.failure_history:
            return {"total_failures": 0, "analysis": "No failures recorded"}

        failure_by_tool = {}
        failure_by_type = {}

        for failure in self.failure_history:
            failure_by_tool[failure.tool_name] = failure_by_tool.get(failure.tool_name, 0) + 1
            failure_by_type[failure.failure_type.value] = failure_by_type.get(failure.failure_type.value, 0) + 1

        return {
            "total_failures": len(self.failure_history),
            "unique_failed_actions": len(self.failed_actions),
            "failure_by_tool": failure_by_tool,
            "failure_by_type": failure_by_type,
            "most_problematic_tool": max(failure_by_tool.items(), key=lambda x: x[1])[0] if failure_by_tool else None,
            "average_failures_per_loop": len(self.failure_history) / max(len(self.loop_times), 1)
        }

    def _generate_performance_insights(self) -> Dict[str, Any]:
        """Generate performance insights and recommendations."""
        insights = {
            "total_execution_time": sum(self.loop_times),
            "average_loop_time": sum(self.loop_times) / max(len(self.loop_times), 1),
            "slowest_loop_time": max(self.loop_times) if self.loop_times else 0,
            "fastest_loop_time": min(self.loop_times) if self.loop_times else 0,
            "tool_success_rates": {}
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

        insights["recommendations"] = recommendations
        return insights

    def _handle_critical_failure(self, error_message: str) -> Dict[str, Any]:
        """Handle critical failures that prevent execution."""
        return {
            "success": False,
            "error": f"Critical failure: {error_message}",
            "conversation_history": self.conversation_history,
            "execution_time": 0,
            "metrics": ExecutionMetrics(0, 0, 1, 1, 1, self.max_consecutive_failures, 0, 0, 0, 0),
            "failure_analysis": {"critical_failure": error_message},
            "performance_insights": {"status": "execution_aborted"}
        }

    def _build_conceptual_map(self) -> Dict[str, str]:
        """Build mapping from conceptual names to real layer names."""
        conceptual_map = {}
        for real_name in self.data_layers.keys():
            # Extract conceptual name: 'schools_1' -> 'schools'
            if '_' in real_name and real_name.rsplit('_', 1)[-1].isdigit():
                conceptual_name = real_name.rsplit('_', 1)[0]
            else:
                conceptual_name = real_name
            conceptual_map[conceptual_name] = real_name
        return conceptual_map

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
                "layer_intelligence": True,  # ✅ NEW: Layer visibility feature
                "max_consecutive_failures": self.max_consecutive_failures
            }
        }


# Enhanced integration test
if __name__ == '__main__':
    print("🧪 === ENHANCED MASTER ORCHESTRATOR INTEGRATION TEST ===")
    
    # Test query: Find schools near parks in Berlin
    test_query = ParsedQuery(
        target='school',
        location='Berlin, Germany',
        constraints=[
            SpatialConstraint(feature_type='park', relationship=SpatialRelationship.NEAR, distance_meters=500)
        ],
        summary_required=True
    )

    # Test with enhanced features including layer intelligence
    orchestrator = MasterOrchestrator(
        max_loops=10,
        max_consecutive_failures=3,
        enable_adaptive_recovery=True,
        enable_performance_metrics=True
    )

    try:
        result = orchestrator.run(test_query)

        print("\n📊 === ENHANCED RESULTS ANALYSIS ===")
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
        print(f"  - Average Loop Time: {metrics.avg_loop_time:.2f}s")

        # Failure analysis
        failure_analysis = result['failure_analysis']
        print(f"\n❌ Failure Analysis:")
        print(f"  - Total Failures: {failure_analysis.get('total_failures', 0)}")
        if failure_analysis.get('most_problematic_tool'):
            print(f"  - Most Problematic Tool: {failure_analysis['most_problematic_tool']}")
        
        # Performance insights
        insights = result['performance_insights']
        print(f"\n💡 Performance Insights:")
        for tool, rates in insights.get('tool_success_rates', {}).items():
            rate_percent = rates['success_rate'] * 100
            print(f"  - {tool}: {rate_percent:.1f}% success rate ({rates['successes']}/{rates['total_attempts']})")
        
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

    print("\n✅ Enhanced MasterOrchestrator test completed!")