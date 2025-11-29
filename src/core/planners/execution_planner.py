#!/usr/bin/env python3
"""
Execution Planner - Task Queue Architecture
==========================================

This module implements the Planner component of the Planner-Executor architecture.
It generates a deterministic task queue upfront, eliminating LLM-based loops.

Key Features:
- Generates linear Task Queue from parsed query
- Supports interleaved task types (LOAD-ANALYZE-LOAD-ANALYZE)
- Each task specifies exactly ONE tool
- Enables single-tool isolation in execution
- INTELLIGENT SHORTCUT: Skips downloads for pure statistical queries
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum

from src.core.planners.query_parser import ParsedQuery, SpatialConstraint, SpatialRelationship
from src.core.agents.data_scout import DataRealityReport, DataProbeResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Task types supported by the planner."""
    LOAD = "LOAD"
    BUFFER = "BUFFER"
    SPATIAL_JOIN = "SPATIAL_JOIN"
    INTERSECT = "INTERSECT"
    FILTER = "FILTER" 
    FINISH = "FINISH"


@dataclass
class ExecutionTask:
    """
    Represents a single task in the execution queue.
    
    Each task is atomic and specifies exactly ONE tool to use.
    """
    id: str  # Unique identifier (e.g., "task_1")
    task_type: TaskType  # Type of task
    tool_name: str  # SPECIFIC tool (e.g., "load_osm_data", "buffer")
    instruction: str  # Specific prompt for LLM to refine parameters
    parameters_hint: Dict[str, Any]  # Suggested parameters (LLM can refine)
    depends_on: List[str] = field(default_factory=list)  # Task IDs this depends on
    output_layer_name: Optional[str] = None  # Expected output layer name
    
    def __repr__(self) -> str:
        return f"ExecutionTask({self.id}: {self.tool_name})"


@dataclass
class TaskQueue:
    """
    Linear queue of execution tasks.
    
    Represents the complete execution plan for a query.
    """
    tasks: List[ExecutionTask]
    original_query: str
    requirements: Dict[str, Any]
    
    def __repr__(self) -> str:
        task_summary = " -> ".join([t.tool_name for t in self.tasks])
        return f"TaskQueue({len(self.tasks)} tasks: {task_summary})"


class ExecutionPlanner:
    """
    Generates a deterministic task queue from a parsed query.
    
    This replaces the LLM-based decision making with upfront planning,
    eliminating infinite loops and context drift.
    """
    
    def __init__(self):
        """Initialize the execution planner."""
        self.task_counter = 0
        logger.info("ExecutionPlanner initialized")
    
    def generate_task_queue(self, parsed_query: ParsedQuery, 
                            data_report: DataRealityReport) -> TaskQueue:
        """
        Generate a complete task queue.
        Includes STATISTICAL SHORTCUT to prevent massive downloads for simple questions.
        Supports multi-target queries natively.
        """
        # === FIX: Handle Multi-Target properly ===
        targets = parsed_query.target if isinstance(parsed_query.target, list) else [parsed_query.target]
        targets_str = ', '.join(targets) if isinstance(targets, list) else targets
        # =========================================
        
        logger.info(f"🎯 Planning task queue for: {targets_str} in {parsed_query.location}")
        
        self.task_counter = 0
        tasks: List[ExecutionTask] = []
        
        # Analyze requirements
        requirements = self._analyze_requirements(parsed_query)
        
        # === MASSIVE DATASET PROTECTION ===
        # Calculate total count across all targets
        total_count = sum(r.count for r in data_report.probe_results 
                         if any(t.lower() in r.original_entity.lower() for t in targets))
        
        SAFETY_THRESHOLD = 50000  # Increased from 10k to allow moderate datasets (like 2k-10k) to generate maps
        is_massive = total_count > SAFETY_THRESHOLD
        
        # If MASSIVE (> 50k), ALWAYS shortcut to avoid timeout - don't attempt download
        if is_massive:
            logger.info(f"⚡ SHORTCUT: Massive dataset detected ({total_count} > {SAFETY_THRESHOLD}). Skipping heavy download to prevent timeout.")
            
            finish_task = ExecutionTask(
                id="task_1",
                task_type=TaskType.FINISH,
                tool_name="finish_task",
                instruction=f"Answer the user directly using the probe data.",
                parameters_hint={
                    "final_layer_name": None,
                    "reason": f"Found {total_count} features. Skipping map download to prevent API timeout (Threshold: {SAFETY_THRESHOLD})."
                }
            )
            
            return TaskQueue(
                tasks=[finish_task],
                original_query=f"Find {targets_str}",
                requirements=requirements
            )
        
        # === INTELLIGENT SHORTCUT: PURE COUNTING ===
        # If user asks "How many?" (summary_required) AND we don't need spatial math AND count is moderate
        if parsed_query.summary_required and not requirements['needs_spatial_analysis']:
            logger.info(f"⚡ SHORTCUT: Statistical query with moderate count ({total_count}). Skipping map generation.")
            
            finish_task = ExecutionTask(
                id="task_1",
                task_type=TaskType.FINISH,
                tool_name="finish_task",
                instruction=f"Answer the user directly using the probe data.",
                parameters_hint={
                    "final_layer_name": None,
                    "reason": f"Found {total_count} features. Statistical query satisfied without map."
                }
            )
            
            return TaskQueue(
                tasks=[finish_task],
                original_query=f"Count {targets_str}",
                requirements=requirements
            )
        # ===========================================
        
        # === PHASE 1: LOAD DATA (Loop through ALL targets) ===
        created_layers = []
        
        for i, target in enumerate(targets):
            load_task = self._create_load_task(
                entity=target,
                location=parsed_query.location,
                data_report=data_report
            )
            tasks.append(load_task)
            created_layers.append(load_task.output_layer_name)
            logger.info(f"  ✅ {load_task.id}: Load {target} data")
        # ====================================================
        
        # Step 2: Handle constraints (if any)
        if parsed_query.constraints:
            constraint_tasks = self._plan_constraint_tasks(
                parsed_query.constraints,
                parsed_query.location,
                data_report,
                primary_task_id=tasks[0].id  # Reference first target task
            )
            tasks.extend(constraint_tasks)
        
        # Step 3: Add finish task
        # For multi-target queries without constraints, just use the last loaded layer
        final_layer = created_layers[-1] if created_layers else None
        finish_task = self._create_finish_task(final_layer)
        tasks.append(finish_task)
        logger.info(f"  ✅ {finish_task.id}: Finish with result layer")
        
        task_queue = TaskQueue(
            tasks=tasks,
            original_query=f"Find {targets_str} in {parsed_query.location}",
            requirements=requirements
        )
        
        logger.info(f"📋 Task queue generated: {len(tasks)} tasks")
        return task_queue
    
    def _analyze_requirements(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """Analyze what the query needs."""
        targets = parsed_query.target if isinstance(parsed_query.target, list) else [parsed_query.target]
        return {
            'primary_target': targets,  # Now a list
            'needs_constraints': bool(parsed_query.constraints),
            'constraint_count': len(parsed_query.constraints) if parsed_query.constraints else 0,
            'needs_spatial_analysis': any(
                c.relationship in [SpatialRelationship.NEAR, SpatialRelationship.WITHIN]
                for c in (parsed_query.constraints or [])
            )
        }
    
    def _create_load_task(self, entity: str, location: str,
                         data_report: DataRealityReport) -> ExecutionTask:
        """Create a data loading task."""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        
        # Find the appropriate tags from data report
        tags = self._get_tags_from_report(entity, data_report)
        
        # Generate clean layer name
        layer_name = f"{entity.lower().replace(' ', '_')}_{location.lower().replace(' ', '_').replace(',', '')}"
        
        task = ExecutionTask(
            id=task_id,
            task_type=TaskType.LOAD,
            tool_name="load_osm_data",
            instruction=f"Load {entity} data in {location}",
            parameters_hint={
                "area_name": location,
                "tags": tags,
                "layer_name": layer_name
            },
            output_layer_name=layer_name
        )
        
        return task
    
    def _create_buffer_task(self, input_layer: str, distance: int,
                           depends_on_id: str) -> ExecutionTask:
        """Create a buffer analysis task."""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        
        output_layer = f"{input_layer}_buffer_{distance}m"
        
        task = ExecutionTask(
            id=task_id,
            task_type=TaskType.BUFFER,
            tool_name="buffer",
            instruction=f"Create {distance}m buffer around {input_layer}",
            parameters_hint={
                "layer_name": input_layer,
                "distance": float(distance),
                "output_layer_name": output_layer
            },
            depends_on=[depends_on_id],
            output_layer_name=output_layer
        )
        
        return task
    
    def _create_spatial_join_task(self, left_layer: str, right_layer: str,
                                 depends_on_ids: List[str], predicate: str = "intersects") -> ExecutionTask:
        """Create a spatial join task."""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        
        output_layer = f"{left_layer}_join_{right_layer}"
        
        task = ExecutionTask(
            id=task_id,
            task_type=TaskType.SPATIAL_JOIN,
            tool_name="spatial_join",
            instruction=f"Find {left_layer} that {predicate} with {right_layer}",
            parameters_hint={
                "left_layer_name": left_layer,
                "right_layer_name": right_layer,
                "predicate": predicate,
                "output_layer_name": output_layer
            },
            depends_on=depends_on_ids,
            output_layer_name=output_layer
        )
        
        return task
    
    def _create_finish_task(self, final_layer_name: str) -> ExecutionTask:
        """Create a finish task."""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        
        task = ExecutionTask(
            id=task_id,
            task_type=TaskType.FINISH,
            tool_name="finish_task",
            instruction=f"Complete the workflow with final result layer: {final_layer_name}",
            parameters_hint={
                "final_layer_name": final_layer_name,
                "reason": "All required analysis completed successfully"
            },
            output_layer_name=final_layer_name
        )
        
        return task
    
    def _plan_constraint_tasks(self, constraints: List[SpatialConstraint],
                               location: str, data_report: DataRealityReport,
                               primary_task_id: str) -> List[ExecutionTask]:
        """
        Plan tasks for handling spatial constraints.
        
        This supports complex multi-step dependencies like:
        "schools near parks" -> LOAD schools, LOAD parks, BUFFER parks, SPATIAL_JOIN
        """
        tasks: List[ExecutionTask] = []
        primary_layer = None  # Will be set from first constraint
        
        for i, constraint in enumerate(constraints):
            # Step 1: Load constraint feature data
            constraint_load_task = self._create_load_task(
                entity=constraint.feature_type,
                location=location,
                data_report=data_report
            )
            tasks.append(constraint_load_task)
            logger.info(f"  ✅ {constraint_load_task.id}: Load constraint ({constraint.feature_type})")
            
            constraint_layer = constraint_load_task.output_layer_name
            
            # Step 2: Handle spatial relationship
            if constraint.relationship == SpatialRelationship.NEAR:
                # Create buffer around constraint features
                distance = constraint.distance_meters or 500  # Default 500m
                buffer_task = self._create_buffer_task(
                    input_layer=constraint_layer,
                    distance=distance,
                    depends_on_id=constraint_load_task.id
                )
                tasks.append(buffer_task)
                logger.info(f"  ✅ {buffer_task.id}: Buffer {constraint.feature_type} by {distance}m")
                
                # Spatial join primary target with buffered constraint
                # Reference the primary task's output using its ID directly
                primary_layer_ref = f"[{primary_task_id}]"
                
                join_task = self._create_spatial_join_task(
                    left_layer=primary_layer_ref,
                    right_layer=buffer_task.output_layer_name,
                    depends_on_ids=[primary_task_id, buffer_task.id],
                    predicate="intersects"
                )
                tasks.append(join_task)
                logger.info(f"  ✅ {join_task.id}: Find primary target within buffered area")
            
            elif constraint.relationship == SpatialRelationship.WITHIN:
                # Reference the primary task's output
                primary_layer_ref = f"[{primary_task_id}]"
                
                join_task = self._create_spatial_join_task(
                    left_layer=primary_layer_ref,
                    right_layer=constraint_layer,
                    depends_on_ids=[primary_task_id, constraint_load_task.id],
                    predicate="within"
                )
                tasks.append(join_task)
                logger.info(f"  ✅ {join_task.id}: Find primary target within {constraint.feature_type}")
        
        return tasks
    
    def _get_tags_from_report(self, entity: str, data_report: DataRealityReport) -> Dict[str, str]:
        """
        Extract proven OSM tags from the data report for an entity.
        Prioritizes tags that were actually successfully probed (count > 0).
        """
        # 1. Look for exact match in SUCCESSFUL probes only
        for probe in data_report.probe_results:
            # Check if this probe matches our entity AND was successful
            if (probe.original_entity.lower() == entity.lower() and 
                probe.count > 0):
                
                logger.info(f"🎯 Using proven tags for '{entity}': {probe.tag} (Count: {probe.count})")
                
                # Parse "key=value" string to dict
                if '=' in probe.tag:
                    key, value = probe.tag.split('=', 1)
                    return {key: value}
        
        # 2. Fallback: Look for partial match in successful probes
        # (e.g. if entity is "parks" but probe was "park")
        for probe in data_report.probe_results:
            if probe.count > 0 and (
                entity.lower() in probe.original_entity.lower() or 
                probe.original_entity.lower() in entity.lower()):
                
                logger.info(f"🎯 Using proven tags for '{entity}' (partial match): {probe.tag} (Count: {probe.count})")
                
                if '=' in probe.tag:
                    key, value = probe.tag.split('=', 1)
                    return {key: value}

        # 3. Fallback: Use common defaults if no successful probe found
        logger.warning(f"⚠️ No proven tags found for '{entity}'. Using defaults.")
        common_tags = {
            'school': {'amenity': 'school'},
            'park': {'leisure': 'park'},  # Default back to leisure=park
            'hospital': {'amenity': 'hospital'},
            'restaurant': {'amenity': 'restaurant'},
        }
        
        return common_tags.get(entity.lower(), {'name': entity})