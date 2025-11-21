#!/usr/bin/env python3
"""
Task Executor - Single-Tool Isolation Architecture
==================================================

This module implements the Executor component with **single-tool isolation**.
For each task, it only exposes ONE tool definition to the LLM, physically
preventing hallucination of other tools.

Key Features:
- Executes tasks sequentially from task queue
- Single-tool isolation: LLM only sees one tool at a time
- State passing between tasks
- Deterministic progression (Python controls flow, not LLM)
"""

import logging
import json
import os
import requests
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

from src.core.planners.execution_planner import ExecutionTask, TaskQueue, TaskType
from src.core.executors.workflow_executor import WorkflowExecutor
from src.gis.tools.definitions import TOOL_REGISTRY, ToolDefinition, get_tool_by_name
import geopandas as gpd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExecutionState:
    """
    Represents the current state of execution.
    Passed between tasks to maintain context.
    """
    available_layers: Dict[str, gpd.GeoDataFrame]  # Current data layers
    task_outputs: Dict[str, str] = field(default_factory=dict)  # Task ID -> Output layer name
    completed_tasks: List[str] = field(default_factory=list)  # Completed task IDs
    
    def get_layer_summary(self) -> str:
        """Get a concise summary of available layers."""
        if not self.available_layers:
            return "No data layers loaded yet"
        
        summaries = []
        for name, gdf in self.available_layers.items():
            geom_types = gdf.geom_type.unique().tolist() if not gdf.empty else ["empty"]
            summaries.append(f"  - {name}: {len(gdf)} features ({', '.join(geom_types)})")
        
        return "\n".join(summaries)


@dataclass
class TaskResult:
    """Result from executing a single task."""
    task_id: str
    success: bool
    output_layer_name: Optional[str]
    observation: str
    updated_layers: Dict[str, gpd.GeoDataFrame]
    error: Optional[str] = None


class TaskExecutor:
    """
    Executes tasks from a task queue with single-tool isolation.
    """
    
    def __init__(self):
        """Initialize the task executor."""
        self.workflow_executor = WorkflowExecutor(enable_reasoning_log=True)
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("OLLAMA_MODEL", "mistral")
        self.timeout = 120
        
        logger.info("TaskExecutor initialized with single-tool isolation")
    
    def execute_task_queue(self, task_queue: TaskQueue, 
                           initial_layers: Dict[str, gpd.GeoDataFrame]) -> Dict[str, Any]:
        """
        Execute a complete task queue sequentially.
        """
        logger.info(f"🚀 Executing task queue: {len(task_queue.tasks)} tasks")
        
        state = ExecutionState(available_layers=initial_layers.copy())
        execution_log: List[Dict[str, Any]] = []
        
        for i, task in enumerate(task_queue.tasks, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Task {i}/{len(task_queue.tasks)}: {task.id} ({task.tool_name})")
            logger.info(f"{'=' * 60}")
            
            # Execute single task
            result = self._execute_single_task(task, state)
            
            # === FIX: FORCE LAYER RENAMING ===
            # The Planner expects specific names (e.g., 'park_berlin_germany')
            # The Tool returns generic names (e.g., 'load_osm_data_output_2')
            # We must bridge this gap here.
            if result.success and result.output_layer_name and task.output_layer_name:
                actual_name = result.output_layer_name
                expected_name = task.output_layer_name
                
                # Check if names match. If not, rename the layer.
                if actual_name != expected_name:
                    if actual_name in result.updated_layers:
                        logger.info(f"🔄 Renaming layer '{actual_name}' -> '{expected_name}' to match Plan")
                        
                        # Move dataframe
                        layer_data = result.updated_layers.pop(actual_name)
                        result.updated_layers[expected_name] = layer_data
                        
                        # Update result pointer
                        result.output_layer_name = expected_name
                    else:
                        logger.warning(f"⚠️ Could not find actual layer '{actual_name}' to rename")
            # =================================
            
            # Update state from result
            if result.success:
                state.available_layers = result.updated_layers
                state.task_outputs[task.id] = result.output_layer_name
                state.completed_tasks.append(task.id)
                logger.info(f"✅ Task {task.id} completed successfully")
            else:
                logger.error(f"Task {task.id} failed: {result.error}")
                return {
                    "success": False,
                    "error": f"Task {task.id} failed: {result.error}",
                    "completed_tasks": len(state.completed_tasks),
                    "total_tasks": len(task_queue.tasks),
                    "execution_log": execution_log
                }
            
            # Log task execution
            execution_log.append({
                "task_id": task.id,
                "tool_name": task.tool_name,
                "success": result.success,
                "observation": result.observation,
                "output_layer": result.output_layer_name
            })
        
        # All tasks completed successfully
        final_layer_name = task_queue.tasks[-2].output_layer_name if len(task_queue.tasks) > 1 else None
        
        return {
            "success": True,
            "final_layer_name": final_layer_name,
            "final_layers": state.available_layers,
            "completed_tasks": len(state.completed_tasks),
            "total_tasks": len(task_queue.tasks),
            "execution_log": execution_log,
            "reasoning_log": self.workflow_executor.get_reasoning_log()
        }
    
    def _execute_single_task(self, task: ExecutionTask, 
                             state: ExecutionState) -> TaskResult:
        """
        Execute a single task with **single-tool isolation**.
        """
        logger.info(f"📌 Executing: {task.instruction}")
        
        # === FIX: HANDLE VIRTUAL TOOLS ===
        # finish_task is a virtual tool (workflow completion signal), not a real tool
        if task.tool_name == "finish_task":
            logger.info("🏁 Encountered finish_task. Marking workflow as complete.")
            return TaskResult(
                task_id=task.id,
                success=True,
                output_layer_name=None,
                observation="Workflow completed successfully.",
                updated_layers=state.available_layers
            )
        # =================================
        
        # CRITICAL: Get ONLY the one tool this task should use
        single_tool_def = get_tool_by_name(task.tool_name)
        
        if not single_tool_def:
            return TaskResult(
                task_id=task.id, success=False, output_layer_name=None,
                observation=f"Tool '{task.tool_name}' not found",
                updated_layers=state.available_layers, error=f"Unknown tool: {task.tool_name}"
            )
        
        # Resolve dependencies (layer references)
        resolved_params = self._resolve_task_parameters(task, state)
        
        # Build task-specific prompt with ONLY ONE TOOL
        prompt = self._build_single_tool_prompt(task, single_tool_def, resolved_params, state)
        
        # Get LLM response (it can only see ONE tool)
        try:
            llm_response = self._call_llm(prompt)
            refined_params = self._parse_llm_response(llm_response, resolved_params)
            
            # Snapshot existing layers to detect what's new
            previous_layers = set(state.available_layers.keys())
            
            # Execute the tool
            observation, updated_layers = self.workflow_executor.execute_single_step(
                tool_call=self._create_tool_call(task.tool_name, refined_params),
                current_data_layers=state.available_layers
            )
            
            success = not observation.startswith("Failure")
            
            # === FIX: DYNAMICALLY DETECT OUTPUT NAME ===
            actual_output_name = None
            if success and updated_layers:
                # Find which key is new
                new_keys = set(updated_layers.keys()) - previous_layers
                if new_keys:
                    actual_output_name = list(new_keys)[0]
                elif task.tool_name == 'buffer': 
                    # Edge case: Buffer might not create a new key if failed or renamed internally? 
                    # Usually it creates a new key. If not, we check if count increased.
                    pass
            # ===========================================
            
            return TaskResult(
                task_id=task.id,
                success=success,
                output_layer_name=actual_output_name, # Use the ACTUAL name found
                observation=observation,
                updated_layers=updated_layers if success else state.available_layers,
                error=None if success else observation
            )
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}", exc_info=True)
            return TaskResult(
                task_id=task.id, success=False, output_layer_name=None,
                observation=f"Execution error: {str(e)}",
                updated_layers=state.available_layers, error=str(e)
            )
    
    def _build_single_tool_prompt(self, task: ExecutionTask, 
                                  tool_def: ToolDefinition,
                                  suggested_params: Dict[str, Any],
                                  state: ExecutionState) -> str:
        # Generate parameter documentation for the ONE tool
        param_docs = []
        for param in tool_def.parameters:
            required_marker = " (REQUIRED)" if param.required else " (optional)"
            param_docs.append(f"  - {param.name}: {param.description}{required_marker}")
        
        params_str = "\n".join(param_docs)
        
        prompt = f"""You are executing a specific GIS task. You have access to EXACTLY ONE tool.

**YOUR TASK:**
{task.instruction}

**AVAILABLE TOOL (ONLY ONE):**
Tool Name: {tool_def.operation_name}
Description: {tool_def.description}
Parameters:
{params_str}

**SUGGESTED PARAMETERS:**
{json.dumps(suggested_params, indent=2)}

**CRITICAL RULES:**
1. You MUST use the tool '{tool_def.operation_name}'.
2. Do NOT add "addr:city", "addr:country" or similar location tags. The location is handled by 'area_name'.
3. Keep tags simple (e.g., use {{'amenity': 'school'}}, NOT {{'amenity': ['school']}}).
4. Do NOT invent new parameters. Use only what is listed above.

**CURRENT DATA LAYERS:**
{state.get_layer_summary()}

Output ONLY a JSON object with the refined parameters:
{{
  "parameters": {{
    "param_name": "value",
    ...
  }}
}}
"""
        return prompt
    
    def _resolve_task_parameters(self, task: ExecutionTask, 
                                 state: ExecutionState) -> Dict[str, Any]:
        resolved = task.parameters_hint.copy()
        for key, value in resolved.items():
            if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                task_ref_id = value[1:-1] # Remove brackets
                # Support bare ID or task_ID format
                if task_ref_id in state.task_outputs:
                    actual_layer = state.task_outputs[task_ref_id]
                    resolved[key] = actual_layer
                    logger.info(f"  Resolved {value} -> {actual_layer}")
        return resolved
    
    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        full_api_url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 512}
        }
        try:
            response = requests.post(full_api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            response_data = response.json()
            raw_response = response_data.get('response', '{}').strip()
            if raw_response.startswith('```json'): raw_response = raw_response[7:]
            if raw_response.endswith('```'): raw_response = raw_response[:-3]
            return json.loads(raw_response.strip())
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {"parameters": {}}
    
    def _parse_llm_response(self, llm_response: Dict[str, Any],
                            suggested_params: Dict[str, Any]) -> Dict[str, Any]:
        if "parameters" in llm_response:
            refined = suggested_params.copy()
            refined.update(llm_response["parameters"])
            return refined
        return suggested_params
    
    def _create_tool_call(self, tool_name: str, parameters: Dict[str, Any]):
        from src.core.planners.workflow_generator import ToolCallIntent
        return ToolCallIntent(tool_name=tool_name, parameters=parameters)