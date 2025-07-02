# src/worker/tasks.py - Geospatial Chain-of-Thought Processing Tasks
import os
import json
import time
from typing import Dict, List, Any, Optional
from celery import current_task
import redis

# Import the Celery app
from .celery_app import celery

# Redis client for storing reasoning logs and progress
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"), 
        port=6379, 
        db=0, 
        decode_responses=True
    )
except Exception as e:
    print(f"Warning: Redis connection failed in tasks: {e}")
    redis_client = None

class GeospatialReasoner:
    """
    Chain-of-Thought reasoning engine for geospatial workflow generation.
    This simulates the LLM-based reasoning process for your project.
    """
    
    def __init__(self):
        self.reasoning_log = []
        self.workflow_steps = []
        self.available_tools = [
            "buffer_analysis", "spatial_join", "clip", "intersection",
            "slope_analysis", "flood_risk_modeling", "site_suitability",
            "land_cover_classification", "proximity_analysis"
        ]
    
    def log_reasoning(self, step: str, thought: str, action: str = None):
        """Log a reasoning step in Chain-of-Thought format"""
        reasoning_entry = {
            "step": len(self.reasoning_log) + 1,
            "timestamp": time.time(),
            "thought": thought,
            "action": action,
            "step_type": step
        }
        self.reasoning_log.append(reasoning_entry)
        return reasoning_entry
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the user query and determine the geospatial workflow needed.
        This simulates LLM-based query understanding.
        """
        self.log_reasoning(
            "query_analysis",
            f"Analyzing user query: '{query}'",
            "identify_spatial_requirements"
        )
        
        # Simulate query analysis (in real implementation, this would use LLM)
        query_lower = query.lower()
        
        analysis = {
            "query_type": "unknown",
            "spatial_operations": [],
            "data_requirements": [],
            "output_format": "geojson"
        }
        
        # Simple keyword-based analysis (replace with LLM in real implementation)
        if "flood" in query_lower:
            analysis["query_type"] = "flood_risk_analysis"
            analysis["spatial_operations"] = ["dem_analysis", "buffer_analysis", "slope_analysis"]
            analysis["data_requirements"] = ["elevation_data", "water_bodies", "rainfall_data"]
            
        elif "site" in query_lower and ("suitable" in query_lower or "selection" in query_lower):
            analysis["query_type"] = "site_suitability"
            analysis["spatial_operations"] = ["proximity_analysis", "overlay_analysis", "scoring"]
            analysis["data_requirements"] = ["land_use", "infrastructure", "constraints"]
            
        elif "buffer" in query_lower:
            analysis["query_type"] = "buffer_analysis"
            analysis["spatial_operations"] = ["buffer_analysis"]
            analysis["data_requirements"] = ["geometry_data"]
        
        self.log_reasoning(
            "query_understanding",
            f"Identified query type: {analysis['query_type']} with operations: {analysis['spatial_operations']}",
            "generate_workflow_plan"
        )
        
        return analysis
    
    def generate_workflow(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a step-by-step geospatial workflow based on the analysis.
        """
        self.log_reasoning(
            "workflow_generation",
            f"Generating workflow for {analysis['query_type']}",
            "create_processing_steps"
        )
        
        workflow = []
        
        if analysis["query_type"] == "flood_risk_analysis":
            workflow = [
                {
                    "step": 1,
                    "operation": "load_elevation_data",
                    "description": "Load Digital Elevation Model (DEM) data",
                    "parameters": {"source": "bhoonidhi", "resolution": "30m"},
                    "reasoning": "DEM is essential for flood modeling and flow direction analysis"
                },
                {
                    "step": 2,
                    "operation": "slope_analysis",
                    "description": "Calculate slope from elevation data",
                    "parameters": {"method": "horn", "units": "degrees"},
                    "reasoning": "Slope affects water flow and flood accumulation patterns"
                },
                {
                    "step": 3,
                    "operation": "flow_accumulation",
                    "description": "Calculate flow accumulation patterns",
                    "parameters": {"fill_sinks": True},
                    "reasoning": "Flow accumulation identifies areas where water naturally collects"
                },
                {
                    "step": 4,
                    "operation": "flood_zone_modeling",
                    "description": "Model potential flood zones",
                    "parameters": {"return_period": "100_year", "sea_level_rise": "0.5m"},
                    "reasoning": "Combine topographic analysis with flood return periods for risk assessment"
                }
            ]
            
        elif analysis["query_type"] == "site_suitability":
            workflow = [
                {
                    "step": 1,
                    "operation": "load_land_use_data",
                    "description": "Load land use/land cover data",
                    "parameters": {"source": "osm", "categories": ["residential", "commercial", "industrial"]},
                    "reasoning": "Land use constraints are primary factors in site selection"
                },
                {
                    "step": 2,
                    "operation": "proximity_analysis",
                    "description": "Calculate distances to key infrastructure",
                    "parameters": {"features": ["roads", "utilities", "services"], "max_distance": "5km"},
                    "reasoning": "Proximity to infrastructure affects site accessibility and development costs"
                },
                {
                    "step": 3,
                    "operation": "constraint_analysis",
                    "description": "Apply environmental and regulatory constraints",
                    "parameters": {"exclude": ["protected_areas", "flood_zones", "steep_slopes"]},
                    "reasoning": "Regulatory and environmental constraints eliminate unsuitable areas"
                },
                {
                    "step": 4,
                    "operation": "suitability_scoring",
                    "description": "Calculate composite suitability scores",
                    "parameters": {"weights": {"proximity": 0.4, "land_use": 0.3, "constraints": 0.3}},
                    "reasoning": "Weighted scoring combines multiple criteria for final site ranking"
                }
            ]
            
        elif analysis["query_type"] == "buffer_analysis":
            workflow = [
                {
                    "step": 1,
                    "operation": "load_geometry_data",
                    "description": "Load input geometry features",
                    "parameters": {"source": "user_input"},
                    "reasoning": "Source geometry is required for buffer operations"
                },
                {
                    "step": 2,
                    "operation": "buffer_analysis",
                    "description": "Create buffer zones around features",
                    "parameters": {"distance": "1000m", "units": "meters"},
                    "reasoning": "Buffer analysis creates zones of influence around spatial features"
                }
            ]
        
        self.workflow_steps = workflow
        
        self.log_reasoning(
            "workflow_complete",
            f"Generated {len(workflow)} workflow steps for processing",
            "execute_workflow"
        )
        
        return workflow
    
    def execute_workflow_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate execution of a workflow step.
        In real implementation, this would call actual geoprocessing tools.
        """
        self.log_reasoning(
            "step_execution",
            f"Executing step {step['step']}: {step['operation']}",
            step['operation']
        )
        
        # Simulate processing time
        time.sleep(0.5)
        
        # Simulate step result
        result = {
            "step": step['step'],
            "operation": step['operation'],
            "status": "completed",
            "output_data": f"mock_output_{step['operation']}.geojson",
            "processing_time": 0.5,
            "features_processed": 100 + step['step'] * 50
        }
        
        self.log_reasoning(
            "step_completed",
            f"Step {step['step']} completed successfully. Processed {result['features_processed']} features.",
            "continue_to_next_step"
        )
        
        return result

@celery.task(bind=True)
def run_geospatial_task(self, query: str, reasoning_mode: str = "chain_of_thought", 
                       include_visualization: bool = True):
    """
    Main Celery task for geospatial Chain-of-Thought processing.
    
    This task:
    1. Analyzes the user query using Chain-of-Thought reasoning
    2. Generates a geospatial workflow
    3. Executes the workflow steps
    4. Returns results with reasoning logs
    """
    
    try:
        # Initialize the reasoning engine
        reasoner = GeospatialReasoner()
        
        # Update task status to PROGRESS
        self.update_state(
            state='PROGRESS',
            meta={'stage': 'initializing', 'progress': 0, 'reasoning_log': reasoner.reasoning_log}
        )
        
        # Store initial status in Redis
        if redis_client:
            redis_client.hset(
                f"job:{self.request.id}",
                mapping={
                    "status": "PROGRESS",
                    "stage": "query_analysis",
                    "progress": "10"
                }
            )
        
        # Step 1: Analyze the query
        reasoner.log_reasoning(
            "task_start",
            f"Starting geospatial analysis task for query: '{query}'",
            "begin_chain_of_thought_processing"
        )
        
        analysis = reasoner.analyze_query(query)
        
        self.update_state(
            state='PROGRESS',
            meta={'stage': 'workflow_generation', 'progress': 25, 'analysis': analysis, 'reasoning_log': reasoner.reasoning_log}
        )
        
        # Step 2: Generate workflow
        workflow = reasoner.generate_workflow(analysis)
        
        self.update_state(
            state='PROGRESS',
            meta={'stage': 'workflow_execution', 'progress': 40, 'workflow': workflow, 'reasoning_log': reasoner.reasoning_log}
        )
        
        # Step 3: Execute workflow steps
        step_results = []
        total_steps = len(workflow)
        
        for i, step in enumerate(workflow):
            progress = 40 + (i / total_steps) * 50  # Progress from 40% to 90%
            
            self.update_state(
                state='PROGRESS',
                meta={
                    'stage': f'executing_step_{step["step"]}',
                    'progress': progress,
                    'current_step': step,
                    'reasoning_log': reasoner.reasoning_log
                }
            )
            
            # Update Redis with current step
            if redis_client:
                redis_client.hset(
                    f"job:{self.request.id}",
                    mapping={
                        "status": "PROGRESS",
                        "current_step": json.dumps(step),
                        "progress": str(int(progress)),
                        "reasoning_log": json.dumps(reasoner.reasoning_log)
                    }
                )
            
            # Execute the step
            step_result = reasoner.execute_workflow_step(step)
            step_results.append(step_result)
        
        # Step 4: Generate final results
        reasoner.log_reasoning(
            "workflow_completion",
            f"All {len(workflow)} workflow steps completed successfully",
            "generate_final_output"
        )
        
        # Simulate visualization generation if requested
        visualization_data = None
        if include_visualization:
            reasoner.log_reasoning(
                "visualization_generation",
                "Generating visualization data for results",
                "create_maps_and_charts"
            )
            
            visualization_data = {
                "maps": [
                    {
                        "type": "choropleth",
                        "title": f"Results Map for {analysis['query_type']}",
                        "data_url": "mock_map_data.geojson"
                    }
                ],
                "charts": [
                    {
                        "type": "bar",
                        "title": "Processing Statistics",
                        "data": {"steps": len(workflow), "features": sum(r['features_processed'] for r in step_results)}
                    }
                ]
            }
        
        # Final result
        final_result = {
            "task_id": self.request.id,
            "query": query,
            "analysis": analysis,
            "workflow": workflow,
            "step_results": step_results,
            "reasoning_log": reasoner.reasoning_log,
            "visualization": visualization_data,
            "summary": {
                "total_steps": len(workflow),
                "total_features_processed": sum(r['features_processed'] for r in step_results),
                "total_processing_time": sum(r['processing_time'] for r in step_results),
                "success": True
            }
        }
        
        # Update Redis with final result
        if redis_client:
            redis_client.hset(
                f"job:{self.request.id}",
                mapping={
                    "status": "SUCCESS",
                    "progress": "100",
                    "reasoning_log": json.dumps(reasoner.reasoning_log),
                    "workflow_steps": json.dumps(workflow),
                    "result": json.dumps(final_result)
                }
            )
        
        reasoner.log_reasoning(
            "task_completion",
            f"Geospatial analysis task completed successfully. Generated {len(reasoner.reasoning_log)} reasoning steps.",
            "return_results"
        )
        
        return final_result
        
    except Exception as e:
        # Log error reasoning
        error_reasoning = {
            "step": "error_handling",
            "timestamp": time.time(),
            "thought": f"An error occurred during processing: {str(e)}",
            "action": "return_error_result",
            "step_type": "error"
        }
        
        # Update Redis with error status
        if redis_client:
            redis_client.hset(
                f"job:{self.request.id}",
                mapping={
                    "status": "FAILURE",
                    "error": str(e),
                    "reasoning_log": json.dumps([error_reasoning])
                }
            )
        
        # Re-raise the exception for Celery to handle
        raise e

@celery.task
def cleanup_old_jobs():
    """
    Cleanup task to remove old job data from Redis.
    This should be run periodically to prevent memory issues.
    """
    if not redis_client:
        return {"status": "skipped", "reason": "Redis not available"}
    
    try:
        # Get all job keys
        job_keys = redis_client.keys("job:*")
        cleaned_count = 0
        
        # In a real implementation, you'd check timestamps and clean old jobs
        # For now, just return the count
        
        return {
            "status": "completed",
            "total_jobs": len(job_keys),
            "cleaned_jobs": cleaned_count
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}