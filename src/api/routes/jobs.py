# src/api/routes/jobs.py - Enhanced with Celery AsyncResult status checking
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import redis
import os
from typing import Optional, Dict, Any
from celery.result import AsyncResult

# Create router instance
router = APIRouter()

# Connect to the Redis service
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"), 
        port=6379, 
        db=0, 
        decode_responses=True
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Redis connection failed: {e}")
    redis_client = None
    REDIS_AVAILABLE = False

class Query(BaseModel):
    query: str
    reasoning_mode: Optional[str] = "chain_of_thought"
    include_visualization: Optional[bool] = True

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[Any, Any]] = None
    reasoning_log: Optional[list] = None
    progress: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

def get_celery_app():
    """
    Lazy import of the Celery app to avoid startup issues.
    """
    try:
        from src.worker.celery_app import celery as celery_app
        return celery_app
    except ImportError as e:
        print(f"Warning: Could not import Celery app: {e}")
        return None

def get_celery_task():
    """
    Lazy import of the Celery task to avoid startup issues.
    """
    try:
        from src.worker.tasks import run_geospatial_task
        return run_geospatial_task
    except ImportError as e:
        print(f"Warning: Could not import Celery task: {e}")
        return None

@router.post("/start", status_code=202, response_model=JobResponse)
def start_job(query: Query):
    """
    Receives a query, dispatches it to the Celery worker for geospatial analysis,
    and returns a job ID for tracking the Chain-of-Thought reasoning process.
    """
    if not REDIS_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Redis service unavailable. Cannot process geospatial jobs."
        )
    
    # Get the Celery task (lazy import)
    run_geospatial_task = get_celery_task()
    if not run_geospatial_task:
        raise HTTPException(
            status_code=503, 
            detail="Celery worker unavailable. Cannot process geospatial analysis."
        )
    
    try:
        # Prepare task parameters for geospatial workflow generation
        task_params = {
            "query": query.query,
            "reasoning_mode": query.reasoning_mode,
            "include_visualization": query.include_visualization
        }
        
        # Dispatch task to Celery queue
        task = run_geospatial_task.delay(**task_params)
        
        # Store initial job metadata in Redis
        job_metadata = {
            "status": "PENDING",
            "query": query.query,
            "reasoning_mode": query.reasoning_mode,
            "created_at": str(os.getpid()),  # Using pid as timestamp placeholder
            "workflow_steps": "[]",
            "reasoning_log": "[]"
        }
        redis_client.hset(f"job:{task.id}", mapping=job_metadata)
        
        return JobResponse(
            job_id=task.id,
            status="PENDING",
            message="Geospatial workflow generation started"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to start geospatial analysis job: {str(e)}"
        )

@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """
    Poll for the status of a Celery task using AsyncResult.
    Returns comprehensive status including Chain-of-Thought reasoning logs
    and geospatial workflow progress.
    """
    if not REDIS_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Redis service unavailable. Cannot retrieve job status."
        )
    
    # Get Celery app for AsyncResult
    celery_app = get_celery_app()
    if not celery_app:
        raise HTTPException(
            status_code=503,
            detail="Celery app unavailable. Cannot check task status."
        )
    
    try:
        # Use Celery's AsyncResult for real-time status
        task_result = AsyncResult(job_id, app=celery_app)
        
        # Get additional metadata from Redis
        redis_data = redis_client.hgetall(f"job:{job_id}")
        if not redis_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Prepare comprehensive response
        response_data = {
            "job_id": job_id,
            "status": task_result.status,
            "result": None,
            "reasoning_log": None,
            "progress": None,
            "error": None
        }
        
        # Handle different task states
        if task_result.ready():
            if task_result.successful():
                result = task_result.result
                response_data["result"] = result
                
                # Extract reasoning log if available
                if isinstance(result, dict) and "reasoning_log" in result:
                    response_data["reasoning_log"] = result["reasoning_log"]
                    
            else:
                # Task failed
                response_data["error"] = str(task_result.info)
                
        elif task_result.state == "PROGRESS":
            # Task is in progress, get progress info
            progress_info = task_result.info
            response_data["progress"] = progress_info
            
            # Check if reasoning log is being updated
            if "reasoning_step" in progress_info:
                response_data["reasoning_log"] = progress_info.get("reasoning_log", [])
        
        # Add Redis metadata
        if "reasoning_log" in redis_data and redis_data["reasoning_log"] != "[]":
            try:
                import json
                response_data["reasoning_log"] = json.loads(redis_data["reasoning_log"])
            except json.JSONDecodeError:
                pass
        
        return JobStatusResponse(**response_data)
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve job status: {str(e)}"
        )

@router.get("/status-redis/{job_id}")
def get_job_status_redis(job_id: str):
    """
    Alternative endpoint that retrieves status directly from Redis.
    Useful for debugging or when Celery AsyncResult is not available.
    """
    if not REDIS_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Redis service unavailable. Cannot retrieve job status."
        )
    
    try:
        status = redis_client.hgetall(f"job:{job_id}")
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Parse JSON fields if they exist
        try:
            import json
            if "workflow_steps" in status:
                status["workflow_steps"] = json.loads(status["workflow_steps"])
            if "reasoning_log" in status:
                status["reasoning_log"] = json.loads(status["reasoning_log"])
        except json.JSONDecodeError:
            pass
            
        return status
    
    except redis.RedisError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve job status from Redis: {str(e)}"
        )

@router.get("/jobs")
def list_jobs():
    """
    List all jobs in the system with their current status.
    Useful for monitoring and debugging geospatial workflow processes.
    """
    if not REDIS_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Redis service unavailable."
        )
    
    try:
        # Get all job keys
        job_keys = redis_client.keys("job:*")
        jobs = []
        
        for key in job_keys:
            job_id = key.replace("job:", "")
            job_data = redis_client.hgetall(key)
            job_data["job_id"] = job_id
            jobs.append(job_data)
        
        return {"jobs": jobs, "total": len(jobs)}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list jobs: {str(e)}"
        )

@router.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    """
    Cancel a running job and clean up associated resources.
    """
    celery_app = get_celery_app()
    if celery_app:
        try:
            # Revoke the Celery task
            celery_app.control.revoke(job_id, terminate=True)
        except Exception as e:
            print(f"Warning: Could not revoke Celery task {job_id}: {e}")
    
    # Clean up Redis data
    if REDIS_AVAILABLE:
        try:
            redis_client.delete(f"job:{job_id}")
        except Exception as e:
            print(f"Warning: Could not delete Redis data for job {job_id}: {e}")
    
    return {"message": f"Job {job_id} cancelled"}

@router.get("/health")
def health_check():
    """
    Check the health of Redis, Celery, and geospatial processing components.
    """
    health_status = {
        "redis": "unavailable",
        "celery_app": "unavailable",
        "celery_task": "unavailable",
        "geospatial_tools": "checking"
    }
    
    # Check Redis
    if REDIS_AVAILABLE:
        try:
            redis_client.ping()
            health_status["redis"] = "healthy"
        except:
            health_status["redis"] = "error"
    
    # Check Celery app
    celery_app = get_celery_app()
    if celery_app:
        health_status["celery_app"] = "available"
    
    # Check Celery task availability
    celery_task = get_celery_task()
    if celery_task:
        health_status["celery_task"] = "available"
    
    # Check geospatial tools (placeholder for actual checks)
    try:
        # You would check imports like geopandas, rasterio, etc. here
        health_status["geospatial_tools"] = "available"
    except:
        health_status["geospatial_tools"] = "error"
    
    return health_status