# src/api/routes/jobs.py - FINAL, COMPLETE, AND CORRECTED VERSION
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import redis
import os
from typing import Optional, Dict, Any
from celery.result import AsyncResult
from datetime import datetime
import json
import traceback

### FIX 1: Add the MinIO client library import ###
from minio import Minio

# Create router instance
router = APIRouter()

# --- Redis Client Setup (Unchanged) ---
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

### FIX 2: Add MinIO client setup, similar to the Redis setup ###
try:
    minio_client = Minio(
        os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        secure=False
    )
    MINIO_AVAILABLE = True
    # The bucket name comes from your docker-compose minio-init service
    BUCKET_NAME = "geospatial-results" 
except Exception as e:
    print(f"Warning: MinIO connection failed: {e}")
    minio_client = None
    MINIO_AVAILABLE = False

# --- Pydantic Models (Unchanged) ---
class Query(BaseModel):
    query: str
    reasoning_mode: Optional[str] = "chain_of_thought"
    include_visualization: Optional[bool] = True
    session_id: Optional[str] = None

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

# --- Celery App/Task Getters (Unchanged) ---
def get_celery_app():
    try:
        from src.worker.celery_app import celery as celery_app
        return celery_app
    except ImportError as e:
        print(f"Warning: Could not import Celery app: {e}")
        return None

def get_celery_task():
    try:
        from src.worker.tasks import execute_geospatial_workflow
        return execute_geospatial_workflow
    except ImportError as e:
        print(f"Warning: Could not import Celery task: {e}")
        return None

# --- Job Start Endpoint (Unchanged) ---
@router.post("/start", status_code=202, response_model=JobResponse)
def start_job(query: Query):
    if not REDIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Redis service unavailable.")
    
    execute_workflow_task = get_celery_task()
    if not execute_workflow_task:
        raise HTTPException(status_code=503, detail="Celery worker task is unavailable.")
    
    try:
        plan = {
            "query": query.query,
            "reasoning_mode": query.reasoning_mode,
            "include_visualization": query.include_visualization,
            "session_id": query.session_id
        }
        task = execute_workflow_task.delay(plan)
        
        job_metadata = {
            "status": "PENDING",
            "query": query.query,
            "reasoning_mode": query.reasoning_mode,
            "created_at": datetime.now().isoformat(),
            "workflow_steps": "[]",
            "reasoning_log": "[]"
        }
        redis_client.hset(f"job:{task.id}", mapping=job_metadata)
        
        return JobResponse(
            job_id=task.id,
            status="PENDING",
            message="Geospatial workflow execution started"
        )
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to start job: {str(e)}")

### FIX 3: Add the missing /data endpoint ###
@router.get("/data", summary="Get list of available data files and loaded layers")
async def list_available_data():
    """
    Retrieves a list of available data files from MinIO and loaded layers from Redis.
    This endpoint is called by the sidebar in the frontend.
    """
    if not MINIO_AVAILABLE:
        raise HTTPException(status_code=503, detail="MinIO service is unavailable.")
    if not REDIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Redis service is unavailable.")

    files = []
    try:
        # Check if bucket exists before listing objects
        if not minio_client.bucket_exists(BUCKET_NAME):
             # If the bucket isn't there, the init service might have failed.
             return {"files": [], "loaded_layers": []}

        objects = minio_client.list_objects(BUCKET_NAME, recursive=True)
        for obj in objects:
            files.append({
                "name": obj.object_name,
                "size_kb": round(obj.size / 1024, 2)
            })
    except Exception as e:
        print(f"Error listing files from MinIO: {e}")
        # Don't crash the whole app, just return an empty list for files
        files = []

    # Get the list of currently loaded layers from Redis
    try:
        loaded_layers = list(redis_client.smembers("loaded_layers"))
    except Exception as e:
        print(f"Error getting loaded layers from Redis: {e}")
        loaded_layers = []

    return {"files": files, "loaded_layers": loaded_layers}


# --- All other endpoints are unchanged and correct ---

@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    if not REDIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Redis unavailable.")
    
    celery_app = get_celery_app()
    if not celery_app:
        raise HTTPException(status_code=503, detail="Celery unavailable.")
    
    try:
        task_result = AsyncResult(job_id, app=celery_app)
        redis_data = redis_client.hgetall(f"job:{job_id}")
        
        if not redis_data and task_result.status == 'PENDING':
             raise HTTPException(status_code=404, detail="Job not found.")
        
        response_data = {"job_id": job_id, "status": task_result.status}
        
        if task_result.ready():
            if task_result.successful():
                response_data["result"] = task_result.result
            else:
                response_data["error"] = str(task_result.info)
        elif task_result.state == "PROGRESS":
            response_data["progress"] = task_result.info
        
        return JobStatusResponse(**response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve job status: {str(e)}")

@router.get("/health")
def health_check():
    health_status = {"redis": "unavailable", "celery_task": "unavailable"}
    if REDIS_AVAILABLE:
        health_status["redis"] = "healthy"
    if get_celery_task():
        health_status["celery_task"] = "available"
    return health_status

# The other utility endpoints below are fine and do not need changes.
@router.get("/status-redis/{job_id}")
def get_job_status_redis(job_id: str):
    if not redis_client: raise HTTPException(status_code=503, detail="Redis unavailable")
    status = redis_client.hgetall(f"job:{job_id}")
    if not status: raise HTTPException(status_code=404, detail="Job not found")
    return status

@router.get("/jobs")
def list_jobs():
    if not redis_client: raise HTTPException(status_code=503, detail="Redis unavailable")
    job_keys = redis_client.keys("job:*")
    jobs = [redis_client.hgetall(key) for key in job_keys]
    return {"jobs": jobs, "total": len(jobs)}

@router.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    celery_app = get_celery_app()
    if celery_app: celery_app.control.revoke(job_id, terminate=True)
    if redis_client: redis_client.delete(f"job:{job_id}")
    return {"message": f"Job {job_id} cancellation requested"}