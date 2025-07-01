# backend/api/routes/jobs.py - FIXED
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import redis
import os
from typing import Optional

# Create router instance
router = APIRouter()

# Connect to the Redis service. The hostname 'redis' is from docker-compose.yml.
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"), 
        port=6379, 
        db=0, 
        decode_responses=True
    )
    # Test the connection
    redis_client.ping()
    REDIS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Redis connection failed: {e}")
    redis_client = None
    REDIS_AVAILABLE = False

class Query(BaseModel):
    query: str

def get_celery_task():
    """
    Lazy import of the Celery task to avoid startup issues.
    This function will import the task only when it's actually needed.
    """
    try:
        from backend.worker.tasks import run_geospatial_task
        return run_geospatial_task
    except ImportError as e:
        print(f"Warning: Could not import Celery task: {e}")
        return None

@router.post("/start", status_code=202)
def start_job(query: Query):
    """
    Receives a query, dispatches it to the Celery worker, and returns a job ID.
    """
    if not REDIS_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Redis service unavailable. Cannot process jobs."
        )
    
    # Get the Celery task (lazy import)
    run_geospatial_task = get_celery_task()
    if not run_geospatial_task:
        raise HTTPException(
            status_code=503, 
            detail="Celery worker unavailable. Cannot process jobs."
        )
    
    try:
        # .delay() sends the task to the Celery queue
        task = run_geospatial_task.delay(query.query)
        
        # Store the initial status in Redis so we can track it
        redis_client.hset(f"job:{task.id}", mapping={"status": "PENDING"})
        
        return {"job_id": task.id}
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to start job: {str(e)}"
        )

@router.get("/status/{job_id}")
def get_job_status(job_id: str):
    """
    Retrieves the status and result of a job from Redis.
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
        return status
    
    except redis.RedisError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve job status: {str(e)}"
        )

@router.get("/health")
def health_check():
    """
    Check the health of Redis and Celery connections
    """
    health_status = {
        "redis": "unavailable",
        "celery": "unavailable"
    }
    
    # Check Redis
    if REDIS_AVAILABLE:
        try:
            redis_client.ping()
            health_status["redis"] = "healthy"
        except:
            health_status["redis"] = "error"
    
    # Check Celery task availability
    celery_task = get_celery_task()
    if celery_task:
        health_status["celery"] = "available"
    
    return health_status