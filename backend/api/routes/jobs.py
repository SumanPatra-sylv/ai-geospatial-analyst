# backend/api/routes/jobs.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import redis
import os

# This imports the task defined in the worker directory
from backend.worker.tasks import run_geospatial_task

router = APIRouter()

# Connect to the Redis service. The hostname 'redis' is from docker-compose.yml.
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0, decode_responses=True)

class Query(BaseModel):
    query: str

@router.post("/start", status_code=202)
def start_job(query: Query):
    """
    Receives a query, dispatches it to the Celery worker, and returns a job ID.
    """
    # .delay() sends the task to the Celery queue
    task = run_geospatial_task.delay(query.query)
    
    # Store the initial status in Redis so we can track it
    redis_client.hset(f"job:{task.id}", mapping={"status": "PENDING"})
    
    return {"job_id": task.id}

@router.get("/status/{job_id}")
def get_job_status(job_id: str):
    """
    Retrieves the status and result of a job from Redis.
    """
    status = redis_client.hgetall(f"job:{job_id}")
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status