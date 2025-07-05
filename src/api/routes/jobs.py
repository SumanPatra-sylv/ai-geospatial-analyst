# src/api/routes/jobs.py - FINAL, ALIGNED WITH CONDUCTOR AGENT ARCHITECTURE

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
import redis
import os
from typing import Optional, Dict, Any, List
from celery.result import AsyncResult
from datetime import datetime
import json
import uuid
from minio import Minio
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter()

# --- Enums for better type safety (Excellent, no changes needed) ---
class ReasoningMode(str, Enum):
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    REACT = "react"  # Reasoning-Acting-Observation loop
    SELF_CONSISTENCY = "self_consistency"

class TaskComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class JobStatus(str, Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    PROGRESS = "PROGRESS"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"

# --- Enhanced Redis Client Setup (Excellent, no changes needed) ---
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"), 
        port=int(os.getenv("REDIS_PORT", 6379)), 
        db=int(os.getenv("REDIS_DB", 0)), 
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
        health_check_interval=30
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis connection established successfully")
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None
    REDIS_AVAILABLE = False

# --- Enhanced MinIO Client Setup (Excellent, no changes needed) ---
try:
    minio_client = Minio(
        os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
    )
    MINIO_AVAILABLE = True
    BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "geospatial-results")
    
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)
    
    logger.info("MinIO connection established successfully")
except Exception as e:
    logger.error(f"MinIO connection failed: {e}")
    minio_client = None
    MINIO_AVAILABLE = False

# --- Enhanced Pydantic Models (Excellent, most are preserved) ---
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant/system)")
    content: str = Field(..., description="Content of the message")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('role')
    def validate_role(cls, v):
        allowed_roles = ['user', 'assistant', 'system']
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v

class SpatialContext(BaseModel):
    bbox: Optional[List[float]] = Field(None, description="Bounding box [minx, miny, maxx, maxy]")
    crs: Optional[str] = Field("EPSG:4326", description="Coordinate Reference System")
    region: Optional[str] = Field(None, description="Named region or location")
    scale: Optional[str] = Field(None, description="Analysis scale (local/regional/global)")

class Query(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language query for geospatial analysis")
    history: List[ChatMessage] = Field(default_factory=list, description="Chat history for context")
    reasoning_mode: ReasoningMode = Field(default=ReasoningMode.CHAIN_OF_THOUGHT, description="LLM reasoning strategy")
    include_visualization: bool = Field(default=True, description="Generate visualizations and maps")
    session_id: Optional[str] = Field(None, description="Session identifier for tracking")
    spatial_context: Optional[SpatialContext] = Field(None, description="Spatial context for the query")
    max_workflow_steps: int = Field(default=10, ge=1, le=50, description="Maximum number of workflow steps")
    priority: int = Field(default=1, ge=1, le=5, description="Job priority (1=highest, 5=lowest)")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing the job")
    
    @validator('history')
    def validate_history_length(cls, v):
        if len(v) > 100:
            raise ValueError("Chat history cannot exceed 100 messages")
        return v

# --- FIX: JobResponse model simplified as per instructions ---
class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: Optional[str] = None

class WorkflowStep(BaseModel):
    step_id: str
    operation: str
    parameters: Dict[str, Any]
    input_data: List[str]
    output_data: List[str]
    reasoning: str
    status: str = "pending"
    execution_time: Optional[float] = None
    error: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: Optional[Dict[Any, Any]] = None
    reasoning_log: Optional[List[str]] = None
    workflow_steps: Optional[List[WorkflowStep]] = None
    progress: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    execution_time: Optional[float] = None
    resource_usage: Optional[Dict[str, Any]] = None

# --- Utility Functions ---
def get_celery_app():
    try:
        from src.worker.celery_app import celery as celery_app
        return celery_app
    except ImportError as e:
        logger.error(f"Could not import Celery app: {e}")
        return None

# --- FIX: Using the new 'execute_agentic_workflow' task name ---
def get_celery_task():
    try:
        from src.worker.tasks import execute_agentic_workflow
        return execute_agentic_workflow
    except ImportError as e:
        logger.error(f"Could not import Celery task: {e}")
        return None

# --- FIX: Simplified function to store only the initial job record ---
def create_initial_job_record(job_id: str, query: Query):
    """Stores the essential, initial job metadata in Redis."""
    if not REDIS_AVAILABLE: 
        return
    try:
        initial_metadata = {
            "job_id": job_id,
            "status": JobStatus.PENDING.value,
            "query": query.query,
            "session_id": query.session_id or "",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        # Store main job data
        redis_client.hset(f"job:{job_id}", mapping=initial_metadata)
        
        # Store history separately for cleanliness and to avoid polluting the main hash
        if query.history:
            history_key = f"job:{job_id}:history"
            history_data = [msg.dict() for msg in query.history]
            redis_client.set(history_key, json.dumps(history_data), ex=86400) # 24 hours TTL
        
        # Set a Time-To-Live for the job record to prevent Redis from filling up
        redis_client.expire(f"job:{job_id}", 86400) # 24 hours
        logger.info(f"Initial job record created for job {job_id}")
    except Exception as e:
        logger.error(f"Failed to store initial job metadata for job {job_id}: {e}")

# --- Main Endpoints ---
@router.post("/start", status_code=202, response_model=JobResponse)
async def start_job(query: Query):
    """
    Job start endpoint. It validates the input, creates a job ID, creates an initial
    record, and dispatches the task to the agentic worker. It does NOT do any analysis itself.
    """
    if not REDIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Redis service is unavailable")
    
    execute_workflow_task = get_celery_task()
    if not execute_workflow_task:
        raise HTTPException(status_code=503, detail="Celery worker task is unavailable")
    
    try:
        job_id = str(uuid.uuid4())
        
        # --- FIX: SIMPLIFY THE TASK INPUT for the Conductor Agent ---
        # The agent only needs the raw query and the context of the chat history.
        task_input = {
            "query": query.query,
            "history": [msg.dict() for msg in query.history],
        }
        
        # Create the initial, basic record for the job in Redis.
        create_initial_job_record(job_id, query)

        # Dispatch the task to the Celery worker.
        execute_workflow_task.apply_async(args=[task_input], task_id=job_id)
        
        logger.info(f"Job {job_id} created and dispatched to Conductor Agent.")
        
        # --- FIX: Return a simple, immediate response. ---
        # The `estimated_duration` and `complexity` have been removed because
        # only the worker can determine them accurately.
        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Request accepted and forwarded to the Conductor Agent for analysis."
        )
        
    except Exception as e:
        logger.error(f"Failed to create and dispatch job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while creating the job.")

# --- All other endpoints below are for STATUS and DATA RETRIEVAL ---
# They are excellent as-is and require NO CHANGES.

@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Enhanced job status endpoint with comprehensive information"""
    if not REDIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    
    celery_app = get_celery_app()
    if not celery_app:
        raise HTTPException(status_code=503, detail="Celery unavailable")
    
    try:
        task_result = AsyncResult(job_id, app=celery_app)
        redis_data = redis_client.hgetall(f"job:{job_id}")
        
        if not redis_data and task_result.status == 'PENDING':
            raise HTTPException(status_code=404, detail="Job not found")
        
        response_data = {
            "job_id": job_id,
            "status": JobStatus(task_result.status) if task_result.status else JobStatus.PENDING
        }
        
        if redis_data:
            response_data.update({
                "created_at": datetime.fromisoformat(redis_data.get("created_at", datetime.now().isoformat())),
                "updated_at": datetime.fromisoformat(redis_data.get("updated_at", datetime.now().isoformat()))
            })
            if redis_data.get("workflow_steps"):
                try:
                    workflow_steps = json.loads(redis_data["workflow_steps"])
                    response_data["workflow_steps"] = [WorkflowStep(**step) for step in workflow_steps]
                except (json.JSONDecodeError, TypeError): pass
            if redis_data.get("reasoning_log"):
                try: response_data["reasoning_log"] = json.loads(redis_data["reasoning_log"])
                except (json.JSONDecodeError, TypeError): pass
        
        if task_result.ready():
            if task_result.successful():
                response_data["result"] = task_result.result
            else:
                response_data["error"] = str(task_result.info)
        elif task_result.state == "PROGRESS":
            response_data["progress"] = task_result.info
        
        return JobStatusResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Failed to retrieve job status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve job status: {str(e)}")

@router.get("/data", summary="Get list of available data files and loaded layers")
async def list_available_data():
    """Enhanced data listing with metadata"""
    if not MINIO_AVAILABLE: raise HTTPException(status_code=503, detail="MinIO service unavailable")
    if not REDIS_AVAILABLE: raise HTTPException(status_code=503, detail="Redis service unavailable")

    files = []
    try:
        if minio_client.bucket_exists(BUCKET_NAME):
            objects = minio_client.list_objects(BUCKET_NAME, recursive=True)
            for obj in objects:
                files.append({
                    "name": obj.object_name, "size_kb": round(obj.size / 1024, 2),
                    "last_modified": obj.last_modified.isoformat(), "etag": obj.etag,
                    "content_type": obj.content_type or "application/octet-stream"
                })
    except Exception as e:
        logger.error(f"Error listing files from MinIO: {e}")
        files = []

    try:
        loaded_layers = list(redis_client.smembers("loaded_layers"))
        layer_metadata = {layer: redis_client.hgetall(f"layer:{layer}") for layer in loaded_layers if redis_client.hgetall(f"layer:{layer}")}
    except Exception as e:
        logger.error(f"Error getting loaded layers from Redis: {e}")
        loaded_layers, layer_metadata = [], {}

    return {
        "files": files, "loaded_layers": loaded_layers, "layer_metadata": layer_metadata,
        "total_files": len(files), "total_size_mb": round(sum(f["size_kb"] for f in files) / 1024, 2)
    }

@router.get("/jobs", summary="List all jobs with filtering and pagination")
async def list_jobs(status: Optional[JobStatus] = None, session_id: Optional[str] = None, limit: int = 50, offset: int = 0):
    if not REDIS_AVAILABLE: raise HTTPException(status_code=503, detail="Redis unavailable")
    try:
        job_keys = redis_client.keys("job:*:*") # More efficient scan
        jobs = []
        for key in job_keys:
            if ":history" in key: continue
            job_data = redis_client.hgetall(key)
            if not job_data: continue
            if status and job_data.get("status") != status.value: continue
            if session_id and job_data.get("session_id") != session_id: continue
            if job_data.get("created_at"):
                try: job_data["created_at"] = datetime.fromisoformat(job_data["created_at"])
                except ValueError: pass
            jobs.append(job_data)
        
        jobs.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)
        total = len(jobs)
        paginated_jobs = jobs[offset:offset + limit]
        
        return {"jobs": paginated_jobs, "total": total, "limit": limit, "offset": offset, "has_more": offset + limit < total}
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")

@router.get("/jobs/{job_id}/history", summary="Get chat history for a specific job")
async def get_job_history(job_id: str):
    if not REDIS_AVAILABLE: raise HTTPException(status_code=503, detail="Redis unavailable")
    try:
        history_data = redis_client.get(f"job:{job_id}:history")
        if not history_data:
            if not redis_client.exists(f"job:{job_id}"):
                raise HTTPException(status_code=404, detail="Job not found")
            return {"history": [], "message": "No chat history available for this job"}
        history = json.loads(history_data)
        return {"history": history, "total_messages": len(history)}
    except json.JSONDecodeError: raise HTTPException(status_code=500, detail="Invalid history data format")
    except Exception as e:
        logger.error(f"Error retrieving job history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve job history: {str(e)}")

@router.delete("/jobs/{job_id}", summary="Cancel and delete a job")
async def cancel_job(job_id: str):
    try:
        celery_app = get_celery_app()
        if celery_app: celery_app.control.revoke(job_id, terminate=True)
        if REDIS_AVAILABLE:
            redis_client.delete(f"job:{job_id}", f"job:{job_id}:history")
            redis_client.zrem("job_priority_queue", job_id)
        logger.info(f"Job {job_id} cancelled and cleaned up")
        return {"message": f"Job {job_id} cancellation requested and data cleaned up"}
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

@router.get("/health", summary="Comprehensive health check")
async def health_check():
    health_status = {"redis": "unavailable", "minio": "unavailable", "celery_task": "unavailable", "timestamp": datetime.now().isoformat()}
    if REDIS_AVAILABLE:
        try:
            redis_client.ping()
            health_status["redis"] = "healthy"
        except Exception: health_status["redis"] = "unhealthy"
    if MINIO_AVAILABLE:
        try:
            minio_client.list_buckets()
            health_status["minio"] = "healthy"
        except Exception: health_status["minio"] = "unhealthy"
    if get_celery_task(): health_status["celery_task"] = "available"
    return health_status