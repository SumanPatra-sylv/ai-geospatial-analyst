# src/worker/tasks.py - FINAL, COMPLETE, AND CORRECTED VERSION

import os
import time
import redis
import io
import geopandas as gpd
from minio import Minio
from celery.utils.log import get_task_logger

# Import the celery app instance
from src.worker.celery_app import celery

# Import all necessary AI pipeline components
from src.core.executors.workflow_executor import WorkflowExecutor
from src.core.planners.workflow_generator import WorkflowGenerator
from src.core.planners.query_parser import QueryParser

# Redis client for storing job status updates
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

# Initialize logger
logger = get_task_logger(__name__)


@celery.task(name="engine.execute_workflow", bind=True)
def execute_geospatial_workflow(self, task_input: dict):
    """
    Receives a raw user query, parses it, generates a workflow plan, executes it,
    and saves the final result to a MinIO bucket. This is the complete pipeline.
    """
    job_id = self.request.id
    logger.info(f"[Job: {job_id}] Starting full workflow process.")
    
    try:
        # STEP 1: Parse the raw query from the user (No change needed here)
        self.update_state(state='PROGRESS', meta={'stage': 'parsing_query', 'progress': 10})
        raw_query = task_input.get('query')
        if not raw_query:
            raise ValueError("Input 'query' from API cannot be empty.")
        logger.info(f"[Job: {job_id}] Parsing raw query: '{raw_query}'")
        parser = QueryParser()
        parsed_query = parser.parse(raw_query) 
        
        # STEP 2: Generate the list of executable steps (No change needed here)
        self.update_state(state='PROGRESS', meta={'stage': 'generating_plan', 'progress': 25})
        logger.info(f"[Job: {job_id}] Generating workflow plan...")
        generator = WorkflowGenerator()
        workflow_steps = generator.generate_workflow(parsed_query)
        
        # STEP 3: Execute the plan to get the GeoDataFrame result.
        self.update_state(state='PROGRESS', meta={'stage': 'executing_plan', 'progress': 50})
        logger.info(f"[Job: {job_id}] Initializing WorkflowExecutor and executing plan...")
        executor = WorkflowExecutor()

        # --- FIX: The `parsed_query` object is now passed to the executor. ---
        # This is the crucial change that provides the executor with the correct 'location'
        # and other contextual information from the AI's plan.
        result_gdf: gpd.GeoDataFrame = executor.execute_workflow(workflow_steps, parsed_query)
        
        # STEP 4: Initialize MinIO Client (No change needed here)
        self.update_state(state='PROGRESS', meta={'stage': 'connecting_to_storage', 'progress': 85})
        logger.info(f"[Job: {job_id}] Initializing MinIO client to save results.")
        minio_client = Minio(
            os.getenv("MINIO_ENDPOINT", "minio:9000"),
            access_key=os.getenv("MINIO_ROOT_USER"),
            secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
            secure=False
        )
        
        # STEP 5: Save the final result to MinIO (No change needed here)
        self.update_state(state='PROGRESS', meta={'stage': 'preparing_output', 'progress': 90})
        logger.info(f"[Job: {job_id}] Converting {len(result_gdf)} features to GeoJSON.")
        result_geojson_str = result_gdf.to_json()
        result_geojson_bytes = result_geojson_str.encode('utf-8')
        
        bucket_name = "geospatial-results"
        object_name = f"{job_id}.geojson"
        
        if not minio_client.bucket_exists(bucket_name):
            logger.warning(f"[Job: {job_id}] Bucket '{bucket_name}' not found. Creating it.")
            minio_client.make_bucket(bucket_name)

        logger.info(f"[Job: {job_id}] Uploading result to MinIO: s3://{bucket_name}/{object_name}")
        minio_client.put_object(
            bucket_name=bucket_name, object_name=object_name,
            data=io.BytesIO(result_geojson_bytes), length=len(result_geojson_bytes),
            content_type="application/geo+json"
        )
        
        # Update Redis with the final success status
        if redis_client:
            redis_client.hset(f"job:{job_id}", mapping={"status": "SUCCESS", "progress": "100"})
        
        logger.info(f"[Job: {job_id}] Successfully saved result to MinIO.")
        
        # Return a rich dictionary that the frontend can use for a nice display message.
        return {
            "response": f"Analysis complete. Found {len(result_gdf)} features for your query about '{parsed_query.target}' in '{parsed_query.location}'.",
            "thinking_process": "Workflow completed and results saved to storage.",
            "bucket": bucket_name,
            "object_name": object_name
        }
        
    except Exception as e:
        logger.error(f"[Job: {job_id}] Workflow execution FAILED. Error: {e}", exc_info=True)
        if redis_client:
            redis_client.hset(f"job:{job_id}", mapping={"status": "FAILURE", "error": str(e)})
        # Re-raise the exception so Celery properly marks the task as failed
        raise

# This cleanup task is unchanged and preserved from your code.
@celery.task
def cleanup_old_jobs():
    """
    Cleanup task to remove old job data from Redis.
    This should be run periodically to prevent memory issues.
    """
    if not redis_client:
        return {"status": "skipped", "reason": "Redis not available"}
    
    try:
        job_keys = redis_client.keys("job:*")
        # In a real scenario, you'd check timestamps and delete old keys.
        cleaned_count = 0
        return {
            "status": "completed",
            "total_jobs": len(job_keys),
            "cleaned_jobs": cleaned_count
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}