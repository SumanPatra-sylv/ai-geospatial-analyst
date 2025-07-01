# backend/worker/tasks.py - FIXED
from .celery_app import celery
import redis
import os
import logging
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

# Set up Redis client with error handling
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"), 
        port=int(os.getenv("REDIS_PORT", "6379")), 
        db=int(os.getenv("REDIS_DB", "0")), 
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    # Test the connection
    redis_client.ping()
    logger.info("Redis connection established successfully")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_client = None

def update_job_status(job_id: str, status: str, **additional_data):
    """Helper function to update job status in Redis"""
    if not redis_client:
        logger.warning(f"Cannot update job {job_id} status: Redis unavailable")
        return
    
    try:
        data = {"status": status, **additional_data}
        redis_client.hset(f"job:{job_id}", mapping=data)
        logger.info(f"Updated job {job_id} status to {status}")
    except Exception as e:
        logger.error(f"Failed to update job {job_id} status: {e}")

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def run_geospatial_task(self, user_query: str) -> Dict[str, Any]:
    """
    The main Celery task that runs the LangGraph agent.
    `bind=True` gives us access to `self`, the task instance.
    
    Args:
        user_query: The user's geospatial query to process
        
    Returns:
        Dict containing status and result/error information
    """
    job_id = self.request.id
    logger.info(f"Starting geospatial task {job_id} with query: {user_query[:100]}...")
    
    # Update the job status in Redis to show we've started
    update_job_status(job_id, "RUNNING", started_at=str(self.request.eta or "now"))
    
    try:
        # Import the agent graph here, just before it's needed.
        # This prevents import errors during worker startup
        logger.info("Importing agent graph...")
        from backend.core.agent import agent_graph
        from langchain_core.messages import HumanMessage
        
        logger.info("Agent graph imported successfully, invoking...")
        
        # This is where the magic happens: we invoke the agent
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "data_store": {}  # The agent starts with an empty data_store for each job
        }
        
        # The invoke call will block until the agent is done
        final_state = agent_graph.invoke(
            initial_state, 
            config={"recursion_limit": 15}
        )
        
        # Get the final response from the agent
        if not final_state.get("messages"):
            raise ValueError("Agent returned no messages")
            
        result = final_state["messages"][-1].content
        logger.info(f"Task {job_id} completed successfully")
        
        # Update Redis with the final, successful status and result
        update_job_status(
            job_id, 
            "COMPLETED", 
            result=result,
            completed_at=str(self.request.eta or "now"),
            message_count=len(final_state["messages"])
        )
        
        return {"status": "COMPLETED", "result": result}

    except ImportError as e:
        # Handle import errors specifically
        error_message = f"Failed to import required modules: {str(e)}"
        logger.error(f"Import error in task {job_id}: {error_message}")
        
        update_job_status(job_id, "FAILED", error=error_message, error_type="ImportError")
        return {"status": "FAILED", "error": error_message}
    
    except Exception as e:
        # Handle all other errors
        error_message = f"An error occurred: {str(e)}"
        logger.error(f"Task {job_id} failed: {error_message}")
        
        # Update Redis with a FAILED status and error message
        update_job_status(
            job_id, 
            "FAILED", 
            error=error_message,
            error_type=type(e).__name__,
            failed_at=str(self.request.eta or "now")
        )
        
        # Re-raise for Celery's retry mechanism if it's a retryable error
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying task {job_id} (attempt {self.request.retries + 1}/{self.max_retries})")
            raise self.retry(exc=e)
        
        return {"status": "FAILED", "error": error_message}

@celery.task
def health_check() -> Dict[str, str]:
    """
    Simple health check task for monitoring worker status
    """
    try:
        # Test Redis connection
        if redis_client:
            redis_client.ping()
            redis_status = "healthy"
        else:
            redis_status = "unavailable"
        
        # Test agent import
        try:
            from backend.core.agent import agent_graph
            agent_status = "available"
        except ImportError:
            agent_status = "unavailable"
        
        return {
            "worker": "healthy",
            "redis": redis_status,
            "agent": agent_status
        }
    except Exception as e:
        return {
            "worker": "error",
            "error": str(e)
        }

@celery.task
def cleanup_old_jobs(max_age_hours: int = 24) -> Dict[str, Any]:
    """
    Clean up old job data from Redis
    
    Args:
        max_age_hours: Maximum age of jobs to keep (default 24 hours)
    """
    if not redis_client:
        return {"status": "failed", "error": "Redis unavailable"}
    
    try:
        # This is a basic implementation - in production you'd want more sophisticated cleanup
        # based on job timestamps
        pattern = "job:*"
        keys = redis_client.keys(pattern)
        
        # For now, just return count - actual cleanup logic would go here
        return {
            "status": "completed",
            "jobs_found": len(keys),
            "message": "Cleanup task completed (monitoring only)"
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)}