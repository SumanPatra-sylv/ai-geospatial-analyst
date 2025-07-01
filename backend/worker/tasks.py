# backend/worker/tasks.py

from .celery_app import celery
import redis
import os

# Import the compiled agent graph from our core logic
from backend.core.agent import agent_graph
from langchain_core.messages import HumanMessage

# Set up a Redis client for this worker process
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0, decode_responses=True)

@celery.task(bind=True)
def run_geospatial_task(self, user_query: str):
    """
    The main Celery task that runs the LangGraph agent.
    `bind=True` gives us access to `self`, the task instance.
    """
    job_id = self.request.id
    
    # Update the job status in Redis to show we've started
    redis_client.hset(f"job:{job_id}", mapping={"status": "RUNNING"})
    
    try:
        # This is where the magic happens: we invoke the agent
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "data_store": {} # The agent starts with an empty data_store for each job
        }
        
        # The invoke call will block until the agent is done
        final_state = agent_graph.invoke(initial_state, {"recursion_limit": 15})
        
        # Get the final response from the agent
        result = final_state["messages"][-1].content
        
        # Update Redis with the final, successful status and result
        redis_client.hset(f"job:{job_id}", mapping={"status": "COMPLETED", "result": result})
        
        return {"status": "COMPLETED", "result": result}

    except Exception as e:
        # If anything goes wrong, update Redis with a FAILED status and error message
        error_message = f"An error occurred: {str(e)}"
        redis_client.hset(f"job:{job_id}", mapping={"status": "FAILED", "error": error_message})
        
        return {"status": "FAILED", "error": error_message}