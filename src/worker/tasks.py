# src/worker/tasks.py - CONDUCTOR AGENT IMPLEMENTATION WITH FOLLOW-UP Q&A

import os
import time
import redis
import io
import json
import requests  # Added for LLM API calls
import geopandas as gpd
from minio import Minio
from celery.utils.log import get_task_logger
from typing import Dict, Any, List, Optional

# Import the celery app instance
from src.worker.celery_app import celery

# Import all necessary AI pipeline components
from src.core.executors.workflow_executor import WorkflowExecutor, LocationNotFoundError
from src.core.planners.workflow_generator import WorkflowGenerator
from src.core.planners.query_parser import QueryParser, QueryParserError
from src.core.knowledge_base import KnowledgeBase # Step 1: Add Necessary Imports

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

# ================================
# CONDUCTOR AGENT SYSTEM PROMPT - UPDATED
# ================================

# Step 3: Upgrade the CONDUCTOR_SYSTEM_PROMPT (Action 3)
# REPLACE THE ENTIRE CONDUCTOR_SYSTEM_PROMPT STRING
CONDUCTOR_SYSTEM_PROMPT = """You are a master conductor for an AI Geospatial Analyst. Your job is to analyze the user's query, conversation history, AND the provided real-time system context to decide which tool to use.

**[RETRIEVED SYSTEM CONTEXT]**
{system_context}
**[END RETRIEVED SYSTEM CONTEXT]**

**TOOLS AVAILABLE:**
1.  `geospatial_tool`: Use for any request that involves finding or analyzing geographic data.
2.  `result_qa_tool`: Use ONLY for follow-up questions about the last analysis result.
3.  `conversational_tool`: Use for greetings or general questions.

**YOUR RESPONSE MUST BE A SINGLE JSON OBJECT WITH ONE KEY:**
- `tool_to_use`: (string) Must be one of ["geospatial_tool", "result_qa_tool", "conversational_tool"].

**Conversation History:**
{chat_history}

**Latest User Query:**
{user_query}

Now, provide your JSON decision.
"""

# ================================
# UTILITY FUNCTIONS
# ================================

def get_minio_client():
    """Helper function to get MinIO client instance"""
    return Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.getenv("MINIO_ROOT_USER"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
        secure=False
    )

# Step 2: Add the find_special_command Helper Function (Action 5)
# ADD THIS HELPER FUNCTION (ideally after get_minio_client)
def find_special_command(user_query: str) -> Optional[Dict[str, Any]]:
    """Checks for hardcoded commands to bypass the LLM for common UI actions."""
    try:
        # Construct an absolute path to the JSON file from this file's location
        dir_path = os.path.dirname(os.path.realpath(__file__))
        json_path = os.path.join(dir_path, '..', 'core', 'planners', 'special_commands.json')
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            for command in data.get("commands", []):
                if command.get("user_phrase") == user_query:
                    return command
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # Log a warning but don't crash the worker if this non-critical file is missing/broken
        logger.warning(f"Could not check for special commands. Reason: {e}")
        return None
    return None

# ================================
# SPECIALIZED TOOL FUNCTIONS
# ================================

def geospatial_planning_and_execution_tool(user_query: str, job_id: str) -> Dict[str, Any]:
    """
    Encapsulates the entire geospatial pipeline and returns rich artifacts and reasoning.
    """
    logger.info(f"Executing geospatial analysis for query: '{user_query}'")
    cot_log = []  # Initialize the Chain-of-Thought log

    try:
        cot_log.append("1. Parsing user query into a structured format.")
        parser = QueryParser()
        parsed_query = parser.parse(user_query)
        
        cot_log.append(f"2. Asking AI Planner to generate a reasoned workflow for target '{parsed_query.target}'.")
        generator = WorkflowGenerator()
        generation_result = generator.generate_workflow(parsed_query)
        workflow_plan = generation_result.get("plan", [])
        true_chain_of_thought = generation_result.get("reasoning", "No reasoning was provided by the planner.")
        cot_log.append("--- AI Planner's Reasoning ---\n" + true_chain_of_thought)
        cot_log.append("3. Executing the workflow using geospatial tools...")
        executor = WorkflowExecutor()
        result_gdf = executor.execute_workflow(workflow_plan, parsed_query)
        feature_count = len(result_gdf)
        cot_log.append(f"4. Execution complete. Found {feature_count} features.")

        # Save the plan and results as artifacts in MinIO
        try:
            minio_client = get_minio_client()
            bucket_name = "geospatial-results"
            
            # Ensure bucket exists
            if not minio_client.bucket_exists(bucket_name):
                minio_client.make_bucket(bucket_name)
            
            # Save plan
            plan_object_name = f"{job_id}_plan.json"
            plan_bytes = json.dumps(workflow_plan, indent=2).encode('utf-8')
            minio_client.put_object(bucket_name, plan_object_name, io.BytesIO(plan_bytes), len(plan_bytes), 'application/json')
            
            # Save result
            result_object_name = f"{job_id}_result.geojson"
            result_bytes = result_gdf.to_json().encode('utf-8')
            minio_client.put_object(bucket_name, result_object_name, io.BytesIO(result_bytes), len(result_bytes), 'application/geo+json')
            cot_log.append(f"5. Saved artifacts to MinIO: '{plan_object_name}' and '{result_object_name}'.")
            
            artifacts = {"plan_file": plan_object_name, "result_file": result_object_name}
        except Exception as minio_error:
            logger.warning(f"Failed to save artifacts to MinIO: {minio_error}")
            artifacts = {}
            cot_log.append("5. Warning: Failed to save artifacts to MinIO.")

        # 1. UPDATE THE SUCCESS RETURN STATEMENT
        # NEW, CORRECTED VERSION
        response_data = {
            "response": f"Analysis complete! I found {feature_count} features for '{parsed_query.target}' in '{parsed_query.location}'.",
            "thinking_process": {"summary": "Successfully executed the geospatial workflow.", "chain_of_thought": cot_log},
            "artifacts": artifacts
        }
        return {
            "status": "SUCCESS", # Use uppercase for consistency
            **response_data 
        }
    
    except QueryParserError as e:
        # 2. UPDATE THE ERROR RETURN STATEMENTS
        # NEW, CORRECTED VERSION
        return {
            "status": "ERROR",
            "response": f"I couldn't understand your query. {str(e)}",
            "thinking_process": {"summary": "Failed to parse the user's query.", "chain_of_thought": [f"Error: {str(e)}"]}
        }
    
    except LocationNotFoundError as e:
        # FLATTENED AND CORRECTED VERSION
        return {
            "status": "ERROR",
            "response": f"I couldn't find the location '{e.location_name}'. Please check the spelling or try a larger city.",
            "thinking_process": {"summary": "Failed to find the specified location.", "chain_of_thought": [f"Error: The geocoding service could not resolve '{e.location_name}'."]}
        }
        
    except Exception as e:
        logger.error(f"[Job: {job_id}] Geospatial analysis failed: {e}", exc_info=True)
        # FLATTENED AND CORRECTED VERSION
        return {
            "status": "ERROR",
            "response": f"I encountered an unexpected server error while analyzing your request. Please try again later.",
            "thinking_process": {"summary": "An unexpected error occurred during workflow execution.", "chain_of_thought": [f"Error: {str(e)}"]}
        }


def result_qa_tool(user_query: str, chat_history: List[Dict]) -> Dict[str, Any]:
    """Answers a user's question based on the previous turn's result."""
    logger.info(f"Using Result Q&A Tool for query: '{user_query}'")
    
    last_assistant_message = ""
    for msg in reversed(chat_history[:-1]):
        if msg.get("role") == "assistant":
            last_assistant_message = msg.get("content", "")
            break
            
    qa_prompt = f"""You are a helpful assistant. Given the previous analysis result, answer the user's follow-up question concisely.

Previous Result: "{last_assistant_message}"
User's Question: "{user_query}"

Your Answer:"""
    
    try:
        decision = make_llm_call(qa_prompt)
        response = decision.get("response", "I'm not sure how to answer that based on the previous result.")
    except Exception as e:
        logger.error(f"Failed to get LLM response for Q&A: {e}")
        response = "I'm having trouble processing your follow-up question. Please try rephrasing it."
    
    return {
        "status": "SUCCESS",
        "response": response,
        "thinking_process": {"summary": "Answered a follow-up question about the last result."}
    }


def conversational_tool(user_query: str) -> Dict[str, Any]:
    """
    Handles non-geospatial queries with conversational responses
    
    Args:
        user_query: User's conversational query
        
    Returns:
        Dictionary containing conversational response
    """
    logger.info(f"Handling conversational query: '{user_query}'")
    
    # Analyze query for appropriate response
    query_lower = user_query.lower().strip()
    
    # Greeting responses
    if any(greeting in query_lower for greeting in ['hello', 'hi', 'hey', 'greetings']):
        return {
            "response": "Hello! I'm your Geospatial AI Analyst. I can help you find and analyze location-based information. Try asking me something like 'Find restaurants in downtown Portland' or 'Show me hospitals near Central Park'.",
            "thinking_process": "User provided a greeting, responding with a friendly introduction and usage examples.",
            "status": "SUCCESS"
        }
    
    # Capability questions
    elif any(word in query_lower for word in ['what can you do', 'capabilities', 'help', 'how to use']):
        return {
            "response": "I specialize in geospatial analysis! I can help you:\n• Find businesses, services, or landmarks in specific locations\n• Analyze spatial relationships and distances\n• Visualize geographic data\n• Perform location-based searches\n\nJust tell me what you're looking for and where you want to search. For example: 'Find coffee shops in Brooklyn' or 'Show me parks near the University of California'.",
            "thinking_process": "User asked about system capabilities, providing comprehensive overview of geospatial analysis features.",
            "status": "SUCCESS"
        }
    
    # Thanks responses
    elif any(word in query_lower for word in ['thank', 'thanks', 'appreciate']):
        return {
            "response": "You're welcome! I'm here whenever you need geospatial analysis or location-based information. Feel free to ask me about finding places, analyzing spatial data, or anything geography-related!",
            "thinking_process": "User expressed gratitude, responding positively and encouraging future geospatial queries.",
            "status": "SUCCESS"
        }
    
    # Generic conversational fallback
    else:
        return {
            "response": "I'm a Geospatial AI Analyst specialized in location-based analysis. While I'd love to chat about other topics, I'm optimized for helping you find and analyze geographic information. Try asking me to find something in a specific location - I'm really good at that!",
            "thinking_process": "User query was conversational but didn't match specific patterns, redirecting to geospatial capabilities.",
            "status": "SUCCESS"
        }


# ================================
# LLM INTEGRATION FUNCTIONS
# ================================

def make_llm_call(prompt: str) -> Dict[str, Any]:
    """
    Makes a REAL call to the LLM service to get the Conductor's decision.
    
    Args:
        prompt: The formatted system prompt for the Conductor.
        
    Returns:
        Dictionary containing the LLM's tool choice decision.
    """
    logger.info("Making a real LLM call for Conductor decision...")
    
    # Get the Ollama URL from the environment variable we set in docker-compose.yml
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if not ollama_url:
        logger.error("OLLAMA_BASE_URL is not set. Cannot connect to LLM.")
        raise ValueError("OLLAMA_BASE_URL environment variable is not configured.")

    full_api_url = f"{ollama_url}/api/generate"
    
    # This payload is specifically for the Ollama API
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False,  # We want the full JSON response at once
        "format": "json"  # Critically important to ensure the output is valid JSON
    }
    
    try:
        # Execute the HTTP POST request to the Ollama service
        response = requests.post(full_api_url, json=payload, timeout=90) # 90-second timeout
        response.raise_for_status() # Raise an error for bad responses (4xx or 5xx)
        
        # Parse the JSON response from Ollama and extract the content
        response_data = response.json()
        decision_json_str = response_data.get('response', '{}')
        
        logger.info(f"LLM decision received: {decision_json_str}")
        
        # Return the parsed dictionary from the LLM's response string
        return json.loads(decision_json_str)

    except requests.exceptions.RequestException as e:
        logger.error(f"LLM call failed: Could not connect to Ollama at {full_api_url}. Error: {e}", exc_info=True)
        # Fallback to conversational tool if the LLM is unreachable
        return {
            "tool_to_use": "conversational_tool",
            "input_for_tool": "hello" # A safe fallback
        }
    except json.JSONDecodeError as e:
        logger.error(f"LLM returned invalid JSON. Response: {decision_json_str}", exc_info=True)
        raise ValueError("LLM failed to produce valid JSON for its decision.") from e


def format_chat_history(chat_history: List[Dict[str, Any]]) -> str:
    """
    Formats chat history for inclusion in the system prompt
    
    Args:
        chat_history: List of chat message dictionaries
        
    Returns:
        Formatted string representation of chat history
    """
    if not chat_history:
        return "No previous conversation history."
    
    formatted_history = []
    for msg in chat_history[-5:]:  # Keep only last 5 messages for context
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        formatted_history.append(f"{role.upper()}: {content}")
    
    return "\n".join(formatted_history)


# ================================
# MAIN CELERY TASK - CONDUCTOR AGENT
# ================================

# Step 4: Overhaul the execute_agentic_workflow Function (Action 4)
# REPLACE THE ENTIRE BODY OF THE execute_agentic_workflow FUNCTION
@celery.task(name="engine.execute_agentic_workflow", bind=True)
def execute_agentic_workflow(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main Conductor Agent task using a Special Command + RAG architecture.
    """
    job_id = self.request.id
    user_query = task_input.get('query', '').strip()
    # The frontend now sends 'history'. Your old code looked for 'chat_history'.
    # This standardizes on 'history' for consistency with the API.
    chat_history = task_input.get('history', []) 

    logger.info(f"[Job: {job_id}] Conductor received query: '{user_query}'")
    self.update_state(state='PROGRESS', meta={'stage': 'Analyzing query...'})

    try:
        # --- Step 1: Check for a hardcoded special command for 100% reliability ---
        special_command = find_special_command(user_query)
        if special_command:
            pre_canned_query = special_command.get("pre-canned_query")
            logger.info(f"[Job: {job_id}] Matched special command. Executing pre-canned geospatial query.")
            # Directly call the tool, bypassing the LLM completely.
            return geospatial_planning_and_execution_tool(pre_canned_query, job_id)

        # --- Step 2: If no special command, perform Retrieve-Augment-Generate (RAG) cycle ---
        # RETRIEVE context
        logger.info(f"[Job: {job_id}] No special command. Starting RAG cycle.")
        knowledge_base = KnowledgeBase()
        system_context = knowledge_base.retrieve_context_for_query(user_query)

        # AUGMENT prompt
        formatted_history = format_chat_history(chat_history)
        conductor_prompt = CONDUCTOR_SYSTEM_PROMPT.format(
            system_context=system_context,
            chat_history=formatted_history,
            user_query=user_query
        )

        # GENERATE decision
        llm_decision = make_llm_call(conductor_prompt)
        tool_to_use = llm_decision.get('tool_to_use')
        
        logger.info(f"[Job: {job_id}] RAG-informed LLM decision: use tool '{tool_to_use}'")

        # --- Step 3: Execute the LLM-chosen tool ---
        if tool_to_use == "geospatial_tool": # <-- IMPORTANT: Matches the new, shorter name in the prompt
            return geospatial_planning_and_execution_tool(user_query, job_id)
        elif tool_to_use == "result_qa_tool":
            return result_qa_tool(user_query, chat_history)
        else: # Default to conversational_tool for safety
            logger.info(f"[Job: {job_id}] Defaulting to conversational tool.")
            return conversational_tool(user_query)

    except Exception as e:
        logger.error(f"[Job: {job_id}] Conductor agent workflow FAILED: {e}", exc_info=True)
        # Your old error handling was good, but Celery propagates exceptions.
        # Raising the exception is often cleaner for the worker to handle failure states.
        # The frontend can then interpret the task failure.
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        # Re-raise the exception to mark the Celery task as failed.
        raise


# ================================
# UTILITY TASKS
# ================================

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
        cleaned_count = 0
        
        # In a real implementation, you would check timestamps and delete old keys
        # For now, just return the count
        
        return {
            "status": "completed",
            "total_jobs": len(job_keys),
            "cleaned_jobs": cleaned_count
        }
    except Exception as e:
        logger.error(f"Cleanup task failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}