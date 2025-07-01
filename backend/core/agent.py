# backend/core/agent.py - FIXED
"""
Geospatial AI Agent using LangGraph
This module defines the agent graph that processes geospatial queries.
"""

import logging
from typing import Dict, Any, List, Optional
from functools import lru_cache

# Set up logging
logger = logging.getLogger(__name__)

# Global variable for the agent graph
agent_graph = None

def create_agent_graph():
    """
    Create and return the LangGraph agent graph.
    This function handles all imports and initialization.
    """
    try:
        # Import LangGraph and related dependencies
        from langgraph.graph import StateGraph, END
        from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
        from typing_extensions import TypedDict
        
        logger.info("Imported LangGraph dependencies successfully")
        
        # Define the state structure
        class AgentState(TypedDict):
            messages: List[BaseMessage]
            data_store: Dict[str, Any]
            
        # Define agent nodes
        def process_query_node(state: AgentState) -> AgentState:
            """Process the user query and generate a response"""
            try:
                messages = state["messages"]
                if not messages:
                    raise ValueError("No messages in state")
                
                user_query = messages[-1].content
                logger.info(f"Processing query: {user_query[:100]}...")
                
                # Simple response for now - replace with your actual AI logic
                response = f"Processed geospatial query: '{user_query}'. This is a placeholder response."
                
                # Add AI response to messages
                new_messages = messages + [AIMessage(content=response)]
                
                return {
                    "messages": new_messages,
                    "data_store": state.get("data_store", {})
                }
                
            except Exception as e:
                logger.error(f"Error in process_query_node: {e}")
                error_response = f"Error processing query: {str(e)}"
                new_messages = state["messages"] + [AIMessage(content=error_response)]
                return {
                    "messages": new_messages,
                    "data_store": state.get("data_store", {})
                }
        
        def data_analysis_node(state: AgentState) -> AgentState:
            """Analyze geospatial data if needed"""
            try:
                messages = state["messages"]
                data_store = state.get("data_store", {})
                
                # Simple data analysis placeholder
                analysis_result = "Geospatial analysis completed."
                data_store["analysis_completed"] = True
                
                # Add analysis result to the last message
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message, AIMessage):
                        enhanced_content = f"{last_message.content}\n\nAnalysis: {analysis_result}"
                        messages[-1] = AIMessage(content=enhanced_content)
                
                return {
                    "messages": messages,
                    "data_store": data_store
                }
                
            except Exception as e:
                logger.error(f"Error in data_analysis_node: {e}")
                return state
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("process_query", process_query_node)
        workflow.add_node("data_analysis", data_analysis_node)
        
        # Define the flow
        workflow.set_entry_point("process_query")
        workflow.add_edge("process_query", "data_analysis")
        workflow.add_edge("data_analysis", END)
        
        # Compile the graph
        graph = workflow.compile()
        logger.info("Agent graph compiled successfully")
        
        return graph
        
    except ImportError as e:
        logger.error(f"Failed to import required dependencies: {e}")
        raise ImportError(f"LangGraph dependencies not available: {e}")
    except Exception as e:
        logger.error(f"Failed to create agent graph: {e}")
        raise RuntimeError(f"Agent graph creation failed: {e}")

@lru_cache(maxsize=1)
def get_agent_graph():
    """
    Get the agent graph, creating it if necessary.
    Uses caching to ensure the graph is only created once.
    """
    global agent_graph
    if agent_graph is None:
        logger.info("Creating agent graph for the first time...")
        agent_graph = create_agent_graph()
    return agent_graph

# Try to initialize the agent graph on module import
try:
    agent_graph = get_agent_graph()
    logger.info("Agent graph initialized successfully on module import")
except Exception as e:
    logger.warning(f"Could not initialize agent graph on import: {e}")
    logger.info("Agent graph will be created when first accessed")
    agent_graph = None

def test_agent_graph():
    """
    Test function to verify the agent graph works
    """
    try:
        from langchain_core.messages import HumanMessage
        
        graph = get_agent_graph()
        test_state = {
            "messages": [HumanMessage(content="Test query")],
            "data_store": {}
        }
        
        result = graph.invoke(test_state)
        return {
            "status": "success",
            "message_count": len(result["messages"]),
            "final_message": result["messages"][-1].content if result["messages"] else "No messages"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# Export the agent graph for use by other modules
__all__ = ["agent_graph", "get_agent_graph", "test_agent_graph"]