#!/usr/bin/env python3
"""
Streamlit UI for AI Geospatial Analyst - Standalone Version
Directly integrates with MasterOrchestrator (no FastAPI backend needed)
"""

import streamlit as st
import time
from datetime import datetime
from src.core.orchestrator import MasterOrchestrator
from src.core.planners.query_parser import QueryParser, ParsedQuery, QueryParserError

# ==============================================================================
# PAGE CONFIG
# ==============================================================================

st.set_page_config(
    page_title="🗺️ AI Geospatial Analyst",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM CSS
# ==============================================================================

st.markdown("""
<style>
.thinking-box { 
    background-color: #1E1E2E;
    border-left: 5px solid #00A2FF;
    padding: 15px; 
    border-radius: 8px; 
    font-family: 'Consolas', 'Courier New', monospace; 
    color: #E0E0E0;
    white-space: pre-wrap; 
    word-wrap: break-word;
    min-height: 100px;
    max-height: 70vh; 
    overflow-y: auto; 
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}
.status-success { 
    background-color: #d4edda; 
    color: #155724; 
    padding: 10px; 
    border-radius: 5px; 
    margin: 5px 0; 
}
.status-error { 
    background-color: #f8d7da; 
    color: #721c24; 
    padding: 10px; 
    border-radius: 5px; 
    margin: 5px 0; 
}
.stChatMessage {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 10px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
    
if "query_parser" not in st.session_state:
    st.session_state.query_parser = None
    
if "thinking_process" not in st.session_state:
    st.session_state.thinking_process = "🤖 Welcome! Ready to analyze geospatial data..."
    
if "stats" not in st.session_state:
    st.session_state.stats = {
        "total_queries": 0,
        "successful_queries": 0,
        "failed_queries": 0,
        "total_execution_time": 0
    }

if "execution_log" not in st.session_state:
    st.session_state.execution_log = []

# ==============================================================================
# INITIALIZATION
# ==============================================================================

@st.cache_resource
def initialize_orchestrator():
    """Initialize the orchestrator (cached for performance)"""
    return MasterOrchestrator(max_loops=15, use_task_queue=True)

@st.cache_resource
def initialize_query_parser():
    """Initialize the query parser (cached for performance)"""
    return QueryParser()

# Initialize components
if st.session_state.orchestrator is None:
    with st.spinner("🔧 Initializing AI Geospatial Analyst..."):
        st.session_state.orchestrator = initialize_orchestrator()
        st.session_state.query_parser = initialize_query_parser()

# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.header("🛠️ Control Panel")
    
    # Architecture Info
    with st.expander("🏗️ Architecture Info", expanded=True):
        st.markdown("""
        **Current Architecture:**
        - ✅ Task Queue (Deterministic)
        - ✅ Single-Tool Isolation
        - ✅ Zero Infinite Loops
        - ✅ Proven OSM Tags
        """)
    
    # Statistics
    st.subheader("📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", st.session_state.stats["total_queries"])
        st.metric("Successful", st.session_state.stats["successful_queries"])
    with col2:
        st.metric("Failed", st.session_state.stats["failed_queries"])
        success_rate = (st.session_state.stats["successful_queries"] / max(st.session_state.stats["total_queries"], 1)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    if st.session_state.stats["total_execution_time"] > 0:
        avg_time = st.session_state.stats["total_execution_time"] / max(st.session_state.stats["successful_queries"], 1)
        st.metric("Avg Execution Time", f"{avg_time:.1f}s")
    
    # Session Management
    st.subheader("🔄 Session")
    if st.button("🆕 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.execution_log = []
        st.session_state.thinking_process = "🤖 Chat cleared. Ready for new queries..."
        st.rerun()
    
    if st.button("🔄 Reset Statistics", use_container_width=True):
        st.session_state.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_execution_time": 0
        }
        st.rerun()
    
    # Execution Log
    with st.expander("📜 Execution Log", expanded=False):
        if st.session_state.execution_log:
            for i, log_entry in enumerate(reversed(st.session_state.execution_log[-10:])):
                st.text(f"{i+1}. {log_entry}")
        else:
            st.info("No execution history yet")

# ==============================================================================
# MAIN INTERFACE
# ==============================================================================

st.title("🗺️ AI Geospatial Analyst")
st.caption("*Powered by Neuro-Symbolic Task Queue Architecture*")

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Messages", len(st.session_state.messages))
with col2:
    st.metric("Architecture", "Task Queue")
with col3:
    st.metric("Loop Prevention", "Guaranteed")
with col4:
    st.metric("Success Rate", f"{(st.session_state.stats['successful_queries'] / max(st.session_state.stats['total_queries'], 1)) * 100:.0f}%")

# Main Layout: Chat + Thinking Process
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("💬 Chat Interface")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Example prompts
    with st.expander("💡 Example Queries", expanded=False):
        examples = [
            "Find schools in Berlin",
            "Find hospitals near parks in London",
            "Find restaurants in Paris",
            "Find universities in New York",
            "Find museums near libraries in Tokyo"
        ]
        for i, example in enumerate(examples):
            if st.button(f"{i+1}. {example}", key=f"example_{i}"):
                # Process the example
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask a geospatial question (e.g., 'Find schools in Berlin')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()
    
    # Process last user message if it hasn't been answered
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_query = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            with st.spinner("🧠 Processing your query..."):
                try:
                    # Update stats
                    st.session_state.stats["total_queries"] += 1
                    
                    # Parse query
                    st.info("📝 Parsing query...")
                    parsed_query = st.session_state.query_parser.parse(user_query)
                    
                    st.success(f"✅ Query parsed: Target='{parsed_query.target}', Location='{parsed_query.location}'")
                    
                    # Execute query
                    st.info("🚀 Executing task queue...")
                    start_time = time.time()
                    
                    result = st.session_state.orchestrator.run(parsed_query)
                    
                    execution_time = time.time() - start_time
                    
                    if result["success"]:
                        # Success!
                        st.session_state.stats["successful_queries"] += 1
                        st.session_state.stats["total_execution_time"] += execution_time
                        
                        # Generate response
                        final_layer = result.get("final_layer_name", "result_layer")
                        final_result = result.get("final_result")
                        
                        if final_result is not None:
                            feature_count = len(final_result)
                            
                            response = f"""
✅ **Query Completed Successfully!**

**Results:**
- Found **{feature_count} features**
- Execution Time: **{execution_time:.2f} seconds**
- Final Layer: `{final_layer}`
- Tasks Completed: **{result.get('loop_count', 'N/A')}**
- Architecture: **{result.get('architecture', 'Task Queue')}**

**Analysis:**
Your query "{user_query}" has been successfully processed using the deterministic Task Queue architecture with zero infinite loops!
"""
                            
                            # Store thinking process
                            st.session_state.thinking_process = f"""
**Last Query:** {user_query}
**Execution Time:** {execution_time:.2f}s
**Tasks:** {result.get('loop_count', 'N/A')}
**Features Found:** {feature_count}
**Status:** ✅ SUCCESS
"""
                        else:
                            response = f"✅ Analysis completed in {execution_time:.2f}s, but no final result layer was produced."
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Log execution
                        st.session_state.execution_log.append(
                            f"{datetime.now().strftime('%H:%M:%S')} - SUCCESS - {user_query[:50]} ({execution_time:.1f}s)"
                        )
                        
                    else:
                        # Failure
                        st.session_state.stats["failed_queries"] += 1
                        error_msg = result.get("error", "Unknown error")
                        
                        response = f"""
❌ **Query Failed**

**Error:** {error_msg}

**Execution Time:** {execution_time:.2f} seconds

Please try rephrasing your query or check the logs for more details.
"""
                        
                        st.error(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Log execution
                        st.session_state.execution_log.append(
                            f"{datetime.now().strftime('%H:%M:%S')} - FAILED - {user_query[:50]}"
                        )
                        
                except QueryParserError as e:
                    st.session_state.stats["failed_queries"] += 1
                    error_response = f"❌ **Query Parsing Failed:** {str(e)}\n\nPlease try rephrasing your query."
                    st.error(error_response)
                    st.session_state.messages.append({"role": "assistant", "content": error_response})
                    
                except Exception as e:
                    st.session_state.stats["failed_queries"] += 1
                    error_response = f"❌ **Error:** {str(e)}"
                    st.error(error_response)
                    st.session_state.messages.append({"role": "assistant", "content": error_response})

with col2:
    st.subheader("🧠 Analysis Status")
    st.markdown(
        f'<div class="thinking-box">{st.session_state.thinking_process}</div>',
        unsafe_allow_html=True
    )
    
    # System Info
    with st.expander("ℹ️ System Info", expanded=False):
        st.markdown("""
        **AI Geospatial Analyst**
        - Version: Task Queue v1.0
        - Architecture: Neuro-Symbolic
        - Loop Prevention: Deterministic
        - Tool Isolation: Single-tool per task
        - OSM Tags: Validated via DataScout
        """)

# Footer
st.markdown("---")
st.caption(f"🤖 AI Geospatial Analyst | Task Queue Architecture | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
