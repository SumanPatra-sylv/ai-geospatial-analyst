# app.py (Refactored Frontend - Pure API Client) - FINAL MODIFIED VERSION

import streamlit as st
import requests
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit.components.v1 as components
import json
import time

# ==============================================================================
# 1. CONFIGURATION AND CONSTANTS
# ==============================================================================

# --- FIX: Read the correct environment variable 'API_BASE_URL' ---
BACKEND_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# The base path for job-related endpoints is /jobs
API_BASE = f"{BACKEND_URL}/api/v1/jobs"

# ==============================================================================
# 2. API CLIENT CLASS
# ==============================================================================

class GeospatialAPIClient:
    """API client for communicating with the FastAPI src."""

    def __init__(self, base_url=API_BASE):
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self):
        """Check if the backend is healthy."""
        try:
            # This correctly calls the root /health endpoint
            response = self.session.get(f"{BACKEND_URL}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_status(self):
        """Get current backend jobs router status."""
        try:
            # Call the available /jobs/health endpoint
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    # SURGICAL FIX 1: Add get_job_status to the API Client
    def get_job_status(self, job_id: str):
        """Get the status of a specific job by polling the backend."""
        try:
            response = self.session.get(f"{self.base_url}/status/{job_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            # If polling fails, return a FAILURE status to stop the loop.
            return {"status": "FAILURE", "error": f"Polling failed: {str(e)}"}

    # CHANGE 1: SIMPLIFY THE API CLIENT
    def process_query(self, query, session_id=None, history=None):
        """Sends query and history to the backend."""
        try:
            # The payload now correctly includes the chat history
            payload = {
                "query": query,

                # This key MUST match the Pydantic model in your jobs.py
                # If your model uses 'history', keep this as 'history'.
                # If your model uses 'chat_history', change it here.
                "history": history or [],

                "session_id": session_id
            }
            response = self.session.post(f"{self.base_url}/start", json=payload, timeout=30)
            
            # We no longer need the special 422 check. We just check for any error.
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            # This will now catch all connection or HTTP errors.
            return {"error": f"Failed to communicate with the backend: {str(e)}"}

    def upload_file(self, file_data, filename):
        """Upload a file to the src."""
        try:
            files = {"file": (filename, file_data)}
            # This now correctly assumes a /jobs/upload endpoint
            response = self.session.post(f"{self.base_url}/upload", files=files)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "success": False}

    def get_available_data(self):
        """Get list of available data files."""
        try:
            # This now correctly assumes a /jobs/data endpoint
            response = self.session.get(f"{self.base_url}/data")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def clear_data(self):
        """Clear all data from the src."""
        try:
            # This now correctly assumes a DELETE /jobs/data endpoint
            response = self.session.delete(f"{self.base_url}/data")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def create_sample_data(self):
        """Request backend to create sample data."""
        try:
            # This now correctly assumes a /jobs/sample-data endpoint
            response = self.session.post(f"{self.base_url}/sample-data")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

# ==============================================================================
# 3. MAIN APPLICATION CLASS
# ==============================================================================

class EnhancedGeospatialApp:
    """Enhanced geospatial application - Pure frontend with API communication."""
    
    def __init__(self):
        self.setup_page_config()
        self.api_client = GeospatialAPIClient()
        self.initialize_session_state()

    def setup_page_config(self):
        """Configure Streamlit page settings and custom CSS."""
        st.set_page_config(
            page_title="🤖 Professional AI Geospatial Analyst",
            layout="wide",
            page_icon="🗺️",
            initial_sidebar_state="expanded"
        )
        
        # Enhanced CSS styling
        st.markdown("""
        <style>
        /* FIX 1: The CSS Change - Replaced the existing .thinking-box style */
        .thinking-box { 
            background-color: #1E1E2E; /* Dark blue/purple background for a modern tech look */
            border-left: 5px solid #00A2FF; /* A vibrant blue accent */
            padding: 15px; 
            border-radius: 8px; 
            font-family: 'Consolas', 'Courier New', monospace; 
            color: #E0E0E0; /* Light grey text for high contrast and readability */
            white-space: pre-wrap; 
            word-wrap: break-word;
            min-height: 100px;
            max-height: 70vh; 
            overflow-y: auto; 
            box-shadow: 0 4px 8px rgba(0,0,0,0.3); /* Deeper shadow for depth */
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
        .tool-box {
            background-color: #e9ecef;
            padding: 8px;
            border-radius: 4px;
            margin: 2px 0;
            font-family: monospace;
        }
        </style>
        """, unsafe_allow_html=True)

    def initialize_session_state(self):
        """Initialize session state variables."""
        default_session_state = {
            "messages": [],
            "thinking_process": "🤖 Welcome! Connect to the backend to begin analysis...",
            "workflow_history": [],
            "current_layers": [],
            "processing_stats": {"total_queries": 0, "successful_queries": 0, "failed_queries": 0},
            "backend_connected": False,
            "backend_status": {},
            "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        for key, default_value in default_session_state.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def check_backend_connection(self):
        """Check and update backend connection status."""
        if self.api_client.health_check():
            if not st.session_state.backend_connected:
                st.session_state.backend_connected = True
                status = self.api_client.get_status()
                st.session_state.backend_status = status
                st.session_state.thinking_process = "✅ Connected to backend! Ready for geospatial analysis..."
        else:
            st.session_state.backend_connected = False
            st.session_state.thinking_process = "❌ Backend not available. Please start the FastAPI server..."

    def render_sidebar(self):
        """Render comprehensive sidebar with all controls and status information."""
        with st.sidebar:
            st.header("🛠️ Control Panel")
            
            # Backend Connection Status
            with st.expander("🔗 Backend Connection", expanded=True):
                self.check_backend_connection()
                
                if st.session_state.backend_connected:
                    st.markdown('<div class="status-success">✅ Backend Connected</div>', unsafe_allow_html=True)
                    if st.session_state.backend_status:
                        status = st.session_state.backend_status
                        st.info(f"🔧 Available Tools: {status.get('available_tools', 0)}")
                        st.info(f"🗺️ Loaded Layers: {status.get('loaded_layers', 0)}")
                else:
                    st.markdown('<div class="status-error">❌ Backend Disconnected</div>', unsafe_allow_html=True)
                    st.info(f"🌐 Backend URL: {BACKEND_URL}")
                    st.info("💡 Make sure FastAPI server is running")
                
                if st.button("🔄 Refresh Connection", use_container_width=True):
                    self.check_backend_connection()
                    st.rerun()
            
            # Session Management
            st.subheader("🔄 Session Management")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🆕 New Session", use_container_width=True, help="Clear all data and start fresh"): 
                    self.reset_session()
                    
            with col2:
                if st.button("📋 Session Info", use_container_width=True, help="Show current session details"): 
                    self.show_session_info()
            
            # Data Management Section
            st.subheader("📁 Data Management")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Sample Data", use_container_width=True, help="Create sample geospatial data"):
                    if st.session_state.backend_connected:
                        result = self.api_client.create_sample_data()
                        if "error" not in result:
                            st.success("✅ Sample data created successfully")
                        else:
                            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                    else:
                        st.error("❌ Backend not connected")
            
            with col2:
                if st.button("🧹 Clear Data", use_container_width=True, help="Clear all loaded data"):
                    if st.session_state.backend_connected:
                        result = self.api_client.clear_data()
                        if "error" not in result:
                            st.success("✅ Data cleared successfully")
                        else:
                            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                        st.rerun()
                    else:
                        st.error("❌ Backend not connected")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload Geospatial Data",
                type=['shp', 'geojson', 'gpkg', 'tif', 'tiff', 'csv'],
                help="Upload vector or raster data files"
            )
            
            if uploaded_file:
                self.handle_file_upload(uploaded_file)
            
            with st.expander("📂 Available Files", expanded=False): 
                self.display_available_data()
            
            # Processing Statistics
            with st.expander("📊 Processing Stats", expanded=False):
                stats = st.session_state.processing_stats
                st.metric("Total Queries", stats['total_queries'])
                st.metric("Successful", stats['successful_queries'])
                st.metric("Failed", stats['failed_queries'])
                
                if stats['total_queries'] > 0:
                    success_rate = (stats['successful_queries'] / stats['total_queries']) * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")

    def handle_file_upload(self, uploaded_file):
        """Handle file upload via API."""
        if not st.session_state.backend_connected:
            st.error("❌ Backend not connected")
            return
            
        try:
            with st.spinner("📤 Uploading file..."):
                result = self.api_client.upload_file(uploaded_file.getvalue(), uploaded_file.name)
                
                if result.get("success"):
                    st.success(f"✅ Successfully uploaded '{uploaded_file.name}'")
                    if result.get("auto_loaded"):
                        st.info(f"📊 Auto-loading: {result.get('message', '')}")
                    st.rerun()
                else:
                    st.error(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
                    
        except Exception as e:
            st.error(f"❌ File upload process failed: {str(e)}")

    def reset_session(self):
        """Reset the current session."""
        try:
            # Clear backend data if connected
            if st.session_state.backend_connected:
                self.api_client.clear_data()
            
            # Reset session state
            keys_to_reset = [
                'messages', 'thinking_process', 'workflow_history', 'current_layers'
            ]
            
            for key in keys_to_reset:
                if key in st.session_state:
                    if key == 'messages':
                        st.session_state[key] = []
                    elif key == 'thinking_process':
                        st.session_state[key] = "🤖 Session reset. Ready for new analysis..."
                    else:
                        st.session_state[key] = [] if isinstance(st.session_state[key], list) else None
            
            st.session_state.processing_stats = {
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0
            }
            
            # Generate new session ID
            st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            st.success("✅ Session reset successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Session reset failed: {str(e)}")

    def show_session_info(self):
        """Display comprehensive session information."""
        try:
            backend_status = "✅ Connected" if st.session_state.backend_connected else "❌ Disconnected"
            
            session_info = f"""
            **📋 Current Session Information:**
            
            **💬 Chat:**
            - Messages: {len(st.session_state.get('messages', []))}
            - Workflow Steps: {len(st.session_state.get('workflow_history', []))}
            - Session ID: {st.session_state.get('session_id', 'Unknown')}
            
            **🔗 Backend:**
            - Status: {backend_status}
            - URL: {BACKEND_URL}
            
            **📊 Statistics:**
            - Total Queries: {st.session_state.processing_stats['total_queries']}
            - Successful: {st.session_state.processing_stats['successful_queries']}
            - Failed: {st.session_state.processing_stats['failed_queries']}
            
            **⏰ Session Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            st.info(session_info)
            
        except Exception as e:
            st.error(f"❌ Could not retrieve session info: {str(e)}")

    def display_available_data(self):
        """Display available data files from src."""
        if not st.session_state.backend_connected:
            st.info("❌ Backend not connected")
            return
            
        try:
            data_info = self.api_client.get_available_data()
            
            if data_info.get("error"):
                st.error(f"❌ Error getting data info: {data_info['error']}")
                return
            
            files = data_info.get("files", [])
            layers = data_info.get("loaded_layers", [])
            
            if not files and not layers:
                st.info("📁 No data files found")
                st.info("💡 Use 'Sample Data' button or upload files to get started")
                return
            
            if files:
                st.write("**📄 Available Files:**")
                for file_info in files:
                    file_size = file_info.get("size_kb", 0)
                    st.code(f"{file_info['name']} ({file_size:.1f} KB)")
                    
            if layers:
                st.write("**🔄 Currently Loaded:**")
                for layer_name in layers:
                    st.success(f"✅ {layer_name}")
                    
        except Exception as e:
            st.error(f"❌ Could not read data information: {str(e)}")

    def render_main_interface(self):
        """Render the main chat interface with enhanced features."""
        st.title("🤖 Professional AI Geospatial Analyst")
        st.markdown("*Frontend client for AI-powered geospatial analysis*")
        
        # Connection status indicator
        if st.session_state.backend_connected:
            st.success("🔗 Connected to backend")
        else:
            st.error("❌ Backend not available - Please start the FastAPI server")
            st.info(f"Expected backend URL: {BACKEND_URL}")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Messages", len(st.session_state.messages))
        with col2:
            backend_status = st.session_state.backend_status
            st.metric("Loaded Layers", backend_status.get("loaded_layers", 0))
        with col3:
            st.metric("Available Tools", backend_status.get("available_tools", 0))
        with col4:
            stats = st.session_state.processing_stats
            success_rate = (stats['successful_queries'] / max(stats['total_queries'], 1)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            st.subheader("💬 Interactive Analysis Chat")
            
            # Display all messages from history
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): 
                    st.markdown(msg["content"])
            
            example_prompts = [
                "Load and analyze the sample data",
                "Generate a visualization of the data",
                "Show and summarize all landuse features in Potsdam, Germany",
                "Create a buffer analysis around points",
                "Calculate area statistics for polygons",
                "Perform spatial intersection analysis"
            ]
            
            with st.expander("💡 Example Queries", expanded=False):
                for i, prompt in enumerate(example_prompts):
                    if st.button(f"{i+1}. {prompt}", key=f"example_{i}"):
                        self.process_user_query(prompt)
                        st.rerun()

            # Step 3: Call the new fragment.
            if prompt := st.chat_input("Describe your geospatial analysis task..."):
                self.process_user_query(prompt)
                st.rerun() # Trigger the rerun after handling the input

            # Add this line to call the new fragment
            self.handle_assistant_response()
        
        with col2:
            st.subheader("🧠 Analysis Status")
            st.markdown(
                f'<div class="thinking-box">{st.session_state.thinking_process}</div>',
                unsafe_allow_html=True
            )
            
            with st.expander("📈 Processing Details", expanded=False):
                if st.session_state.workflow_history:
                    st.write("**Recent Workflow Steps:**")
                    for i, step in enumerate(reversed(st.session_state.workflow_history[-3:])):
                        st.write(f"**Step {len(st.session_state.workflow_history)-i}:** {step.get('timestamp', 'Unknown')[:19]}")
                        st.write(f"   Query: `{step.get('query', 'N/A')[:50]}...`")
                else:
                    st.info("No workflow history yet")

    # Step 1: Replace the process_user_query method.
    def process_user_query(self, prompt: str):
        """Adds user query to chat history and triggers a re-run to process it."""
        if not st.session_state.backend_connected:
            st.error("❌ Backend not connected")
            return
        
        st.session_state.processing_stats['total_queries'] += 1
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Don't do any processing here, just rerun.
        # The main script flow will handle displaying the new message and calling the assistant.

    # Step 2: Add the handle_assistant_response method.
    @st.fragment
    def handle_assistant_response(self):
        """A self-contained fragment to handle polling and displaying the assistant's response."""
        # This logic only runs if the last message was from the user
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            prompt = st.session_state.messages[-1]["content"]

            with st.chat_message("assistant"):
                start_time = time.time()
                job_id = None
                try:
                    # Step 1: Submit the job
                    with st.spinner("🚀 Submitting job to the Conductor Agent..."):
                        initial_response = self.api_client.process_query(
                            query=prompt,
                            history=st.session_state.messages[:-1]
                        )

                    if "job_id" not in initial_response:
                        error_msg = f"❌ Request Rejected: {initial_response.get('error', 'Failed to start job.')}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        st.session_state.processing_stats['failed_queries'] += 1
                        return

                    job_id = initial_response["job_id"]
                    st.info(f"✅ Job `{job_id}` accepted. The AI Analyst is now at work...")

                    # Step 2: Poll for the result
                    final_result = None
                    progress_bar = st.progress(0, text="🧠 AI processing...")
                    while True:
                        status_response = self.api_client.get_job_status(job_id)
                        status = status_response.get("status")
                        if status in ["SUCCESS", "FAILURE"]:
                            final_result = status_response.get("result", {})
                            progress_bar.progress(100, text="✅ Complete!")
                            break
                        time.sleep(1.5)

                    # Step 3: Process and Display the final rich result
                    if final_result and final_result.get("status") == "SUCCESS":
                        st.session_state.processing_stats['successful_queries'] += 1
                        response_text = final_result.get("response", "Analysis complete!")
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        
                        thinking_data = final_result.get("thinking_process", {})
                        if isinstance(thinking_data, dict) and "chain_of_thought" in thinking_data:
                            summary = thinking_data.get("summary", "Done.")
                            cot_log = "\n".join(thinking_data.get("chain_of_thought", []))
                            st.session_state.thinking_process = f"**Summary:** {summary}\n\n**Chain of Thought:**\n{cot_log}"
                        
                        if final_result.get("artifacts"):
                            with st.expander("📄 View Generated Artifacts", expanded=True):
                                for key, value in final_result["artifacts"].items():
                                    st.markdown(f"- **{key.replace('_', ' ').title()}:** `{value}`")
                    else: # Handle job failure
                        st.session_state.processing_stats['failed_queries'] += 1
                        error_response = final_result.get("response", "An unknown error occurred.")
                        st.error(f"**Analysis Failed:** {error_response}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_response}"})
                except Exception as e:
                    st.session_state.processing_stats['failed_queries'] += 1
                    st.error(f"A critical error occurred: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
    
    def render_advanced_features(self):
        """Render advanced features panel."""
        st.subheader("🚀 Advanced Features")
        
        with st.expander("🔧 Backend Configuration", expanded=False):
            st.write("**Backend Settings:**")
            current_url = st.text_input("Backend URL:", value=BACKEND_URL)
            
            if st.button("🔄 Update Backend URL"):
                # Note: This would require restarting the app to take effect
                st.info("Backend URL updated. Restart the app to apply changes.")
            
            if st.button("🔍 Test Connection"):
                test_client = GeospatialAPIClient(f"{current_url}/api/v1")
                if test_client.health_check():
                    st.success("✅ Connection successful")
                else:
                    st.error("❌ Connection failed")
        
        with st.expander("📊 Quick Analysis", expanded=False):
            st.write("**Quick Analysis Tools:**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📈 Data Summary", use_container_width=True):
                    summary_prompt = "Provide a comprehensive summary of all loaded geospatial data."
                    self.process_user_query(summary_prompt)
                    st.rerun()
            
            with col2:
                if st.button("🗺️ Quick Visualization", use_container_width=True):
                    viz_prompt = "Create a basic visualization of the loaded geospatial data."
                    self.process_user_query(viz_prompt)
                    st.rerun()
        
        with st.expander("🎯 Specialized Analysis", expanded=False):
            analysis_type = st.selectbox(
                "Analysis Type:",
                ["Spatial Statistics", "Network Analysis", "Terrain Analysis", "Time Series", "Custom"]
            )
            
            if st.button(f"🚀 Run {analysis_type}", use_container_width=True):
                if analysis_type == "Spatial Statistics":
                    prompt = "Perform comprehensive spatial statistics analysis on the loaded data."
                elif analysis_type == "Network Analysis":
                    prompt = "Conduct network analysis on the available geospatial data."
                elif analysis_type == "Terrain Analysis":
                    prompt = "Perform terrain analysis on raster data if available."
                elif analysis_type == "Time Series":
                    prompt = "Analyze temporal patterns in the geospatial data if time attributes exist."
                else:
                    prompt = st.text_input("Enter custom analysis prompt:")
                
                if prompt:
                    self.process_user_query(prompt)
                    st.rerun()

    def export_session_data(self):
        """Export session data and results."""
        try:
            export_data = {
                "session_info": {
                    "timestamp": datetime.now().isoformat(),
                    "session_id": st.session_state.session_id,
                    "total_messages": len(st.session_state.messages),
                    "backend_connected": st.session_state.backend_connected,
                    "processing_stats": st.session_state.processing_stats
                },
                "chat_history": st.session_state.messages,
                "workflow_history": st.session_state.workflow_history,
                "backend_url": BACKEND_URL
            }
            
            json_str = json.dumps(export_data, indent=2, default=str)
            
            st.download_button(
                label="📥 Download Session Data",
                data=json_str,
                file_name=f"geospatial_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")

    def render_footer(self):
        """Render application footer with additional options."""
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**🤖 AI Geospatial Analyst**")
            st.caption("Frontend Client for FastAPI Backend")
        
        with col2:
            st.download_button(
                label="📥 Export Session",
                data="{}", # Dummy data, real logic in on_click
                on_click=self.export_session_data,
                file_name="session_export.json", # Dummy name
                help="Download session data and chat history"
            )

        with col3:
            if st.button("🔄 Refresh App", help="Refresh the entire application"):
                st.rerun()
        
        backend_status = "🟢 Connected" if st.session_state.backend_connected else "🔴 Disconnected"
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"Backend: {backend_status} | "
                  f"URL: {BACKEND_URL}")

    def run(self):
        """Run the complete application with all features."""
        try:
            self.render_sidebar()
            self.render_main_interface()
            
            with st.expander("🚀 Advanced Features", expanded=False):
                self.render_advanced_features()
            
            self.render_footer()
            
        except Exception as e:
            st.error(f"❌ Application error: {str(e)}")
            st.info("💡 Try refreshing the page or checking backend connection")

# ==============================================================================
# 4. APPLICATION ENTRY POINT
# ==============================================================================

def main():
    """Main application entry point."""
    try:
        app = EnhancedGeospatialApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Critical application error: {str(e)}")
        st.info("💡 Please check your configuration and backend connection")
        
        with st.expander("🔍 Debug Information", expanded=False):
            import traceback
            st.code(traceback.format_exc())
        
        st.subheader("🛠️ Recovery Options")
        if st.button("🔄 Restart Application"):
            st.rerun()

if __name__ == "__main__":
    main()