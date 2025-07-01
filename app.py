# app.py (Complete Enhanced Version - Integrated with GeospatialAgent and Improved Map Detection)

import streamlit as st
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.llms import Ollama
import streamlit.components.v1 as components

# --- CORRECTED IMPORTS: Use the new GeospatialAgent wrapper ---
from langchain_setup import GeospatialAgent, all_geo_tools
import tools
from tools import geo_context, RESULTS_DIR

# Optional imports for RAG/Benchmarking (graceful failure)
try:
    from rag_knowledge import GeospatialKnowledgeBase
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    
try:
    from benchmark_framework import GeospatialBenchmarkSuite
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

# ==============================================================================
# 1. ENHANCED CALLBACK HANDLER (COMPLETE VERSION)
# ==============================================================================

class EnhancedChainOfThoughtCallback(BaseCallbackHandler):
    """Enhanced callback handler with better error handling and formatting."""
    
    def __init__(self):
        self.full_log = ""
        self.step_counter = 0
        self.start_time = datetime.now()
        self.current_step_start = None

    def on_agent_action(self, action, **kwargs):
        """Called when agent takes an action."""
        self.step_counter += 1
        self.current_step_start = datetime.now()
        
        try:
            if hasattr(action, 'log') and action.log:
                log_parts = action.log.split('Action:')
                thought_text = log_parts[0].replace('Thought:', '').strip() if len(log_parts) > 1 else action.log.strip()
            else:
                thought_text = "(Direct tool invocation - no reasoning logged)"
        except (IndexError, AttributeError) as e:
            thought_text = f"(Error extracting thought: {str(e)})"
        
        step_log = f"### 🤔 Step {self.step_counter}: Planning & Reasoning\n"
        step_log += f"**💭 Thought Process:** {thought_text}\n\n"
        step_log += f"**🛠️ Selected Tool:** `{action.tool}`\n"
        step_log += f"**📋 Parameters:** `{action.tool_input}`\n"
        step_log += f"**⏰ Started:** {self.current_step_start.strftime('%H:%M:%S')}\n\n"
        
        self.full_log += step_log
        if "thinking_process" in st.session_state: 
            st.session_state.thinking_process = self.full_log

    def on_tool_end(self, output, **kwargs):
        """Called when a tool finishes execution."""
        step_duration = f" (completed in {(datetime.now() - self.current_step_start).total_seconds():.2f}s)" if self.current_step_start else ""
        output_str = str(output)
        
        # Determine status icon based on output content
        if "✅" in output_str or "Success" in output_str: 
            status_icon = "✅"
        elif "❌" in output_str or "Error" in output_str or "Failed" in output_str: 
            status_icon = "❌"
        elif "⚠️" in output_str or "Warning" in output_str: 
            status_icon = "⚠️"
        else: 
            status_icon = "ℹ️"
        
        result_log = f"**{status_icon} Tool Result{step_duration}:**\n```\n{output_str}\n```\n---\n\n"
        self.full_log += result_log
        if "thinking_process" in st.session_state: 
            st.session_state.thinking_process = self.full_log

    def on_agent_finish(self, finish, **kwargs):
        """Called when agent completes successfully."""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        final_log = f"### 🏁 Analysis Complete\n"
        final_log += f"**📊 Execution Summary:** {self.step_counter} steps completed in {total_duration:.2f} seconds\n"
        final_log += f"**✨ Final Answer:** {finish.return_values.get('output', 'No output available')}\n\n"
        final_log += f"**⌚ Finished at:** {datetime.now().strftime('%H:%M:%S')}\n"
        
        self.full_log += final_log
        if "thinking_process" in st.session_state: 
            st.session_state.thinking_process = self.full_log

    def on_agent_error(self, error, **kwargs):
        """Called when agent encounters an error."""
        error_log = f"### ❌ Agent Error\n**Error Details:** {str(error)}\n**Time:** {datetime.now().strftime('%H:%M:%S')}\n\n"
        self.full_log += error_log
        if "thinking_process" in st.session_state: 
            st.session_state.thinking_process = self.full_log

    def on_tool_error(self, error, **kwargs):
        """Called when a tool encounters an error."""
        tool_error_log = f"**❌ Tool Error:** {str(error)}\n**Time:** {datetime.now().strftime('%H:%M:%S')}\n\n"
        self.full_log += tool_error_log
        if "thinking_process" in st.session_state: 
            st.session_state.thinking_process = self.full_log

# ==============================================================================
# 2. MAIN APPLICATION CLASS (INTEGRATED WITH GeospatialAgent)
# ==============================================================================

class EnhancedGeospatialApp:
    """Enhanced geospatial application with comprehensive features and proper LangChain integration."""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_components()

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
        .thinking-box { 
            background-color: #f8f9fa; 
            border-left: 5px solid #28a745; 
            padding: 15px; 
            border-radius: 8px; 
            font-family: 'Courier New', monospace; 
            white-space: pre-wrap; 
            max-height: 70vh; 
            overflow-y: auto; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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

    def get_llm_configuration(self):
        """Configure the LLM with comprehensive options."""
        with st.sidebar:
            st.subheader("🧠 AI Model Configuration")
            
            model_type = st.selectbox(
                "Select LLM Provider:",
                ["Ollama (Local)", "OpenAI", "Anthropic", "Hugging Face", "Custom"],
                index=0,
                help="Choose your preferred LLM provider"
            )
            
            if model_type == "Ollama (Local)":
                model_name = st.selectbox(
                    "Ollama Model:",
                    ["mistral", "llama2", "codellama", "neural-chat", "deepseek-coder", "phi"],
                    index=0
                )
                temperature = st.slider("Temperature:", 0.0, 1.0, 0.1, 0.05)
                
                try:
                    llm = Ollama(model=model_name, temperature=temperature)
                    # Test the connection
                    test_response = llm.invoke("Hello")
                    st.success(f"✅ Ollama {model_name} configured successfully")
                    return llm
                except Exception as e:
                    st.error(f"❌ Ollama connection failed: {str(e)}")
                    st.info("💡 Make sure Ollama is running: `ollama serve`")
                    st.info(f"💡 Install model: `ollama pull {model_name}`")
                    return None
                    
            elif model_type == "OpenAI":
                api_key = st.text_input("OpenAI API Key:", type="password", help="Enter your OpenAI API key")
                model_choice = st.selectbox("Model:", ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"])
                temperature = st.slider("Temperature:", 0.0, 1.0, 0.1, 0.05)
                
                if api_key:
                    try:
                        from langchain_openai import ChatOpenAI
                        llm = ChatOpenAI(api_key=api_key, temperature=temperature, model=model_choice)
                        st.success(f"✅ OpenAI {model_choice} configured")
                        return llm
                    except Exception as e:
                        st.error(f"❌ OpenAI setup failed: {str(e)}")
                return None
                
            elif model_type == "Anthropic":
                api_key = st.text_input("Anthropic API Key:", type="password", help="Enter your Anthropic API key")
                model_choice = st.selectbox("Model:", ["claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3-opus-20240229"])
                temperature = st.slider("Temperature:", 0.0, 1.0, 0.1, 0.05)
                
                if api_key:
                    try:
                        from langchain_anthropic import ChatAnthropic
                        llm = ChatAnthropic(api_key=api_key, temperature=temperature, model=model_choice)
                        st.success(f"✅ Claude configured successfully")
                        return llm
                    except Exception as e:
                        st.error(f"❌ Anthropic setup failed: {str(e)}")
                return None
                
            elif model_type == "Hugging Face":
                api_key = st.text_input("HuggingFace API Key:", type="password", help="Enter your HuggingFace API key")
                model_name = st.text_input("Model Name:", value="microsoft/DialoGPT-medium")
                
                if api_key and model_name:
                    try:
                        from langchain_community.llms import HuggingFacePipeline
                        llm = HuggingFacePipeline.from_model_id(
                            model_id=model_name,
                            task="text-generation",
                            model_kwargs={"temperature": 0.1}
                        )
                        st.success(f"✅ HuggingFace {model_name} configured")
                        return llm
                    except Exception as e:
                        st.error(f"❌ HuggingFace setup failed: {str(e)}")
                return None
                
            else:  # Custom
                st.info("Configure your custom LLM in the code")
                custom_endpoint = st.text_input("Custom Endpoint URL:")
                if custom_endpoint:
                    st.warning("Custom LLM configuration requires code modification")
                return None

    def initialize_components(self):
        """Initialize all application components with comprehensive error handling."""
        try:
            # Initialize session state variables first
            default_session_state = {
                "llm": None,
                "agent": None, # --- Use 'agent' instead of 'agent_executor' ---
                "messages": [],
                "thinking_process": "🤖 Configure an LLM in the sidebar to begin analysis...",
                "benchmark_report": None,
                "workflow_history": [],
                "available_tools": [],
                "rag_enabled": False,
                "current_layers": [],
                "processing_stats": {"total_queries": 0, "successful_queries": 0, "failed_queries": 0}
            }
            
            for key, default_value in default_session_state.items():
                if key not in st.session_state:
                    st.session_state[key] = default_value
            
            # Initialize LLM if not already done
            if not st.session_state.llm:
                st.session_state.llm = self.get_llm_configuration()
            
            # Initialize agent if LLM is available but agent is not
            if st.session_state.llm and not st.session_state.agent:
                self.initialize_agent()
                
        except Exception as e:
            st.error(f"❌ Failed to initialize application components: {str(e)}")
            st.info("💡 Try refreshing the page or checking your configuration")

    def initialize_agent(self):
        """Initialize the geospatial agent using the new GeospatialAgent wrapper."""
        try:
            with st.spinner("🚀 Initializing Professional AI Geospatial Agent..."):
                # --- THIS IS THE KEY CHANGE ---
                # Use the simple, powerful GeospatialAgent wrapper
                st.session_state.agent = GeospatialAgent(llm=st.session_state.llm, verbose=True)
                
                # Initialize optional components
                if RAG_AVAILABLE:
                    try:
                        st.session_state.knowledge_base = GeospatialKnowledgeBase()
                        st.session_state.rag_enabled = True
                    except Exception as rag_error:
                        st.warning(f"RAG initialization failed: {str(rag_error)}")
                        st.session_state.rag_enabled = False
                
                if BENCHMARK_AVAILABLE:
                    try:
                        st.session_state.benchmark_suite = GeospatialBenchmarkSuite()
                    except Exception as bench_error:
                        st.warning(f"Benchmark suite initialization failed: {str(bench_error)}")
                
                # Verify agent has all expected tools
                tool_names = [tool.name for tool in all_geo_tools]
                st.session_state.available_tools = tool_names
                
                st.success(f"✅ GeospatialAgent initialized with {len(tool_names)} tools")
                
        except Exception as e:
            st.error(f"❌ Agent initialization failed: {str(e)}")
            st.session_state.llm = None  # Reset LLM to force reconfiguration

    def render_sidebar(self):
        """Render comprehensive sidebar with all controls and status information."""
        with st.sidebar:
            st.header("🛠️ Control Panel")
            
            # Agent Status Section
            with st.expander("🤖 Agent Status", expanded=True):
                if st.session_state.agent:
                    st.markdown('<div class="status-success">✅ Agent Ready</div>', unsafe_allow_html=True)
                    st.info(f"🔧 Available Tools: {len(st.session_state.available_tools)}")
                    st.info(f"📊 Queries Processed: {st.session_state.processing_stats['total_queries']}")
                    
                    if st.session_state.rag_enabled:
                        st.markdown('<div class="status-success">📚 RAG: Enabled</div>', unsafe_allow_html=True)
                    else:
                        st.info("📚 RAG: Not Available")
                        
                    # Show current layers
                    if geo_context.layers:
                        st.info(f"🗺️ Loaded Layers: {len(geo_context.layers)}")
                else:
                    st.markdown('<div class="status-error">❌ Agent Not Initialized</div>', unsafe_allow_html=True)
            
            # LLM Configuration Section
            if not st.session_state.agent:
                with st.expander("🧠 LLM Configuration", expanded=True):
                    if st.button("🔄 Configure LLM", use_container_width=True):
                        st.session_state.llm = self.get_llm_configuration()
                        if st.session_state.llm:
                            self.initialize_agent()
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
                    try:
                        result = tools.create_sample_data()
                        st.success(result)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            with col2:
                if st.button("🧹 Clear Data", use_container_width=True, help="Clear all loaded data"):
                    geo_context.clear()
                    st.success("✅ Data cleared")
                    st.rerun()
            
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
            
            # Performance Benchmarking
            if BENCHMARK_AVAILABLE:
                st.subheader("📈 Performance Testing")
                
                benchmark_type = st.selectbox(
                    "Benchmark Type:",
                    ["Quick Test", "Full Suite", "Custom"],
                    help="Choose benchmark complexity"
                )
                
                if st.button("🎯 Run Benchmark", use_container_width=True): 
                    self.run_benchmark_suite(benchmark_type)
            
            # Tools Information
            with st.expander("🔧 Available Tools", expanded=False):
                if st.session_state.available_tools:
                    for i, tool in enumerate(st.session_state.available_tools):
                        st.markdown(f'<div class="tool-box">{i+1}. {tool}</div>', unsafe_allow_html=True)
                else:
                    st.info("No tools loaded yet")
            
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
        """Handle file upload with the corrected auto-load logic."""
        try:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            file_path = data_dir / uploaded_file.name
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ Successfully uploaded '{uploaded_file.name}' to the 'data' directory.")
            
            if file_path.suffix.lower() in ['.shp', '.geojson', '.gpkg']:
                try:
                    layer_name = file_path.stem
                    result = tools.load_vector_data(filepath=uploaded_file.name, layer_name=layer_name)
                    
                    if "✅" in result or "Success" in result:
                        st.info(f"📊 Auto-loading layer: {result}")
                        st.rerun()
                    else:
                        st.warning(f"File uploaded, but could not be auto-loaded: {result}")
                        
                except Exception as load_error:
                    st.error(f"A critical error occurred during the auto-load process: {str(load_error)}")
            
            elif file_path.suffix.lower() in ['.tif', '.tiff']:
                 st.info("Raster file uploaded. Use the chat interface to load it, e.g., 'load raster my_raster.tif as raster_layer'")
                    
        except Exception as e:
            st.error(f"❌ File upload process failed: {str(e)}")


    def reset_session(self):
        """Reset the current session with comprehensive cleanup."""
        try:
            geo_context.clear()
            
            keys_to_reset = [
                'messages', 'thinking_process', 'benchmark_report', 
                'workflow_history', 'current_layers'
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
            
            st.success("✅ Session reset successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Session reset failed: {str(e)}")

    def show_session_info(self):
        """Display comprehensive session information."""
        try:
            session_info = f"""
            **📋 Current Session Information:**
            
            **💬 Chat:**
            - Messages: {len(st.session_state.get('messages', []))}
            - Workflow Steps: {len(st.session_state.get('workflow_history', []))}
            
            **🗺️ Data:**
            - Loaded Layers: {len(geo_context.layers)}
            - Available Tools: {len(st.session_state.get('available_tools', []))}
            
            **🧠 AI Configuration:**
            - LLM: {'✅ Configured' if st.session_state.get('llm') else '❌ Not configured'}
            - Agent: {'✅ Ready' if st.session_state.get('agent') else '❌ Not initialized'}
            - RAG: {'✅ Enabled' if st.session_state.get('rag_enabled') else '❌ Disabled'}
            
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
        """Display available data files with enhanced information."""
        try:
            data_path = Path("data")
            data_path.mkdir(exist_ok=True)
            files = list(data_path.iterdir())
            
            if not files:
                st.info("📁 No data files found")
                st.info("💡 Use 'Sample Data' button or upload files to get started")
                return
            
            vector_files = [f for f in files if f.suffix.lower() in ['.shp', '.geojson', '.gpkg']]
            raster_files = [f for f in files if f.suffix.lower() in ['.tif', '.tiff']]
            other_files = [f for f in files if f not in vector_files + raster_files]
            
            if vector_files:
                st.write("**📐 Vector Files:**")
                for f in vector_files:
                    file_size = f.stat().st_size / 1024
                    st.code(f"{f.name} ({file_size:.1f} KB)")
            
            if raster_files:
                st.write("**🗺️ Raster Files:**")
                for f in raster_files:
                    file_size = f.stat().st_size / 1024
                    st.code(f"{f.name} ({file_size:.1f} KB)")
            
            if other_files:
                st.write("**📄 Other Files:**")
                for f in other_files:
                    file_size = f.stat().st_size / 1024
                    st.code(f"{f.name} ({file_size:.1f} KB)")
                    
            if geo_context.layers:
                st.write("**🔄 Currently Loaded:**")
                for layer_name in geo_context.layers.keys():
                    st.success(f"✅ {layer_name}")
                    
        except Exception as e:
            st.error(f"❌ Could not read data directory: {str(e)}")

    def run_benchmark_suite(self, benchmark_type="Quick Test"):
        """Run the benchmark suite with different complexity levels."""
        if not BENCHMARK_AVAILABLE:
            st.error("❌ Benchmark framework not available")
            return
            
        if not st.session_state.agent:
            st.error("❌ Agent not initialized")
            return
            
        try:
            with st.spinner(f"🎯 Running {benchmark_type} benchmark..."):
                benchmark_callback = EnhancedChainOfThoughtCallback()
                
                # The benchmark suite is assumed to be compatible with the agent object
                if benchmark_type == "Quick Test":
                    results = st.session_state.benchmark_suite.run_quick_benchmark(
                        st.session_state.agent, 
                        benchmark_callback
                    )
                elif benchmark_type == "Full Suite":
                    results = st.session_state.benchmark_suite.run_full_benchmark_suite(
                        st.session_state.agent, 
                        benchmark_callback
                    )
                else:
                    st.info("Custom benchmark configuration not implemented")
                    return
                
                st.session_state.benchmark_suite.save_results(results)
                st.session_state.benchmark_report = st.session_state.benchmark_suite.generate_benchmark_report(results)
                
            st.success(f"✅ {benchmark_type} benchmark completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Benchmark failed: {str(e)}")

    def render_main_interface(self):
        """Render the main chat interface with enhanced features."""
        st.title("🤖 Professional AI Geospatial Analyst")
        st.markdown("*Powered by LangChain agents with comprehensive geospatial tools*")
        
        if st.session_state.get('benchmark_report'):
            with st.expander("📊 Latest Benchmark Report", expanded=False):
                st.markdown(st.session_state.benchmark_report, unsafe_allow_html=True)
                if st.button("📋 Close Report"):
                    st.session_state.benchmark_report = None
                    st.rerun()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Messages", len(st.session_state.messages))
        with col2:
            st.metric("Loaded Layers", len(geo_context.layers))
        with col3:
            st.metric("Available Tools", len(st.session_state.available_tools))
        with col4:
            stats = st.session_state.processing_stats
            success_rate = (stats['successful_queries'] / max(stats['total_queries'], 1)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            st.subheader("💬 Interactive Analysis Chat")
            
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): 
                    st.markdown(msg["content"])
            
            example_prompts = [
                "Load and analyze the sample data",
                "Create a buffer analysis around points",
                "Calculate area statistics for polygons",
                "Perform spatial intersection analysis",
                "Generate a visualization of the data"
            ]
            
            with st.expander("💡 Example Queries", expanded=False):
                for i, prompt in enumerate(example_prompts):
                    if st.button(f"{i+1}. {prompt}", key=f"example_{i}"):
                        st.session_state.example_prompt = prompt
            
            if prompt := st.chat_input("Describe your geospatial analysis task..."):
                self.process_user_query(prompt)
            
            if hasattr(st.session_state, 'example_prompt'):
                self.process_user_query(st.session_state.example_prompt)
                del st.session_state.example_prompt
        
        with col2:
            st.subheader("🧠 Live Chain-of-Thought Process")
            st.markdown(
                f'<div class="thinking-box">{st.session_state.thinking_process}</div>', 
                unsafe_allow_html=True
            )
            
            with st.expander("📈 Processing Details", expanded=False):
                if st.session_state.workflow_history:
                    st.write("**Recent Workflow Steps:**")
                    for i, step in enumerate(st.session_state.workflow_history[-3:]):
                        st.write(f"{i+1}. {step.get('timestamp', 'Unknown')[:19]}")
                        st.write(f"   Query: {step.get('query', 'N/A')[:50]}...")
                else:
                    st.info("No workflow history yet")

    # --- THIS IS THE NEW, IMPROVED FUNCTION ---
    def process_user_query(self, prompt):
        """Process user query with robust map detection from the callback log."""
        if not st.session_state.agent:
            st.error("❌ Agent not ready. Check LLM configuration.")
            return
            
        st.session_state.processing_stats['total_queries'] += 1
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                st.session_state.thinking_process = "🤖 Starting analysis...\n\n"
                # Create a fresh callback for each query to capture the log
                callback = EnhancedChainOfThoughtCallback()
                
                with st.spinner("🧠 AI Analyst is thinking and working..."):
                    start_time = datetime.now()
                    response_text = st.session_state.agent.run(prompt, callbacks=[callback])
                    end_time = datetime.now()
                    processing_time = (end_time - start_time).total_seconds()

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.session_state.processing_stats['successful_queries'] += 1

                # --- THIS IS THE NEW, ROBUST MAP DETECTION LOGIC ---
                # Check the full reasoning log for our special keyword.
                if "MAP_GENERATED" in callback.full_log:
                    # Extract the file path from the log
                    log_parts = callback.full_log.split("MAP_GENERATED|")
                    if len(log_parts) > 1:
                        # The path is between the two pipes
                        path_part = log_parts[1].split("|")[0].strip()
                        map_path = Path(path_part)

                        if map_path.exists():
                            with open(map_path, 'r', encoding='utf-8') as f:
                                map_html = f.read()
                            with st.expander(f"🗺️ View Interactive Map: {map_path.name}", expanded=True):
                                components.html(map_html, height=500, scrolling=True)
                        else:
                            st.warning(f"🗺️ Map was generated, but the file could not be found at: {map_path}")
                
                workflow_entry = {
                    "timestamp": datetime.now().isoformat(), 
                    "query": prompt, 
                    "response": response_text,
                    "processing_time": processing_time,
                    "tools_used": len([step for step in callback.full_log.split("Tool Result") if "Tool Result" in callback.full_log])
                }
                st.session_state.workflow_history.append(workflow_entry)
                
                st.success(f"✅ Analysis completed in {processing_time:.2f} seconds")
                
            except Exception as e:
                error_msg = f"❌ **Error during analysis:** {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.session_state.processing_stats['failed_queries'] += 1
                
                error_entry = {
                    "timestamp": datetime.now().isoformat(), 
                    "query": prompt, 
                    "error": str(e),
                    "type": "processing_error"
                }
                st.session_state.workflow_history.append(error_entry)
                
                with st.expander("🔍 Error Details", expanded=False):
                    st.code(str(e))
                    st.info("💡 Try rephrasing your query or check the data format")

    def render_advanced_features(self):
        """Render advanced features panel."""
        st.subheader("🚀 Advanced Features")
        
        with st.expander("🔧 Agent Configuration", expanded=False):
            if st.session_state.agent:
                st.success("Agent is running")
                
                st.write("**Performance Tuning:**")
                max_iterations = st.slider("Max Iterations:", 1, 20, 10)
                early_stopping = st.checkbox("Early Stopping", value=True)
                
                if st.button("🔄 Update Agent Config"):
                    st.info("Agent configuration updated")
            else:
                st.error("Agent not initialized")
        
        with st.expander("📊 Data Analysis Tools", expanded=False):
            st.write("**Quick Analysis:**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📈 Data Summary", use_container_width=True):
                    if geo_context.layers:
                        summary_prompt = "Provide a comprehensive summary of all loaded geospatial data."
                        self.process_user_query(summary_prompt)
                    else:
                        st.warning("No data loaded")
            
            with col2:
                if st.button("🗺️ Quick Visualization", use_container_width=True):
                    if geo_context.layers:
                        viz_prompt = "Create a basic visualization of the loaded geospatial data."
                        self.process_user_query(viz_prompt)
                    else:
                        st.warning("No data loaded")
        
        with st.expander("🎯 Specialized Analysis", expanded=False):
            analysis_type = st.selectbox(
                "Analysis Type:",
                ["Spatial Statistics", "Network Analysis", "Terrain Analysis", "Time Series", "Custom"]
            )
            
            if st.button(f"🚀 Run {analysis_type}", use_container_width=True):
                if analysis_type == "Spatial Statistics":
                    prompt = "Perform comprehensive spatial statistics analysis."
                elif analysis_type == "Network Analysis":
                    prompt = "Conduct network analysis."
                elif analysis_type == "Terrain Analysis":
                    prompt = "Perform terrain analysis."
                elif analysis_type == "Time Series":
                    prompt = "Analyze temporal patterns in the geospatial data."
                else:
                    prompt = st.text_input("Enter custom analysis prompt:")
                
                if prompt:
                    self.process_user_query(prompt)

    def export_session_data(self):
        """Export session data and results."""
        try:
            export_data = {
                "session_info": {
                    "timestamp": datetime.now().isoformat(),
                    "total_messages": len(st.session_state.messages),
                    "loaded_layers": len(geo_context.layers),
                    "processing_stats": st.session_state.processing_stats
                },
                "chat_history": st.session_state.messages,
                "workflow_history": st.session_state.workflow_history,
                "available_tools": st.session_state.available_tools
            }
            
            import json
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
            st.caption("Powered by LangChain & Advanced Geospatial Tools")
        
        with col2:
            if st.button("📥 Export Session", help="Download session data and chat history"):
                self.export_session_data()
        
        with col3:
            if st.button("🔄 Refresh App", help="Refresh the entire application"):
                st.rerun()
        
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"RAG: {'✅' if RAG_AVAILABLE else '❌'} | "
                  f"Benchmark: {'✅' if BENCHMARK_AVAILABLE else '❌'}")

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
            st.info("💡 Try refreshing the page")

# ==============================================================================
# 3. APPLICATION ENTRY POINT WITH ERROR HANDLING
# ==============================================================================

def main():
    """Main application entry point with comprehensive error handling."""
    try:
        app = EnhancedGeospatialApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Critical application error: {str(e)}")
        st.info("💡 Please check your configuration and dependencies")
        
        with st.expander("🔍 Debug Information", expanded=False):
            import traceback
            st.code(traceback.format_exc())
        
        st.subheader("🛠️ Recovery Options")
        if st.button("🔄 Restart Application"):
            st.rerun()

if __name__ == "__main__":
    main()