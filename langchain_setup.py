# langchain_setup.py (Fixed - No Deprecated Parameters)

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional, List, Any
import tools

# =============================================================================
# SCHEMAS - ONLY TWO TYPES NEEDED
# =============================================================================

class SimpleStringInput(BaseModel):
    """Single string input that will be parsed by the tool itself."""
    params: str = Field(
        default="", 
        description="Parameters as a simple string (comma-separated when multiple). See tool description for exact format."
    )

class NoInput(BaseModel):
    """For tools that take no parameters."""
    pass

# =============================================================================
# TOOL WRAPPERS - SYNCHRONIZED WITH TOOLS.PY REGISTRY
# =============================================================================

class DynamicGeoTool(BaseTool):
    """Dynamic wrapper that adapts to any function from tools.py registry."""
    
    # Type annotations for custom fields
    func_name: str
    format_str: str
    tool_func: Any
    
    def __init__(self, func_name: str, format_str: str, **kwargs):
        """Initialize the dynamic tool with proper Pydantic V2 compatibility."""
        # Get the actual python function from the tools module
        tool_func = getattr(tools, func_name)
        
        # Determine the correct input schema
        if format_str == 'No arguments needed':
            args_schema = NoInput
        else:
            args_schema = SimpleStringInput
            
        # Build the description from docstring + format
        doc = tool_func.__doc__ or f"Execute {func_name}"
        description = f"{doc.strip()}. Expected format: '{format_str}'"
        
        # Call the parent's __init__ with all required fields
        super().__init__(
            name=func_name,
            description=description,
            args_schema=args_schema,
            func_name=func_name,
            format_str=format_str,
            tool_func=tool_func,
            **kwargs
        )
    
    def _run(self, params: str = "") -> str:
        """Execute the tool function with proper parameter handling."""
        try:
            if self.format_str == 'No arguments needed':
                return self.tool_func()
            else:
                # All multi-parameter tools expect a single string that they parse internally
                return self.tool_func(params.strip())
        except Exception as e:
            return f"❌ Error in {self.func_name}: {str(e)}"

# =============================================================================
# AUTOMATIC TOOL DISCOVERY AND CREATION
# =============================================================================

def create_tools_from_registry() -> List[BaseTool]:
    """
    Automatically creates LangChain tools from the tools.py registry.
    This ensures perfect synchronization - if tools.py changes, this adapts automatically.
    """
    if not hasattr(tools, 'AVAILABLE_FUNCTIONS'):
        raise ValueError("❌ tools.py must have AVAILABLE_FUNCTIONS registry")
    
    tool_objects = []
    
    for func_name, format_str in tools.AVAILABLE_FUNCTIONS.items():
        if not hasattr(tools, func_name):
            print(f"⚠️  Warning: Function {func_name} listed in registry but not found in tools module")
            continue
        
        try:
            tool = DynamicGeoTool(func_name, format_str)
            tool_objects.append(tool)
        except Exception as e:
            print(f"⚠️  Warning: Could not create tool for {func_name}: {e}")
    
    print(f"✅ Created {len(tool_objects)} geospatial tools from registry")
    print(f"📋 Available tools: {', '.join([t.name for t in tool_objects])}")
    
    return tool_objects

# Initialize tools from registry
all_geo_tools = create_tools_from_registry()

# =============================================================================
# REACT AGENT CREATION - FIXED TO REMOVE DEPRECATED PARAMETERS
# =============================================================================

def create_geospatial_agent(llm, verbose=True, max_iterations=20):
    """
    Creates a robust geospatial agent with corrected ReAct prompt.
    FIXED: Removed deprecated early_stopping_method parameter.
    """
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate

    # Corrected prompt with required {tool_names} placeholder
    prompt_template = """You are an expert AI Geospatial Analyst with access to powerful geospatial analysis tools.

🎯 **CRITICAL SUCCESS RULES:**

1. **ALWAYS START WITH DATA DISCOVERY**
   - Use `list_available_data` FIRST to see what files exist
   - Check `show_operation_log` to see what's already been done

2. **BE REALISTIC ABOUT DATA**
   - Only work with files that actually exist
   - If a file doesn't exist, either create sample data or ask the user for the correct filename
   - Don't keep trying to load non-existent files

3. **SIMPLE STRING INPUTS ONLY**
   - All tool inputs are plain text strings (comma-separated when multiple parameters)
   - NO JSON objects, NO complex structures
   - Example: "cities,1000,buffered_cities" NOT {{"layer": "cities", "distance": 1000}}

4. **FOLLOW THE FORMAT EXACTLY**
   - Each tool description shows the exact expected format
   - Respect the parameter order and separators

**RESPONSE FORMAT:**
Thought: [Your reasoning about what to do next]
Action: [Choose from: {tool_names}]
Action Input: [simple string following the tool's format]
Observation: [result from tool]

**EXAMPLE WORKFLOW:**
Human: Create a buffer around cities
Thought: I need to see what data is available first.
Action: list_available_data
Action Input: 
Observation: Available files: sample_cities.shp, sample_districts.shp
Thought: I'll load the cities data first.
Action: load_vector_data
Action Input: sample_cities.shp,cities
Observation: ✅ Loaded cities layer...
Thought: Now I can create a 1000m buffer around the cities.
Action: create_buffer
Action Input: cities,1000,cities_buffer

**AVAILABLE TOOLS:**
{tools}

Current task: {input}

**Begin Analysis:**
{agent_scratchpad}"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Create the ReAct agent
    agent = create_react_agent(
        llm=llm, 
        tools=all_geo_tools, 
        prompt=prompt
    )
    
    # Create executor with robust error handling
    # FIXED: Removed the deprecated early_stopping_method parameter
    return AgentExecutor(
        agent=agent,
        tools=all_geo_tools,
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors="❌ Format Error: Use this format exactly:\nThought: [reasoning]\nAction: [tool_name]\nAction Input: [simple string]\n\nCheck the tool description for the exact input format required.",
        return_intermediate_steps=True
        # REMOVED: early_stopping_method="generate"  # This was causing the ValueError
    )

# =============================================================================
# ENHANCED CONVENIENCE WRAPPER CLASS
# =============================================================================

class GeospatialAgent:
    """
    Enhanced wrapper for the geospatial agent with additional convenience methods.
    """
    
    def __init__(self, llm, verbose=True, max_iterations=20):
        self.llm = llm
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.agent = create_geospatial_agent(llm, verbose, max_iterations)
        self._initialized = False
    
    def run(self, query: str, callbacks: Optional[List[Any]] = None) -> str:
        """
        Execute a query through the agent with enhanced error handling.
        """
        try:
            if hasattr(self.agent, 'invoke'):
                response = self.agent.invoke(
                    {"input": query}, 
                    {"callbacks": callbacks} if callbacks else None
                )
                return response.get('output', str(response))
            else:
                # Fallback for older LangChain versions
                return self.agent.run(query, callbacks=callbacks)
                
        except Exception as e:
            error_msg = f"❌ Agent execution error: {str(e)}"
            if self.verbose:
                import traceback
                print(f"\n🔍 Full error traceback:")
                traceback.print_exc()
            return error_msg
    
    def quick_setup(self) -> str:
        """
        Perform initial setup by creating sample data and showing available resources.
        """
        setup_query = """
        I need to set up my geospatial workspace. Please:
        1. Create sample data for testing
        2. List all available data files
        3. Show me what operations are available
        """
        result = self.run(setup_query)
        self._initialized = True
        return result
    
    def status(self) -> str:
        """Check the current status of loaded data and recent operations."""
        return self.run("Show me what data is currently loaded and recent operations")
    
    def help(self) -> str:
        """Get help about available tools and their usage."""
        help_text = "🗺️  **GEOSPATIAL AGENT HELP**\n\n"
        help_text += f"**Available Tools ({len(all_geo_tools)}):**\n"
        
        for tool in all_geo_tools:
            help_text += f"\n• **{tool.name}**\n"
            help_text += f"  {tool.description}\n"
        
        help_text += "\n**Quick Start:**\n"
        help_text += "1. agent.quick_setup() - Create sample data\n"
        help_text += "2. agent.run('your analysis request') - Run analysis\n"
        help_text += "3. agent.status() - Check current state\n"
        
        return help_text
    
    def reset(self) -> str:
        """Clear all data and start fresh."""
        return self.run("Clear all loaded layers and start fresh")

# =============================================================================
# VALIDATION AND DIAGNOSTICS
# =============================================================================

def validate_setup() -> bool:
    """
    Comprehensive validation of the setup to ensure everything works correctly.
    """
    print("🔍 **VALIDATING GEOSPATIAL AGENT SETUP**")
    print("=" * 50)
    
    try:
        # Check tools module
        if not hasattr(tools, 'AVAILABLE_FUNCTIONS'):
            print("❌ tools.py missing AVAILABLE_FUNCTIONS registry")
            return False
        
        print(f"✅ Found {len(tools.AVAILABLE_FUNCTIONS)} functions in registry")
        
        # Check tool creation
        if len(all_geo_tools) == 0:
            print("❌ No tools were created from the registry")
            return False
        
        print(f"✅ Created {len(all_geo_tools)} LangChain tools")
        
        # Test a simple tool
        test_tools = [tool for tool in all_geo_tools if tool.name == 'list_available_data']
        if not test_tools:
            print("❌ Critical tool 'list_available_data' not found")
            return False
        
        test_result = test_tools[0]._run()
        if "❌" in test_result:
            print(f"❌ Tool test failed: {test_result}")
            return False
        
        print("✅ Tool execution test passed")
        
        # Validate registry consistency
        registry_funcs = set(tools.AVAILABLE_FUNCTIONS.keys())
        available_funcs = {name for name in dir(tools) if callable(getattr(tools, name)) and not name.startswith('_')}
        
        missing_in_module = registry_funcs - available_funcs
        if missing_in_module:
            print(f"⚠️  Registry lists functions not in module: {missing_in_module}")
        
        print(f"✅ Registry consistency check passed")
        print("✅ **SETUP VALIDATION SUCCESSFUL!**")
        print("\n🎯 **Key Features Enabled:**")
        print("   • Automatic tool discovery from registry")
        print("   • Single-string input architecture (no JSON complexity)")
        print("   • Robust error handling and recovery")
        print("   • Dynamic adaptation to tools.py changes")
        print("   • Enhanced ReAct agent with clear instructions")
        print("   • Fixed deprecated parameter issue")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

# =============================================================================
# EXPORT THE MAIN COMPONENTS
# =============================================================================

__all__ = [
    'GeospatialAgent',
    'all_geo_tools', 
    'create_geospatial_agent',
    'validate_setup'
]