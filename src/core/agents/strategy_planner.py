#!/usr/bin/env python3
"""
Strategy Planner Agent
Responsible for communicating with the LLM to generate high-level strategic plans.
"""

import os
import json
import jinja2
import requests
from typing import Dict, Any
from pydantic import BaseModel, Field


class ToolCallIntent(BaseModel):
    """Represents the LLM's intent to use a specific tool."""
    tool_name: str = Field(..., description="The exact name of the tool from the AVAILABLE TOOLS list.")
    parameters: Dict[str, Any] = Field(..., description="A dictionary of conceptual parameters for the tool.")


class LLMStrategyResponse(BaseModel):
    """The LLM's high-level strategic plan as a sequence of tool-use intents."""
    reasoning: str = Field(..., description="High-level explanation of the overall strategy.")
    tool_calls: list[ToolCallIntent] = Field(..., description="The sequence of tools to be called to execute the strategy.")


class StrategyPlanner:
    """
    An agent whose sole responsibility is to communicate with the LLM,
    taking a factual context and generating a high-level strategic plan.
    """
    
    SYSTEM_PROMPT = """You are a master GIS Strategist with access to past successful workflows. You have been given a factual Data Reality Report from a DataScout agent{{ " and guidance from similar past problems" if guidance_from_rag else "" }}. Your task is to devise a high-level strategy to answer the user's request by selecting the correct sequence of tools.

**Your Goal:** Decide WHICH tools to use and in WHAT order. You will NOT generate the final code, only the intent to use a tool.

**RULES:**
1.  Your output MUST be a JSON object with two keys: "reasoning" and "tool_calls".
2.  "tool_calls" must be a list of objects, where each object represents one tool you intend to use.
3.  Each tool call object MUST have "tool_name" and "parameters".
4.  For "parameters", use conceptual layer names that are human-readable (e.g., "schools_data", "parks_data"). The executor will handle the real variables. Use the `canonical_name` from the report for locations.
5.  Use intuitive parameter names - the system will translate them to internal formats.

**AVAILABLE TOOLS (Your Palette):**
{{ tool_documentation }}

**Data Reality Report (Ground Truth):**
{{ data_reality_report }}

{% if guidance_from_rag %}
**Guidance from Past Successful Workflows:**
{{ guidance_from_rag }}
{% endif %}

---
**OUTPUT FORMAT EXAMPLE (This is what you must generate):**

```json
{
  "reasoning": "Based on the data reality report and similar past workflows, I will first load the school and park data because the report shows they are available. Then, to find features 'near' each other, I will create a buffer around the parks. Finally, I will select only the schools that fall within these park buffers to get the final result.",
  "tool_calls": [
    {
      "tool_name": "load_osm_data",
      "parameters": {
        "area_name": "Potsdam, Brandenburg, Deutschland",
        "tags": {"amenity": "school"},
        "conceptual_name": "all_schools_in_area"
      }
    },
    {
      "tool_name": "load_osm_data",
      "parameters": {
        "area_name": "Potsdam, Brandenburg, Deutschland",
        "tags": {"leisure": "park"},
        "conceptual_name": "all_parks_in_area"
      }
    },
    {
      "tool_name": "buffer",
      "parameters": {
        "input_layer": "all_parks_in_area",
        "distance_meters": 500,
        "conceptual_name": "park_buffer_zones"
      }
    },
    {
      "tool_name": "spatial_join",
      "parameters": {
        "left_layer": "all_schools_in_area",
        "right_layer": "park_buffer_zones",
        "predicate": "intersects",
        "conceptual_name": "final_schools_near_parks"
      }
    }
  ]
}
```

TASK: Generate the strategic JSON plan based on the Data Reality Report{{ " and leverage the guidance from past successful workflows" if guidance_from_rag else "" }}.
"""

    def __init__(self):
        """Initialize the Strategy Planner with LLM configuration."""
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("OLLAMA_MODEL", "mistral")
        self.timeout = 120

    def generate_strategy(self, prompt_context: str, tool_docs: str, rag_guidance: str = "") -> LLMStrategyResponse:
        """
        Takes a fully built context and generates the strategic tool calls.
        
        Args:
            prompt_context: Formatted data reality report
            tool_docs: Available tools documentation
            rag_guidance: Guidance from past successful workflows
            
        Returns:
            LLMStrategyResponse: The strategic plan from the LLM
        """
        template = jinja2.Environment(
            loader=jinja2.BaseLoader(), 
            trim_blocks=True, 
            lstrip_blocks=True
        ).from_string(self.SYSTEM_PROMPT)
        
        prompt = template.render(
            data_reality_report=prompt_context,
            tool_documentation=tool_docs,
            guidance_from_rag=rag_guidance
        )
        
        llm_response_dict = self._make_llm_call(prompt)
        return LLMStrategyResponse(**llm_response_dict)

    def _make_llm_call(self, prompt: str) -> Dict:
        """
        Enhanced LLM call with better error handling and response validation.
        """
        if not self.ollama_base_url:
            raise ValueError("OLLAMA_BASE_URL environment variable is not configured.")
        
        full_api_url = f"{self.ollama_base_url}/api/generate"
        payload = { 
            "model": self.model_name, 
            "prompt": prompt, 
            "stream": False, 
            "format": "json", 
            "options": {
                "temperature": 0.1, 
                "top_p": 0.9,
                "num_predict": 3072,  # Increased for complex responses
                "stop": ["```", "---"]  # Stop tokens to prevent over-generation
            } 
        }

        try:
            response = requests.post(full_api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            response_data = response.json()
            
            if 'response' not in response_data:
                raise ValueError("Invalid response format from LLM: 'response' key missing.")
            
            raw_llm_json_str = response_data['response'].strip()
            
            # Clean up common LLM response artifacts
            if raw_llm_json_str.startswith('```json'):
                raw_llm_json_str = raw_llm_json_str[7:]
            if raw_llm_json_str.endswith('```'):
                raw_llm_json_str = raw_llm_json_str[:-3]
            raw_llm_json_str = raw_llm_json_str.strip()
            
            print("--- RAW LLM JSON RESPONSE ---")
            print(raw_llm_json_str)
            print("---------------------------")
            
            # Parse and validate the JSON structure
            parsed_response = json.loads(raw_llm_json_str)
            
            # Enhanced structure validation
            if not isinstance(parsed_response, dict):
                raise ValueError("LLM response must be a JSON object")
            
            required_keys = {"reasoning", "tool_calls"}
            if not all(key in parsed_response for key in required_keys):
                missing_keys = required_keys - set(parsed_response.keys())
                raise ValueError(f"LLM response missing required keys: {missing_keys}")
            
            if not isinstance(parsed_response["tool_calls"], list):
                raise ValueError("'tool_calls' must be a list")
            
            if not parsed_response["tool_calls"]:
                raise ValueError("'tool_calls' cannot be empty")
            
            # Validate each tool call structure
            for i, tool_call in enumerate(parsed_response["tool_calls"]):
                if not isinstance(tool_call, dict):
                    raise ValueError(f"Tool call {i+1} must be a dictionary")
                if "tool_name" not in tool_call or "parameters" not in tool_call:
                    raise ValueError(f"Tool call {i+1} missing required keys 'tool_name' or 'parameters'")
                if not isinstance(tool_call["parameters"], dict):
                    raise ValueError(f"Tool call {i+1} 'parameters' must be a dictionary")
            
            return parsed_response
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Could not connect to Ollama at {self.ollama_base_url}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"LLM request timed out after {self.timeout} seconds")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"HTTP error from LLM service: {e}")