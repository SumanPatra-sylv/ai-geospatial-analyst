# In new file: src/core/knowledge_base.py
import json

class KnowledgeBase:
    """A simple RAG component to provide real-time context to the Conductor Agent."""
    def __init__(self):
        # In a real system, this could connect to Redis or other sources.
        self.tool_definitions = {
            "geospatial_tool": "Used for any task involving finding, analyzing, or processing geographic data in a specific, named location.",
            "result_qa_tool": "Used ONLY to answer a follow-up question about the immediately preceding analysis result.",
            "conversational_tool": "Used for greetings, general questions about capabilities, or any non-geospatial query."
        }

    def retrieve_context_for_query(self, query: str) -> str:
        """Retrieves relevant facts to help the LLM make a better decision."""
        context = "### System Capabilities and Tool Definitions:\n"
        context += json.dumps(self.tool_definitions, indent=2)
        # Future enhancement: Add real-time data here, e.g., list of loaded layers from Redis.
        return context