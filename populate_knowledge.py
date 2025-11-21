import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.core.knowledge.knowledge_base import SpatialKnowledgeBase
from src.gis.tools.definitions import TOOL_REGISTRY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def populate():
    print("🚀 Initializing Knowledge Base...")
    kb = SpatialKnowledgeBase()
    
    # 1. Populate Expert Docs (Tool Definitions)
    print("\n📚 Populating Expert Documentation (Tool Definitions)...")
    if kb.chroma_client and kb.embedding_model:
        collection = kb.chroma_client.get_or_create_collection(name=kb.collection_name)
        
        ids = []
        documents = []
        metadatas = []
        
        for tool_name, tool_def in TOOL_REGISTRY.items():
            doc_text = f"Tool: {tool_name}\nDescription: {tool_def.description}\n"
            doc_text += "Parameters:\n"
            for param in tool_def.parameters:
                doc_text += f"- {param.name} ({param.type}): {param.description}\n"
            
            ids.append(f"tool_{tool_name}")
            documents.append(doc_text)
            metadatas.append({"type": "tool_definition", "tool_name": tool_name})
            
        # Add some general RAG guidance
        general_docs = [
            {
                "id": "guide_hospitals",
                "text": "To find hospitals in a specific city, use the 'load_osm_data' tool. Set 'area_name' to the city (e.g., 'Pune, India') and 'tags' to {'amenity': 'hospital'}. Do NOT use 'load_bhoonidhi_data'.",
                "metadata": {"type": "guide", "topic": "hospitals"}
            },
            {
                "id": "guide_buffer",
                "text": "To create a buffer zone, use the 'buffer' tool. Ensure the input layer exists. Specify 'distance' and 'unit'.",
                "metadata": {"type": "guide", "topic": "buffer"}
            }
        ]
        
        for doc in general_docs:
            ids.append(doc["id"])
            documents.append(doc["text"])
            metadatas.append(doc["metadata"])

        # Upsert to ChromaDB
        # We use the same embedding model as the KB to ensure compatibility
        embeddings = kb.embedding_model.encode(documents).tolist()
        
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"✅ Added {len(documents)} documents to ChromaDB.")
    else:
        print("❌ ChromaDB or Embedding Model not available.")

    # 2. Populate Successful Workflows
    print("\n🧠 Populating Successful Workflows...")
    
    # Workflow: Find hospitals in Pune
    workflow_hospitals = [
        {
            "tool_name": "load_osm_data",
            "parameters": {
                "area_name": "Pune, India",
                "tags": {"amenity": "hospital"},
                "layer_name": "hospitals_pune"
            },
            "reasoning": "Load hospital data for Pune using OSM."
        },
        {
            "tool_name": "finish_task",
            "parameters": {
                "final_layer_name": "hospitals_pune"
            },
            "reasoning": "Task complete."
        }
    ]
    
    kb.store_successful_workflow(
        original_query="Find hospitals in Pune",
        workflow_plan=workflow_hospitals,
        execution_time=10.5
    )
    
    print("✅ Added workflow: Find hospitals in Pune")

if __name__ == "__main__":
    populate()
