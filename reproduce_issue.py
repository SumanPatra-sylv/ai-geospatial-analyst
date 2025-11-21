
import sys
import os

# Add the current directory to sys.path
sys.path.append(os.getcwd())

try:
    import chromadb
    print("✅ chromadb is installed.")
except ImportError:
    print("❌ chromadb is NOT installed.")

try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence_transformers is installed.")
except ImportError:
    print("❌ sentence_transformers is NOT installed.")

try:
    from src.core.knowledge.knowledge_base import SpatialKnowledgeBase
    kb = SpatialKnowledgeBase()
    print("✅ SpatialKnowledgeBase initialized.")
    
    # Test Expert Docs Search
    print("\nTesting Expert Docs Search...")
    docs = kb.search_expert_docs("How to calculate flood risk?")
    print(f"Found {len(docs)} docs.")
    for doc in docs:
        print(f"- {doc[:100]}...")

    # Test Workflow Search
    print("\nTesting Workflow Search...")
    workflows = kb.search_similar_workflows("flood risk analysis")
    print(f"Found {len(workflows)} workflows.")
    for wf in workflows:
        print(f"- {wf['original_query']}")

except Exception as e:
    print(f"❌ Error initializing or using SpatialKnowledgeBase: {e}")
