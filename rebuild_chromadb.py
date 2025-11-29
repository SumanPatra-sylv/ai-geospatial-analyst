#!/usr/bin/env python3
"""
Quick script to rebuild the ChromaDB GIS documentation collection.
"""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def rebuild_collection():
    """Rebuild the gis_documentation ChromaDB collection."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        from chromadb.config import Settings
        
        logger.info("Starting ChromaDB collection rebuild...")
        
        # Initialize ChromaDB
        vector_db_path = Path("data/vector_db")
        vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # Delete existing db if present
        import shutil
        db_file = vector_db_path / "chroma.db"
        if db_file.exists():
            logger.info(f"Removing existing database: {db_file}")
            shutil.rmtree(vector_db_path, ignore_errors=True)
            vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # Create fresh client
        logger.info(f"Initializing ChromaDB at: {vector_db_path}")
        client = chromadb.PersistentClient(
            path=str(vector_db_path),
            settings=Settings(
                allow_reset=False,
                anonymized_telemetry=False
            )
        )
        
        # Create collection
        logger.info("Creating 'gis_documentation' collection...")
        collection = client.get_or_create_collection(
            name="gis_documentation",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Sample expert documentation
        docs = [
            {
                "id": "doc_1",
                "text": "Calculate area for polygon features and add as a new attribute. Useful for analyzing region sizes, measuring buffer zones, or computing land coverage statistics. Use geopandas dissolve() for aggregation.",
                "metadata": {"tool": "calculate_area", "category": "spatial_analysis"}
            },
            {
                "id": "doc_2",
                "text": "Create buffer zones around features at specified distances. Useful for proximity analysis, creating protective zones, or expanding geographic boundaries. Buffer distance in meters can be positive (outward) or negative (inward).",
                "metadata": {"tool": "buffer", "category": "spatial_analysis"}
            }
        ]
        
        # Load embedding model
        logger.info("Loading SentenceTransformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Add documents to collection
        logger.info(f"Adding {len(docs)} documents to collection...")
        for doc in docs:
            embedding = model.encode(doc["text"]).tolist()
            collection.add(
                ids=[doc["id"]],
                embeddings=[embedding],
                metadatas=[doc["metadata"]],
                documents=[doc["text"]]
            )
        
        count = collection.count()
        logger.info(f"✅ ChromaDB collection rebuilt successfully with {count} documents")
        
    except Exception as e:
        logger.error(f"❌ Failed to rebuild collection: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = rebuild_collection()
    sys.exit(0 if success else 1)
