#!/usr/bin/env python3
"""
Knowledge Corpus Ingestion Script for AI Geospatial Analyst
===========================================================

This script processes expert GIS knowledge documents and creates a searchable
vector database for the RAG (Retrieval-Augmented Generation) system.

The script:
1. Loads the sentence-transformer embedding model
2. Reads all .txt files from the knowledge corpus directory
3. Chunks text content by double newlines
4. Generates embeddings for each chunk
5. Stores everything in a persistent ChromaDB vector database

Usage:
    python src/core/knowledge/ingest_docs.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import uuid

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("Please install with: pip install sentence-transformers chromadb")
    sys.exit(1)


class KnowledgeIngestionEngine:
    """
    Handles the ingestion of GIS knowledge documents into a vector database.
    """
    
    def __init__(self, 
                 corpus_dir: str = "data/knowledge_corpus",
                 vector_db_path: str = "data/vector_db",
                 collection_name: str = "gis_documentation",
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the ingestion engine.
        
        Args:
            corpus_dir: Directory containing knowledge corpus .txt files
            vector_db_path: Path for the persistent ChromaDB database
            collection_name: Name of the ChromaDB collection
            model_name: Sentence transformer model name
        """
        self.corpus_dir = Path(corpus_dir)
        self.vector_db_path = Path(vector_db_path)
        self.collection_name = collection_name
        self.model_name = model_name
        
        self.model = None
        self.client = None
        self.collection = None
        
    def initialize_components(self):
        """Initialize the embedding model and vector database."""
        print("🚀 Initializing Knowledge Ingestion Engine...")
        
        # Load the sentence transformer model
        print(f"📦 Loading embedding model: {self.model_name}...")
        try:
            self.model = SentenceTransformer(self.model_name)
            print("✅ Embedding model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            raise
        
        # Create the vector database directory if it doesn't exist
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistent storage
        print(f"🗄️  Initializing ChromaDB at: {self.vector_db_path}...")
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.vector_db_path),
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )
            )
            
            # Create or get the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "GIS expert knowledge and best practices"}
            )
            print(f"✅ ChromaDB collection '{self.collection_name}' ready!")
            
        except Exception as e:
            print(f"❌ Failed to initialize ChromaDB: {e}")
            raise
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text by splitting on double newlines.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # Split on double newlines and filter out empty chunks
        chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
        return chunks
    
    def process_file(self, file_path: Path) -> int:
        """
        Process a single knowledge file and add its chunks to the database.
        
        Args:
            file_path: Path to the text file to process
            
        Returns:
            Number of chunks processed
        """
        print(f"📖 Processing file: {file_path.name}")
        
        try:
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print(f"⚠️  Warning: File {file_path.name} is empty, skipping...")
                return 0
            
            # Chunk the content
            chunks = self.chunk_text(content)
            
            if not chunks:
                print(f"⚠️  Warning: No chunks found in {file_path.name}, skipping...")
                return 0
            
            print(f"   📝 Found {len(chunks)} chunks in {file_path.name}")
            
            # Generate embeddings for all chunks
            print(f"   🧮 Generating embeddings...")
            embeddings = self.model.encode(chunks, show_progress_bar=False)
            
            # Prepare data for ChromaDB
            chunk_ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = [
                {
                    "source_file": file_path.name,
                    "chunk_index": i,
                    "content_length": len(chunk),
                    "ingestion_timestamp": str(Path(__file__).stat().st_mtime)
                }
                for i, chunk in enumerate(chunks)
            ]
            
            # Add to the collection
            print(f"   💾 Adding chunks to vector database...")
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings.tolist(),
                documents=chunks,
                metadatas=metadatas
            )
            
            print(f"   ✅ Successfully processed {len(chunks)} chunks from {file_path.name}")
            return len(chunks)
            
        except Exception as e:
            print(f"   ❌ Error processing {file_path.name}: {e}")
            return 0
    
    def ingest_corpus(self) -> Dict[str, Any]:
        """
        Ingest all .txt files from the knowledge corpus directory.
        
        Returns:
            Dictionary with ingestion statistics
        """
        print(f"\n🔍 Scanning knowledge corpus directory: {self.corpus_dir}")
        
        if not self.corpus_dir.exists():
            print(f"❌ Knowledge corpus directory does not exist: {self.corpus_dir}")
            print("Please create the directory and add .txt files with expert knowledge.")
            return {"success": False, "error": "Corpus directory not found"}
        
        # Find all .txt files
        txt_files = list(self.corpus_dir.glob("*.txt"))
        
        if not txt_files:
            print(f"⚠️  No .txt files found in {self.corpus_dir}")
            print("Please add knowledge files with .txt extension to the corpus directory.")
            return {"success": False, "error": "No .txt files found"}
        
        print(f"📚 Found {len(txt_files)} knowledge files to process:")
        for file_path in txt_files:
            print(f"   - {file_path.name}")
        
        # Process each file
        print(f"\n⚙️  Starting ingestion process...")
        total_chunks = 0
        processed_files = 0
        failed_files = 0
        
        for file_path in txt_files:
            chunks_processed = self.process_file(file_path)
            if chunks_processed > 0:
                total_chunks += chunks_processed
                processed_files += 1
            else:
                failed_files += 1
        
        # Generate ingestion report
        collection_count = self.collection.count()
        
        results = {
            "success": True,
            "files_found": len(txt_files),
            "files_processed": processed_files,
            "files_failed": failed_files,
            "total_chunks": total_chunks,
            "collection_size": collection_count
        }
        
        return results
    
    def display_ingestion_report(self, results: Dict[str, Any]):
        """Display the final ingestion report."""
        print("\n" + "="*60)
        print("🎉 KNOWLEDGE INGESTION COMPLETE!")
        print("="*60)
        
        if results["success"]:
            print(f"📊 Ingestion Statistics:")
            print(f"   • Files Found: {results['files_found']}")
            print(f"   • Files Processed: {results['files_processed']}")
            print(f"   • Files Failed: {results['files_failed']}")
            print(f"   • Total Chunks Ingested: {results['total_chunks']}")
            print(f"   • Vector Database Size: {results['collection_size']} documents")
            print(f"   • Database Location: {self.vector_db_path}")
            print(f"   • Collection Name: {self.collection_name}")
            
            if results['files_processed'] > 0:
                print(f"\n✅ SUCCESS: Knowledge corpus successfully ingested!")
                print(f"   The RAG system can now retrieve expert GIS knowledge.")
            else:
                print(f"\n⚠️  WARNING: No files were successfully processed.")
        else:
            print(f"❌ FAILED: {results.get('error', 'Unknown error')}")
    
    def test_retrieval(self, query: str = "spatial join best practices"):
        """Test the retrieval functionality with a sample query."""
        print(f"\n🔍 Testing retrieval with query: '{query}'")
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])
            
            # Search the collection
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=2
            )
            
            if results['documents']:
                print(f"✅ Retrieval test successful! Found {len(results['documents'][0])} relevant chunks:")
                for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    print(f"   {i+1}. Source: {metadata['source_file']}")
                    print(f"      Content: {doc[:100]}...")
            else:
                print(f"⚠️  No results found for the test query.")
                
        except Exception as e:
            print(f"❌ Retrieval test failed: {e}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("🎓 GIS KNOWLEDGE CORPUS INGESTION")
    print("=" * 60)
    print("This script will process expert GIS knowledge and create a searchable vector database.")
    print("Make sure you have .txt files in the data/knowledge_corpus directory.\n")
    
    # Initialize the ingestion engine
    ingestion_engine = KnowledgeIngestionEngine()
    
    try:
        # Initialize components
        ingestion_engine.initialize_components()
        
        # Ingest the knowledge corpus
        results = ingestion_engine.ingest_corpus()
        
        # Display the report
        ingestion_engine.display_ingestion_report(results)
        
        # Test retrieval if ingestion was successful
        if results["success"] and results["total_chunks"] > 0:
            ingestion_engine.test_retrieval()
            
            print(f"\n🎯 Next Steps:")
            print(f"   1. Add more .txt files to {ingestion_engine.corpus_dir}")
            print(f"   2. Re-run this script to update the knowledge base")
            print(f"   3. Integrate with the RAG system in your workflow generator")
        
        print(f"\n🏁 Ingestion process completed!")
        
    except Exception as e:
        print(f"\n💥 Fatal error during ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()