#!/usr/bin/env python3
import argparse
import sys
import logging
import os
from src.core.orchestrator import MasterOrchestrator
from src.core.planners.query_parser import ParsedQuery, QueryParser, QueryParserError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="AI Geospatial Analyst CLI")
    parser.add_argument("query", nargs="?", help="The geospatial query to execute")
    parser.add_argument("--location", help="Target location (if not in query)")
    parser.add_argument("--target", help="Target entity (if not in query)")
    parser.add_argument("--max-loops", type=int, default=15, help="Maximum number of reasoning loops")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG features (if dependencies are missing)")
    
    args = parser.parse_args()
    
    # Check for Ollama connection if query parsing is needed
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    if not args.query and not (args.location and args.target):
        print("Error: Please provide a query or specify --location and --target")
        sys.exit(1)

    parsed_query = None

    if args.query:
        print(f"🧠 Parsing query: '{args.query}'...")
        try:
            # Try to use the LLM-based parser first
            qp = QueryParser()
            parsed_query = qp.parse(args.query)
            print(f"✅ Query parsed successfully: Target='{parsed_query.target}', Location='{parsed_query.location}'")
        except (QueryParserError, Exception) as e:
            logger.warning(f"LLM Query Parsing failed: {e}")
            logger.info("Falling back to simple keyword extraction...")
            
            # Fallback logic
            query_text = args.query
            target = args.target or "amenity" 
            location = args.location or "London"
            
            if " in " in query_text:
                parts = query_text.split(" in ")
                target = parts[0].replace("Find ", "").strip()
                location = parts[1].strip()
            
            parsed_query = ParsedQuery(
                target=target,
                location=location,
                constraints=[],
                summary_required=True
            )
    else:
        parsed_query = ParsedQuery(
            target=args.target,
            location=args.location,
            constraints=[],
            summary_required=True
        )
        
    print(f"🚀 Starting Analyst with query: Find '{parsed_query.target}' in '{parsed_query.location}'")
    
    # Initialize Orchestrator
    # Note: We might want to pass args.no_rag to the orchestrator if we modify it to accept that flag
    orchestrator = MasterOrchestrator(max_loops=args.max_loops)
    
    try:
        result = orchestrator.run(parsed_query)
        
        if result["success"]:
            print("\n✅ Mission Accomplished!")
            if result.get('final_result') is not None:
                print(f"Final Result Layer: {result.get('final_layer_name')}")
                # print(result['final_result']) # Might be too large to print
        else:
            print("\n❌ Mission Failed.")
            if 'error' in result:
                print(f"Error: {result['error']}")
            
    except Exception as e:
        logger.error(f"Orchestrator execution failed: {e}", exc_info=True)
        sys.exit(1)
        
if __name__ == "__main__":
    main()
