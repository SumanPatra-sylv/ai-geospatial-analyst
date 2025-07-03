# rag_knowledge.py - Enhanced RAG Integration for Geospatial Workflows
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sqlite3
from sentence_transformers import SentenceTransformer
import faiss

class GeospatialKnowledgeBase:
    """
    RAG-enhanced knowledge base for geospatial workflow patterns and best practices.
    Stores successful workflows, common patterns, and expert knowledge.
    """
    
    def __init__(self, db_path: str = "geospatial_knowledge.db"):
        self.db_path = db_path
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.workflows = []
        self._init_database()
        self._load_workflows()
        
    def _init_database(self):
        """Initialize SQLite database for workflow storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workflows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                workflow_steps TEXT NOT NULL,
                success_rate REAL DEFAULT 1.0,
                execution_time REAL,
                data_types TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workflow_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT NOT NULL,
                description TEXT,
                tools_sequence TEXT,
                use_cases TEXT,
                complexity_level INTEGER
            )
        ''')
        
        # Insert common workflow patterns
        patterns = [
            ("flood_risk_analysis", "Identify flood-prone areas using elevation and proximity to water bodies", 
             "LoadRasterData,LoadVectorData,CreateBuffer,IntersectLayers,CalculateZonalStatistics", 
             "flood mapping,risk assessment,emergency planning", 3),
            ("site_suitability", "Find optimal locations based on multiple criteria",
             "LoadVectorData,FilterByAttribute,CreateBuffer,IntersectLayers",
             "urban planning,facility location,development", 2),
            ("proximity_analysis", "Analyze spatial relationships and distances",
             "LoadVectorData,CreateBuffer,IntersectLayers",
             "accessibility,catchment analysis,service areas", 1),
            ("land_cover_statistics", "Calculate statistics within administrative boundaries",
             "LoadVectorData,LoadRasterData,CalculateZonalStatistics",
             "environmental monitoring,agriculture,urban studies", 2)
        ]
        
        cursor.executemany('''
            INSERT OR IGNORE INTO workflow_patterns 
            (pattern_name, description, tools_sequence, use_cases, complexity_level)
            VALUES (?, ?, ?, ?, ?)
        ''', patterns)
        
        conn.commit()
        conn.close()
        
    def _load_workflows(self):
        """Load existing workflows and build FAISS index."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT query, workflow_steps FROM workflows')
        results = cursor.fetchall()
        
        if results:
            queries = [row[0] for row in results]
            self.workflows = results
            
            # Create embeddings and FAISS index
            embeddings = self.encoder.encode(queries)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings.astype('float32'))
            
        conn.close()
        
    def find_similar_workflows(self, query: str, k: int = 3) -> List[Tuple[str, str, float]]:
        """Find similar workflows using semantic search."""
        if not self.index or len(self.workflows) == 0:
            return []
            
        query_embedding = self.encoder.encode([query])
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.workflows):
                original_query, workflow_steps = self.workflows[idx]
                results.append((original_query, workflow_steps, float(score)))
                
        return results
        
    def get_workflow_suggestions(self, query: str) -> str:
        """Generate workflow suggestions based on query analysis."""
        suggestions = []
        query_lower = query.lower()
        
        # Pattern matching for common scenarios
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT pattern_name, description, tools_sequence 
            FROM workflow_patterns 
            WHERE use_cases LIKE ?
        ''', (f'%{query_lower}%',))
        
        patterns = cursor.fetchall()
        
        if patterns:
            suggestions.append("🔍 RECOMMENDED WORKFLOW PATTERNS:")
            for pattern_name, description, tools_sequence in patterns:
                tools = tools_sequence.split(',')
                suggestions.append(f"• {pattern_name.replace('_', ' ').title()}: {description}")
                suggestions.append(f"  Typical sequence: {' → '.join(tools)}")
                
        # Similar workflows
        similar = self.find_similar_workflows(query, k=2)
        if similar:
            suggestions.append("\n📚 SIMILAR PAST WORKFLOWS:")
            for orig_query, workflow, score in similar:
                if score > 0.7:  # High similarity threshold
                    suggestions.append(f"• Query: {orig_query}")
                    suggestions.append(f"  Workflow: {workflow}")
                    
        conn.close()
        
        return "\n".join(suggestions) if suggestions else "No specific workflow patterns found for this query."
        
    def store_successful_workflow(self, query: str, workflow_steps: List[str], 
                                execution_time: float, data_types: List[str]):
        """Store a successful workflow for future reference."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        workflow_json = json.dumps(workflow_steps)
        data_types_json = json.dumps(data_types)
        
        cursor.execute('''
            INSERT INTO workflows (query, workflow_steps, execution_time, data_types)
            VALUES (?, ?, ?, ?)
        ''', (query, workflow_json, execution_time, data_types_json))
        
        conn.commit()
        conn.close()
        
        # Rebuild index with new workflow
        self._load_workflows()

# Enhanced prompt templates with RAG integration
ENHANCED_GEOSPATIAL_PROMPTS = {
    "system_prompt": """You are an Expert AI Geospatial Analyst with access to a knowledge base of successful workflows.

ENHANCED CAPABILITIES:
- Access to proven workflow patterns and best practices
- Knowledge of common geospatial analysis scenarios
- Understanding of data relationships and processing sequences

WORKFLOW PLANNING PRINCIPLES:
1. **Consult Knowledge Base**: Consider similar past workflows and established patterns
2. **Data-Driven Decisions**: Always check available data first using ListAvailableData
3. **Logical Sequencing**: Follow proven sequences (load → process → analyze → save)
4. **Error Prevention**: Anticipate common issues like CRS mismatches, empty results
5. **Efficiency**: Choose the most direct path to the solution

COMMON WORKFLOW PATTERNS:
• Flood Risk: DEM + Water Bodies → Buffer → Intersection → Zonal Stats
• Site Suitability: Multiple Criteria → Filters → Buffers → Intersection
• Accessibility: Points of Interest → Buffer → Population → Zonal Stats
• Land Use Analysis: Boundaries + Land Cover → Intersection → Statistics

Remember: Break complex problems into logical, sequential steps.""",

    "flood_analysis_template": """For flood risk analysis, follow this proven pattern:
1. Load elevation data (DEM/raster) and water body vectors
2. Create buffer zones around water bodies (typical: 100-500m)
3. Load administrative boundaries or study area
4. Intersect buffer zones with study area
5. Calculate zonal statistics of elevation within flood zones
6. Filter areas below flood threshold elevation""",

    "suitability_template": """For site suitability analysis:
1. Define all criteria layers (roads, land use, slopes, etc.)
2. Apply filters to each layer based on suitability criteria
3. Create appropriate buffer zones for proximity factors
4. Intersect all suitable areas to find optimal locations
5. Rank results if multiple criteria weights are specified"""
}

def enhance_agent_with_rag(agent_executor, knowledge_base: GeospatialKnowledgeBase):
    """Enhance existing agent with RAG capabilities."""
    
    def enhanced_planning_tool(query: str) -> str:
        """Tool that provides workflow suggestions based on knowledge base."""
        suggestions = knowledge_base.get_workflow_suggestions(query)
        return f"📋 WORKFLOW GUIDANCE:\n{suggestions}"
    
    # Add the RAG tool to existing tools
    from langchain.tools import Tool
    
    rag_tool = Tool(
        name="GetWorkflowGuidance",
        func=enhanced_planning_tool,
        description="Get workflow suggestions and patterns based on the query. Use this after ListAvailableData to plan your approach."
    )
    
    # Insert RAG tool at the beginning of tools list
    agent_executor.tools.insert(1, rag_tool)
    
    return agent_executor

# Usage example for integration
def create_enhanced_geospatial_agent():
    """Create agent with RAG enhancement."""
    from src.core.agent import create_geospatial_agent # Import base agent
    
    # Initialize knowledge base
    kb = GeospatialKnowledgeBase()
    
    # Create base agent
    agent = create_geospatial_agent()
    
    # Enhance with RAG
    enhanced_agent = enhance_agent_with_rag(agent, kb)
    
    return enhanced_agent, kb