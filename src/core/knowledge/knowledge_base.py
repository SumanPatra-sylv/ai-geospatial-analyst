import json
import re
import requests
import logging
import inflect
import sqlite3
from pathlib import Path
from difflib import get_close_matches, SequenceMatcher
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import hashlib


# Optional SpaCy import for advanced entity extraction
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class SpatialKnowledgeBase:
    """
    A state-of-the-art, self-expanding component for mapping natural language to OSM tags
    with enhanced workflow memory capabilities for RAG-based query planning.
    
    Features:
    - Advanced NLP entity/modifier extraction with SpaCy
    - Query explainability with chain-of-thought reasoning
    - Fuzzy matching and normalization
    - Resilient external API lookups with quality gates
    - Persistent caching and SQLite database storage
    - Workflow pattern storage and retrieval for RAG systems
    - Semantic similarity matching for query patterns
    - Performance metrics and analytics
    - Robust database operations with transactions and context managers
    - Enhanced error handling and validation
    """
    
    def __init__(self, cache_file: str = "data/cache/tag_cache.json", 
                 db_path: str = "data/knowledge_base/geospatial_knowledge.db"):
        """
        Initializes the knowledge base with persistent storage for tags and workflows.
        
        Args:
            cache_file: Path to JSON cache file for entity-tag mappings
            db_path: Path to SQLite database for workflow storage
        """
        self.logger = logging.getLogger(__name__)
        self.inflect_engine = inflect.engine()
        
        # Validate inputs
        if not cache_file or not db_path:
            raise ValueError("cache_file and db_path cannot be empty")
        
        # Core entity-to-tags mapping with confidence scores
        self.entity_to_tags_map = {
            "city": [{"tag": {"place": "city"}, "source": "local", "confidence": 1.0}],
            "town": [{"tag": {"place": "town"}, "source": "local", "confidence": 1.0}],
            "village": [{"tag": {"place": "village"}, "source": "local", "confidence": 1.0}],
            "restaurant": [{"tag": {"amenity": "restaurant"}, "source": "local", "confidence": 1.0}],
            "cafe": [{"tag": {"amenity": "cafe"}, "source": "local", "confidence": 1.0}],
            "bar": [{"tag": {"amenity": "bar"}, "source": "local", "confidence": 1.0}],
            "pub": [{"tag": {"amenity": "pub"}, "source": "local", "confidence": 1.0}],
            "school": [{"tag": {"amenity": "school"}, "source": "local", "confidence": 1.0}],
            "hospital": [{"tag": {"amenity": "hospital"}, "source": "local", "confidence": 1.0}],
            "clinic": [{"tag": {"amenity": "clinic"}, "source": "local", "confidence": 1.0}],
            "pharmacy": [{"tag": {"amenity": "pharmacy"}, "source": "local", "confidence": 1.0}],
            "bank": [{"tag": {"amenity": "bank"}, "source": "local", "confidence": 1.0}],
            "atm": [{"tag": {"amenity": "atm"}, "source": "local", "confidence": 1.0}],
            "post office": [{"tag": {"amenity": "post_office"}, "source": "local", "confidence": 1.0}],
            "police station": [{"tag": {"amenity": "police"}, "source": "local", "confidence": 1.0}],
            "fire station": [{"tag": {"amenity": "fire_station"}, "source": "local", "confidence": 1.0}],
            "cinema": [{"tag": {"amenity": "cinema"}, "source": "local", "confidence": 1.0}],
            "theater": [{"tag": {"amenity": "theatre"}, "source": "local", "confidence": 1.0}],
            "park": [{"tag": {"leisure": "park"}, "source": "local", "confidence": 1.0}],
            "playground": [{"tag": {"leisure": "playground"}, "source": "local", "confidence": 1.0}],
            "hotel": [{"tag": {"tourism": "hotel"}, "source": "local", "confidence": 1.0}],
            "supermarket": [{"tag": {"shop": "supermarket"}, "source": "local", "confidence": 1.0}],
            "shopping mall": [{"tag": {"shop": "mall"}, "source": "local", "confidence": 1.0}],
            "bakery": [{"tag": {"shop": "bakery"}, "source": "local", "confidence": 1.0}],
            "bus stop": [{"tag": {"highway": "bus_stop"}, "source": "local", "confidence": 1.0}],
            "train station": [{"tag": {"railway": "station"}, "source": "local", "confidence": 1.0}],
            "airport": [{"tag": {"aeroway": "aerodrome"}, "source": "local", "confidence": 1.0}],
            "building": [{"tag": {"building": "yes"}, "source": "local", "confidence": 1.0}],
            "residential area": [{"tag": {"landuse": "residential"}, "source": "local", "confidence": 1.0}],
            "commercial area": [{"tag": {"landuse": "commercial"}, "source": "local", "confidence": 1.0}],
            "industrial area": [{"tag": {"landuse": "industrial"}, "source": "local", "confidence": 1.0}],
            "office": [{"tag": {"building": "office"}, "source": "local", "confidence": 1.0}]
        }
        
        # Initialize database and cache
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_cache()

        # Initialize SpaCy if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("SpaCy model 'en_core_web_sm' loaded successfully.")
            except OSError:
                self.logger.warning("SpaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'. Falling back to regex.")
                self.nlp = None

    def _init_database(self) -> None:
        """Initializes the SQLite database with tables for tags and successful workflows."""
        with sqlite3.connect(self.db_path) as conn:  # ✅ Context manager
            cursor = conn.cursor()
            
            # Table for storing successful workflow patterns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS successful_workflows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT NOT NULL UNIQUE,
                    user_query TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    workflow_plan TEXT NOT NULL,
                    success_count INTEGER DEFAULT 1,
                    avg_execution_time REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table for storing learned entity-tag mappings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learned_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity TEXT NOT NULL,
                    tag_key TEXT NOT NULL,
                    tag_value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(entity, tag_key, tag_value)
                )
            """)
            
            # Table for query analytics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_workflow_query_hash ON successful_workflows(query_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_workflow_normalized ON successful_workflows(normalized_query)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_learned_tags_entity ON learned_tags(entity)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analytics_query_hash ON query_analytics(query_hash)")
            
            # ✅ Add schema versioning
            cursor.execute("CREATE TABLE IF NOT EXISTS schema_info (version INTEGER PRIMARY KEY)")
            cursor.execute("INSERT OR IGNORE INTO schema_info (version) VALUES (1)")
            # For future migrations: Add logic here to check version and alter tables if needed
            
            conn.commit()  # ✅ Explicit commit

    def _normalize_query(self, query: str) -> str:
        """
        Normalizes a query for better matching.
        
        Args:
            query: Raw input query
            
        Returns:
            Normalized query string
        """
        if not query or not isinstance(query, str):
            return ""
        
        # Remove extra whitespace, convert to lowercase, remove punctuation
        normalized = re.sub(r'[^\w\s]', '', query.lower().strip())
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def _get_query_hash(self, query: str) -> str:
        """
        Generates a hash for a query for efficient storage and retrieval.
        
        Args:
            query: Query string to hash
            
        Returns:
            SHA-256 hash of the query
        """
        return hashlib.sha256(query.encode()).hexdigest()  # ✅ Upgraded to SHA-256

    def store_successful_workflow(self, original_query: str, workflow_plan: List[Dict], 
                                  execution_time: float = 0.0) -> bool:
        """
        Stores a successful query and its corresponding workflow plan in the database.
        
        Args:
            original_query: The original user query  # ✅ Fixed parameter name
            workflow_plan: The successful workflow plan as a list of dictionaries
            execution_time: Time taken to execute the workflow
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        # Input validation
        if not original_query or not isinstance(original_query, str):
            self.logger.error("Invalid original_query provided", extra={'query': original_query})
            return False
        
        if not workflow_plan or not isinstance(workflow_plan, list):
            self.logger.error("Invalid workflow_plan provided", extra={'plan': workflow_plan})
            return False
        
        try:
            normalized_query = self._normalize_query(original_query)
            query_hash = self._get_query_hash(normalized_query)
            plan_json = json.dumps(workflow_plan, indent=2)
            
            with sqlite3.connect(self.db_path) as conn:  # ✅ Context manager
                cursor = conn.cursor()
                
                # Check if workflow already exists
                cursor.execute("SELECT id, success_count, avg_execution_time FROM successful_workflows WHERE query_hash = ?", 
                             (query_hash,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing workflow with new execution data
                    workflow_id, success_count, avg_time = existing
                    new_count = success_count + 1
                    new_avg_time = ((avg_time * success_count) + execution_time) / new_count
                    
                    cursor.execute("""
                        UPDATE successful_workflows 
                        SET success_count = ?, avg_execution_time = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (new_count, new_avg_time, workflow_id))
                    
                    self.logger.info(f"Updated existing workflow pattern (#{workflow_id}) success count: {new_count}", 
                                   extra={'workflow_id': workflow_id, 'new_count': new_count})
                else:
                    # Insert new workflow
                    cursor.execute("""
                        INSERT INTO successful_workflows 
                        (query_hash, user_query, normalized_query, workflow_plan, avg_execution_time)
                        VALUES (?, ?, ?, ?, ?)
                    """, (query_hash, original_query, normalized_query, plan_json, execution_time))
                    
                    self.logger.info(f"Stored new workflow pattern for query: '{original_query}'", 
                                   extra={'query': original_query, 'execution_time': execution_time})
                
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Database error storing workflow: {e}", 
                            extra={'error_type': type(e).__name__, 'query': original_query})
            return False
        except json.JSONEncodeError as e:
            self.logger.error(f"JSON encoding error for workflow plan: {e}", 
                            extra={'error_type': type(e).__name__})
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error storing workflow: {e}", 
                            extra={'error_type': type(e).__name__})
            return False

    def retrieve_similar_workflows(self, user_query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves similar successful workflows based on query similarity.
        
        Args:
            user_query: The user's query
            limit: Maximum number of workflows to return
            
        Returns:
            List of similar workflows with metadata
        """
        if not user_query or not isinstance(user_query, str):
            self.logger.warning("Invalid user_query provided for similarity search")
            return []
        
        if limit <= 0:
            limit = 5
        
        try:
            normalized_query = self._normalize_query(user_query)
            with sqlite3.connect(self.db_path) as conn:  # ✅ Context manager
                cursor = conn.cursor()
                
                # Get workflows ordered by success count and recency (optimized limit)
                cursor.execute("""
                    SELECT user_query, normalized_query, workflow_plan, success_count, 
                           avg_execution_time, created_at, updated_at
                    FROM successful_workflows
                    ORDER BY success_count DESC, updated_at DESC
                    LIMIT ?
                """, (limit * 2,))  # Get more than needed for similarity filtering
                
                workflows = cursor.fetchall()
                similar_workflows = []
                
                for workflow in workflows:
                    (orig_query, norm_query, plan_json, success_count, 
                     avg_time, created_at, updated_at) = workflow
                    
                    # Calculate similarity score with configurable parameters
                    similarity = self._calculate_query_similarity(normalized_query, norm_query)
                    
                    if similarity > 0.3:  # Threshold for similarity
                        try:
                            workflow_plan = json.loads(plan_json)
                            similar_workflows.append({
                                'original_query': orig_query,
                                'normalized_query': norm_query,
                                'workflow_plan': workflow_plan,
                                'success_count': success_count,
                                'avg_execution_time': avg_time,
                                'similarity_score': similarity,
                                'created_at': created_at,
                                'updated_at': updated_at
                            })
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Failed to decode workflow plan JSON: {e}")
                            continue
                
                # Sort by similarity score and return top results
                similar_workflows.sort(key=lambda x: x['similarity_score'], reverse=True)
                return similar_workflows[:limit]
                
        except sqlite3.Error as e:
            self.logger.error(f"Database error retrieving workflows: {e}", 
                            extra={'error_type': type(e).__name__})
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error retrieving workflows: {e}", 
                            extra={'error_type': type(e).__name__})
            return []

    def _calculate_query_similarity(self, query1: str, query2: str, 
                                    jaccard_weight: float = 0.7, seq_weight: float = 0.3, 
                                    min_threshold: float = 0.0) -> float:
        """
        Calculates similarity between two queries using multiple methods.
        
        Args:
            query1: First query (normalized)
            query2: Second query (normalized)
            jaccard_weight: Weight for Jaccard similarity
            seq_weight: Weight for sequence similarity
            min_threshold: Minimum similarity to consider (returns 0 below this)
            
        Returns:
            Similarity score between 0 and 1
        """
        # Input validation
        if not isinstance(query1, str) or not isinstance(query2, str):
            return 0.0
        
        # Jaccard similarity for word overlap
        words1 = set(query1.split())
        words2 = set(query2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard = intersection / union if union > 0 else 0.0
        
        # Sequence similarity using difflib
        sequence_similarity = SequenceMatcher(None, query1, query2).ratio()
        
        # Combined score (weighted average)
        combined = (jaccard * jaccard_weight) + (sequence_similarity * seq_weight)
        return combined if combined > min_threshold else 0.0  # ✅ Apply configurable threshold

    def store_query_analytics(self, query: str, execution_time: float, 
                            success: bool, error_message: Optional[str] = None) -> None:
        """
        Stores query execution analytics for performance monitoring.
        
        Args:
            query: The executed query
            execution_time: Time taken to execute
            success: Whether execution was successful
            error_message: Error message if execution failed
        """
        if not query or not isinstance(query, str):
            return
        
        try:
            query_hash = self._get_query_hash(self._normalize_query(query))
            with sqlite3.connect(self.db_path) as conn:  # ✅ Context manager
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO query_analytics (query_hash, execution_time, success, error_message)
                    VALUES (?, ?, ?, ?)
                """, (query_hash, execution_time, success, error_message))
                
                conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to store query analytics: {e}", 
                            extra={'error_type': type(e).__name__})

    def get_workflow_statistics(self) -> Dict[str, Any]:
        """
        Returns statistics about stored workflows and performance.
        
        Returns:
            Dictionary containing workflow statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:  # ✅ Context manager
                cursor = conn.cursor()
                
                # Get workflow statistics
                cursor.execute("SELECT COUNT(*) FROM successful_workflows")
                total_workflows = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(success_count) FROM successful_workflows")
                avg_success_count = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT AVG(avg_execution_time) FROM successful_workflows")
                avg_execution_time = cursor.fetchone()[0] or 0
                
                # Get analytics statistics
                cursor.execute("SELECT COUNT(*) FROM query_analytics WHERE success = 1")
                successful_queries = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM query_analytics WHERE success = 0")
                failed_queries = cursor.fetchone()[0]
                
                success_rate = (successful_queries / (successful_queries + failed_queries) * 100) if (successful_queries + failed_queries) > 0 else 0
                
                return {
                    'total_workflows': total_workflows,
                    'avg_success_count': round(avg_success_count, 2),
                    'avg_execution_time': round(avg_execution_time, 3),
                    'successful_queries': successful_queries,
                    'failed_queries': failed_queries,
                    'success_rate': round(success_rate, 2)
                }
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get workflow statistics: {e}", 
                            extra={'error_type': type(e).__name__})
            return {}

    def cleanup_database(self, max_age_days: int = 30) -> None:
        """
        Prune old analytics and workflows to prevent database bloat.
        
        Args:
            max_age_days: Maximum age in days for records to keep
        """
        try:
            cutoff = datetime.now() - timedelta(days=max_age_days)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clean up old analytics
                cursor.execute("DELETE FROM query_analytics WHERE created_at < ?", (cutoff.isoformat(),))
                deleted_analytics = cursor.rowcount
                
                # Clean up old workflows (but keep successful ones longer)
                cursor.execute("DELETE FROM successful_workflows WHERE updated_at < ? AND success_count < 2", 
                             (cutoff.isoformat(),))
                deleted_workflows = cursor.rowcount
                
                conn.commit()
                self.logger.info(f"Cleaned up database: {deleted_analytics} analytics, {deleted_workflows} workflows older than {max_age_days} days.", 
                               extra={'deleted_analytics': deleted_analytics, 'deleted_workflows': deleted_workflows})
        except sqlite3.Error as e:
            self.logger.error(f"Failed to clean up database: {e}", 
                            extra={'error_type': type(e).__name__})

    # === Enhanced existing methods ===
    
    def _load_cache(self) -> None:
        """Load cached entity-tag mappings from JSON file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    cached_data = json.load(f)
                    self.entity_to_tags_map.update(cached_data)
                    self.logger.info(f"Loaded {len(cached_data)} cached entity mappings", 
                                   extra={'cache_size': len(cached_data)})
            except (json.JSONDecodeError, IOError) as e:
                self.logger.error(f"Failed to load cache file: {e}", 
                                extra={'error_type': type(e).__name__})

    def _save_cache(self) -> None:
        """Save entity-tag mappings to JSON cache file."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.entity_to_tags_map, f, indent=2)
            self.logger.debug("Cache saved successfully")
        except (json.JSONEncodeError, IOError) as e:
            self.logger.error(f"Failed to save cache file: {e}", 
                            extra={'error_type': type(e).__name__})

    def add_entity(self, entity: str, tags_with_metadata: List[Dict[str, Any]]) -> bool:
        """
        Add a new entity-tag mapping to the knowledge base.
        
        Args:
            entity: Entity name to add
            tags_with_metadata: List of tag dictionaries with metadata
            
        Returns:
            bool: True if added successfully
        """
        if not entity or not tags_with_metadata:
            return False
        
        try:
            normalized_entity = self._normalize_entity(entity)
            self.entity_to_tags_map[normalized_entity] = tags_with_metadata
            self._save_cache()
            
            # Also store in database
            self._store_learned_tags(normalized_entity, tags_with_metadata)
            return True
        except Exception as e:
            self.logger.error(f"Failed to add entity '{entity}': {e}", 
                            extra={'error_type': type(e).__name__, 'entity': entity})
            return False

    def _store_learned_tags(self, entity: str, tags_with_metadata: List[Dict[str, Any]]) -> None:
        """Store learned entity-tag mappings in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:  # ✅ Context manager
                cursor = conn.cursor()
                conn.execute("BEGIN")  # ✅ Start transaction for batch
                
                for tag_data in tags_with_metadata:
                    tag = tag_data.get('tag', {})
                    confidence = tag_data.get('confidence', 0.5)
                    source = tag_data.get('source', 'unknown')
                    
                    for key, value in tag.items():
                        # Check if exists
                        cursor.execute("""
                            SELECT usage_count FROM learned_tags 
                            WHERE entity = ? AND tag_key = ? AND tag_value = ?
                        """, (entity, key, value))
                        existing = cursor.fetchone()
                        
                        if existing:
                            new_count = existing[0] + 1  # ✅ Increment usage_count
                            cursor.execute("""
                                UPDATE learned_tags 
                                SET usage_count = ?, confidence = ?, source = ?, created_at = CURRENT_TIMESTAMP
                                WHERE entity = ? AND tag_key = ? AND tag_value = ?
                            """, (new_count, confidence, source, entity, key, value))
                        else:
                            cursor.execute("""
                                INSERT INTO learned_tags 
                                (entity, tag_key, tag_value, confidence, source, usage_count)
                                VALUES (?, ?, ?, ?, ?, 1)
                            """, (entity, key, value, confidence, source))
                
                conn.commit()  # ✅ Commit transaction
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to store learned tags: {e}", 
                            extra={'error_type': type(e).__name__})
            if 'conn' in locals():
                conn.rollback()  # ✅ Rollback on error

    def _normalize_entity(self, entity: str) -> str:
        """
        Normalize entity name to singular form and lowercase.
        
        Args:
            entity: Entity name to normalize
            
        Returns:
            Normalized entity name
        """
        if not entity or not isinstance(entity, str):
            return ""
        
        singular = self.inflect_engine.singular_noun(entity.lower().strip())
        return singular or entity.lower().strip()

    def extract_entities(self, sentence: str) -> List[Dict[str, Any]]:
        """
        Extract entities and modifiers from a sentence.
        
        Args:
            sentence: Input sentence to analyze
            
        Returns:
            List of extracted entities with metadata
        """
        if not sentence or not isinstance(sentence, str):
            return []
        
        if self.nlp:
            return self._extract_entities_spacy(sentence)
        else:
            return self._extract_entities_regex(sentence)

    def _extract_entities_spacy(self, sentence: str) -> List[Dict[str, Any]]:
        """Use SpaCy to extract noun chunks and identify entities/modifiers."""
        doc = self.nlp(sentence)
        found_entities = []
        
        for chunk in doc.noun_chunks:
            root_entity_normalized = self._normalize_entity(chunk.root.text)
            if root_entity_normalized in self.entity_to_tags_map:
                modifiers = [token.text for token in chunk if token.text != chunk.root.text and not token.is_stop]
                found_entities.append({
                    "entity": root_entity_normalized,
                    "modifiers": modifiers,
                    "original_phrase": chunk.text
                })
        
        return found_entities

    def _extract_entities_regex(self, sentence: str) -> List[Dict[str, Any]]:
        """Fallback regex-based entity extraction."""
        self.logger.warning("Using fallback regex extraction. Install SpaCy for better results.")
        lower_sentence = sentence.lower()
        known_entities = sorted(self.entity_to_tags_map.keys(), key=len, reverse=True)
        found = []
        
        for entity in known_entities:
            if re.search(rf"\b{re.escape(entity)}\b", lower_sentence):
                found.append({
                    "entity": entity,
                    "modifiers": [],
                    "original_phrase": entity
                })
        
        return found

    def get_candidate_tags(self, entity: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieve scored tags for an entity with chain-of-thought explanation.
        
        Args:
            entity: Entity to get tags for
            
        Returns:
            Tuple of (candidate_tags, explanation)
        """
        normalized = self._normalize_entity(entity)
        explanation = {
            "original_input": entity,
            "normalized": normalized,
            "timestamp": datetime.now().isoformat()
        }

        if not normalized:
            explanation["source"] = "error"
            explanation["reasoning"] = "Invalid entity input"
            return [], explanation

        # Direct match
        if normalized in self.entity_to_tags_map:
            explanation["source"] = "local_direct"
            explanation["reasoning"] = f"Direct match found for '{normalized}' in local knowledge base."
            return self.entity_to_tags_map[normalized], explanation

        # Fuzzy match
        match = self.get_closest_key(normalized)
        if match:
            explanation["source"] = "local_fuzzy"
            explanation["reasoning"] = f"Fuzzy match found: '{match}' for input '{normalized}'."
            tags = self.entity_to_tags_map[match]
            scored_tags = [{"tag": t["tag"], "source": "local_fuzzy", "confidence": 0.9} for t in tags]
            return scored_tags, explanation

        # External API query
        explanation["source"] = "external_api"
        explanation["reasoning"] = "No local match found. Querying external Taginfo API."
        tags = self.query_external_sources(normalized)
        return tags, explanation

    def query_external_sources(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Query Taginfo API for tag suggestions with quality filtering.
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            List of quality-filtered tag suggestions
        """
        if not keyword:
            return []
        
        self.logger.info(f"Querying Taginfo API for '{keyword}'...")
        url = "https://taginfo.openstreetmap.org/api/4/search/by_keyword"
        params = {
            "q": keyword,
            "lang": "en",
            "sortname": "count_all",
            "sortorder": "desc"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json().get("data", [])
            
            results = []
            for item in data:
                # Enhanced quality gates
                is_documented = item.get("in_wiki", False)
                has_usage = item.get("count_all", 0) > 1000
                has_key_value = item.get("key") and item.get("value")
                
                if has_key_value and (is_documented or has_usage):
                    confidence = 0.8 if is_documented else 0.6
                    results.append({
                        "tag": {item["key"]: item["value"]},
                        "source": "external",
                        "confidence": confidence
                    })
                
                if len(results) >= 3:
                    break
            
            if results:
                self.add_entity(keyword, results)
                self.logger.info(f"Found {len(results)} quality tags for '{keyword}'", 
                               extra={'keyword': keyword, 'results_count': len(results)})
            else:
                self.logger.warning(f"No quality results from Taginfo for '{keyword}'", 
                                  extra={'keyword': keyword})
            
            return results
            
        except requests.RequestException as e:
            self.logger.error(f"Taginfo API request failed for '{keyword}': {e}", 
                            extra={'error_type': type(e).__name__, 'keyword': keyword})
            return []

    def get_closest_key(self, query: str) -> Optional[str]:
        """
        Find the closest matching key using fuzzy string matching.
        
        Args:
            query: Query to find matches for
            
        Returns:
            Closest matching key or None
        """
        if not query:
            return None
        
        all_keys = list(self.entity_to_tags_map.keys())
        matches = get_close_matches(query, all_keys, n=1, cutoff=0.8)
        return matches[0] if matches else None

    def tag_entities_in_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        """
        Extract entities from sentence and get their candidate tags.
        
        Args:
            sentence: Input sentence to analyze
            
        Returns:
            List of entities with their candidate tags and explanations
        """
        if not sentence:
            return []
        
        extracted_groups = self.extract_entities(sentence)
        results = []
        seen_entities = set()
        
        for group in extracted_groups:
            entity = group["entity"]
            if entity in seen_entities:
                continue
            seen_entities.add(entity)
            
            tags, explanation = self.get_candidate_tags(entity)
            results.append({
                **group,
                "candidate_tags": tags,
                "explanation": explanation
            })
        
        return results


# === Enhanced Example Usage & Testing ===
if __name__ == "__main__":
    kb = SpatialKnowledgeBase()
    
    print("=== Enhanced SpatialKnowledgeBase with Workflow Memory ===\n")
    
    # Test workflow storage
    print("1. Testing Workflow Storage")
    sample_workflow = [
        {"step": "extract_entities", "entities": ["restaurant", "hospital"]},
        {"step": "get_osm_tags", "tags": {"amenity": "restaurant"}},
        {"step": "execute_query", "query": "overpass query"}
    ]
    
    success = kb.store_successful_workflow(
        "Find restaurants near hospitals",  # ✅ Uses original_query
        sample_workflow,
        execution_time=1.23
    )
    print(f"Workflow stored: {success}")
    
    # Test workflow retrieval
    print("\n2. Testing Workflow Retrieval")
    similar = kb.retrieve_similar_workflows("restaurants close to medical facilities")
    print(f"Found {len(similar)} similar workflows")
    for i, workflow in enumerate(similar):
        print(f"  {i+1}. Query: '{workflow['original_query']}' (similarity: {workflow['similarity_score']:.2f})")
    
    # Test statistics
    print("\n3. Testing Statistics")
    stats = kb.get_workflow_statistics()
    print(f"Workflow Statistics: {json.dumps(stats, indent=2)}")
    
    # Test existing functionality
    print("\n4. Testing Entity Extraction")
    sentence = "Show me restaurants and hospitals near parks"
    tagged = kb.tag_entities_in_sentence(sentence)
    print(f"Extracted entities: {len(tagged)}")
    for entity_data in tagged:
        print(f"  - {entity_data['entity']}: {len(entity_data['candidate_tags'])} tags")
    
    # ✅ Added: Test edge cases
    print("\n5. Testing Edge Cases")
    # Empty query
    similar_empty = kb.retrieve_similar_workflows("")
    print(f"Similar workflows for empty query: {len(similar_empty)}")
    
    # DB cleanup
    kb.cleanup_database(max_age_days=1)  # Test with small age for demo
    
    # Invalid storage
    success_invalid = kb.store_successful_workflow("Invalid", [], execution_time=0.0)
    print(f"Invalid workflow stored: {success_invalid}")
    
    # Test analytics
    kb.store_query_analytics("test query", 0.5, True)
    kb.store_query_analytics("failed query", 2.0, False, "Test error")
    
    print("\n=== Enhanced Knowledge Base Ready for RAG Integration ===")