import json
import re
import requests
import logging
import inflect
from pathlib import Path
from difflib import get_close_matches
from typing import List, Dict, Optional, Any, Tuple

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
    A state-of-the-art, self-expanding component for mapping natural language to OSM tags.
    Features: Advanced NLP entity/modifier extraction, query explainability (chain-of-thought),
    normalization, fuzzy matching, resilient external lookups with quality gates,
    persistent caching, and high-level integration hooks.
    """
    def __init__(self, cache_file: str = "kb_cache.json"):
        self.logger = logging.getLogger(__name__)
        self.inflect_engine = inflect.engine()
        
        # --- (Knowledge base dictionary remains the same, with scored tags) ---
        self.entity_to_tags_map = {
            "city": [{"tag": {"place": "city"}, "source": "local", "confidence": 1.0}], "town": [{"tag": {"place": "town"}, "source": "local", "confidence": 1.0}], "village": [{"tag": {"place": "village"}, "source": "local", "confidence": 1.0}],
            "restaurant": [{"tag": {"amenity": "restaurant"}, "source": "local", "confidence": 1.0}], "cafe": [{"tag": {"amenity": "cafe"}, "source": "local", "confidence": 1.0}], "bar": [{"tag": {"amenity": "bar"}, "source": "local", "confidence": 1.0}], "pub": [{"tag": {"amenity": "pub"}, "source": "local", "confidence": 1.0}], "school": [{"tag": {"amenity": "school"}, "source": "local", "confidence": 1.0}], "hospital": [{"tag": {"amenity": "hospital"}, "source": "local", "confidence": 1.0}], "clinic": [{"tag": {"amenity": "clinic"}, "source": "local", "confidence": 1.0}], "pharmacy": [{"tag": {"amenity": "pharmacy"}, "source": "local", "confidence": 1.0}], "bank": [{"tag": {"amenity": "bank"}, "source": "local", "confidence": 1.0}], "atm": [{"tag": {"amenity": "atm"}, "source": "local", "confidence": 1.0}],
            "post office": [{"tag": {"amenity": "post_office"}, "source": "local", "confidence": 1.0}], "police station": [{"tag": {"amenity": "police"}, "source": "local", "confidence": 1.0}], "fire station": [{"tag": {"amenity": "fire_station"}, "source": "local", "confidence": 1.0}], "cinema": [{"tag": {"amenity": "cinema"}, "source": "local", "confidence": 1.0}], "theater": [{"tag": {"amenity": "theatre"}, "source": "local", "confidence": 1.0}], "park": [{"tag": {"leisure": "park"}, "source": "local", "confidence": 1.0}], "playground": [{"tag": {"leisure": "playground"}, "source": "local", "confidence": 1.0}], "hotel": [{"tag": {"tourism": "hotel"}, "source": "local", "confidence": 1.0}],
            "supermarket": [{"tag": {"shop": "supermarket"}, "source": "local", "confidence": 1.0}], "shopping mall": [{"tag": {"shop": "mall"}, "source": "local", "confidence": 1.0}], "bakery": [{"tag": {"shop": "bakery"}, "source": "local", "confidence": 1.0}],
            "bus stop": [{"tag": {"highway": "bus_stop"}, "source": "local", "confidence": 1.0}], "train station": [{"tag": {"railway": "station"}, "source": "local", "confidence": 1.0}], "airport": [{"tag": {"aeroway": "aerodrome"}, "source": "local", "confidence": 1.0}],
            "building": [{"tag": {"building": "yes"}, "source": "local", "confidence": 1.0}], "residential area": [{"tag": {"landuse": "residential"}, "source": "local", "confidence": 1.0}], "commercial area": [{"tag": {"landuse": "commercial"}, "source": "local", "confidence": 1.0}], "industrial area": [{"tag": {"landuse": "industrial"}, "source": "local", "confidence": 1.0}], "office": [{"tag": {"building": "office"}, "source": "local", "confidence": 1.0}]
        }
        self.cache_file = Path(cache_file)
        self._load_cache()

        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("SpaCy model 'en_core_web_sm' loaded successfully.")
            except OSError:
                self.logger.warning("SpaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'. Falling back to regex.")
                self.nlp = None
    
    # --- Core Data and Cache Methods ---
    def _load_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f: self.entity_to_tags_map.update(json.load(f))
            except Exception as e: self.logger.error(f"Failed to load cache file: {e}")

    def _save_cache(self):
        try:
            with open(self.cache_file, "w") as f: json.dump(self.entity_to_tags_map, f, indent=2)
        except Exception as e: self.logger.error(f"Failed to save cache file: {e}")

    def add_entity(self, entity: str, tags_with_metadata: List[Dict[str, Any]]):
        if not entity or not tags_with_metadata: return
        normalized_entity = self._normalize_entity(entity)
        self.entity_to_tags_map[normalized_entity] = tags_with_metadata
        self._save_cache()

    # --- Entity Normalization and Extraction ---
    def _normalize_entity(self, entity: str) -> str:
        singular = self.inflect_engine.singular_noun(entity.lower().strip())
        return singular or entity.lower().strip()

    def extract_entities(self, sentence: str) -> List[Dict[str, Any]]:
        """Extracts entities and modifiers, using SpaCy if available."""
        if self.nlp:
            return self._extract_entities_spacy(sentence)
        else:
            return self._extract_entities_regex(sentence)

    def _extract_entities_spacy(self, sentence: str) -> List[Dict[str, Any]]:
        """Uses SpaCy to find noun chunks and identify entities/modifiers."""
        doc = self.nlp(sentence)
        found_entities = []
        for chunk in doc.noun_chunks:
            root_entity_normalized = self._normalize_entity(chunk.root.text)
            if root_entity_normalized in self.entity_to_tags_map:
                modifiers = [token.text for token in chunk if token.text != chunk.root.text]
                found_entities.append({
                    "entity": root_entity_normalized, "modifiers": modifiers, "original_phrase": chunk.text
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
                found.append({"entity": entity, "modifiers": [], "original_phrase": entity})
        return found
    
    # --- Tag Retrieval and Explainability ---
    def get_candidate_tags(self, entity: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Retrieves scored tags for an entity and generates a 'chain-of-thought' explanation."""
        normalized = self._normalize_entity(entity)
        explanation = {"original_input": entity, "normalized": normalized}

        if normalized in self.entity_to_tags_map:
            explanation["source"] = "local_direct"
            explanation["reasoning"] = f"Matched directly to '{normalized}' in the local knowledge base."
            return self.entity_to_tags_map[normalized], explanation

        match = self.get_closest_key(normalized)
        if match:
            explanation["source"] = "local_fuzzy"
            explanation["reasoning"] = f"No direct match found. Fuzzy matching identified the closest key: '{match}'."
            tags = self.entity_to_tags_map[match]
            scored_tags = [{"tag": t["tag"], "source": "local_fuzzy", "confidence": 0.9} for t in tags]
            return scored_tags, explanation

        explanation["source"] = "external_api"
        explanation["reasoning"] = "Entity not found locally. Querying the Taginfo API for real-time suggestions."
        tags = self.query_external_sources(normalized)
        return tags, explanation

    def query_external_sources(self, keyword: str) -> List[Dict[str, Any]]:
        """Queries Taginfo, filters for quality, scores results, and caches them."""
        self.logger.info(f"Querying external source (Taginfo) for '{keyword}'...")
        url = "https://taginfo.openstreetmap.org/api/4/search/by_keyword"
        params = {"q": keyword, "lang": "en", "sortname": "count_all", "sortorder": "desc"}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json().get("data", [])
            results = []
            for item in data:
                is_doc = item.get("in_wiki", False)
                has_usage = item.get("count_all", 0) > 1000
                if item.get("key") and item.get("value") and (is_doc or has_usage):
                    results.append({"tag": {item["key"]: item["value"]}, "source": "external", "confidence": 0.7})
                if len(results) >= 3: break
            if results: self.add_entity(keyword, results)
            else: self.logger.warning(f"No high-quality results from external source for '{keyword}'.")
            return results
        except requests.RequestException as e:
            self.logger.error(f"External query failed for '{keyword}': {e}")
            return []

    def get_closest_key(self, query: str) -> Optional[str]:
        all_keys = list(self.entity_to_tags_map.keys())
        matches = get_close_matches(query, all_keys, n=1, cutoff=0.8)
        return matches[0] if matches else None

    # --- High-Level Integration Methods ---
    def tag_entities_in_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        """High-level hook to extract entities/modifiers and get their tags, avoiding duplicates."""
        extracted_groups = self.extract_entities(sentence)
        results = []
        seen_entities = set()
        for group in extracted_groups:
            entity = group["entity"]
            if entity in seen_entities:
                continue
            seen_entities.add(entity)
            tags, _ = self.get_candidate_tags(entity)
            results.append({**group, "candidate_tags": tags})
        return results

# --- Example Usage & Unit Tests ---
if __name__ == "__main__":
    kb = SpatialKnowledgeBase()

    # 1. Test Query Explainability and Updated Return Signature
    print("\n--- 1. Testing Query Explainability & Assertions ---")
    
    print("\n[A] Direct Match:")
    tags, explanation = kb.get_candidate_tags("hospital")
    print(json.dumps(explanation, indent=2))
    assert explanation["source"] == "local_direct"
    assert tags[0]["tag"] == {"amenity": "hospital"}

    print("\n[B] Fuzzy Match:")
    tags, explanation = kb.get_candidate_tags("restaurante") # Typo
    print(json.dumps(explanation, indent=2))
    assert explanation["source"] == "local_fuzzy"
    assert "closest key: 'restaurant'" in explanation["reasoning"]
    assert tags[0]["confidence"] == 0.9

    print("\n[C] External API Match:")
    tags, explanation = kb.get_candidate_tags("fire hydrant")
    print(json.dumps(explanation, indent=2))
    assert explanation["source"] == "external_api"

    # 2. Test Advanced Entity/Modifier Extraction
    print("\n--- 2. Testing Advanced Phrase Extraction (SpaCy) ---")
    sentence = "Show me a government school and any private hospitals."
    tagged_sentence = kb.tag_entities_in_sentence(sentence)
    print(json.dumps(tagged_sentence, indent=2))
    if kb.nlp: # Only run SpaCy specific tests if it loaded
        assert any(g['entity'] == 'school' and 'government' in g['modifiers'] for g in tagged_sentence)
        assert any(g['entity'] == 'hospital' and 'private' in g['modifiers'] for g in tagged_sentence)
        
    # 3. Test Duplicate Entity Handling in a Sentence
    print("\n--- 3. Testing Duplicate Entity Handling ---")
    dup_sentence = "I want to find parks, parks, and more parks."
    tagged_dups = kb.tag_entities_in_sentence(dup_sentence)
    print(json.dumps(tagged_dups, indent=2))
    assert len(tagged_dups) == 1, "Should only return the 'park' entity once."
    assert tagged_dups[0]['entity'] == 'park'