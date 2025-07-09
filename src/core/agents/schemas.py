# src/core/agents/schemas.py

from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ClarificationAsk:
    """A structured object created when an agent is uncertain and needs user input."""
    original_entity: str
    message: str
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    requires_user_intervention: bool = field(default=True, init=False)