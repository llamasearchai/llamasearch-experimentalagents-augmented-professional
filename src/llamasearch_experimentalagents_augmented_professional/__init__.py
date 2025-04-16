"""
LlamaSearch ExperimentalAgents: AI Knowledge Democratization Engine

This package provides professional-grade AI-powered knowledge search
and response generation capabilities.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing LlamaSearch ExperimentalAgents")

# Import public API
from .agents.agents_assistant import LlamaAssistant
from .agents.agents_retriever import SemanticRetriever
from .models.models_knowledge import KnowledgeBase, KnowledgeChunk
from .models.models_responses import ProfessionalResponse
from .integrations.knowledge_manager import KnowledgeManager

__version__ = "1.0.0"

__all__ = [
    "LlamaAssistant", 
    "SemanticRetriever", 
    "KnowledgeBase", 
    "KnowledgeChunk", 
    "ProfessionalResponse",
    "KnowledgeManager",
]

# Import CLI for entry point
from .cli import app as cli_app
