"""
LlamaSearch ExperimentalAgents Augmented Professional
---------------------------------------------------

Professional AI knowledge democratization engine with hardware-accelerated semantic search,
engaging llama animations, and structured responses.
"""

__version__ = "1.0.0"
__author__ = "LlamaSearch Team"
__license__ = "Apache-2.0"

from .models.knowledge import KnowledgeBase, KnowledgeChunk, RunContextWrapper
from .models.responses import ProfessionalResponse, SourceReference, SuggestedAction
from .agents.retriever import SemanticRetriever
from .agents.assistant import LlamaAssistant

# Expose main entry points
__all__ = [
    "KnowledgeBase",
    "KnowledgeChunk",
    "RunContextWrapper",
    "ProfessionalResponse",
    "SourceReference",
    "SuggestedAction",
    "SemanticRetriever",
    "LlamaAssistant",
]

# Import CLI for entry point
from .cli import app as cli_app
