"""
Knowledge representation models for the LlamaSearch system.

This module defines the core data structures used to represent knowledge
chunks and their embeddings for semantic search operations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid
from datetime import datetime


@dataclass
class KnowledgeChunk:
    """A chunk of knowledge from a document with its embedding vector."""
    
    # Content and metadata
    content: str
    source: str
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Ensure metadata has basic fields."""
        if not self.metadata.get("source_type"):
            # Try to infer source type from extension
            if "." in self.source:
                ext = self.source.split(".")[-1].lower()
                if ext in ["md", "markdown"]:
                    self.metadata["source_type"] = "markdown"
                elif ext in ["txt", "text"]:
                    self.metadata["source_type"] = "plaintext"
                elif ext in ["py", "python"]:
                    self.metadata["source_type"] = "code"
                else:
                    self.metadata["source_type"] = "unknown"
            else:
                self.metadata["source_type"] = "unknown"


@dataclass
class KnowledgeBase:
    """A collection of knowledge chunks with search capabilities."""
    
    chunks: List[KnowledgeChunk] = field(default_factory=list)
    name: str = "default"
    description: str = "Default knowledge base"
    
    def add_chunk(self, chunk: KnowledgeChunk) -> None:
        """Add a chunk to the knowledge base."""
        self.chunks.append(chunk)
    
    def add_chunks(self, chunks: List[KnowledgeChunk]) -> None:
        """Add multiple chunks to the knowledge base."""
        self.chunks.extend(chunks)
    
    def get_all_embeddings(self) -> List[List[float]]:
        """Get all embeddings as a list."""
        return [chunk.embedding for chunk in self.chunks if chunk.embedding is not None]
    
    def __len__(self) -> int:
        return len(self.chunks)


class RunContextWrapper:
    """Wrapper for the KnowledgeBase to be used with OpenAI's RunContext."""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
    
    def get_context(self) -> Dict[str, Any]:
        """Return a dictionary with the knowledge base data for the run context."""
        return {
            "knowledge_base": {
                "name": self.knowledge_base.name,
                "description": self.knowledge_base.description,
                "size": len(self.knowledge_base),
                "sources": list(set(chunk.source for chunk in self.knowledge_base.chunks))
            }
        }


class QueryConfig(BaseModel):
    """Configuration for a semantic search query."""
    
    query: str = Field(..., description="The search query string")
    top_k: int = Field(3, description="Number of results to return", ge=1, le=20)
    threshold: float = Field(0.6, description="Similarity threshold for results", ge=0, le=1)
    use_mlx: bool = Field(True, description="Whether to use MLX for acceleration if available")
