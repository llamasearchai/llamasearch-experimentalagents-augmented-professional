"""
Structured response models for the LlamaSearch system.

This module defines Pydantic models for structured responses from the agent,
ensuring consistent and typed outputs for professional use.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any


class SourceReference(BaseModel):
    """A reference to a source document used in generating a response."""
    
    source: str = Field(..., description="Source document name or path")
    relevance: float = Field(..., description="Relevance score from semantic search", ge=0, le=1)
    excerpt: Optional[str] = Field(None, description="Short excerpt from the source")
    
    class Config:
        frozen = True


class SuggestedAction(BaseModel):
    """A suggested action for the user to take based on the response."""
    
    title: str = Field(..., description="Short title for the action")
    description: str = Field(..., description="Detailed description of the action")
    priority: str = Field("medium", description="Priority of the action")
    
    @validator("priority")
    def validate_priority(cls, v):
        """Validate priority value."""
        allowed = ["low", "medium", "high"]
        if v.lower() not in allowed:
            raise ValueError(f"Priority must be one of {allowed}")
        return v.lower()
    
    class Config:
        frozen = True


class ProfessionalResponse(BaseModel):
    """A professional structured response from the LlamaSearch agent."""
    
    answer: str = Field(..., description="Main answer text")
    confidence: float = Field(..., description="Confidence in the answer", ge=0, le=1)
    sources: List[SourceReference] = Field(
        default_factory=list, 
        description="References to sources used"
    )
    suggested_actions: List[SuggestedAction] = Field(
        default_factory=list, 
        description="Suggested follow-up actions"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional response metadata"
    )
    
    def format_sources(self) -> str:
        """Format sources as a readable string."""
        if not self.sources:
            return "No sources referenced."
        
        result = "Sources:\n"
        for i, source in enumerate(self.sources, 1):
            result += f"{i}. {source.source} (relevance: {source.relevance:.2f})\n"
            if source.excerpt:
                result += f"   Excerpt: \"{source.excerpt}\"\n"
        return result
    
    def format_suggested_actions(self) -> str:
        """Format suggested actions as a readable string."""
        if not self.suggested_actions:
            return "No suggested actions."
        
        result = "Suggested Actions:\n"
        for i, action in enumerate(self.suggested_actions, 1):
            priority_marker = {
                "low": "🔵",
                "medium": "🟡",
                "high": "🔴"
            }.get(action.priority, "•")
            
            result += f"{i}. {priority_marker} {action.title}\n"
            result += f"   {action.description}\n"
        return result
    
    class Config:
        frozen = True


class SearchResults(BaseModel):
    """Results from a semantic search operation."""
    
    query: str = Field(..., description="Original search query")
    results: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Search results with scores"
    )
    execution_time_ms: float = Field(..., description="Search execution time in milliseconds")
    backend_used: str = Field(..., description="Backend used for search (MLX, JAX, etc.)")
    
    class Config:
        frozen = True
