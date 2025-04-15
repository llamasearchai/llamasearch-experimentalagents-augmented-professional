"""
Semantic search retriever with MLX/JAX acceleration.

This module provides optimized semantic search functionality for the LlamaSearch system,
dynamically selecting the best backend based on hardware availability and performance.
"""

import time
import logging
from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np

# Conditional imports for hardware acceleration
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from ..models.knowledge import KnowledgeBase, KnowledgeChunk
from ..models.responses import SearchResults

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Retriever for semantic search operations with hardware acceleration."""

    def __init__(self, knowledge_base: KnowledgeBase):
        """Initialize the retriever with a knowledge base."""
        self.knowledge_base = knowledge_base
        self._embeddings_cache = None
        self._backend_capabilities = self._detect_backends()
        
        logger.info(f"Initialized SemanticRetriever with {len(knowledge_base)} chunks")
        logger.info(f"Backend capabilities: {self._backend_capabilities}")

    def _detect_backends(self) -> Dict[str, bool]:
        """Detect available hardware acceleration backends."""
        backends = {
            "mlx": HAS_MLX,
            "jax": HAS_JAX,
            "numpy": True  # Fallback
        }
        
        # Check if MLX has Apple Silicon hardware acceleration
        if HAS_MLX:
            try:
                backends["mlx_metal"] = mx.metal.is_available()
            except AttributeError:
                backends["mlx_metal"] = False
        
        # Check if JAX has GPU or TPU acceleration
        if HAS_JAX:
            backends["jax_gpu"] = "gpu" in jax.devices()[0].device_kind.lower()
            backends["jax_tpu"] = "tpu" in jax.devices()[0].device_kind.lower()
        
        return backends

    def _select_backend(self, query_embedding: Any, prefer_backend: Optional[str] = None) -> str:
        """Select the best backend based on availability and preference."""
        if prefer_backend and prefer_backend in self._backend_capabilities:
            if self._backend_capabilities[prefer_backend]:
                return prefer_backend
        
        # Default selection prioritization
        if self._backend_capabilities.get("mlx_metal", False):
            return "mlx"
        elif self._backend_capabilities.get("jax_gpu", False):
            return "jax"
        elif self._backend_capabilities.get("mlx", False):
            return "mlx"
        elif self._backend_capabilities.get("jax", False):
            return "jax"
        else:
            return "numpy"

    def _ensure_embeddings_cache(self) -> None:
        """Ensure embeddings are cached in appropriate array format."""
        if self._embeddings_cache is not None:
            return
        
        # Get raw embeddings
        embeddings = self.knowledge_base.get_all_embeddings()
        if not embeddings or not all(e is not None for e in embeddings):
            raise ValueError("Knowledge base contains chunks without embeddings")
        
        # Create backend-specific cache
        self._embeddings_cache = {
            "numpy": np.array(embeddings, dtype=np.float32),
        }
        
        # Add backend-specific arrays
        if HAS_MLX:
            self._embeddings_cache["mlx"] = mx.array(embeddings, dtype=mx.float32)
        
        if HAS_JAX:
            self._embeddings_cache["jax"] = jnp.array(embeddings, dtype=jnp.float32)

    @staticmethod
    def _numpy_cosine_sim(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity with NumPy."""
        # Normalize vectors for cosine similarity
        query_norm = np.linalg.norm(query)
        docs_norm = np.linalg.norm(docs, axis=1)
        
        # Compute dot product and divide by product of norms
        return np.dot(query, docs.T) / (query_norm * docs_norm + 1e-8)

    def _mlx_cosine_sim(self, query: Any, docs: Any) -> np.ndarray:
        """Compute cosine similarity with MLX, optimized for Apple Silicon."""
        if not HAS_MLX:
            raise RuntimeError("MLX is not available")
        
        # Cast to MLX arrays if needed
        if not isinstance(query, mx.array):
            query = mx.array(query, dtype=mx.float32)
        if not isinstance(docs, mx.array):
            docs = mx.array(docs, dtype=mx.float32)
        
        # Optimize with MLX compilation if possible
        @mx.compile
        def _compute_sim(q, d):
            q_norm = mx.linalg.norm(q)
            d_norm = mx.linalg.norm(d, axis=1)
            return mx.dot(q, d.T) / (q_norm * d_norm + 1e-8)
        
        # Compute and convert to numpy
        result = _compute_sim(query, docs)
        return np.array(result)

    def _jax_cosine_sim(self, query: Any, docs: Any) -> np.ndarray:
        """Compute cosine similarity with JAX, optimized for GPU/TPU."""
        if not HAS_JAX:
            raise RuntimeError("JAX is not available")
        
        # Cast to JAX arrays if needed
        if not isinstance(query, jnp.ndarray):
            query = jnp.array(query, dtype=jnp.float32)
        if not isinstance(docs, jnp.ndarray):
            docs = jnp.array(docs, dtype=jnp.float32)
        
        # JIT compile the similarity computation
        @jax.jit
        def _compute_sim(q, d):
            q_norm = jnp.linalg.norm(q)
            d_norm = jnp.linalg.norm(d, axis=1)
            return jnp.dot(q, d.T) / (q_norm * d_norm + 1e-8)
        
        # Compute and convert to numpy
        result = _compute_sim(query, docs)
        return np.array(result)

    def semantic_search(
        self, 
        query_embedding: List[float],
        top_k: int = 3,
        score_threshold: float = 0.6,
        backend: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], str, float]:
        """
        Perform semantic search against the knowledge base.
        
        Args:
            query_embedding: The embedding vector of the query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score for results
            backend: Preferred backend (mlx, jax, or numpy)
            
        Returns:
            A tuple of (search_results, backend_used, execution_time_ms)
        """
        start_time = time.time()
        
        # Ensure embeddings are cached
        self._ensure_embeddings_cache()
        
        # Convert query to numpy array for initial processing
        query_np = np.array(query_embedding, dtype=np.float32)
        
        # Select the best backend
        selected_backend = self._select_backend(query_np, prefer_backend=backend)
        logger.debug(f"Selected backend for search: {selected_backend}")
        
        # Get embeddings for the selected backend
        docs_embeddings = self._embeddings_cache[selected_backend if selected_backend in self._embeddings_cache else "numpy"]
        
        # Compute similarity scores with the selected backend
        try:
            if selected_backend == "mlx":
                scores = self._mlx_cosine_sim(query_np, docs_embeddings)
            elif selected_backend == "jax":
                scores = self._jax_cosine_sim(query_np, docs_embeddings)
            else:  # fallback to numpy
                scores = self._numpy_cosine_sim(query_np, docs_embeddings)
        except Exception as e:
            logger.warning(f"Error with {selected_backend} backend: {e}, falling back to numpy")
            scores = self._numpy_cosine_sim(query_np, self._embeddings_cache["numpy"])
            selected_backend = "numpy"

        # Convert scores to list
        scores_list = scores.tolist()
        
        # Get top-k indices and filter by threshold
        results = []
        for i, score in enumerate(scores_list):
            if score >= score_threshold:
                chunk = self.knowledge_base.chunks[i]
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "source": chunk.source,
                    "score": float(score),
                    "metadata": chunk.metadata
                })
        
        # Sort by score descending and limit to top_k
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        
        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000
        
        return results, selected_backend, execution_time_ms

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        score_threshold: float = 0.6,
        backend: Optional[str] = None
    ) -> SearchResults:
        """
        Execute a query and return a structured SearchResults object.
        
        Args:
            query_embedding: The embedding vector of the query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score for results
            backend: Preferred backend (mlx, jax, or numpy)
            
        Returns:
            A SearchResults object with the results and metadata
        """
        results, backend_used, execution_time_ms = self.semantic_search(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            backend=backend
        )
        
        return SearchResults(
            query="Embedding query",  # We don't have the original query text here
            results=results,
            execution_time_ms=execution_time_ms,
            backend_used=backend_used
        )
