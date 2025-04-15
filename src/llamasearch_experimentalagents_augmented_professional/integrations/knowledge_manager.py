"""
Manages the loading, chunking, embedding, and retrieval of knowledge.
"""
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import time

from openai import OpenAI

from ..models.knowledge import KnowledgeBase, KnowledgeChunk
from ..agents.retriever import SemanticRetriever # May consolidate later

logger = logging.getLogger(__name__)

# Define default chunking parameters (can be made configurable)
DEFAULT_CHUNK_MIN_LENGTH = 20

class KnowledgeManager:
    """Handles knowledge base operations: loading, embedding, searching."""

    def __init__(
        self,
        openai_client: OpenAI,
        embedding_model: str = "text-embedding-3-small",
        knowledge_base: Optional[KnowledgeBase] = None,
    ):
        self.client = openai_client
        self.embedding_model = embedding_model
        self.kb = knowledge_base or KnowledgeBase(name="Managed KB")
        self.retriever = SemanticRetriever(self.kb) # Retriever uses the KB
        logger.info(f"KnowledgeManager initialized with embedding model: {embedding_model}")

    def load_documents_from_directory(self, dir_path: str, embed_immediately: bool = True):
        """Loads documents from a directory, chunks them, and optionally embeds."""
        knowledge_path = Path(dir_path)
        if not knowledge_path.exists() or not knowledge_path.is_dir():
            logger.error(f"Knowledge directory not found or not a directory: {dir_path}")
            return

        text_files = list(knowledge_path.glob("**/*.md")) + list(knowledge_path.glob("**/*.txt"))
        if not text_files:
            logger.warning(f"No .md or .txt files found in {dir_path}")
            return

        logger.info(f"Loading {len(text_files)} files from {dir_path}...")
        new_chunks = []
        for file_path in text_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                # Simple chunking by paragraphs
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

                for i, paragraph in enumerate(paragraphs):
                    if len(paragraph) < DEFAULT_CHUNK_MIN_LENGTH:
                        continue

                    chunk = KnowledgeChunk(
                        content=paragraph,
                        source=str(file_path.relative_to(knowledge_path)),
                        metadata={
                            "chunk_index": i,
                            "filename": file_path.name,
                        }
                        # embedding is added later
                    )
                    # Add chunk to KB directly or collect first?
                    # self.kb.add_chunk(chunk) # Add directly
                    new_chunks.append(chunk)

            except Exception as e:
                logger.error(f"Error loading or chunking file {file_path}: {e}")

        if new_chunks:
            self.kb.add_chunks(new_chunks)
            logger.info(f"Added {len(new_chunks)} new chunks to the knowledge base.")
            if embed_immediately:
                self.generate_embeddings_for_new_chunks()
        else:
             logger.info("No new chunks were added from the directory.")


    def generate_embeddings_for_new_chunks(self, batch_size: int = 20):
        """Generates embeddings for chunks in the KB that don't have them."""
        chunks_to_embed = [chunk for chunk in self.kb.chunks if chunk.embedding is None]
        if not chunks_to_embed:
            logger.info("No new chunks require embedding.")
            return

        logger.info(f"Generating embeddings for {len(chunks_to_embed)} chunks...")
        total_embedded = 0
        try:
            for i in range(0, len(chunks_to_embed), batch_size):
                batch = chunks_to_embed[i:i+batch_size]
                batch_texts = [chunk.content for chunk in batch]

                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts
                )

                # Assign embeddings back to the original chunks in the KB
                for j, embedding_data in enumerate(response.data):
                    # Find the original chunk object and assign the embedding
                    # This assumes chunk objects are stable in self.kb.chunks
                    # (might need optimization if KB is very large or changes often)
                    batch[j].embedding = embedding_data.embedding
                    total_embedded += 1

                logger.debug(f"Embedded batch {i//batch_size + 1}/{len(chunks_to_embed)//batch_size + 1}")
                if i + batch_size < len(chunks_to_embed):
                    time.sleep(0.5) # Avoid rate limits

            logger.info(f"Successfully generated embeddings for {total_embedded} chunks.")
            # Important: Clear the retriever's cache as embeddings have changed
            self.retriever._embeddings_cache = None

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Optionally, decide how to handle partial failure

    def search(self, query: str, top_k: int = 3, score_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Search the knowledge base for relevant chunks."""
        if not self.kb or not self.kb.chunks:
             logger.warning("Search attempted on empty or non-existent knowledge base.")
             return []
        if all(c.embedding is None for c in self.kb.chunks):
            logger.warning("Search attempted, but no chunks have embeddings.")
            return []

        try:
            query_embedding = self.client.embeddings.create(
                model=self.embedding_model,
                input=query,
            ).data[0].embedding

            results, backend_used, execution_time_ms = self.retriever.semantic_search(
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=score_threshold,
                # backend preference can be added here if needed
            )
            logger.info(f"Search for '{query[:50]}...' found {len(results)} results in {execution_time_ms:.2f}ms using {backend_used}")
            return results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    @property
    def knowledge_base_size(self) -> int:
        """Return the number of chunks in the knowledge base."""
        return len(self.kb) if self.kb else 0

# TODO:
# - Add method to save/load knowledge base with embeddings (e.g., to JSON or SQLite)
# - Implement more sophisticated chunking strategies
# - Add support for other document types (PDF, etc.)
# - Integrate SQLite for persistent storage of chunks and embeddings
# - Refine error handling 