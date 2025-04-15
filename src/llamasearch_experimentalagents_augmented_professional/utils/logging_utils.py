"""
Utility for logging agent interactions to a SQLite database.
"""

import sqlite_utils
import json
from datetime import datetime
import os
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "local_agent_logs.db"
LOG_TABLE_NAME = "agent_interactions"

def get_db(db_path: Optional[str] = None) -> sqlite_utils.Database:
    """Get a connection to the SQLite database."""
    path = db_path or os.environ.get("SQLITE_DB_PATH", DEFAULT_DB_PATH)
    db = sqlite_utils.Database(path)
    return db

def initialize_log_table(db: sqlite_utils.Database):
    """Initialize the agent interactions log table if it doesn't exist."""
    if not db[LOG_TABLE_NAME].exists():
        try:
            db[LOG_TABLE_NAME].create({
                "timestamp": str,
                "interaction_id": str, # Optional: unique ID for each interaction
                "query": str,
                "search_query": str,   # Query used for knowledge search
                "search_results_count": int,
                "model_used": str,
                "response_answer": str,
                "response_confidence": float,
                "response_sources": str,  # Store as JSON string
                "response_suggested_actions": str, # Store as JSON string
                "error_message": str,
                "execution_time_ms": float # Optional: track performance
            }, pk="timestamp") # Use timestamp as primary key for simplicity, or add interaction_id
            db[LOG_TABLE_NAME].enable_fts(["query", "response_answer", "error_message"], create_triggers=True)
            logger.info(f"Initialized SQLite log table '{LOG_TABLE_NAME}' in {db.path}")
        except Exception as e:
            logger.error(f"Failed to initialize log table '{LOG_TABLE_NAME}': {e}")

def log_interaction(
    db: sqlite_utils.Database,
    query: str,
    search_query: Optional[str] = None,
    search_results_count: Optional[int] = None,
    model_used: Optional[str] = None,
    response_answer: Optional[str] = None,
    response_confidence: Optional[float] = None,
    response_sources: Optional[List[Dict[str, Any]]] = None,
    response_suggested_actions: Optional[List[Dict[str, Any]]] = None,
    error_message: Optional[str] = None,
    execution_time_ms: Optional[float] = None,
    interaction_id: Optional[str] = None,
):
    """Log a single agent interaction to the database."""
    initialize_log_table(db) # Ensure table exists

    try:
        record = {
            "timestamp": datetime.now().isoformat(),
            "interaction_id": interaction_id,
            "query": query,
            "search_query": search_query,
            "search_results_count": search_results_count,
            "model_used": model_used,
            "response_answer": response_answer,
            "response_confidence": response_confidence,
            "response_sources": json.dumps(response_sources) if response_sources else None,
            "response_suggested_actions": json.dumps(response_suggested_actions) if response_suggested_actions else None,
            "error_message": error_message,
            "execution_time_ms": execution_time_ms,
        }
        # Remove None values to avoid inserting nulls unless the column allows it
        record = {k: v for k, v in record.items() if v is not None}

        db[LOG_TABLE_NAME].insert(record)
        logger.debug(f"Logged interaction for query: {query[:50]}...")
    except Exception as e:
        logger.error(f"Failed to log interaction to SQLite: {e}")

# Example Usage (can be removed or put under if __name__ == "__main__")
if __name__ == "__main__":
    db = get_db()
    log_interaction(
        db=db,
        query="Test query",
        search_query="test query internal",
        search_results_count=3,
        model_used="gpt-4-test",
        response_answer="This is a test answer.",
        response_confidence=0.95,
        response_sources=[{"source": "doc1.txt", "relevance": 0.8}],
        response_suggested_actions=[{"title": "Test Action"}],
        execution_time_ms=123.45
    )
    print(f"Test log inserted into {db.path}")
    # To view with Datasette: datasette {db.path} 