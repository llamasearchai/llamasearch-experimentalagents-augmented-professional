"""
Utility for routing requests to multiple Language Models with fallback.

Leverages Simon Willison's `llm` library for accessing various models.
"""

import llm
import os
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Configuration --- #

# Default model aliases (can be overridden by environment variables)
DEFAULT_PRIMARY_MODEL = os.environ.get("LLM_PRIMARY_MODEL", "gpt-4-turbo-preview") # Default to OpenAI
DEFAULT_FALLBACK_MODEL = os.environ.get("LLM_FALLBACK_MODEL", "gpt-3.5-turbo") # Default fallback
# Example for local model: "llama-3-8b-instruct-mlx" if llm-mlx is installed

# --- Model Interaction Logic --- #

def get_model(model_id: str) -> Optional[llm.Model]:
    """Get an llm model instance by ID, handling potential errors."""
    try:
        model = llm.get_model(model_id)
        # Ensure API keys are configured if needed (llm handles this for known providers)
        # For OpenAI, it checks OPENAI_API_KEY environment variable.
        # Add checks or setup for other providers if necessary.
        return model
    except llm.UnknownModelError:
        logger.error(f"Unknown LLM model: {model_id}. Is it installed or configured correctly?")
        return None
    except Exception as e:
        logger.error(f"Error getting LLM model {model_id}: {e}")
        return None

def execute_llm_prompt(
    prompt: str,
    system_prompt: Optional[str] = None,
    model_id: str = DEFAULT_PRIMARY_MODEL,
    fallback_model_id: Optional[str] = DEFAULT_FALLBACK_MODEL,
    **kwargs # Pass additional args like temperature, max_tokens to model.prompt()
) -> Tuple[Optional[str], str]:
    """
    Execute a prompt against the primary LLM, with optional fallback.

    Args:
        prompt: The user prompt.
        system_prompt: An optional system prompt.
        model_id: The ID of the primary model to use.
        fallback_model_id: The ID of the fallback model.
        **kwargs: Additional keyword arguments for the model's prompt method.

    Returns:
        A tuple containing (response_text, model_id_used).
        Returns (None, model_id_used) if both primary and fallback fail.
    """
    primary_model = get_model(model_id)
    model_used = model_id

    if primary_model:
        try:
            logger.info(f"Attempting prompt with primary model: {model_id}")
            response = primary_model.prompt(prompt, system=system_prompt, **kwargs)
            response_text = response.text() # Get the text part of the response
            logger.info(f"Successfully received response from {model_id}")
            return response_text, model_used
        except Exception as e:
            logger.warning(f"Primary model {model_id} failed: {e}. Attempting fallback.")
            # Fall through to fallback if primary fails
    else:
        logger.warning(f"Primary model {model_id} not available. Attempting fallback.")

    # Try fallback model if primary failed or wasn't available
    if fallback_model_id:
        fallback_model = get_model(fallback_model_id)
        model_used = fallback_model_id
        if fallback_model:
            try:
                logger.info(f"Attempting prompt with fallback model: {fallback_model_id}")
                response = fallback_model.prompt(prompt, system=system_prompt, **kwargs)
                response_text = response.text()
                logger.info(f"Successfully received response from fallback {fallback_model_id}")
                return response_text, model_used
            except Exception as e:
                logger.error(f"Fallback model {fallback_model_id} also failed: {e}")
                return None, model_used # Fallback failed
        else:
             logger.error(f"Fallback model {fallback_model_id} not available.")
             return None, model_used # Fallback unavailable
    else:
        logger.error("Primary model failed and no fallback model configured.")
        return None, model_used # Primary failed, no fallback

# Note: This basic implementation uses model.prompt(), which might not directly
# support advanced features like OpenAI's function calling or JSON mode out-of-the-box
# via the `llm` library's generic interface for all models.
# For features specific to OpenAI (like function calling and JSON mode), we might need
# to bypass this router and use the openai client directly, or enhance this router
# to handle provider-specific parameters and response parsing.

# Example Usage:
if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in environment for this example
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. OpenAI models may fail.")

    response, model_name = execute_llm_prompt(
        prompt="Explain the theory of relativity in simple terms.",
        system_prompt="You are a helpful assistant.",
        temperature=0.7 # Example of passing kwargs
    )

    if response:
        print(f"Response (from {model_name}):\n{response}")
    else:
        print(f"Failed to get response from configured models (tried primary: {DEFAULT_PRIMARY_MODEL}, fallback: {DEFAULT_FALLBACK_MODEL})")

    # Example with a non-existent primary model to test fallback
    response_fallback, model_name_fallback = execute_llm_prompt(
        prompt="What is 2+2?",
        model_id="non-existent-model-123",
        fallback_model_id="gpt-3.5-turbo"
    )
    if response_fallback:
        print(f"\nResponse (from {model_name_fallback}):\n{response_fallback}")
    else:
         print(f"\nFailed to get response even from fallback.") 