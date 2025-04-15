"""
Entry point for the Python backend logic when run via Tauri GUI.
"""
import os
import sys
import logging

# Ensure the src directory is in the path if running directly
# This might not be needed when run via tauri-plugin-python
# script_dir = os.path.dirname(os.path.abspath(__file__))
# src_dir = os.path.dirname(os.path.dirname(script_dir))
# if src_dir not in sys.path:
#     sys.path.insert(0, src_dir)

try:
    from openai import OpenAI
    from .agents.assistant import LlamaAssistant
    from .integrations.knowledge_manager import KnowledgeManager
    # Assume .env is loaded by the environment Tauri runs in or load explicitly
    from dotenv import load_dotenv
    load_dotenv() # Load .env file from project root
except ImportError as e:
    # Log error - this might be tricky depending on how Tauri captures stdout/stderr
    print(f"ERROR: Failed to import necessary modules: {e}")
    # If imports fail, the agent won't work. Exit or handle gracefully.
    # We might need a way to communicate this failure back to the frontend.
    raise

# --- Global Initialization (Consider managing state better) --- #

logger = logging.getLogger("llamasearch_gui_backend")
logging.basicConfig(level=logging.INFO) # Configure logging level

# Initialize components once
# TODO: Handle potential errors during initialization gracefully
AGENT_INSTANCE = None

def get_agent_instance():
    global AGENT_INSTANCE
    if AGENT_INSTANCE is None:
        logger.info("Initializing Agent Instance for GUI...")
        try:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set.")

            client = OpenAI(api_key=openai_api_key)
            # TODO: Make knowledge_dir configurable or determine dynamically
            knowledge_dir = "./knowledge_base" # Default path relative to project root

            knowledge_manager = KnowledgeManager(openai_client=client)
            # Load documents - consider doing this async or on demand
            knowledge_manager.load_documents_from_directory(knowledge_dir, embed_immediately=True)

            AGENT_INSTANCE = LlamaAssistant(
                knowledge_manager=knowledge_manager,
                openai_client=client
            )
            logger.info("Agent Instance Initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}", exc_info=True)
            # How to signal failure to frontend?
            # Maybe return an error state in handle_frontend_request
            AGENT_INSTANCE = None # Ensure it remains None on failure
    return AGENT_INSTANCE

# --- Function callable from Tauri --- #

def handle_frontend_request(query: str) -> dict:
    """
    Handles a query from the frontend, interacts with the LlamaAssistant,
    and returns the response.

    Args:
        query: The user's query string.

    Returns:
        A dictionary containing the response or an error.
        Example success: {"status": "success", "response": professional_response.dict()}
        Example error: {"status": "error", "message": "Error details"}
    """
    logger.info(f"Received query from frontend: {query[:100]}...")
    agent = get_agent_instance()

    if agent is None:
        logger.error("Agent instance is not available.")
        return {"status": "error", "message": "Agent initialization failed. Check backend logs."}

    try:
        # Call the agent's generate_response method
        # Note: Callbacks like on_search_start won't work directly here
        # Need a different mechanism (e.g., websocket) for streaming updates to frontend
        professional_response = agent.generate_response(query)

        logger.info("Successfully generated response.")
        # Return the response data as a dictionary
        return {"status": "success", "response": professional_response.dict()}

    except Exception as e:
        logger.error(f"Error handling frontend request: {e}", exc_info=True)
        return {"status": "error", "message": f"An error occurred: {str(e)}"}

# --- Optional: Direct execution for testing --- #
if __name__ == "__main__":
    print("Testing backend script...")
    # Ensure API key is set for testing
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable to run tests.")
    else:
        test_query = "Explain the basics of the KnowledgeManager."
        print(f"Sending test query: {test_query}")
        result = handle_frontend_request(test_query)
        print("\nResult:")
        import json
        print(json.dumps(result, indent=2))

        if result['status'] == 'success':
            print("\nAgent Answer:")
            print(result['response']['answer']) 