"""
Professional assistant agent implementation for LlamaSearch.

This module implements the main AI assistant functionality, coordinating between
semantic search, embedding generation, and response formatting.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Callable
import time

import openai
from openai import OpenAI
from openai.types.function_definition import FunctionDefinition
from openai.types.function_parameters import FunctionParameters

from ..models.models_knowledge import KnowledgeBase, RunContextWrapper, KnowledgeChunk
from ..models.models_responses import ProfessionalResponse, SourceReference, SuggestedAction
from ..integrations.knowledge_manager import KnowledgeManager
from ..utils.logging_utils import get_db, log_interaction
from ..utils.llm_router import execute_llm_prompt

logger = logging.getLogger(__name__)


class LlamaAssistant:
    """Professional AI assistant with semantic search capabilities."""
    
    def __init__(
        self,
        knowledge_manager: KnowledgeManager,
        openai_client: Optional[OpenAI] = None,
        assistant_model: str = "gpt-4-turbo-preview",
    ):
        """
        Initialize the LlamaAssistant.
        
        Args:
            knowledge_manager: The knowledge manager instance
            openai_client: Optional OpenAI client (created if not provided)
            assistant_model: The OpenAI model to use for the assistant
        """
        self.knowledge_manager = knowledge_manager
        self.client = openai_client or OpenAI()
        self.assistant_model = assistant_model
        
        # Initialize SQLite logging database
        self.db = get_db()
        
        logger.info(f"Initialized LlamaAssistant with KnowledgeManager (KB size: {knowledge_manager.knowledge_base_size})")
        logger.info(f"Using assistant model: {assistant_model}")
    
    def search_knowledge_base(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant content using the KnowledgeManager.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score for results
            
        Returns:
            A list of search results with scores
        """
        # Use the KnowledgeManager's search method
        results = self.knowledge_manager.search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )
        return results
    
    def _define_search_function(self) -> FunctionDefinition:
        """Define the search function for OpenAI function calling."""
        return FunctionDefinition(
            name="search_knowledge_base",
            description="Search the knowledge base for relevant information to answer the user's query",
            parameters=FunctionParameters(
                type="object",
                properties={
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant information"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return",
                        "default": 3
                    }
                },
                required=["query"]
            )
        )
    
    def _format_sources_for_context(self, results: List[Dict[str, Any]]) -> str:
        """Format search results as context for the LLM."""
        if not results:
            return "No relevant information found in the knowledge base."
        
        context = "Here is relevant information from the knowledge base:\n\n"
        for i, result in enumerate(results, 1):
            context += f"[Source {i}: {result['source']}]\n"
            context += f"{result['content']}\n\n"
        
        return context
    
    def _parse_suggested_actions(self, actions_text: str) -> List[SuggestedAction]:
        """Parse suggested actions from the LLM response."""
        try:
            actions_data = json.loads(actions_text)
            if not isinstance(actions_data, list):
                return []
            
            result = []
            for action_data in actions_data:
                if not isinstance(action_data, dict):
                    continue
                
                # Get required fields with defaults
                title = action_data.get("title", "Unnamed action")
                description = action_data.get("description", "No description provided")
                priority = action_data.get("priority", "medium")
                
                # Create and validate the action
                try:
                    action = SuggestedAction(
                        title=title,
                        description=description,
                        priority=priority
                    )
                    result.append(action)
                except ValueError:
                    # Skip invalid actions
                    continue
            
            return result
        except Exception as e:
            logger.warning(f"Failed to parse suggested actions: {e}")
            return []
    
    def generate_response(
        self,
        query: str,
        on_search_start: Optional[Callable[[], None]] = None,
        on_search_complete: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        on_thinking_start: Optional[Callable[[], None]] = None,
        on_thinking_complete: Optional[Callable[[], None]] = None
    ) -> ProfessionalResponse:
        """
        Generate a professional response to a user query.
        
        Args:
            query: The user query
            on_search_start: Optional callback when search starts
            on_search_complete: Optional callback when search completes
            on_thinking_start: Optional callback when thinking starts
            on_thinking_complete: Optional callback when thinking completes
            
        Returns:
            A professional structured response
        """
        # Define search function
        search_fn = self._define_search_function()
        
        # Initial system prompt
        system_prompt = f"""
        You are a professional AI assistant powered by LlamaSearch.
        Your task is to provide helpful, accurate, and detailed responses to questions based on the knowledge base provided.
        
        When answering:
        1. Be clear, concise, and professional
        2. Cite your sources from the knowledge base when possible
        3. If you don't know something or it's not in the knowledge base, be honest about it
        4. Suggest follow-up actions when appropriate
        
        Knowledge base description: {self.knowledge_manager.kb.description if self.knowledge_manager and self.knowledge_manager.kb else 'No description available'}
        """
        
        # Step 1: Call the model to determine if search is needed
        if on_thinking_start:
            on_thinking_start()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        initial_response = self.client.chat.completions.create(
            model=self.assistant_model,
            messages=messages,
            tools=[{"type": "function", "function": search_fn}],
            tool_choice={"type": "function", "function": {"name": "search_knowledge_base"}},
        )
        
        # Extract search parameters
        tool_call = initial_response.choices[0].message.tool_calls[0]
        search_args = json.loads(tool_call.function.arguments)
        search_query = search_args.get("query", query)
        top_k = search_args.get("top_k", 3)
        
        # Step 2: Search knowledge base
        if on_search_start:
            on_search_start()
        
        search_results = self.search_knowledge_base(search_query, top_k=top_k)
        
        if on_search_complete:
            on_search_complete(search_results)
        
        # Step 3: Generate final response with search results
        knowledge_context = self._format_sources_for_context(search_results)
        
        messages.append({
            "role": "assistant", 
            "content": None, 
            "tool_calls": [tool_call]
        })
        
        messages.append({
            "role": "tool", 
            "tool_call_id": tool_call.id,
            "content": knowledge_context
        })
        
        # Additional system instruction for structured output
        messages.append({
            "role": "system", 
            "content": """
            Please format your response as a JSON object with the following structure:
            
            {
                "answer": "Your detailed answer here",
                "confidence": 0.85,  # A float between 0 and 1 representing your confidence in the answer
                "sources": [
                    {
                        "source": "Source identifier",
                        "relevance": 0.9,  # A float between 0 and 1
                        "excerpt": "Brief excerpt if applicable"
                    }
                ],
                "suggested_actions": [
                    {
                        "title": "Action title",
                        "description": "Detailed description of the action",
                        "priority": "medium"  # Can be 'low', 'medium', or 'high'
                    }
                ]
            }

            Ensure your response is a valid JSON object. Use your best judgment to populate the confidence score
            and sources based on the information from the knowledge base.
            """
        })

        # Generate final response
        final_response = self.client.chat.completions.create(
            model=self.assistant_model,
            messages=messages,
            response_format={"type": "json_object"}
        )

        if on_thinking_complete:
            on_thinking_complete()

        # Parse the response
        try:
            response_content = final_response.choices[0].message.content
            response_data = json.loads(response_content)

            # Create source references
            sources = []
            for src in response_data.get("sources", []):
                source_ref = SourceReference(
                    source=src.get("source", "Unknown"),
                    relevance=src.get("relevance", 0.0),
                    excerpt=src.get("excerpt", None)
                )
                sources.append(source_ref)

            # Create suggested actions
            actions_data = response_data.get("suggested_actions", [])
            suggested_actions = []
            for action in actions_data:
                try:
                    suggested_action = SuggestedAction(
                        title=action.get("title", "Unnamed action"),
                        description=action.get("description", "No description"),
                        priority=action.get("priority", "medium")
                    )
                    suggested_actions.append(suggested_action)
                except Exception:
                    # Skip invalid actions
                    pass

            # Create the professional response
            interaction_id = log_interaction(
                db=self.db,
                interaction_id=None,
                query=query,
                search_query=search_query if 'search_query' in locals() and tool_call else None,
                search_results_count=len(search_results) if 'search_results' in locals() else 0,
                model_used=self.assistant_model,
                response_answer=response_data.get("answer", "No answer provided") if response_data else None,
                knowledge_base_name=self.knowledge_manager.kb.name if self.knowledge_manager and self.knowledge_manager.kb else None
            )

            professional_response = ProfessionalResponse(
                answer=response_data.get("answer", "No answer provided"),
                confidence=response_data.get("confidence", 0.0),
                sources=sources,
                suggested_actions=suggested_actions,
                metadata={
                    "query": query,
                    "search_query": search_query if tool_call else None,
                    "search_results_count": len(search_results),
                    "model": self.assistant_model,
                    "interaction_id": interaction_id,
                    "knowledge_base_name": self.knowledge_manager.kb.name if self.knowledge_manager and self.knowledge_manager.kb else None
                }
            )

            return professional_response

        except Exception as e:
            logger.error(f"Failed during response generation: {e}", exc_info=True)
            error_message = str(e)

            # --- Fallback Logic --- #
            fallback_answer = None
            model_used_for_fallback = "N/A"
            try:
                logger.info(f"Attempting fallback LLM execution for query: {query[:50]}...")
                # Use a simpler system prompt for fallback
                fallback_system_prompt = "You are a helpful assistant. Please answer the user's query based on your general knowledge."
                # Execute using the router (which handles primary/fallback model selection)
                fallback_response_text, model_used_for_fallback = execute_llm_prompt(
                    prompt=query,
                    system_prompt=fallback_system_prompt,
                    # Router uses env vars for primary/fallback models
                )
                if fallback_response_text:
                    fallback_answer = f"I encountered an error with the primary process. However, I can offer this general information: {fallback_response_text}"
                    logger.info(f"Fallback LLM ({model_used_for_fallback}) provided a response.")
                else:
                    logger.error(f"Fallback LLM execution failed or returned no text.")

            except Exception as fallback_e:
                logger.error(f"Error during fallback LLM execution: {fallback_e}")
            # --- End Fallback Logic --- #

            # Original Fallback response (if LLM fallback also fails)
            final_answer = fallback_answer if fallback_answer else f"I encountered an error while processing your query: {error_message}"

            professional_response = ProfessionalResponse(
                answer=final_answer,
                confidence=0.1 if fallback_answer else 0.0, # Low confidence for fallback
                sources=[],
                suggested_actions=[
                    SuggestedAction(
                        title="Try a different query or check logs",
                        description=f"The primary process failed ({error_message}). Please check logs (Interaction ID: {interaction_id}) or try rephrasing.",
                        priority="high"
                    )
                ],
                metadata={"error": error_message, "interaction_id": interaction_id, "fallback_model_used": model_used_for_fallback}
            )

        finally:
            execution_time_ms = (time.time() - start_time) * 1000

        return professional_response
