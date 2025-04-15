"""
Command-line interface for the LlamaSearch system.

This module provides a Rich/Textual-powered CLI for interacting with
the LlamaSearch assistant, including animations and professional formatting.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich import print as rich_print
from rich.logging import RichHandler
import openai
from openai import OpenAI

from .agents.assistant import LlamaAssistant
from .models.knowledge import KnowledgeBase, KnowledgeChunk
from .models.responses import ProfessionalResponse
from .integrations.knowledge_manager import KnowledgeManager
from .llama_animations.thinking import LlamaThinking
from .llama_animations.typing_effect import LlamaResponseTypingEffect

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("llamasearch")

# Create the CLI app
app = typer.Typer(
    name="llamasearch",
    help="Professional AI knowledge search powered by LlamaSearch",
    add_completion=False,
)

# Create console for rich output
console = Console()


def get_openai_api_key() -> str:
    """Get the OpenAI API key from environment or prompt."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = Prompt.ask(
            "[bold yellow]Please enter your OpenAI API key[/]",
            password=True
        )
    return api_key


def format_response(response: ProfessionalResponse, detailed: bool = False) -> str:
    """
    Format a professional response as markdown.
    
    Args:
        response: The professional response to format
        detailed: Whether to include detailed information
        
    Returns:
        Formatted markdown string
    """
    confidence_emoji = "🟢" if response.confidence >= 0.8 else "🟡" if response.confidence >= 0.6 else "🔴"
    confidence_text = f"{confidence_emoji} Confidence: {response.confidence:.0%}"
    
    markdown = f"""
# LlamaSearch Response

{response.answer}

## {confidence_text}
"""
    
    if response.sources and detailed:
        markdown += "\n## Sources\n\n"
        for i, source in enumerate(response.sources, 1):
            markdown += f"{i}. **{source.source}** (relevance: {source.relevance:.0%})\n"
            if source.excerpt:
                markdown += f"   > {source.excerpt}\n"
            markdown += "\n"
    
    if response.suggested_actions:
        markdown += "\n## Suggested Actions\n\n"
        for i, action in enumerate(response.suggested_actions, 1):
            priority_marker = {
                "low": "🔵",
                "medium": "🟡",
                "high": "🔴"
            }.get(action.priority, "•")
            
            markdown += f"{i}. {priority_marker} **{action.title}**\n"
            markdown += f"   {action.description}\n\n"
    
    return markdown


@app.command("ask")
def ask_query(
    query: str = typer.Option(None, "--query", "-q", help="Question to ask"),
    knowledge_dir: str = typer.Option(
        "./knowledge_base",
        "--knowledge", "-k",
        help="Directory containing knowledge base files"
    ),
    visual: str = typer.Option(
        "basic",
        "--visual", "-v",
        help="Visual mode (basic, animated, or minimal)"
    ),
    detailed: bool = typer.Option(
        False,
        "--detailed", "-d",
        help="Show detailed information in the response"
    ),
    backend: str = typer.Option(
        "auto",
        "--backend", "-b",
        help="Backend for semantic search (auto, mlx, jax, or numpy)"
    ),
    results: int = typer.Option(
        3,
        "--results", "-r",
        help="Number of search results to use"
    ),
    api_key: str = typer.Option(
        None,
        "--api-key",
        help="OpenAI API key (will use OPENAI_API_KEY env var if not provided)"
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Path to the SQLite database for logging (defaults to local_agent_logs.db)"
    ),
):
    """Ask a question and get an answer from the knowledge base."""
    # Get API key
    openai_api_key = api_key or get_openai_api_key()
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Create OpenAI client
    try:
        client = OpenAI(api_key=openai_api_key)
    except Exception as e:
        console.print(f"[bold red]Error creating OpenAI client:[/] {e}")
        sys.exit(1)
    
    # Check if interactive mode (no query provided)
    interactive = query is None
    
    # Setup thinking animation
    thinking_animation = LlamaThinking(
        title="🧠 Llamasearch ExperimentalAgents",
        subtitle="Initializing..."
    )
    
    # Set DB path environment variable if provided
    if db_path:
        os.environ["SQLITE_DB_PATH"] = db_path
    
    try:
        # Initialize Knowledge Manager
        knowledge_manager = KnowledgeManager(openai_client=client)
        
        # Load knowledge base using Knowledge Manager
        with console.status(f"Loading knowledge from [bold blue]{knowledge_dir}[/]..."):
            knowledge_manager.load_documents_from_directory(knowledge_dir, embed_immediately=True)
        
        if knowledge_manager.knowledge_base_size == 0:
            console.print("[bold yellow]Warning:[/] No knowledge chunks loaded. Assistant will rely only on its internal knowledge.")
            # Decide if we should exit in non-interactive mode if no KB
            # if not interactive:
            #     return
        else:
            console.print(f"[bold green]Knowledge base initialized with {knowledge_manager.knowledge_base_size} chunks[/]")
        
        # Create assistant with Knowledge Manager
        assistant = LlamaAssistant(
            knowledge_manager=knowledge_manager,
            openai_client=client,
            assistant_model="gpt-4-turbo-preview" # Or make this configurable
        )
        
        # Interactive mode
        if interactive:
            console.print(Panel(
                Markdown("# 🦙 LlamaSearch Professional\n\nAsk questions about your knowledge base!"),
                border_style="blue"
            ))
            
            while True:
                # Get query from user
                query = Prompt.ask("\n[bold blue]Ask a question[/]")
                if not query:
                    continue
                
                if query.lower() in ("exit", "quit", "q"):
                    break
                
                # Process the query
                if visual == "animated":
                    thinking_animation.start()
                    
                    def on_search_complete(results):
                        thinking_animation.subtitle = f"Found {len(results)} relevant knowledge chunks..."
                    
                    response = assistant.generate_response(
                        query,
                        on_search_complete=on_search_complete
                    )
                    
                    thinking_animation.stop()
                    
                    # Display response with typing effect
                    markdown = format_response(response, detailed=detailed)
                    typing_effect = LlamaResponseTypingEffect(markdown)
                    typing_effect.start()
                    
                    if typing_effect.thread:
                        typing_effect.thread.join()
                else:
                    # Basic mode
                    with console.status("[bold blue]Generating response...[/]"):
                        response = assistant.generate_response(query)
                    
                    # Display response
                    markdown = format_response(response, detailed=detailed)
                    console.print(Panel(Markdown(markdown), border_style="green"))
                
                # Ask for another query
                if not Confirm.ask("\n[bold blue]Ask another question?[/]", default=True):
                    break
        
        # Single query mode
        else:
            # Process the query
            if visual == "animated":
                thinking_animation.start()
                
                def on_search_complete(results):
                    thinking_animation.subtitle = f"Found {len(results)} relevant knowledge chunks..."
                
                response = assistant.generate_response(
                    query,
                    on_search_complete=on_search_complete
                )
                
                thinking_animation.stop()
                
                # Display response with typing effect
                markdown = format_response(response, detailed=detailed)
                typing_effect = LlamaResponseTypingEffect(markdown)
                typing_effect.start()
                
                if typing_effect.thread:
                    typing_effect.thread.join()
            else:
                # Basic mode
                with console.status("[bold blue]Generating response...[/]"):
                    response = assistant.generate_response(query)
                
                # Display response
                markdown = format_response(response, detailed=detailed)
                console.print(Panel(Markdown(markdown), border_style="green"))
    
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user[/]")
        if thinking_animation.thinking:
            thinking_animation.stop()
    
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        if thinking_animation.thinking:
            thinking_animation.stop()
        # Log error if possible (though logging might fail if DB connection is issue)
        try:
            db_conn = get_db()
            log_interaction(db=db_conn, query=query if query else "CLI Startup Error", error_message=str(e))
        except Exception as log_e:
            console.print(f"[bold red]Additionally, failed to log error to DB:[/] {log_e}")


@app.command("version")
def version():
    """Show the version of LlamaSearch."""
    console.print("[bold blue]LlamaSearch ExperimentalAgents Augmented Professional[/] v1.0.0")


if __name__ == "__main__":
    app()
