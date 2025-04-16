"""
Llama thinking animation for the LlamaSearch CLI.

This module provides a rich-powered animation to display while the assistant
is processing a query or searching the knowledge base.
"""

import time
import threading
from typing import Optional, List, Callable

from rich.live import Live
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.spinner import Spinner
from rich.align import Align

# ASCII art frames for llama animation
LLAMA_FRAMES = [
    r"""
    //\\
   _(o o)_
   /|/\/\|\
     |  |
    / \/ \
    """,
    r"""
    //\\
   _(- o)_
   /|/\/\|\
     |  |
    / \/ \
    """,
    r"""
    //\\
   _(o -)_
   /|/\/\|\
     |  |
    / \/ \
    """,
]

THINKING_THOUGHTS = [
    "Consulting knowledge base...",
    "Retrieving relevant information...",
    "Processing your query...",
    "Searching for answers...",
    "Analyzing knowledge base...",
    "Finding connections...",
    "Computing semantic similarity...",
]


class LlamaThinking:
    """A rich-powered thinking animation featuring a llama."""
    
    def __init__(
        self,
        title: str = "🧠 Augmented Professional",
        subtitle: str = "Generating expert response...",
        frame_rate: float = 0.2,
    ):
        """
        Initialize the thinking animation.
        
        Args:
            title: The title to display in the panel
            subtitle: The subtitle to display under the animation
            frame_rate: The animation frame rate in seconds
        """
        self.title = title
        self.subtitle = subtitle
        self.frame_rate = frame_rate
        self.current_frame = 0
        self.current_thought = 0
        self.thought_change_counter = 0
        self.spinner = Spinner("dots")
        self.thinking = False
        self.thread: Optional[threading.Thread] = None
        self.live: Optional[Live] = None
        self.stop_event = threading.Event()
        self.on_stop_callbacks: List[Callable[[], None]] = []
    
    def get_llama_frame(self) -> str:
        """Get the current llama ASCII art frame."""
        return LLAMA_FRAMES[self.current_frame]
    
    def get_thought(self) -> str:
        """Get the current thinking thought."""
        return THINKING_THOUGHTS[self.current_thought]
    
    def __rich__(self) -> Panel:
        """Generate a rich panel with the current animation frame."""
        # Get the current animation elements
        llama_art = Text(self.get_llama_frame(), style="bright_yellow")
        thought = Text(self.get_thought(), style="italic bright_blue")
        
        # Create the panel content
        header = Text(self.title, style="bold")
        footer = Text(self.subtitle, style="italic")
        
        # Put everything together
        content = Group(
            Align.center(header),
            Align.center(llama_art),
            Align.center(Group(
                Align.center(self.spinner),
                Align.center(thought),
            ))
        )
        
        return Panel(
            content,
            border_style="blue",
            subtitle=footer,
            subtitle_align="center",
        )
    
    def _animate(self) -> None:
        """Animation loop to run in a separate thread."""
        console = Console()
        
        with Live(self.__rich__(), console=console, refresh_per_second=10) as live:
            self.live = live
            
            while not self.stop_event.is_set():
                # Update animation
                self.current_frame = (self.current_frame + 1) % len(LLAMA_FRAMES)
                self.thought_change_counter += 1
                
                # Change the thought text occasionally
                if self.thought_change_counter >= 10:
                    self.current_thought = (self.current_thought + 1) % len(THINKING_THOUGHTS)
                    self.thought_change_counter = 0
                
                # Update the display
                live.update(self.__rich__())
                
                # Wait for next frame
                time.sleep(self.frame_rate)
            
            # Final update before stopping
            live.update(Panel(
                Align.center(Text("✅ Processing complete!", style="bold green")),
                border_style="green"
            ))
            time.sleep(0.5)
        
        # Call any registered stop callbacks
        for callback in self.on_stop_callbacks:
            callback()
    
    def start(self) -> None:
        """Start the thinking animation in a separate thread."""
        if self.thinking:
            return
        
        self.thinking = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self) -> None:
        """Stop the thinking animation."""
        if not self.thinking:
            return
        
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        self.thinking = False
    
    def on_stop(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when the animation stops."""
        self.on_stop_callbacks.append(callback)


if __name__ == "__main__":
    # Example usage
    animation = LlamaThinking()
    animation.start()
    
    try:
        time.sleep(5)  # Let it run for 5 seconds
    finally:
        animation.stop()
