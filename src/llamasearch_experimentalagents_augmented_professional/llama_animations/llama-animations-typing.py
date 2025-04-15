"""
Llama typing effect animation for the LlamaSearch CLI.

This module provides a Rich-powered typing effect to simulate the assistant
typing out responses in real-time.
"""

import time
import threading
import random
from typing import Optional, Callable, List

from rich.live import Live
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown


class LlamaTypingEffect:
    """A rich-powered typing effect for displaying text progressively."""
    
    def __init__(
        self,
        text: str,
        typing_speed: float = 0.03,
        variance: float = 0.02,
        use_markdown: bool = True,
        title: str = "🦙 LlamaSearch",
    ):
        """
        Initialize the typing effect.
        
        Args:
            text: The text to display with typing effect
            typing_speed: Base time between characters in seconds
            variance: Random variance in typing speed
            use_markdown: Whether to render the text as markdown
            title: The title to display in the panel
        """
        self.full_text = text
        self.current_text = ""
        self.typing_speed = typing_speed
        self.variance = variance
        self.use_markdown = use_markdown
        self.title = title
        self.typing = False
        self.thread: Optional[threading.Thread] = None
        self.live: Optional[Live] = None
        self.stop_event = threading.Event()
        self.on_complete_callbacks: List[Callable[[], None]] = []
    
    def get_display_element(self) -> any:
        """Get the current text as a Rich element."""
        if self.use_markdown:
            return Markdown(self.current_text)
        return Text(self.current_text)
    
    def __rich__(self) -> Panel:
        """Generate a rich panel with the current text."""
        return Panel(
            self.get_display_element(),
            title=self.title,
            border_style="green",
        )
    
    def _type(self) -> None:
        """Animation loop to run in a separate thread."""
        console = Console()
        
        with Live(self.__rich__(), console=console, refresh_per_second=20) as live:
            self.live = live
            current_length = 0
            full_length = len(self.full_text)
            
            while current_length < full_length and not self.stop_event.is_set():
                # Calculate how many characters to add in this frame
                # Usually 1, but occasionally more for a natural-looking rhythm
                chars_to_add = 1
                if random.random() < 0.1:  # 10% chance for a "burst" of typing
                    chars_to_add = random.randint(2, 5)
                
                # Ensure we don't exceed the full text length
                next_length = min(current_length + chars_to_add, full_length)
                self.current_text = self.full_text[:next_length]
                current_length = next_length
                
                # Update the display
                live.update(self.__rich__())
                
                # If we're done, break
                if current_length >= full_length:
                    break
                
                # Calculate and apply delay with variance
                delay = self.typing_speed
                if self.variance > 0:
                    delay += random.uniform(-self.variance, self.variance)
                delay = max(0.005, delay)  # Ensure minimum delay
                
                # Add occasional longer pause for punctuation
                if current_length > 0 and self.full_text[current_length - 1] in ".!?,:;":
                    delay += random.uniform(0.1, 0.3)
                
                time.sleep(delay)
            
            # Ensure the full text is shown at the end
            self.current_text = self.full_text
            live.update(self.__rich__())
            
            # The animation is complete
            self.typing = False
            
            # Call any registered complete callbacks
            for callback in self.on_complete_callbacks:
                callback()
    
    def start(self) -> None:
        """Start the typing effect in a separate thread."""
        if self.typing:
            return
        
        self.typing = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._type)
        self.thread.daemon = True
        self.thread.start()
    
    def complete_immediately(self) -> None:
        """Show the full text immediately."""
        self.current_text = self.full_text
        if self.live:
            self.live.update(self.__rich__())
    
    def stop(self) -> None:
        """Stop the typing effect and display the full text."""
        if not self.typing:
            return
        
        self.complete_immediately()
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        self.typing = False
    
    def on_complete(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when typing is complete."""
        self.on_complete_callbacks.append(callback)


class LlamaResponseTypingEffect(LlamaTypingEffect):
    """A specialized typing effect for assistant responses."""
    
    def __init__(
        self,
        response_text: str,
        typing_speed: float = 0.02,
        title: str = "🦙 LlamaSearch Response",
    ):
        """
        Initialize the response typing effect.
        
        Args:
            response_text: The response text to display
            typing_speed: Base time between characters in seconds
            title: The title to display in the panel
        """
        # Format the response as markdown with styling
        formatted_text = f"""
# {title}

{response_text}
        """
        
        super().__init__(
            text=formatted_text,
            typing_speed=typing_speed,
            use_markdown=True,
            title=""  # Empty title since it's in the markdown
        )


if __name__ == "__main__":
    # Example usage
    sample_text = """
# Example Response

The **LlamaSearch** system is designed to provide:

1. Fast semantic search
2. Natural language responses
3. Professional formatting

Would you like to know more about any specific feature?
    """
    
    effect = LlamaTypingEffect(sample_text, typing_speed=0.01)
    effect.start()
    
    try:
        # Wait for typing to complete
        if effect.thread:
            effect.thread.join()
    except KeyboardInterrupt:
        effect.stop()
