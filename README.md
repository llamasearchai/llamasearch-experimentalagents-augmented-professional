# LlamaSearch ExperimentalAgents Augmented Professional

<div align="center">

![LlamaSearch Logo](docs/assets/logo.png)

**AI-Powered Knowledge Democratization Engine**

*A cutting-edge CLI tool that makes AI knowledge accessible through semantic search, accelerated by MLX/JAX, with Rich-powered visual interactions*

[![CI](https://github.com/example/llamasearch-experimentalagents-augmented-professional/actions/workflows/ci.yml/badge.svg)](https://github.com/example/llamasearch-experimentalagents-augmented-professional/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/llamasearch-experimentalagents-augmented-professional.svg)](https://badge.fury.io/py/llamasearch-experimentalagents-augmented-professional)
[![Documentation Status](https://readthedocs.org/projects/llamasearch-experimentalagents-augmented-professional/badge/?version=latest)](https://llamasearch-experimentalagents-augmented-professional.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

## 🚀 Features

- **Hybrid MLX/JAX Semantic Search**: Dynamically selects the fastest hardware acceleration based on your device
- **Local Knowledge Base**: Process your own documents and create a searchable knowledge base
- **Professional-Grade Output**: Structured responses with confidence scores and suggested actions
- **Engaging CLI**: Rich-powered animations and interactive experience
- **Comprehensive Testing**: Property-based testing with Hypothesis and performance benchmarks

## 🧠 How It Works

LlamaSearch delivers a professional AI knowledge search experience through:

1. **Knowledge Ingestion**: Load your text documents into a structured knowledge base
2. **Semantic Embedding**: Convert content into vector embeddings for similarity search
3. **Hardware-Accelerated Search**: Find the most relevant content using MLX (Apple Silicon) or JAX
4. **AI-Powered Responses**: Generate professional, structured responses with sources

## 🔧 Installation

```bash
# Basic installation
pip install llamasearch-experimentalagents-augmented-professional

# With MLX support (Apple Silicon)
pip install "llamasearch-experimentalagents-augmented-professional[mlx]"

# With JAX support
pip install "llamasearch-experimentalagents-augmented-professional[jax]"

# Full installation with all extras
pip install "llamasearch-experimentalagents-augmented-professional[all]"
```

## 🚀 Quick Start

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Ask a question with animated interface
llamasearch ask --query "Explain neural networks" --visual animated

# Interactive mode with your own knowledge base
llamasearch ask --knowledge ./my_documents --visual animated
```

## 📚 Core Components

- **SemanticRetriever**: Hardware-accelerated semantic search engine
- **LlamaAssistant**: AI agent that combines search with natural language generation
- **KnowledgeBase**: Structured representation of your documents
- **LlamaAnimations**: Engaging visual feedback for CLI interactions

## 📊 Performance

LlamaSearch leverages hardware acceleration for maximum performance:

| Backend | 1K Documents | 10K Documents |
|---------|--------------|---------------|
| MLX     | ~2.1ms       | ~15ms         |
| JAX     | ~2.8ms       | ~22ms         |
| NumPy   | ~5.4ms       | ~45ms         |

*Benchmark results on M2 Ultra. Your results may vary based on hardware.*

## 🏗️ Architecture

```
llamasearch-experimentalagents-augmented-professional/
├── src/llamasearch/
│   ├── agents/
│   │   ├── retriever.py     # MLX/JAX-accelerated search
│   │   └── assistant.py     # Main agent logic
│   ├── models/
│   │   ├── knowledge.py     # Dataclass + Pydantic
│   │   └── responses.py     # Structured output
│   ├── llama_animations/
│   │   ├── thinking.py      # Animated thinking state
│   │   └── typing_effect.py # Typing animation
│   └── cli.py               # Command-line interface
├── tests/                   # Comprehensive test suite
├── docs/                    # Documentation
└── knowledge_base/          # Sample knowledge files
```

## 🔍 Example Usage

### Basic Query

```python
from llamasearch import KnowledgeBase, LlamaAssistant
from openai import OpenAI

# Initialize components
client = OpenAI()
kb = KnowledgeBase(name="My Knowledge Base")

# Load your knowledge (simplified example)
kb.add_chunk(content="Neural networks are computational systems inspired by the brain...", 
             source="ai_fundamentals.md")

# Create assistant
assistant = LlamaAssistant(knowledge_base=kb, openai_client=client)

# Generate response
response = assistant.generate_response("Explain neural networks")

# Access structured data
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence:.0%}")
print(f"Sources: {[s.source for s in response.sources]}")
```

### CLI Experience

```bash
# Interactive mode
llamasearch ask

# With specific options
llamasearch ask --query "How do transformers work?" --knowledge ./ml_papers --detailed
```

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

Apache 2.0. See [LICENSE](LICENSE) for details.

## ✨ Acknowledgements

- Built with [OpenAI API](https://openai.com/api/) for embeddings and completion
- Powered by [MLX](https://github.com/ml-explore/mlx) and [JAX](https://github.com/google/jax) for acceleration
- CLI experience enhanced by [Rich](https://github.com/Textualize/rich) and [Textual](https://github.com/Textualize/textual)
