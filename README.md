# 🔬 Multi-Agent Research Assistant

A PhD-level autonomous research assistant powered by multiple AI agents working in concert. The system breaks down complex research questions, searches the web, analyzes documents, and produces comprehensive research reports with proper citations — all with human oversight at critical decision points.

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.59-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

- **🤖 Multi-Agent Architecture** — 5 specialized AI agents orchestrated through LangGraph
- **🔍 Comprehensive Web Search** — Parallel searches across 20-25 sources including academic papers, news, and company websites
- **📄 Intelligent Document Analysis** — Automatic chunking, embedding, and semantic storage in vector database
- **📝 Research Report Generation** — Detailed reports with executive summaries, key findings, and proper citations
- **👤 Human-in-the-Loop (HITL)** — Strategic interrupt points for human oversight and approval
- **🔄 Self-Reflection** — Quality evaluation with automatic iteration when needed

---

## 🏗️ Architecture

### Agent Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Clarification  │────▶│   Web Search    │────▶│    Document     │
│     Agent       │     │     Agent       │     │    Analyzer     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Output      │◀────│   Reflection    │◀────│    Synthesis    │
│                 │     │     Agent       │     │     Agent       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Agent Descriptions

| Agent | Role |
|-------|------|
| **Clarification Agent** | Refines vague queries by asking intelligent follow-up questions |
| **Web Search Agent** | Generates diverse search queries and executes parallel searches |
| **Document Analyzer** | Downloads, chunks, and stores documents in vector database |
| **Synthesis Agent** | Performs semantic search and generates comprehensive reports |
| **Reflection Agent** | Evaluates report quality and decides on iteration needs |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Orchestration** | LangChain, LangGraph |
| **LLM** | Claude Sonnet 4 (Anthropic API) |
| **Vector Database** | Qdrant |
| **Web Search** | Tavily |
| **Embeddings** | OpenAI Embeddings |
| **UI** | Streamlit |
| **Package Manager** | UV |
| **Terminal Output** | Rich |

---

## 📁 Project Structure

```
multi-agent-research/
├── agents/              # Individual agent implementations
│   ├── clarification.py
│   ├── web_search.py
│   ├── document_analyzer.py
│   ├── synthesis.py
│   └── reflection.py
├── graph/               # LangGraph workflow orchestration
│   ├── state.py         # Shared state definitions
│   └── workflow.py      # Agent graph and transitions
├── utils/               # Utilities and helpers
│   ├── config.py        # Configuration management
│   ├── vector_store.py  # Qdrant integration
│   └── metrics.py       # Performance tracking
├── storage/             # Local Qdrant database files
├── ui/                  # Streamlit web interface
│   └── streamlit_app.py
├── tests/               # Unit and integration tests
├── notebooks/           # Jupyter notebooks for experimentation
├── docs/                # Documentation
├── main.py              # Entry point
├── pyproject.toml       # Project configuration (UV)
└── requirements.txt     # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- [UV](https://github.com/astral-sh/uv) package manager
- API keys for Anthropic, OpenAI, and Tavily

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/multi-agent-research.git
   cd multi-agent-research
   ```

2. **Create and activate virtual environment**
   ```bash
   uv venv
   .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   # or
   source .venv/bin/activate      # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   ANTHROPIC_API_KEY=your_anthropic_key
   OPENAI_API_KEY=your_openai_key
   TAVILY_API_KEY=your_tavily_key
   ```

5. **Run the application**
   ```bash
   # CLI mode
   python main.py
   
   # Web UI mode
   streamlit run ui/streamlit_app.py
   ```

---

## 💡 Usage

### Basic Research Query

```python
from graph.workflow import run_research

result = run_research(
    query="What are the latest developments in quantum computing for drug discovery?"
)

print(result.report)
```

### Expected Output

For a query like *"What are the latest developments in quantum computing for drug discovery?"*, the system will:

1. 🔍 Search 25+ sources (Nature, ArXiv, TechCrunch, company blogs)
2. 📄 Process hundreds of pages of content
3. 💾 Store 250+ searchable document chunks in vector database
4. 📝 Generate a comprehensive research report including:
   - Executive summary
   - Key findings from multiple sources
   - Recent breakthroughs (last 12 months)
   - Commercial applications and partnerships
   - Technical analysis
   - Limitations and challenges
   - Future directions
   - 20+ properly cited sources

---

## 👤 Human-in-the-Loop Checkpoints

The system has strategic interrupt points where humans can:

- ✅ Review and approve/reject discovered sources before analysis
- 🔄 Provide feedback to refine search strategies
- 📋 Review draft reports and request revisions
- 🎯 Decide whether to continue research iterations or accept output
- ⚙️ Modify search parameters based on intermediate results

---

## 📊 Success Metrics

| Metric | Target |
|--------|--------|
| Research any topic | ✅ Coherent reports |
| Response time | < 3 minutes |
| Report quality | Junior researcher level |
| Source handling | 90%+ success rate |
| HITL interrupts | Smooth workflow |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=agents --cov=graph --cov=utils
```

---

## 📝 Development Roadmap

- [x] Week 1: Setup infrastructure, implement basic agents, test vector database
- [ ] Week 2: Build LangGraph workflow, integrate all agents, add basic UI
- [ ] Week 3: Implement HITL features, add quality checks, testing
- [ ] Week 4: Polish UI, create documentation, record demo, deploy

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the amazing LLM framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for multi-agent orchestration
- [Anthropic](https://anthropic.com/) for Claude
- [Qdrant](https://qdrant.tech/) for vector database
- [Tavily](https://tavily.com/) for AI-optimized web search

---

<p align="center">
  Built with ❤️ for advanced AI research
</p>
