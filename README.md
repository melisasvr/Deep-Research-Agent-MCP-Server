# 🔍 Deep Research Agent MCP Server

**An AI-powered deep research agent built with Python, FastMCP, and Streamlit.**  
Search → Fetch → Cluster → Report. Fully automated. Fully open source.

![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python)
![FastMCP](https://img.shields.io/badge/FastMCP-3.x-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

</div>

---

## 📖 Overview

Deep Research Agent is a **Python MCP server** that automates the full research pipeline from web search to a structured executive report in under 10 seconds. It uses a 4-tool pipeline, powered by the [Tavily Search API](https://tavily.com) for retrieval and a pure-Python TF-IDF + K-Means engine for semantic clustering. The frontend is a sleek **Streamlit chat interface** that shows every step live.

> 💡 Built as a portfolio project demonstrating: FastMCP tool design, async web scraping, NLP clustering without heavy ML dependencies, and full-stack Python app architecture.

---

## ✨ Features

- 🔎 **Multi-angle web search** — 3 query variations per topic for broader coverage
- 🌐 **Async URL fetching** — parallel page retrieval with smart HTML cleaning
- 🧹 **Text denoising** — strips HTML entities, SVG labels, nav boilerplate, repeated patterns
- 🧠 **Semantic clustering** — pure-Python TF-IDF + K-Means, zero ML framework required
- 🏷️ **Auto cluster labeling** — 13 topic categories (Quantum, Biotech, Climate, AI, Policy...)
- 📄 **Structured reports** — markdown or JSON output with sources, keywords, evidence
- ⬇️ **One-click download** — export `.md` report directly from the UI
- ⚡ **Fast** — full pipeline typically completes in 6–12 seconds

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              Streamlit Frontend                 │
│              app.py  —  Chat UI                 │
│         Live step cards · Download button       │
└───────────────────┬─────────────────────────────┘
                    │  FastMCP Client (protocol-aware)
┌───────────────────▼─────────────────────────────┐
│           FastMCP Server  (server.py)           │
│                                                 │
│  🔎 search_web        🌐 fetch_and_chunk        │
│     Tavily API           httpx + HTML parser    │
│                                                 │
│  🧠 cluster_findings  📄 generate_report        │
│     TF-IDF + K-Means     Markdown / JSON        │
└─────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| 🐍 Language | Python 3.12+ |
| 🔌 MCP Framework | FastMCP 3.x |
| 🖥️ Frontend | Streamlit |
| 🔎 Search | Tavily Search API |
| 🌐 HTTP Client | httpx (async) |
| 🧠 Clustering | Pure-Python TF-IDF + K-Means |
| ⚙️ Config | python-dotenv |

---

## 🚀 Quick Start

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/deep-research-agent.git
cd deep-research-agent
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your Tavily API key:

```env
TAVILY_API_KEY=tvly-your-key-here
MCP_HOST=localhost
MCP_PORT=8000
TAVILY_SEARCH_DEPTH=advanced
```

> 🔑 Get a **free** Tavily API key at [app.tavily.com](https://app.tavily.com)

### 4️⃣ Start the MCP server

```bash
# Terminal 1
python server.py
```

You should see:
```
🔍 Deep Research Agent MCP Server
   Host : localhost
   Port : 8000
   Tools: search_web, fetch_and_chunk, cluster_findings, generate_report
   Tavily key: ✓ set
```

### 5️⃣ Launch the Streamlit UI

```bash
# Terminal 2
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) and start researching! 🎉

---

## 🔧 MCP Tools Reference

### 🔎 `search_web(query, max_results)`
- Searches the web via Tavily API and returns ranked results with scores.

```python
search_web(
    query="AI energy crisis 2026",
    max_results=8,              # 1–15
    include_domains=None,       # e.g. [".edu", ".gov"]
    exclude_domains=None
)
```

### 🌐 `fetch_and_chunk(urls, chunk_size)`
- Fetches pages asynchronously and splits content into overlapping text chunks.

```python
fetch_and_chunk(
    urls=["https://example.com/article"],
    chunk_size=400,             # words per chunk
    chunk_overlap=50,           # overlap between chunks
    max_chunks_per_url=6
)
```

### 🧠 `cluster_findings(chunks, n_clusters)`
- Groups chunks into semantic themes using TF-IDF vectorization + K-Means clustering.

```python
cluster_findings(
    chunks=[...],               # from fetch_and_chunk
    n_clusters=4,               # 2–6 themes
    top_terms_per_cluster=8
)
```

### 📄 `generate_report(topic, clusters)`
- Synthesizes all clusters into a structured research report.

```python
generate_report(
    topic="AI energy crisis 2026",
    clusters=[...],             # from cluster_findings
    format="markdown",          # or "json"
    include_sources=True
)
```

---

## 🔄 Research Pipeline

```
📝 User enters topic
        │
        ▼
🔎 search_web() × 3 queries ──────────► 24 ranked sources
        │
        ▼
🌐 fetch_and_chunk() on top 5 URLs ───► 20–30 text chunks
        │
        ▼
🧠 cluster_findings() ────────────────► 4 semantic themes
        │
        ▼
📄 generate_report() ─────────────────► Structured .md report
        │
        ▼
⬇️  Download / Display in UI
```

---

## 📁 Project Structure

```
deep-research-agent/
│
├── 📄 server.py          # FastMCP server — all 4 tools
├── 🖥️  app.py             # Streamlit frontend
├── 📋 requirements.txt   # Python dependencies
├── 🔒 .env.example       # Environment variable template
└── 📖 README.md          # This file
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TAVILY_API_KEY` | *(required)* | Your Tavily search API key |
| `MCP_HOST` | `localhost` | MCP server host |
| `MCP_PORT` | `8000` | MCP server port |
| `TAVILY_SEARCH_DEPTH` | `advanced` | `basic` (faster) or `advanced` (thorough) |

---

## 🤝 Contributing
- Contributions are welcome and appreciated! Here's how to get involved:

### 🐛 Reporting Bugs

1. Check the [Issues](https://github.com/yourusername/deep-research-agent/issues) page to see if it's already reported
2. Open a new issue with:
   - A clear title and description
   - Steps to reproduce
   - Expected vs actual behaviour
   - Your Python version and OS

### 💡 Suggesting Features

Open an issue with the `enhancement` label and describe:
- The problem you're trying to solve
- Your proposed solution
- Why would it benefit other users

### 🔧 Submitting Pull Requests

1. **Fork** the repository
2. **Create** a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make** your changes with clear, descriptive commits
   ```bash
   git commit -m "feat: add BM25 ranking to cluster_findings"
   ```
4. **Test** your changes thoroughly
5. **Push** to your fork
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open** a Pull Request with a clear description of what you changed and why

### 📐 Code Style

- Follow [PEP 8](https://pep8.org/) for Python code
- Use type hints wherever possible
- Add docstrings to all new functions
- Keep functions focused — one responsibility per function

### 🌱 Good First Issues
- Looking for a place to start? Check issues tagged `good first issue`:
- Adding more cluster label categories to `_infer_cluster_label()`
- Improving the HTML cleaning regex patterns
- Adding a progress bar to the Streamlit UI
- Supporting additional output formats (PDF, DOCX)
- Writing unit tests for the clustering functions

---

## 📜 License

```
MIT License

Copyright (c) 2026 Deep Research Agent Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including, without limitation, the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgements

- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP server framework
- [Tavily](https://tavily.com) — AI-optimised search API
- [Streamlit](https://streamlit.io) — Python web app framework
- [httpx](https://www.python-httpx.org/) — Async HTTP client

---

<div align="center">

- Made with ❤️ and Python · Star ⭐ this repo if you found it useful!

</div>
