# 🔍 AEGIS: Advanced Entity-aware Generative Intelligent Search

## 🚀 Project Overview

AEGIS is a cutting-edge AI-powered semantic search engine with multi-hop reasoning capabilities, designed for complex entity relationship discovery and context-aware information retrieval. It leverages advanced technologies to provide comprehensive insights into entities discovered through intelligent web searches.

## ✨ Key Features

- **Neural Query Expansion**: Automatically enriches search terms using contextual embeddings
- **Multi-Hop Reasoning Engine**: Chains related concepts across documents through probabilistic inference
- **Entity-Aware Crawling**: Focused web harvesting with dynamic priority queuing (Crawl4AI integration)
- **Semantic Indexing**: Hybrid vector-relational storage for fast concept retrieval
- **3D Latent Space Visualization**: Explore semantic relationships between entities
- **AI-Powered Analysis**: Generate structured insights using advanced language models

## 🛠 Technologies Used

- **Web Search**: SearXNG
- **Crawler**: Crawl4AI
- **Entity Extraction**: SpaCy
- **Embedding**: Sentence Transformers
- **Dimensionality Reduction**: UMAP
- **Visualization**: Plotly
- **Language Model**: Ollama (Llama3.2)
- **Frontend**: Streamlit

## 📦 Installation

```bash
git clone https://github.com/lucky-verma/Aegis.git
cd Aegis
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

## 🔧 Configuration

Create a `.env` file in the project root with the following:

```bash
SEARXNG_URL=http://localhost:8080
OLLAMA_URL=http://localhost:11434/api/generate
LLM_MODEL=llama3.2
VECTOR_DIMENSION=384
```

## 🚀 Running AEGIS

```bash
# Start the search service
cd searxng_docker
docker compose up -d

# Start other services
docker compose up -d

# Run the Streamlit app
streamlit run src/app.py
```

## 🌟 Usage

1. Enter an entity or query in the search bar
2. Explore results through:
   - Structured Output
   - 3D Entity Visualization
   - Raw Search Results

## 📊 Entity Visualization

The 3D latent space chart provides:

- Semantic proximity between entities
- Entity frequency and importance (color-coded)
- Interactive exploration

## 🔮 Future Developments

- Enhance Haystack implementation for improved vector store and retrieval
- Optimize multi-hop reasoning and query expansion algorithms
- Implement advanced evaluation metrics for search quality
- Expand UI features for more intuitive data exploration

## 📂 Project Structure

```bash
Aegis/
├── .github/
├── .streamlit/
├── scraper_service/
├── searxng_docker/
├── src/
│   ├── agents/
│   │   ├── retriever.py    # Hybrid search + context management
│   │   ├── reasoner.py     # Cognitive processing pipeline
│   │   └── evaluator.py    # Quality assessment framework
│   ├── utils/
│   ├── app.py              # Orchestration & UI
│   └── advanced_visualizations.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```
