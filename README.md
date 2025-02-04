# 🔍 Entity Search and Analysis App

## 🚀 Project Overview

This advanced AI-powered Entity Search and Analysis application leverages cutting-edge technologies to provide comprehensive insights into entities discovered through web searches.

## ✨ Key Features

- **Web Search**: Perform intelligent searches across multiple search engines
- **Entity Extraction**: Identify and categorize key entities from search results
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
# Clone the repository
git clone https://github.com/lucky-verma/Agentic-Entity-Search.git

# Navigate to project directory
cd Agentic-Entity-Search

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_trf
```

## 🔧 Configuration

1. Create a `.env` file in the project root
2. Add the following configurations:

``` bash
SEARXNG_URL=http://localhost:8080
OLLAMA_URL=http://localhost:11434/api/generate
LLM_MODEL=llama3.2
VECTOR_DIMENSION=384
```

## 🚀 Running the Application

```bash
# Start the search service
cd searxng_docker
docker compose up -d
```

```bash
# Start the crawler service
cd src
python web_scraper_service.py
```

```bash
# Run the Streamlit app
streamlit run app.py
```

## 🌟 Usage

1. Enter an entity name in the search bar
2. Click "Search"
3. Explore results through different tabs:
   - Structured Output
   - 3D Entity Visualization
   - Raw Search Results

## 📊 Entity Visualization

The 3D latent space chart provides:

- Semantic proximity between entities
- Entity frequency
- Color-coded importance
- Interactive exploration
