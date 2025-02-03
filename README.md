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
python -m spacy download en_core_web_sm
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
cd src
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


&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

# ProjectTemplate-Python

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/xinntao/HandyView.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/xinntao/HandyView/context:python)
[![download](https://img.shields.io/github/downloads/xinntao/Real-ESRGAN/total.svg)](https://github.com/xinntao/Real-ESRGAN/releases)
[![Open issue](https://isitmaintained.com/badge/open/xinntao/basicsr.svg)](https://github.com/xinntao/basicsr/issues)
[![PyPI](https://img.shields.io/pypi/v/basicsr)](https://pypi.org/project/basicsr/)
[![LICENSE](https://img.shields.io/github/license/xinntao/basicsr.svg)](https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE)
[![python lint](https://github.com/xinntao/BasicSR/actions/workflows/pylint.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/xinntao/BasicSR/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/publish-pip.yml)
[![gitee mirror](https://github.com/xinntao/BasicSR/actions/workflows/gitee-mirror.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/gitee-mirror.yml)

[English](README.md) **|** [简体中文](README_CN.md) &emsp; [GitHub](https://github.com/xinntao/ProjectTemplate-Python) **|** [Gitee码云](https://gitee.com/xinntao/ProjectTemplate-Python)

## File Modification

1. Setup *pre-commit* hook
    1. If necessary, modify `.pre-commit-config.yaml`
    1. In the repository root path, run
    > pre-commit install
1. Modify the `.gitignore` file
1. Modify the `LICENSE` file
    This repository uses the *MIT* license, you may change it to other licenses
1. Modify the *setup* files
    1. `setup.cfg`
    1. `setup.py`, especially the `basicsr` keyword
1. Modify the `requirements.txt` files
1. Modify the `VERSION` file

## GitHub Workflows

1. [pylint](./github/workflows/pylint.yml)
1. [gitee-repo-mirror](./github/workflow/gitee-repo-mirror.yml) - Support Gitee码云
    1. Clone GitHub repo in the [Gitee](https://gitee.com/) website
    1. Modify [gitee-repo-mirror](./github/workflow/gitee-repo-mirror.yml)
    1. In Github *Settings* -> *Secrets*, add `SSH_PRIVATE_KEY`

## Other Procedures

1. The `description`, `website`, `topics` in the main page
1. Support Chinese documents, for example, `README_CN.md`

## Emoji

[Emoji cheat-sheet](https://github.com/ikatyang/emoji-cheat-sheet)

| Emoji | Meaning |
| :---         |     :---:      |
| :rocket:   | Used for [BasicSR](https://github.com/xinntao/BasicSR) Logo |
| :sparkles: | Features |
| :zap: | HOWTOs |
| :wrench: | Installation / Usage |
| :hourglass_flowing_sand: | TODO list |
| :turtle: | Dataset preparation |
| :computer: | Commands |
| :european_castle: | Model zoo |
| :memo: | Designs |
| :scroll: | License and acknowledgement |
| :earth_asia: | Citations |
| :e-mail: | Contact |
| :m: | Models |
| :arrow_double_down: | Download |
| :file_folder: | Datasets |
| :chart_with_upwards_trend: | Curves|
| :eyes: | Screenshot |
| :books: |References |

## Useful Image Links

<img src="https://colab.research.google.com/assets/colab-badge.svg" height="28" alt="google colab logo">  Google Colab Logo <br>
<img src="https://upload.wikimedia.org/wikipedia/commons/8/8d/Windows_darkblue_2012.svg" height="28" alt="google colab logo">  Windows Logo <br>
<img src="https://upload.wikimedia.org/wikipedia/commons/3/3a/Logo-ubuntu_no%28r%29-black_orange-hex.svg" alt="Ubuntu" height="24">  Ubuntu Logo <br>

## Other Useful Tips

1. `More` drop-down menu
    <details>
    <summary>More</summary>
    <ul>
    <li>Nov 19, 2020. Set up ProjectTemplate-Python.</li>
    </ul>
    </details>
