# <center>Agentic RAG using CrewAI</center>
<!-- # Agentic RAG System -->

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project seeks to leverage CrewAI to build an Agentic RAG that can search through your docs and fallbacks to web search in case it doesn't find the answer in the docs, have option to use either of deep-seek-r1 or llama 3.2, both hosted on the Groq Inference LPU. 

## Features

- PDF Document Processing Pipeline
- Semantic Search with Qdrant
- CrewAI Agent Orchestration
- Secure API Key Management
- Streamlit Web Interface

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt