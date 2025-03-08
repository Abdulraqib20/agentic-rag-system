# config/appconfig.py
import os
import logging

# Simple direct exports with validation
try:
    import streamlit as st
    GROQ_API_KEY = st.secrets.GROQ.API_KEY
    FIRECRAWL_API_KEY = st.secrets.FIRECRAWL.API_KEY
    SERPER_API_KEY = st.secrets.SERPER.API_KEY
    QDRANT_API_KEY = st.secrets.QDRANT.API_KEY
    QDRANT_LOCATION = st.secrets.QDRANT.LOCATION
    MODEL = st.secrets.LLM.MODEL_NAME
except (ImportError, KeyError) as e:
    raise RuntimeError(f"Missing configuration: {str(e)}") from e

__all__ = [
    'GROQ_API_KEY',
    'FIRECRAWL_API_KEY',
    'SERPER_API_KEY',
    'QDRANT_API_KEY',
    'QDRANT_LOCATION',
    'MODEL'
]