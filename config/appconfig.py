import streamlit as st

import sys
import os
import logging
from pathlib import Path

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Secret Sources Priority: 1. Streamlit Secrets 2. Environment Variables
try:
    config_source = st.secrets
    logger.info("Using Streamlit secrets for configuration")
except ImportError:
    config_source = os.environ
    logger.info("Using environment variables for configuration")

# Required configuration keys (match secrets.toml structure)
REQUIRED_KEYS = {
    "groq": ["api_key"],
    "firecrawl": ["api_key"],
    "serper": ["api_key"],
    "qdrant": ["api_key", "location"],
    "model": ["name"],
    "embeddings": ["model"]
}

# Build config from secrets
config = {}
try:
    # Groq Configuration
    config.update({
        "GROQ_API_KEY": config_source["groq"]["api_key"],
        "MODEL": config_source["model"]["name"]
    })
    
    # Firecrawl Configuration
    config["FIRECRAWL_API_KEY"] = config_source["firecrawl"]["api_key"]
    
    # Serper Configuration
    config["SERPER_API_KEY"] = config_source["serper"]["api_key"]
    
    # Qdrant Configuration
    config.update({
        "QDRANT_API_KEY": config_source["qdrant"]["api_key"],
        "QDRANT_LOCATION": config_source["qdrant"]["location"]
    })
    
    # Add to your config class
    EMBEDDING_MODEL = config_source["embeddings"]["model"]

except KeyError as e:
    missing = str(e).strip("'")
    logger.error(f"Missing required configuration key: {missing}")
    raise RuntimeError(f"Configuration error: Missing {missing}") from e

# Export variables
globals().update(config)
__all__ = list(config.keys())

logger.info("Configuration loaded successfully")
logger.debug("Loaded configuration:\n" + "\n".join(
    f"{k}: {v[:3]}...{v[-3:] if len(v) > 6 else '***'}" 
    for k, v in config.items()
))