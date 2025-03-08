import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

try:
    root_dir = Path(__file__).parent.parent  # Running from a .py file
except NameError:
    root_dir = Path(os.getcwd()).parent  # Running from a Jupyter notebook

# Ensure logs directory exists
log_dir = root_dir / 'logs'
log_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'config.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = root_dir / '.env'
if not env_path.exists():
    logger.error(f".env file not found at {env_path}. Please ensure it exists.")
    raise FileNotFoundError(f".env file not found at {env_path}")

load_dotenv(env_path)

# Required environment variables
REQUIRED_VARS = [
    'GROQ_API_KEY',
    'FIRECRAWL_API_KEY',
    'MODEL',
    'QDRANT_API_KEY',
    'QDRANT_LOCATION'
]

# Load and validate environment variables
config = {var: os.getenv(var) for var in REQUIRED_VARS}
missing_vars = [var for var, value in config.items() if not value]

if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
    logger.error(error_msg)
    raise ValueError(error_msg)

# Export variables explicitly
GROQ_API_KEY = config['GROQ_API_KEY']
FIRECRAWL_API_KEY = config['FIRECRAWL_API_KEY']
MODEL = "groq/llama-3.3-70b-versatile"
QDRANT_API_KEY = config['QDRANT_API_KEY']
QDRANT_LOCATION = config['QDRANT_LOCATION']

logger.info("Configuration loaded successfully")
logger.info("Loaded configuration values:")
for key, value in config.items():
    masked_value = f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}" if value and len(value) > 4 else value
    logger.info(f"{key}: {masked_value}")

__all__ = [
    'GROQ_API_KEY',
    'FIRECRAWL_API_KEY',
    'MODEL',
    'QDRANT_API_KEY',
    'QDRANT_LOCATION'
]
