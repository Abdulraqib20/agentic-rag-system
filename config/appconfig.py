import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import sys

try:
    # This works when running from a .py file
    root_dir = Path(__file__).parent.parent
except NameError:
    # This works when running from a Jupyter notebook
    root_dir = Path(os.getcwd()).parent

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
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler(log_dir / 'config.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


sys.path.append(str(root_dir))

env_path = root_dir / '.env'
if not env_path.exists():
    logger.error(f"Environment file not found at {env_path}. Please ensure it exists.")
    raise FileNotFoundError(f".env file not found at {env_path}")

load_dotenv(env_path)

# Required environment variables
REQUIRED_VARS = [
    'GROQ_API_KEY', 
    'FIRECRAWL_API_KEY',
    'MODEL',
    'QDRANT_API_KEY',
    'QDRANT_LOCATION',
    'SERPER_API_KEY'
]

# Load and validate environment variables
config = {}
missing_vars = []

for var in REQUIRED_VARS:
    value = os.getenv(var)
    if not value:
        missing_vars.append(var)
    config[var] = value

if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
    logger.error(error_msg)
    raise ValueError(error_msg)

# Export variables explicitly
GROQ_API_KEY = config['GROQ_API_KEY']
FIRECRAWL_API_KEY = config['FIRECRAWL_API_KEY']
# MODEL = "groq/llama-3.3-70b-versatile"
MODEL = config['MODEL']
QDRANT_API_KEY = config['QDRANT_API_KEY']
QDRANT_LOCATION = config['QDRANT_LOCATION']
SERPER_API_KEY = config['SERPER_API_KEY']

logger.info("Configuration loaded successfully")
logger.info("Loaded configuration values:")
for key, value in config.items():
    # Mask sensitive data for logging
    display_value = value if len(value) <= 4 else f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"
    logger.info(f"{key}: {display_value}")

__all__ = [
    'GROQ_API_KEY',
    'FIRECRAWL_API_KEY',
    'MODEL',
    'QDRANT_API_KEY',
    'QDRANT_LOCATION',
    'SERPER_API_KEY'
]