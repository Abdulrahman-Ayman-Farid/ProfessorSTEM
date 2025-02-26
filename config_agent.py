import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Flask Configuration
SECRET_KEY = '16a4a22af5c3ffa329130ffb50b64d75bf2733e370844174dea6ab456a0da06a'

# API Keys Configuration
CENTRAL_API_KEY = os.getenv("CENTRAL_API_KEY")
MATH_API_KEY = os.getenv("MATH_API_KEY")
SCIENCE_API_KEY = os.getenv("SCIENCE_API_KEY")

if not all([CENTRAL_API_KEY, MATH_API_KEY, SCIENCE_API_KEY]):
    raise ValueError("API keys not found in environment variables.")
