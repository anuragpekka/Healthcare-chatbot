import logging

MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"
CHUNK_SIZE = 300 #1-500
CHUNK_OVERLAP = 50

DATA_DIR = r".\Data"
GLOB_PATTERNS = ["*pdf"]

INDEX_NAME = "medicalbot-cs300-ol50"
INDEX_EMBEDDING_SIZE = 768
INDEX_METRIC = "cosine"
INDEX_CLOUD = "aws"
INDEX_REGION = "us-east-1"
VECTOR_STORE_SEARCH_TYPE = "similarity"
NUM_DOCS = 10

LOG_LEVEL = logging.DEBUG
LOG_FILE = "logs/flask_app.log"
LOG_MAXBYTES = 1024*1024 # 1MB max
LOG_BACKUP_COUNT = 5 # keep 5 backups

LLM_TEMPERATURE = 0.4
LLM_MAX_TOKENS = 500

from  src.prompt import system_prompt_detailed
SYSTEM_PROMPT = system_prompt_detailed

HOST_IP = "0.0.0.0"
HOST_PORT = 8080