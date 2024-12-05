import os

from dotenv import load_dotenv

load_dotenv()

# Qdrant config
# QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
# QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_EMBEDDING_SIZE = 512
QDRANT_THEMES_REVIEWS_ASSOCIATION_SCORE_TRESHOLD = 0.5
MAX_AMOUNT_OF_REVIEWS_TO_ASSOCIATE_TO_THEME = 200

# Embedder config
EMBEDDER_MODEL_NAME = "text-embedding-3-small"

# OpenAI config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Postgre config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

# Groq config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Phospho config
PHOSPHO_API_KEY = os.getenv("PHOSPHO_API_KEY")
