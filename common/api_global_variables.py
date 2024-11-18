from groq import Groq
from common.embedder import Embedder
from qdrant_client import QdrantClient


class ApiGlobalVariables:
    """Global variables for the API."""

    qdrant_client: QdrantClient
    embedder: Embedder
    llm: Groq


api_global_variables = ApiGlobalVariables()
