from groq import Groq
from common.ai_agents import RequirementsExtractor
from common.embedder import Embedder
from qdrant_client import QdrantClient


class ApiGlobalVariables:
    """Global variables for the API."""

    qdrant_client: QdrantClient
    embedder: Embedder
    llm: Groq
    requirement_extractor: RequirementsExtractor


api_global_variables = ApiGlobalVariables()
