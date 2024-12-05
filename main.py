import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import fitz
import phospho
import pymupdf4llm
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from supabase import create_client

from common import rfp_utils
from common.api_global_variables import api_global_variables
from common.constants import (
    GROQ_API_KEY,
    PHOSPHO_API_KEY,
    SUPABASE_API_KEY,
    SUPABASE_URL,
)
from common.embedder import Embedder
from common.llm import call_groq
from common.schemas import UserDto


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Load clients when the app starts.

    FastAPI doc about lifespan events: https://fastapi.tiangolo.com/advanced/events/.
    """

    if SUPABASE_URL and SUPABASE_API_KEY:
        api_global_variables.supabase_client = create_client(
            SUPABASE_URL, SUPABASE_API_KEY
        )
    else:
        raise ValueError("Supabase URL or Key missing")

    api_global_variables.llm = Groq(
        api_key=GROQ_API_KEY,
    )

    phospho.init(api_key=PHOSPHO_API_KEY, project_id="377e4f22774446849175109f663ad991")

    # api_global_variables.qdrant_client = QdrantClient(
    #     host=QDRANT_HOST, port=QDRANT_PORT
    # )
    api_global_variables.embedder = Embedder()

    yield

    # api_global_variables.qdrant_client.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "OK"}, 200


@app.post("/embedding")
async def embbed_file(request: Request):
    data = await request.json()
    vector = data.get("vector")
    payload = data.get("payload")
    if vector:
        api_global_variables.qdrant_client.upsert(
            collection_name="your_collection",
            points=[{"id": "unique-id", "vector": vector, "payload": payload}],
        )
        return {"message": "Vector added successfully"}
    raise HTTPException(status_code=400, detail="Vector data missing")


@app.post("/search")
async def search_vector(request: Request):
    data = await request.json()
    query_vector = data.get("vector")
    if query_vector:
        result = api_global_variables.qdrant_client.search(
            collection_name="your_collection", query_vector=query_vector, limit=5
        )
        return result
    raise HTTPException(status_code=400, detail="Query vector missing")


@app.get("/users")
async def get_users():
    response = api_global_variables.supabase_client.table("users").select("*").execute()
    return response.data


@app.post("/users")
async def create_user(userDto: UserDto):
    response = (
        api_global_variables.supabase_client.table("users")
        .insert({"id": userDto.id, "name": userDto.name})
        .execute()
    )

    if not response.data:
        raise HTTPException(status_code=400, detail="Failed to create user")

    return response.data[0]


@app.post("/rfp_analysis")
async def rfp_analysis(document: UploadFile):
    try:
        # Read the uploaded file
        rfp_content = await document.read()

        if not rfp_content.startswith(b"%PDF"):
            raise ValueError("The uploaded file is not a valid PDF.")

        rfp_pdf = fitz.open(stream=rfp_content, filetype="pdf")

        rfp_md = pymupdf4llm.to_markdown(rfp_pdf)

        rfp_title = str(uuid.uuid4())

        rfp_summary = rfp_utils.summarize("AO groupe", rfp_md)

        response = (
            api_global_variables.supabase_client.table("documents")
            .insert(
                {
                    "title": rfp_title,
                    "content": rfp_summary,  # Convert bytes to string
                    "user_id": 1,
                }
            )
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=400, detail="Failed to store document")

        return rfp_summary
    except RuntimeError as e:
        raise ValueError("Failed to open the PDF. Ensure the file is valid.") from e
    except Exception as e:
        raise ValueError(f"An error occurred while processing the PDF: {e}") from e


@app.post("/test-phospho")
async def test_phospho(question: str):
    return call_groq(question)
