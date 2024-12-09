from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import phospho
import pymupdf4llm
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from qdrant_client import QdrantClient
from supabase import create_client

from common import rfp_utils
from common.api_global_variables import api_global_variables
from common.constants import (
    GROQ_API_KEY,
    PHOSPHO_API_KEY,
    SUPABASE_API_KEY,
    SUPABASE_URL,
    QDRANT_HOST,
    QDRANT_PORT,
)
from common.embedder import Embedder
from common.llm import call_groq
from common.schemas import CompanyDto, RfpAnalysis, UserDto
from pydantic_ai import Agent


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

    api_global_variables.qdrant_client = QdrantClient(
        host=QDRANT_HOST, port=QDRANT_PORT
    )

    api_global_variables.embedder = Embedder()

    api_global_variables.rfp_analysis_agent = Agent(
        "groq:llama3-8b-8192",
        result_type=RfpAnalysis,
        system_prompt=(
            "Tu es un agent qui analyse un appel d'offre et donne les requirements et dates ainsi qu'un risque à répondre entre 1 et 10"
        ),
    )

    phospho.init(api_key=PHOSPHO_API_KEY, project_id="377e4f22774446849175109f663ad991")

    yield

    api_global_variables.qdrant_client.close()


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
async def rfp_analysis(rfp_file: UploadFile):
    try:
        (rfp_pdf, rfp_title) = await rfp_utils.read_file(rfp_file)

        rfp_md = pymupdf4llm.to_markdown(rfp_pdf)
        rfp_chunks = rfp_utils.chunk_and_store_embeddings(rfp_md, rfp_title)

        rfp_analysis = await api_global_variables.rfp_analysis_agent.run(rfp_md)

        print(rfp_analysis.data)

        rfp_file = (
            api_global_variables.supabase_client.table("documents")
            .insert(
                {
                    "title": rfp_title,
                    "content": str(rfp_analysis.data),
                    "user_id": 1,
                }
            )
            .execute()
        )

        if not rfp_file.data:
            raise HTTPException(status_code=400, detail="Failed to store document")

        return rfp_chunks
    except RuntimeError as e:
        raise ValueError("Failed to open the PDF. Ensure the file is valid.") from e
    except Exception as e:
        raise ValueError(f"An error occurred while processing the PDF: {e}") from e


@app.post("/test-phospho")
async def test_phospho(question: str):
    return call_groq(question)


@app.get("/companies")
async def get_companies():
    response = (
        api_global_variables.supabase_client.table("companies").select("*").execute()
    )
    return response.data


@app.post("/companies")
async def create_company(
    company_dto: CompanyDto,
):
    """Handles new company requests"""
    response = (
        api_global_variables.supabase_client.table("companies")
        .insert(
            {
                "name": company_dto.name,
                "email": company_dto.email,
                "industry": company_dto.industry,
                "size": company_dto.size,
                "location": company_dto.location,
                "founded": company_dto.founded,
            }
        )
        .execute()
    )

    if not response.data:
        raise HTTPException(status_code=400, detail="Failed to create company")

    return response.data[0]
