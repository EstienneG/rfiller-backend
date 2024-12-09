import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import fitz
import phospho
import pymupdf4llm
from fastapi import FastAPI, HTTPException, UploadFile
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

    phospho.init(api_key=PHOSPHO_API_KEY, project_id="377e4f22774446849175109f663ad991")

    # api_global_variables.qdrant_client = QdrantClient(
    #     host=QDRANT_HOST, port=QDRANT_PORT
    # )
    api_global_variables.embedder = Embedder()

    api_global_variables.test_agent = Agent(
        "groq:llama3-8b-8192",
        result_type=RfpAnalysis,
        system_prompt=(
            "Tu es un agent qui analyse un appel d'offre et donne les requirements et dates ainsi qu'un risque à répondre entre 1 et 10"
        ),
    )

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
        rfp_chunks = rfp_utils.chunk(rfp_md)

        rfp_title = str(uuid.uuid4())

        rfp_summary = await api_global_variables.test_agent.run(rfp_md)

        print(rfp_summary.data)
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
):  # input must be a ConversationDto object
    """Handles new users requests"""
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
