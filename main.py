from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import os
import tempfile

import phospho
import pymupdf4llm
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from qdrant_client import QdrantClient
from supabase import create_client

from common import utils
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
from common.llm import call_groq, summarize_chunks, summarize_chunks_summaries
from common.schemas import CompanyDto, UserDto


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


@app.post("/initialize-analysis")
async def create_embeddings(rfp_file: UploadFile, company_id: int = 1):
    try:
        print(1)

        (rfp_pdf, rfp_title) = await utils.read_file(rfp_file)
        print(1)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(rfp_pdf.write())
            temp_file_path = temp_file.name

            try:
                # Convert to markdown using the temporary file path
                rfp_md = pymupdf4llm.to_markdown(temp_file_path)
            finally:
                # Clean up the temporary file
                os.unlink(temp_file_path)

        print(1)

        rfp_chunks = utils.chunk(rfp_md)

        print(1)
        rfp_chunks_summaries = summarize_chunks(rfp_chunks)
        print(2)
        rfp_summary = summarize_chunks_summaries(rfp_chunks_summaries)
        print(3)

        rfp = (
            api_global_variables.supabase_client.table("documents")
            .upsert(
                {
                    "title": rfp_title,
                    "summary": rfp_summary,
                    "company_id": company_id,
                }
            )
            .execute()
        )

        utils.create_embeddings(rfp.data[0]["id"], rfp_chunks)

        if not rfp.data:
            raise HTTPException(status_code=400, detail="Failed to store document")

        return rfp
    except RuntimeError as e:
        raise ValueError("Failed to open the PDF. Ensure the file is valid.") from e
    except Exception as e:
        raise ValueError(f"An error occurred while processing the PDF: {e}") from e


@app.post("/requirements")
async def create_requirements(rfp_id: int):
    existing_requirements = (
        api_global_variables.supabase_client.table("requirements")
        .select("*")
        .eq("rfp_id", rfp_id)
        .execute()
    )
    if existing_requirements.data:
        return existing_requirements.data

    return await utils.create_requirements(rfp_id)


@app.post("/evaluation-criteria")
async def create_evaluation_criteria(rfp_id: int):
    existing_criteria = (
        api_global_variables.supabase_client.table("evaluation_criteria")
        .select("*")
        .eq("rfp_id", rfp_id)
        .execute()
    )
    if existing_criteria.data:
        return existing_criteria

    return await utils.create_evaluation_criteria(rfp_id)


@app.post("/generate-answer")
async def generate_answer(requirementId: int, company_id: int = 1):
    requirement = (
        api_global_variables.supabase_client.table("requirements")
        .select("*")
        .eq("id", requirementId)
        .execute()
    )
    if not requirement.data:
        raise HTTPException(status_code=400, detail="Requirement not found")
    requirement = requirement.data[0]

    rfp = (
        api_global_variables.supabase_client.table("documents")
        .select("*")
        .eq("id", requirement["rfp_id"])
        .execute()
    )
    if not rfp.data:
        raise HTTPException(status_code=400, detail="Document not found")
    rfp_summary = rfp.data[0]["summary"]

    company = (
        api_global_variables.supabase_client.table("companies")
        .select("*")
        .eq("id", company_id)
        .execute()
    )
    if not company.data:
        raise HTTPException(status_code=400, detail="Company not found")
    company = company.data[0]

    company_attributes = (
        f"Nom de l'entreprise: {company['name']}, "
        f"Industrie: {company['industry']}, "
        f"Taille: {company['size']}, "
        f"Emplacement: {company['location']}, "
        f"Année de création: {company['founded']}"
    )

    return await utils.generate_answer(
        rfp_summary, requirement["description"], company_attributes
    )


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
