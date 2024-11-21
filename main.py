from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import fitz
import pymupdf4llm

from common.llm import call_groq
from common.constants import GROQ_API_KEY
from common import rfp_utils
from common.api_global_variables import api_global_variables
from common.dependencies import get_db
from common.embedder import Embedder
from common.schemas import UserDto
from database.db import Base, engine
from database.models import Document, User
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session
from groq import Groq


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Load clients when the app starts.

    FastAPI doc about lifespan events: https://fastapi.tiangolo.com/advanced/events/.
    """

    Base.metadata.create_all(bind=engine)

    api_global_variables.llm = Groq(
        api_key=GROQ_API_KEY,
    )

    # api_global_variables.qdrant_client = QdrantClient(
    #     host=QDRANT_HOST, port=QDRANT_PORT
    # )
    api_global_variables.embedder = Embedder()

    yield

    # api_global_variables.qdrant_client.close()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def read_root():
    return {"message": "Welcome to the FastAPI + PostgreSQL + Qdrant app"}


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


@app.get("/user")
async def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users


@app.post("/user")
async def create_user(userDto: UserDto, db: Session = Depends(get_db)):
    db_user = User(**userDto.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.post("/rfp_analysis")
async def rfp_analysis(document: UploadFile, db: Session = Depends(get_db)):
    try:
        # Read the uploaded file
        rfp_content = await document.read()

        if not rfp_content.startswith(b"%PDF"):
            raise ValueError("The uploaded file is not a valid PDF.")

        rfp_pdf = fitz.open(stream=rfp_content, filetype="pdf")

        metadata = rfp_pdf.metadata
        rfp_title = metadata.get("title", None)

        rfp_md = pymupdf4llm.to_markdown(rfp_pdf)

        rfp_summary = rfp_utils.summarize(rfp_title, rfp_md)

        # rfp_utils.chunk(rfp_content)

        db_document = Document(title=rfp_title, content=rfp_content, user_id=2)

        db.add(db_document)
        db.commit()
        db.refresh(db_document)

        return rfp_summary
    except fitz.FileDataError as e:
        raise ValueError("Failed to open the PDF. Ensure the file is valid.") from e
    except Exception as e:
        raise ValueError(f"An error occurred while processing the PDF: {e}") from e


@app.post("/test-phospho")
async def test_phospho(question: str):
    return call_groq(question)
