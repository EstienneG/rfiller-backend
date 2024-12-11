import re
from typing import List, Tuple

from fastapi import UploadFile
import fitz
from qdrant_client.http import models
from common.constants import QDRANT_EMBEDDING_SIZE
from common import llm
from common.api_global_variables import api_global_variables
from qdrant_client.http.models import PointStruct


def summarize(rfp_title: str, rfp: bytes) -> str:
    return llm.summarize_chunks(rfp_title, rfp)


async def read_file(rfp_file: UploadFile) -> Tuple[fitz.Document, str]:
    rfp_bytes = await rfp_file.read()

    if not rfp_bytes.startswith(b"%PDF"):
        raise ValueError("The uploaded file is not a valid PDF.")

    if not rfp_file.filename:
        raise ValueError("File must have a name")

    rfp_title = rfp_file.filename.replace(".pdf", "")

    rfp_pdf = fitz.open(stream=rfp_bytes, filetype="pdf")

    return rfp_pdf, rfp_title


def chunk_and_store_embeddings(md_content: str, rfp_title: str) -> List[str]:
    chunks = []
    current_chunk = []
    lines = md_content.split("\n")
    code_block = False

    for line in lines:
        if line.lstrip().startswith("```"):
            code_block = not code_block

        header_match = re.match(r"^(#+)\s(.*)", line)

        if header_match and not code_block and current_chunk:
            chunk_content = "\n".join(current_chunk).strip()
            chunks.append(chunk_content)
            current_chunk = []

        current_chunk.append(line)

    if current_chunk:
        chunk_content = "\n".join(current_chunk).strip()
        chunks.append(chunk_content)

    api_global_variables.qdrant_client.recreate_collection(
        collection_name=rfp_title,
        vectors_config=models.VectorParams(
            size=QDRANT_EMBEDDING_SIZE, distance=models.Distance.COSINE
        ),
    )

    embeddings = api_global_variables.embedder.embed(chunks)

    api_global_variables.qdrant_client.upsert(
        collection_name=rfp_title,
        points=[
            PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "previous_chunk": chunks[idx - 1] if idx > 0 else None,
                    "chunk": chunks[idx],
                    "next_chunk": chunks[idx + 1] if idx < len(chunks) - 1 else None,
                },
            )
            for idx, embedding in enumerate(embeddings)
        ],
    )

    return chunks
