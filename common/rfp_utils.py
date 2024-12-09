import re
from typing import List
from common import llm


def summarize(rfp_title: str, rfp: bytes) -> str:
    return llm.summarize(rfp_title, rfp)


def chunk(md_content: str) -> List[str]:
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

    return chunks
