from common import llm


def summarize(rfp_title: str, rfp: bytes) -> str:
    return llm.summarize(rfp_title, rfp)
