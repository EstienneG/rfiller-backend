from common import llm


def summarize(rfp_title: str, rfp_content: str) -> str:
    return llm.summarize(rfp_title, rfp_content)


# def chunk(rfp_content: str: bytes) -> None:
