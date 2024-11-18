from common.api_global_variables import api_global_variables


def summarize_rfp(rfp_title: str, rfp: bytes) -> str:
    rfp_content: str = rfp.decode("utf-8")

    rfp_summary = (
        api_global_variables.llm.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "You are a summariser AI. Please summarize this RFP."
                    + "Title: "
                    + rfp_title
                    + "Content: "
                    + rfp_content,
                }
            ],
            model="llama3-8b-8192",
        )
        .choices[0]
        .message.content
    )

    print(rfp_summary)
    return rfp_summary
