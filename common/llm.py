import phospho
from prompts.system_prompts import (
    create_consolidate_requirements_system_prompt,
    create_extract_requirements_system_prompt,
    create_summarize_chunks_summaries_system_prompt,
    create_summarize_chunks_system_prompt,
)
from common.api_global_variables import api_global_variables


def summarize_chunks(rfp_chunks: list[str]) -> list[str]:
    rfp_summaries = []
    for i in range(0, len(rfp_chunks), 8):
        chunk_group = rfp_chunks[max(0, i - 1) : i + 8]
        summarize_chunks_system_prompt = create_summarize_chunks_system_prompt(
            chunk_group
        )

        rfp_summary = (
            api_global_variables.llm.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": summarize_chunks_system_prompt,
                    }
                ],
                model="llama3-groq-8b-8192-tool-use-preview",
            )
            .choices[0]
            .message.content
        )

        phospho.log(input=summarize_chunks_system_prompt, output=rfp_summary)
        rfp_summaries.append(rfp_summary)

    return rfp_summaries


def summarize_chunks_summaries(rfp_chunks_summaries: list[str]) -> str:
    input_str = create_summarize_chunks_summaries_system_prompt(rfp_chunks_summaries)

    rfp_summary = (
        api_global_variables.llm.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": input_str,
                }
            ],
            model="llama3-groq-8b-8192-tool-use-preview",
        )
        .choices[0]
        .message.content
    )

    phospho.log(input=input_str, output=rfp_summary)

    print("extracted rfp_summary :" + rfp_summary)
    return rfp_summary


async def extract_requirements(rfp_chunks: list[str]) -> list[str]:
    requirements_descriptions = []
    for i in range(0, len(rfp_chunks), 13):
        chunk_group = rfp_chunks[max(0, i - 1) : i + 13]
        extract_requirements_system_prompt = create_extract_requirements_system_prompt(
            chunk_group
        )

        requirements_description = (
            api_global_variables.llm.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": extract_requirements_system_prompt,
                    }
                ],
                model="llama3-groq-8b-8192-tool-use-preview",
            )
            .choices[0]
            .message.content
        )

        phospho.log(
            input=extract_requirements_system_prompt, output=requirements_description
        )
        requirements_descriptions.append(requirements_description)

    consolidate_requirements_system_prompt = (
        create_consolidate_requirements_system_prompt(requirements_descriptions)
    )

    consolidated_requirements = (
        await api_global_variables.requirement_extractor.extract(
            consolidate_requirements_system_prompt
        )
    )

    print("extracted requirements :" + str(consolidated_requirements.data))

    return consolidated_requirements.data.requirements_and_dates


def call_groq(question: str) -> str:
    answer = (
        api_global_variables.llm.chat.completions.create(
            messages=[{"role": "user", "content": question}],
            model="llama3-groq-8b-8192-tool-use-preview",
        )
        .choices[0]
        .message.content
    )
    phospho.log(input=question, output=answer)

    return answer
