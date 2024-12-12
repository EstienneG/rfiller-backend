import phospho
from common.ai_agents import (
    CriteriaExtractor,
    RequirementsExtractor,
    RfpAnsweringOrchestrator,
)
from prompts.system_prompts import (
    create_consolidate_evaluation_criteria_system_prompt,
    create_consolidate_requirements_system_prompt,
    create_extract_evaluation_criteria_system_prompt,
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
                model="mixtral-8x7b-32768",
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
            model="llama-3.3-70b-versatile",
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
                model="mixtral-8x7b-32768",
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

    requirements_extractor = RequirementsExtractor()
    consolidated_requirements = await requirements_extractor.extract(
        consolidate_requirements_system_prompt
    )

    print("extracted requirements :" + str(consolidated_requirements.data))

    return consolidated_requirements.data.requirements_and_dates


async def extract_evaluation_criterias(rfp_chunks: list[str]) -> list[str]:
    evaluation_criteria = []
    for i in range(0, len(rfp_chunks), 20):
        chunk_group = rfp_chunks[max(0, i - 1) : min(i + 20, len(rfp_chunks) - 1)]

        extract_evaluation_criteria_system_prompt = (
            create_extract_evaluation_criteria_system_prompt(chunk_group)
        )

        evaluation_criteria_description = (
            api_global_variables.llm.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": extract_evaluation_criteria_system_prompt,
                    }
                ],
                model="llama-guard-3-8b",
            )
            .choices[0]
            .message.content
        )

        phospho.log(
            input=extract_evaluation_criteria_system_prompt,
            output=evaluation_criteria_description,
        )
        evaluation_criteria.append(evaluation_criteria_description)

    consolidate_evaluation_criterias_system_prompt = (
        create_consolidate_evaluation_criteria_system_prompt(evaluation_criteria)
    )

    criteria_extractor = CriteriaExtractor()
    consolidated_evaluation_criteria = await criteria_extractor.extract(
        consolidate_evaluation_criterias_system_prompt
    )

    print(
        "extracted evaluation criterias:" + str(consolidated_evaluation_criteria.data)
    )

    return consolidated_evaluation_criteria.data.evaluation_criteria


async def generate_requirement_answer(
    rfp_summary: str, requirement_description: str, company_attributes: str
):
    rfp_answering_orchestrator = RfpAnsweringOrchestrator()
    tasks_prompts = (
        await rfp_answering_orchestrator.generate_requirement_answer_material(
            rfp_summary, requirement_description, company_attributes
        )
    )
    final_answer = ""
    for task_prompt in tasks_prompts:
        task_answer = (
            api_global_variables.llm.chat.completions.create(
                messages=[{"role": "user", "content": task_prompt}],
                model="llama-3.3-70b-versatile",
            )
            .choices[0]
            .message.content
        )
        final_answer += task_answer + "\n"

    return final_answer


def call_groq(question: str) -> str:
    answer = (
        api_global_variables.llm.chat.completions.create(
            messages=[{"role": "user", "content": question}],
            model="mixtral-8x7b-32768",
        )
        .choices[0]
        .message.content
    )
    phospho.log(input=question, output=answer)

    return answer
