from pydantic_ai import Agent
from prompts.system_prompts import (
    create_decompose_task_system_prompt,
    create_gather_task_information_system_prompt,
    create_write_task_response_system_prompt,
)
from common.schemas import RfpEvaluationCriteria, RfpRequirements, Subtasks


class RequirementsExtractor:
    def __init__(
        self,
        model_name: str = "groq:llama-3.3-70b-versatile",
        result_type=RfpRequirements,
    ):
        self.agent = Agent(
            model=model_name,
            result_type=result_type,
        )

    async def extract(self, text: str) -> RfpRequirements:
        return await self.agent.run(text)


class CriteriaExtractor:
    def __init__(
        self,
        model_name: str = "groq:llama3-8b-8192",
        result_type=RfpEvaluationCriteria,
    ):
        self.agent = Agent(
            model=model_name,
            result_type=result_type,
        )

    async def extract(self, text: str) -> RfpEvaluationCriteria:
        return await self.agent.run(text)


class TaskDecompositionAgent:
    def __init__(
        self,
        model_name: str = "groq:llama-3.3-70b-versatile",
        result_type=Subtasks,
    ):
        self.agent = Agent(model=model_name, result_type=result_type)

    async def decompose(
        self,
        requirement_description: str,
        enterprise_information: str,
        rfp_summary: str,
    ) -> Subtasks:
        return await self.agent.run(
            create_decompose_task_system_prompt(
                requirement_description, enterprise_information, rfp_summary
            )
        )


class InformationGathererAgent:
    def __init__(
        self,
        model_name: str = "groq:llama-3.3-70b-versatile",
    ):
        self.agent = Agent(
            model=model_name,
            # tools=[web_search, natural_language_search],
        )

    async def gather_information(
        self,
        requirement_subtask: str,
        enterprise_information: str,
        rfp_summary: str,
    ) -> str:
        return await self.agent.run(
            create_gather_task_information_system_prompt(
                requirement_subtask, enterprise_information, rfp_summary
            )
        )


class RfpAnsweringOrchestrator:
    def __init__(
        self,
    ):
        pass

    async def generate_requirement_answer_material(
        self, requirement_description, enterprise_data, rfp_summary
    ) -> list[str]:
        subtask_decomposition_agent = TaskDecompositionAgent()
        try:
            requirement_subtasks = await subtask_decomposition_agent.decompose(
                requirement_description, enterprise_data, rfp_summary
            )
        except Exception as e:
            print(f"Error decomposing tasks: {e}")
            requirement_subtasks = e.error.failed_generation

        task_answer_prompts = []

        for requirement_subtask in requirement_subtasks.data:
            gap_identification_agent = InformationGathererAgent()

            try:
                subtask_information = await gap_identification_agent.gather_information(
                    requirement_subtask,
                    enterprise_data,
                    rfp_summary,
                )
                subtask_information = subtask_information.data
            except Exception as e:
                print(
                    f"Error gathering information for subtask {requirement_subtask}: {e}"
                )
                subtask_information = e.error.failed_generation

            task_response_prompt = create_write_task_response_system_prompt(
                subtask_information
            )

            task_answer_prompts.append(task_response_prompt)
            print(task_answer_prompts)

        return task_answer_prompts
