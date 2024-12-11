from pydantic_ai import Agent
from common.schemas import RfpRequirements


class RequirementsExtractor:
    def __init__(
        self,
        model_name: str = "groq:llama3-groq-8b-8192-tool-use-preview",
        result_type=RfpRequirements,
    ):
        self.agent = Agent(
            model=model_name,
            result_type=result_type,
        )

    async def extract(self, text: str) -> RfpRequirements:
        return await self.agent.run(text)
