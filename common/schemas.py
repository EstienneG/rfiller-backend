from typing import Optional

from pydantic import BaseModel, Field


class UserDto(BaseModel):
    id: Optional[int]
    name: str

    class Config:
        from_attributes = True


class CompanyDto(BaseModel):
    name: str
    email: str
    industry: str
    size: int
    location: str
    founded: int

    class Config:
        from_attributes = True


class DocumentDto(BaseModel):
    id: int
    title: str
    content: bytes
    user_id: int

    class Config:
        from_attributes = True


class RfpRequirement(BaseModel):
    requirement: str = Field(description="Description du rendu.")
    due_date: Optional[str] = Field(description="Date limite de rendu.")


class RfpRequirements(BaseModel):
    requirements_and_dates: list[RfpRequirement] = Field(
        description="Liste de paires de rendus et de leurs dates associées."
    )


class RfpEvaluationCriterion(BaseModel):
    criterion: str = Field(description="Critère d'évaluation.")
    weight: str = Field(description="Poids du critère dans l'évaluation.")


class RfpEvaluationCriteria(BaseModel):
    evaluation_criteria: list[RfpEvaluationCriterion] = Field(
        description="Liste de critères d'évaluation et leur poids dans l'évaluation."
    )


class Subtasks(BaseModel):
    subtasks: list[str] = Field(description="List of subtasks.")
