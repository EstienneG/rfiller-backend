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
        description="Liste de paires de rendus et de dates associées."
    )
