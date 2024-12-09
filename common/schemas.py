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


class RfpAnalysis(BaseModel):
    risk: int = Field(description="Risk level of answering the RFP", ge=0, le=10)
    requirements: list[str] = Field(
        description="List of requirements that have to be respected"
    )
    dates: list[str] = Field(
        description="List of dates associated with each requirement"
    )
