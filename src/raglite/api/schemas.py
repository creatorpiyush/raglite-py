from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    query: str = Field(..., min_length=1)
    topK: Optional[int] = Field(default=None, alias="topK", gt=0, le=50)
    scoreThreshold: Optional[float] = Field(
        default=None, alias="scoreThreshold", ge=-1.0, le=1.0
    )
    mode: Optional[Literal["vector", "keyword", "hybrid"]] = None


class AskRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    question: str = Field(..., min_length=1)
    topK: Optional[int] = Field(default=None, alias="topK", gt=0, le=50)
    scoreThreshold: Optional[float] = Field(
        default=None, alias="scoreThreshold", ge=-1.0, le=1.0
    )
    mode: Optional[Literal["vector", "keyword", "hybrid"]] = None
    includeCitations: Optional[bool] = Field(
        default=None, alias="includeCitations"
    )
    stream: Optional[bool] = None
