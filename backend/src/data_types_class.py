from typing import Literal, Optional, Union, TextIO, List,Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum

class QueryRequest(BaseModel):
    userquery: str
    query_context: Optional[str] = None  

class Credentials(BaseModel):
    deployment: Literal["Weaviate", "Docker", "Local"]
    url: str
    key: str

class ConfigSetting(BaseModel):
    type: str
    value: str | int
    description: str
    values: list[str]

class RAGComponentConfig(BaseModel):
    name: str
    variables: list[str]
    library: list[str]
    description: str
    config: dict[str, ConfigSetting]
    type: str
    available: bool

class RAGComponentClass(BaseModel):
    selected: str
    components: dict[str, RAGComponentConfig]

class RAGConfig(BaseModel):
    Reader: RAGComponentClass
    Chunker: RAGComponentClass
    Embedder: RAGComponentClass
    Retriever: RAGComponentClass
    Generator: RAGComponentClass

class DocumentFilter(BaseModel):
    title: str
    uuid: str


class QueryPayload(BaseModel):
    query: str
    RAG: dict[str, RAGComponentClass]
    labels: list[str]
    documentFilter: list[DocumentFilter]
    credentials: Credentials
