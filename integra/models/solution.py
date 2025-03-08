from typing import List, Dict, Optional
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl


class SolutionType(str, Enum):
    """Type of solution."""
    AGENT = "agent"
    API = "api"
    LIBRARY = "library"
    FRAMEWORK = "framework"
    MODEL = "model"
    MCP_SERVER = "mcp_server"
    TOOL = "tool"
    OTHER = "other"


class SolutionCandidate(BaseModel):
    """Represents a potential solution found by the SearchAgent."""
    
    # Basic information
    name: str = Field(..., description="Name of the solution")
    description: str = Field(..., description="Brief description of the solution")
    solution_type: SolutionType = Field(..., description="Type of solution")
    
    # Repository information
    repository_url: Optional[str] = Field(None, description="URL to the source code repository")
    stars: Optional[int] = Field(None, description="Number of GitHub stars (if applicable)")
    forks: Optional[int] = Field(None, description="Number of forks (if applicable)")
    last_updated: Optional[str] = Field(None, description="Date of last update")
    
    # Package information
    pypi_name: Optional[str] = Field(None, description="PyPI package name (if available)")
    current_version: Optional[str] = Field(None, description="Current version of the solution")
    python_version: Optional[str] = Field(None, description="Required Python version")
    
    # Documentation
    documentation_url: Optional[str] = Field(None, description="URL to documentation")
    homepage_url: Optional[str] = Field(None, description="URL to project homepage")
    
    # Search metadata
    relevance_score: float = Field(default=0.0, description="Relevance score from 0 to 1")
    compatibility_score: float = Field(default=0.0, description="Compatibility score from 0 to 1")
    quality_score: float = Field(default=0.0, description="Quality score from 0 to 1")
    search_source: Optional[str] = Field(None, description="Source where this candidate was found")
    
    # Extended metadata
    tags: List[str] = Field(default_factory=list, description="Tags associated with the solution")
    license: Optional[str] = Field(None, description="License type")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies required by the solution") 