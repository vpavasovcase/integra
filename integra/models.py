from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl

class SolutionType(str, Enum):
    AGENT = "agent"
    MCP_SERVER = "mcp_server"
    API = "api"
    LIBRARY = "library"

class CompatibilityStatus(str, Enum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    NEEDS_ADAPTATION = "needs_adaptation"
    UNKNOWN = "unknown"

class SolutionCandidate(BaseModel):
    """Represents a potential solution found by the Search Agent."""
    name: str
    description: str
    repository_url: HttpUrl
    solution_type: SolutionType
    stars: Optional[int] = Field(default=None, description="GitHub stars if available")
    last_updated: Optional[datetime] = None
    license: Optional[str] = None
    compatibility_score: float = Field(ge=0, le=1)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BestSolution(BaseModel):
    """Represents the best solution selected by the Evaluation Agent."""
    candidate: SolutionCandidate
    compatibility_status: CompatibilityStatus
    adaptation_required: bool = False
    adaptation_notes: Optional[str] = None
    estimated_integration_time: str = Field(
        description="Estimated time for integration in human-readable format"
    )
    requirements: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)

class NoViableSolution(BaseModel):
    """Represents the case when no viable solution is found."""
    reason: str
    suggestions: List[str] = Field(default_factory=list)
    alternative_approaches: List[str] = Field(default_factory=list)

class DocumentationContent(BaseModel):
    """Represents processed documentation content."""
    content: str
    source_url: HttpUrl
    format: str = Field(description="Format of the documentation (markdown, rst, etc.)")
    sections: Dict[str, str] = Field(default_factory=dict)
    code_examples: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IntegrationStep(BaseModel):
    """Represents a single step in the integration process."""
    order: int
    description: str
    code: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    validation_steps: List[str] = Field(default_factory=list)

class IntegrationInstructions(BaseModel):
    """Represents complete integration instructions."""
    solution: BestSolution
    prerequisites: List[str] = Field(default_factory=list)
    steps: List[IntegrationStep]
    configuration: Dict[str, Any] = Field(default_factory=dict)
    troubleshooting: List[str] = Field(default_factory=list)
    testing_instructions: List[str] = Field(default_factory=list)

class ProjectContext(BaseModel):
    """Represents the context of the user's project."""
    project_name: str
    python_version: str
    dependencies: Dict[str, str] = Field(default_factory=dict)
    project_structure: Dict[str, Any] = Field(default_factory=dict)
    framework_versions: Dict[str, str] = Field(default_factory=dict)
    environment_variables: List[str] = Field(default_factory=list)

class IntegraSession(BaseModel):
    """Represents an active Integra session."""
    session_id: str = Field(default_factory=lambda: datetime.now().isoformat())
    project_context: ProjectContext
    search_query: str
    candidates: List[SolutionCandidate] = Field(default_factory=list)
    selected_solution: Union[BestSolution, NoViableSolution, None] = None
    documentation: Optional[DocumentationContent] = None
    integration_instructions: Optional[IntegrationInstructions] = None
    status: str = Field(default="initialized")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now) 