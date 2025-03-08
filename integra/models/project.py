from typing import List, Dict, Optional, Set
from pydantic import BaseModel, Field


class FrameworkDetails(BaseModel):
    """Information about a framework used in the project."""
    name: str = Field(..., description="Name of the framework")
    version: Optional[str] = Field(None, description="Version of the framework")
    usage_level: str = Field(default="unknown", description="How extensively the framework is used (core, peripheral, etc.)")


class PydanticInfo(BaseModel):
    """Information about Pydantic usage in the project."""
    version: str = Field(..., description="Pydantic version")
    models: List[str] = Field(default_factory=list, description="Names of key Pydantic models")
    features_used: List[str] = Field(default_factory=list, description="Pydantic features being used")


class PydanticAIInfo(BaseModel):
    """Information about PydanticAI usage in the project."""
    version: str = Field(..., description="PydanticAI version")
    agents: List[str] = Field(default_factory=list, description="Agent names")
    models_used: List[str] = Field(default_factory=list, description="Models being used (Gemini, GPT, etc.)")
    features_used: List[str] = Field(default_factory=list, description="PydanticAI features being used")


class ProjectContext(BaseModel):
    """Context of the user's project for evaluating solutions."""
    
    # Basic project information
    name: str = Field(..., description="Name of the project")
    description: str = Field(..., description="Brief description of the project")
    
    # Technical information
    python_version: str = Field(..., description="Python version used in the project")
    dependencies: Dict[str, str] = Field(default_factory=dict, description="Project dependencies and their versions")
    dev_dependencies: Dict[str, str] = Field(default_factory=dict, description="Development dependencies")
    
    # Framework information
    frameworks: List[FrameworkDetails] = Field(default_factory=list, description="Frameworks used in the project")
    
    # Pydantic/PydanticAI specific information
    pydantic: Optional[PydanticInfo] = Field(None, description="Information about Pydantic usage")
    pydantic_ai: Optional[PydanticAIInfo] = Field(None, description="Information about PydanticAI usage")
    
    # Project structure
    file_count: Optional[int] = Field(None, description="Number of files in the project")
    directory_structure: Optional[List[str]] = Field(None, description="Key directories in the project")
    
    # Extra context
    notes: Optional[str] = Field(None, description="Additional notes or context about the project") 