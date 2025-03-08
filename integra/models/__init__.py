"""
Models for the Integra system.

This module contains data models used throughout the
Integra system to represent projects, solutions,
and integration processes.
"""

from .solution import SolutionCandidate, SolutionType
from .project import ProjectContext, PydanticInfo, PydanticAIInfo, FrameworkDetails 