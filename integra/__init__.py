"""
Integra: A Multi-Agent System for Integrating Open-Source Solutions into PydanticAI Projects.

Integra helps users integrate existing open-source solutions (agents, MCP servers, or APIs)
into their PydanticAI projects by discovering, evaluating, and providing integration instructions
for appropriate solutions.

Core Components:
- Coordinator Agent: Orchestrates workflow and interfaces with Cursor
- Search Agent: Discovers relevant open source solutions
- Evaluation Agent: Assesses solution viability for projects
- Documentation Agent: Retrieves and processes documentation
- Integration Agent: Creates step-by-step integration instructions
"""

__version__ = "0.1.0"

from .models import (
    SolutionCandidate, 
    SolutionType,
    ProjectContext,
    PydanticInfo,
    PydanticAIInfo
)

from .agents import (
    SearchAgent,
    DocumentationAgent
)

from .search_tools import (
    SearchToolsAgent,
    SearchToolInput,
    GithubSearchInput,
    SearchResult,
    DuckDuckGoResult,
    GitHubResult,
    PyPIResult,
)

# Example usage of the SearchToolsAgent with tool decorators:
"""
from integra import SearchToolsAgent, SearchToolInput, GithubSearchInput
import asyncio

async def example_search():
    # Initialize the search tools agent
    search_agent = SearchToolsAgent()
    
    # Use the search_duckduckgo tool
    ddg_results = await search_agent.search_duckduckgo(
        SearchToolInput(query="pydantic-ai agents", max_results=5)
    )
    print(f"Found {len(ddg_results.results)} results from DuckDuckGo")
    
    # Use the search_github tool (requires GitHub token)
    github_results = await search_agent.search_github(
        GithubSearchInput(
            query="pydantic-ai",
            github_token="your_github_token",
            max_results=5
        )
    )
    print(f"Found {len(github_results.repositories)} repositories from GitHub")
    
    # Use the search_pypi tool
    pypi_results = await search_agent.search_pypi(
        SearchToolInput(query="pydantic-ai", max_results=5)
    )
    print(f"Found {len(pypi_results.packages)} packages from PyPI")

# Run the example
# asyncio.run(example_search())
""" 