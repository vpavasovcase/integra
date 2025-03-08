from typing import List, Optional, Dict, Any
from datetime import datetime
import httpx
from pydantic import BaseModel, Field, HttpUrl, validator
from duckduckgo_search import DDGS
from github import Github, Repository
import pypi_simple
from pydantic_ai import Agent, Tool
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from .models import SolutionType, SolutionCandidate
from enum import Enum
import asyncio
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys and configuration from environment
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
PYDANTIC_AI_MODEL = os.getenv("PYDANTIC_AI_MODEL", "google-gla:gemini-1.5-flash")

class SearchError(Exception):
    """Base exception for search-related errors."""
    pass

class SearchResult(BaseModel):
    """Base model for search results."""
    source: str
    query: str
    timestamp: datetime = Field(default_factory=datetime.now)
    raw_results: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None

class DuckDuckGoResult(SearchResult):
    """Structured results from DuckDuckGo search."""
    results: List[Dict[str, str]] = Field(default_factory=list)

class GitHubResult(SearchResult):
    """Structured results from GitHub repository search."""
    repositories: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int = 0
    rate_limit_remaining: Optional[int] = None

class PyPIResult(SearchResult):
    """Structured results from PyPI package search."""
    packages: List[Dict[str, Any]] = Field(default_factory=list)

class SearchToolInput(BaseModel):
    """Input for search tool functions."""
    query: str = Field(..., description="Search query string")
    max_results: int = Field(default=10, description="Maximum number of results to return")

class GithubSearchInput(SearchToolInput):
    """Input for GitHub search."""
    github_token: str = Field(..., description="GitHub API token")

class SearchToolsAgent(Agent):
    """Agent providing search tools via PydanticAI's common tools."""
    
    system_prompt = """You are a search tools agent for Integra, providing various search capabilities 
    to discover open-source solutions for Pydantic AI projects. You help find relevant packages, repositories, 
    and information across multiple sources."""
    
    def __init__(self):
        # Initialize with model from environment variable
        super().__init__(
            PYDANTIC_AI_MODEL,
            tools=[duckduckgo_search_tool()],
        )
        
        # Add GitHub search tool
        self.github_search_tool = Tool(
            name="github_search",
            description="Search for GitHub repositories",
            parameters={
                "query": {
                    "type": "string", 
                    "description": "Search query for GitHub"
                },
                "github_token": {
                    "type": "string", 
                    "description": "GitHub API token"
                },
                "max_results": {
                    "type": "integer", 
                    "description": "Maximum number of results"
                }
            },
            function=self._github_search_function
        )
        
        # Add PyPI search tool
        self.pypi_search_tool = Tool(
            name="pypi_search",
            description="Search for Python packages on PyPI",
            parameters={
                "query": {
                    "type": "string", 
                    "description": "Search query for PyPI"
                },
                "max_results": {
                    "type": "integer", 
                    "description": "Maximum number of results"
                }
            },
            function=self._pypi_search_function
        )
        
        # Add converter tool
        self.convert_tool = Tool(
            name="convert_to_solution_candidate",
            description="Convert search result to a solution candidate",
            parameters={
                "result": {
                    "type": "object", 
                    "description": "Search result to convert"
                },
                "source": {
                    "type": "string", 
                    "description": "Source of the result"
                },
                "solution_type": {
                    "type": "string", 
                    "description": "Type of solution"
                }
            },
            function=self._convert_function
        )
    
    async def search_duckduckgo(self, input: SearchToolInput) -> DuckDuckGoResult:
        """
        Search DuckDuckGo for relevant documentation, tutorials, and discussions.
        
        Args:
            input: SearchToolInput containing query and max_results
            
        Returns:
            DuckDuckGoResult containing search results
        """
        try:
            result = await self.run(f"Search for: {input.query} and return up to {input.max_results} results")
            
            # Process the results from the DuckDuckGo search tool
            results = []
            for item in result.data.get('results', [])[:input.max_results]:
                results.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })
                
            return DuckDuckGoResult(
                source="duckduckgo",
                query=input.query,
                results=results,
                raw_results=result.data
            )
        except Exception as e:
            return DuckDuckGoResult(
                source="duckduckgo",
                query=input.query,
                error=str(e)
            )
    
    def _github_search_function(self, query: str, github_token: str, max_results: int = 10) -> Dict[str, Any]:
        """Search GitHub repositories."""
        try:
            g = Github(github_token)
            repositories = g.search_repositories(query, sort="stars", order="desc")
            results = []
            
            # Get the most relevant repositories
            for repo in repositories[:max_results]:
                results.append({
                    'name': repo.name,
                    'full_name': repo.full_name,
                    'description': repo.description,
                    'url': repo.html_url,
                    'stars': repo.stargazers_count,
                    'last_updated': repo.updated_at.isoformat(),
                    'license': repo.license.name if repo.license else None,
                    'topics': repo.get_topics()
                })
            
            return {
                'repositories': results,
                'total_count': repositories.totalCount,
                'rate_limit_remaining': g.get_rate_limit().search.remaining
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def search_github(self, input: GithubSearchInput) -> GitHubResult:
        """
        Search GitHub for relevant repositories and projects.
        
        Args:
            input: GithubSearchInput containing query, github_token, and max_results
            
        Returns:
            GitHubResult containing repository information
        """
        try:
            result = await self.run(
                f"Use github_search to search for: {input.query}",
                tools=[self.github_search_tool],
                tool_choice={"type": "function", "function": {"name": "github_search"}},
                tool_args={
                    "query": input.query,
                    "github_token": input.github_token,
                    "max_results": input.max_results
                }
            )
            
            if 'error' in result.data:
                return GitHubResult(
                    source="github",
                    query=input.query,
                    error=result.data['error']
                )
            
            return GitHubResult(
                source="github",
                query=input.query,
                repositories=result.data.get('repositories', []),
                total_count=result.data.get('total_count', 0),
                rate_limit_remaining=result.data.get('rate_limit_remaining'),
                raw_results=result.data
            )
        except Exception as e:
            return GitHubResult(
                source="github", 
                query=input.query,
                error=str(e)
            )
    
    def _pypi_search_function(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Search PyPI packages."""
        try:
            client = pypi_simple.PyPISimple()
            packages = []
            
            # Search for packages
            for pkg in client.search(query):
                if len(packages) >= max_results:
                    break
                
                # Get package information
                package_info = {
                    'name': pkg.name,
                    'version': pkg.version if hasattr(pkg, 'version') else None,
                    'url': f"https://pypi.org/project/{pkg.name}/",
                    'description': None  # PyPI Simple doesn't provide descriptions
                }
                packages.append(package_info)
                
            return {'packages': packages}
        except Exception as e:
            return {'error': str(e)}
    
    async def search_pypi(self, input: SearchToolInput) -> PyPIResult:
        """
        Search PyPI packages using PyPI Simple API.
        
        Args:
            input: SearchToolInput containing query and max_results
            
        Returns:
            PyPIResult containing package information
        """
        try:
            result = await self.run(
                f"Use pypi_search to search for: {input.query}",
                tools=[self.pypi_search_tool],
                tool_choice={"type": "function", "function": {"name": "pypi_search"}},
                tool_args={
                    "query": input.query,
                    "max_results": input.max_results
                }
            )
            
            if 'error' in result.data:
                return PyPIResult(
                    source="pypi",
                    query=input.query,
                    error=result.data['error']
                )
            
            return PyPIResult(
                source="pypi",
                query=input.query,
                packages=result.data.get('packages', []),
                raw_results=result.data
            )
        except Exception as e:
            return PyPIResult(
                source="pypi",
                query=input.query,
                error=str(e)
            )
    
    def _convert_function(self, result: Dict[str, Any], source: str, solution_type: str) -> Dict[str, Any]:
        """Convert search result to a solution candidate."""
        try:
            # Convert solution_type string to enum
            solution_type_enum = SolutionType(solution_type)
            
            if source == "github":
                return {
                    'name': result.get('name', ''),
                    'description': result.get('description', ''),
                    'solution_type': solution_type_enum,
                    'repository_url': result.get('url', ''),
                    'stars': result.get('stars', 0),
                    'last_updated': result.get('last_updated', ''),
                    'pypi_name': None,
                    'current_version': None,
                    'license': result.get('license'),
                    'documentation_url': None,
                    'tags': result.get('topics', [])
                }
            elif source == "pypi":
                return {
                    'name': result.get('name', ''),
                    'description': result.get('description', ''),
                    'solution_type': solution_type_enum,
                    'repository_url': None,
                    'stars': 0,
                    'last_updated': None,
                    'pypi_name': result.get('name'),
                    'current_version': result.get('version'),
                    'license': None,
                    'documentation_url': None,
                    'tags': []
                }
            elif source == "duckduckgo":
                return {
                    'name': result.get('title', '').split(' - ')[0],
                    'description': result.get('snippet', ''),
                    'solution_type': solution_type_enum,
                    'repository_url': None,
                    'stars': 0,
                    'last_updated': None,
                    'pypi_name': None,
                    'current_version': None,
                    'license': None,
                    'documentation_url': result.get('link'),
                    'tags': []
                }
            else:
                return {'error': f"Unknown source: {source}"}
        except Exception as e:
            return {'error': str(e)}
    
    async def convert_to_solution_candidate(self, result: Dict[str, Any], source: str, solution_type: SolutionType) -> SolutionCandidate:
        """
        Convert a search result into a standardized SolutionCandidate.
        
        Args:
            result: Individual result from a search operation
            source: Source of the result (github, pypi, duckduckgo)
            solution_type: Type of solution the result represents
            
        Returns:
            SolutionCandidate with standardized fields
        """
        try:
            solution_type_str = solution_type.value
            
            convert_result = await self.run(
                f"Convert this search result to a solution candidate",
                tools=[self.convert_tool],
                tool_choice={"type": "function", "function": {"name": "convert_to_solution_candidate"}},
                tool_args={
                    "result": result,
                    "source": source,
                    "solution_type": solution_type_str
                }
            )
            
            if 'error' in convert_result.data:
                raise ValueError(convert_result.data['error'])
            
            return SolutionCandidate(**convert_result.data)
        except Exception as e:
            raise ValueError(f"Failed to convert result: {str(e)}")

# For backward compatibility, keeping the standalone functions that delegate to the agent
async def search_duckduckgo(query: str, max_results: int = 10) -> DuckDuckGoResult:
    """Backward compatibility function that delegates to the SearchToolsAgent."""
    agent = SearchToolsAgent()
    return await agent.search_duckduckgo(SearchToolInput(query=query, max_results=max_results))

async def search_github(
    query: str,
    github_token: str,
    max_results: int = 10
) -> GitHubResult:
    """Backward compatibility function that delegates to the SearchToolsAgent."""
    agent = SearchToolsAgent()
    return await agent.search_github(GithubSearchInput(
        query=query,
        github_token=github_token,
        max_results=max_results
    ))

async def search_pypi(query: str, max_results: int = 10) -> PyPIResult:
    """Backward compatibility function that delegates to the SearchToolsAgent."""
    agent = SearchToolsAgent()
    return await agent.search_pypi(SearchToolInput(query=query, max_results=max_results))

def convert_to_solution_candidate(
    result: Dict[str, Any],
    source: str,
    solution_type: SolutionType
) -> SolutionCandidate:
    """Backward compatibility function that delegates to the SearchToolsAgent."""
    agent = SearchToolsAgent()
    return agent.convert_to_solution_candidate(result, source, solution_type) 