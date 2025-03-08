from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import asyncio
from datetime import datetime
import re
import os
from dotenv import load_dotenv
from ..models import SolutionCandidate, SolutionType, ProjectContext
from ..search_tools import (
    SearchToolsAgent,
    SearchToolInput,
    GithubSearchInput
)

# Load environment variables
load_dotenv()

# Get API keys from environment
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
PYDANTIC_AI_MODEL = os.getenv("PYDANTIC_AI_MODEL", "google-gla:gemini-1.5-flash")

class SearchDependencies(BaseModel):
    """Dependencies required by the Search Agent."""
    github_token: str = Field(default=GITHUB_TOKEN, description="GitHub API token for repository searches")
    project_context: ProjectContext = Field(..., description="Context of the user's project")
    max_results_per_source: int = Field(default=10, description="Maximum results to fetch per source")
    min_stars: int = Field(default=5, description="Minimum GitHub stars for considering a solution")
    
class SearchQuery(BaseModel):
    """Structured search query generated from user requirements."""
    main_query: str = Field(..., description="Primary search query")
    keywords: List[str] = Field(default_factory=list, description="Relevant keywords")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Search filters")
    solution_type_hint: Optional[SolutionType] = Field(
        default=None, 
        description="Hint about the type of solution needed"
    )

class SearchInput(BaseModel):
    """Input for search requests."""
    requirement: str = Field(..., description="Natural language description of what the user needs")
    project_context: ProjectContext = Field(..., description="User's project context")

class SearchOutput(BaseModel):
    """Output from search process."""
    candidates: List[SolutionCandidate] = Field(..., description="List of found solution candidates")
    query: SearchQuery = Field(..., description="Generated search query")
    total_results: int = Field(..., description="Total number of results found")

class SearchAgent(Agent[SearchInput, SearchOutput]):
    """Agent responsible for discovering relevant open-source solutions."""
    
    system_message = """You are the Search Agent for Integra, an expert at discovering relevant open-source solutions for Pydantic AI projects. Your role is to:

1. Analyze user requirements and project context to formulate effective search strategies
2. Search across multiple sources (GitHub, PyPI, documentation) to find potential solutions
3. Filter and rank results based on relevance, quality, and compatibility
4. Present a curated list of the most promising solutions

Each solution should be relevant to the user's specific needs, be of high quality, and likely to integrate well with their project.

When searching, prioritize:
- Solutions specifically designed for Pydantic AI projects
- Well-maintained repositories (recent updates, active contributors)
- Solutions with good documentation and examples
- Production-ready code over prototypes

Ensure all candidates are properly evaluated for technical compatibility with the user's project context."""

    def __init__(self, dependencies: SearchDependencies):
        """Initialize the Search Agent."""
        super().__init__(
            model=PYDANTIC_AI_MODEL,  # Use the model from environment
            deps_type=SearchInput,
            result_type=SearchOutput,
        )
        
        self.github_token = dependencies.github_token
        self.project_context = dependencies.project_context
        self.max_results_per_source = dependencies.max_results_per_source
        self.min_stars = dependencies.min_stars
        self.search_tools = SearchToolsAgent()

    def _extract_keywords(self, requirement: str) -> List[str]:
        """Extract relevant keywords from the requirement text."""
        # Remove common words and punctuation
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = re.findall(r'\w+', requirement.lower())
        keywords = [word for word in words if word not in common_words]
        
        # Add relevant technical terms
        if 'api' in keywords:
            keywords.extend(['rest', 'http', 'client'])
        if 'agent' in keywords:
            keywords.extend(['ai', 'autonomous', 'assistant'])
        if 'mcp' in keywords:
            keywords.extend(['model', 'context', 'protocol'])
            
        return list(set(keywords))

    def _generate_search_query(self, requirement: str) -> SearchQuery:
        """Generate structured search query from user requirement."""
        # Use Gemini to analyze the requirement
        analysis = self.generate(
            """Analyze this requirement and extract key search components:
            {requirement}
            
            Focus on identifying:
            1. The main functionality or feature needed
            2. Any specific technologies or frameworks mentioned
            3. Integration points with Pydantic or PydanticAI
            4. Technical constraints or requirements
            
            Return the analysis in JSON format with these keys:
            - mainQuery: A clear search query that captures the core need
            - keywords: A list of relevant keywords for searching
            - filters: Any specific filters to apply (e.g., language, framework)
            """.format(requirement=requirement)
        )
        
        # Extract keywords both programmatically and from the model's analysis
        keywords = self._extract_keywords(requirement)
        
        # Determine solution type hint
        solution_type_hint = None
        for type_name in SolutionType.__members__:
            if type_name.lower() in requirement.lower():
                solution_type_hint = SolutionType[type_name]
                break
        
        return SearchQuery(
            main_query=requirement,
            keywords=keywords,
            filters={
                'min_stars': self.min_stars,
                'language': 'python',
                'has_documentation': True
            },
            solution_type_hint=solution_type_hint
        )

    def _deduplicate_candidates(
        self,
        candidates: List[SolutionCandidate]
    ) -> List[SolutionCandidate]:
        """Remove duplicate solutions based on repository URL."""
        seen_urls = set()
        unique_candidates = []
        
        for candidate in candidates:
            url = str(candidate.repository_url)
            if url not in seen_urls:
                seen_urls.add(url)
                unique_candidates.append(candidate)
                
        return unique_candidates

    async def _search_all_sources(self, query: SearchQuery) -> List[SolutionCandidate]:
        """Search across multiple sources and compile results."""
        candidates = []
        
        # Search GitHub repositories
        github_result = await self.search_tools.search_github(GithubSearchInput(
            query=query.main_query,
            github_token=self.github_token,
            max_results=self.max_results_per_source
        ))
        if not github_result.error and github_result.repositories:
            for repo in github_result.repositories:
                if repo['stars'] >= self.min_stars:
                    solution_type = query.solution_type_hint or SolutionType.LIBRARY
                    candidate = self.search_tools.convert_to_solution_candidate(
                        repo, "github", solution_type
                    )
                    candidates.append(candidate)
                    
        # Search PyPI packages
        pypi_result = await self.search_tools.search_pypi(SearchToolInput(
            query=query.main_query,
            max_results=self.max_results_per_source
        ))
        if not pypi_result.error and pypi_result.packages:
            for pkg in pypi_result.packages:
                solution_type = query.solution_type_hint or SolutionType.LIBRARY
                candidate = self.search_tools.convert_to_solution_candidate(
                    pkg, "pypi", solution_type
                )
                candidates.append(candidate)
                
        # Search documentation and tutorials
        ddg_result = await self.search_tools.search_duckduckgo(SearchToolInput(
            query=f"{query.main_query} python pydantic documentation examples",
            max_results=self.max_results_per_source
        ))
        
        # Deduplicate and validate candidates
        candidates = self._deduplicate_candidates(candidates)
        candidates = self._validate_candidates(candidates)
        
        return candidates

    def _validate_candidates(
        self,
        candidates: List[SolutionCandidate]
    ) -> List[SolutionCandidate]:
        """Validate and filter solution candidates."""
        valid_candidates = []
        
        for candidate in candidates:
            # Check for required fields
            if not candidate.name or not candidate.description:
                continue
                
            # Validate repository URL
            if not candidate.repository_url:
                continue
                
            # Check compatibility with project context
            if candidate.metadata.get('requires_python'):
                required_python = candidate.metadata['requires_python']
                project_python = self.project_context.python_version
                # Simple version compatibility check
                if not self._is_python_compatible(required_python, project_python):
                    continue
            
            valid_candidates.append(candidate)
            
        return valid_candidates

    def _is_python_compatible(self, required: str, current: str) -> bool:
        """Check if the required Python version is compatible with current version."""
        # Simple version comparison - could be enhanced with packaging.version
        required = required.replace('>=', '').replace('<=', '').replace('~=', '')
        required_parts = [int(p) for p in required.split('.')[0:2]]
        current_parts = [int(p) for p in current.split('.')[0:2]]
        
        return current_parts >= required_parts

    async def run(self, input: SearchInput) -> SearchOutput:
        """
        Run the search process for a requirement.
        
        This is the main entry point required by the Agent base class.
        """
        # Generate structured search query
        search_query = self._generate_search_query(input.requirement)
        
        # Execute searches across all sources
        candidates = await self._search_all_sources(search_query)
        
        # Sort by stars (if available) and last updated
        candidates.sort(
            key=lambda x: (x.stars or 0, x.last_updated or datetime.min),
            reverse=True
        )
        
        return SearchOutput(
            candidates=candidates,
            query=search_query,
            total_results=len(candidates)
        )
        
    async def find_solutions(self, requirement: str) -> List[SolutionCandidate]:
        """
        Legacy method for backward compatibility.
        Use run() for new code.
        """
        result = await self.run(SearchInput(
            requirement=requirement,
            project_context=self.project_context
        ))
        return result.candidates 