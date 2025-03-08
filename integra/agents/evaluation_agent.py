from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel, Field, HttpUrl
from pydantic_ai import Agent
import httpx
from datetime import datetime, timedelta
from packaging import version
import asyncio
import os
from dotenv import load_dotenv

from ..models import (
    SolutionCandidate,
    BestSolution,
    NoViableSolution,
    ProjectContext,
    CompatibilityStatus
)

# Load environment variables
load_dotenv()

# Get API keys from environment
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
PYDANTIC_AI_MODEL = os.getenv("PYDANTIC_AI_MODEL", "google-gla:gemini-1.5-flash")

class EvaluationMetrics(BaseModel):
    """Metrics used to evaluate a solution candidate."""
    maintenance_score: float = Field(ge=0, le=1, description="Score based on maintenance activity")
    community_score: float = Field(ge=0, le=1, description="Score based on community engagement")
    documentation_score: float = Field(ge=0, le=1, description="Score based on documentation quality")
    compatibility_score: float = Field(ge=0, le=1, description="Score based on technical compatibility")
    integration_complexity: float = Field(ge=0, le=1, description="Score based on integration difficulty")
    
    def weighted_average(self) -> float:
        """Calculate weighted average of all metrics."""
        weights = {
            'maintenance_score': 0.25,
            'community_score': 0.2,
            'documentation_score': 0.2,
            'compatibility_score': 0.25,
            'integration_complexity': 0.1
        }
        return sum(
            getattr(self, metric) * weight 
            for metric, weight in weights.items()
        )

class EvaluationDependencies(BaseModel):
    """Dependencies required by the Evaluation Agent."""
    github_token: str = Field(default=GITHUB_TOKEN, description="GitHub API token for repository analysis")
    project_context: ProjectContext = Field(..., description="Context of the user's project")
    min_evaluation_score: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Minimum score for a solution to be considered viable"
    )
    max_integration_time_hours: int = Field(
        default=16,
        ge=0,
        description="Maximum acceptable integration time in hours"
    )

class EvaluationInput(BaseModel):
    """Input for evaluation requests."""
    candidate: SolutionCandidate = Field(..., description="Solution candidate to evaluate")
    project_context: ProjectContext = Field(..., description="User's project context")

class EvaluationOutput(BaseModel):
    """Output from evaluation process."""
    metrics: EvaluationMetrics = Field(..., description="Calculated evaluation metrics")
    compatibility_status: CompatibilityStatus = Field(..., description="Compatibility assessment")
    integration_time: str = Field(..., description="Estimated integration time")
    issues: List[str] = Field(default_factory=list, description="Identified compatibility issues")

class EvaluationAgent(Agent[EvaluationInput, EvaluationOutput]):
    """Agent responsible for evaluating solution candidates for compatibility and viability."""
    
    system_prompt = """You are the Evaluation Agent for Integra, an expert at assessing open-source solutions for compatibility with Pydantic AI projects. Your role is to:

1. Analyze solution candidates against project requirements
2. Assess technical compatibility with the user's environment
3. Evaluate maintenance status and community support
4. Identify potential integration challenges
5. Estimate time and effort required for integration

For each solution, consider:
- Compatibility with Python, Pydantic, and PydanticAI versions
- Quality and completeness of documentation
- Recent maintenance and update history
- Community size and engagement
- Integration complexity and required adaptations

Provide detailed metrics and reasoning for each evaluation. Be critical but fair in your assessments, and always prioritize solutions that will provide the best developer experience after integration.
"""

    def __init__(self, dependencies: EvaluationDependencies):
        """Initialize the Evaluation Agent."""
        super().__init__(
            model=PYDANTIC_AI_MODEL,  # Use the model from environment
            deps_type=EvaluationInput,
            result_type=EvaluationOutput,
        )
        
        self.github_token = dependencies.github_token
        self.project_context = dependencies.project_context
        self.min_evaluation_score = dependencies.min_evaluation_score
        self.max_integration_time_hours = dependencies.max_integration_time_hours
        self.http_client = httpx.AsyncClient()
        
    async def _get_github_metrics(self, repo_url: HttpUrl) -> Dict[str, Any]:
        """Fetch and analyze GitHub repository metrics."""
        # Extract owner and repo from URL
        parts = str(repo_url).split('/')
        owner, repo = parts[-2], parts[-1]
        
        headers = {'Authorization': f'token {self.github_token}'}
        
        # Fetch repository data
        repo_response = await self.http_client.get(
            f'https://api.github.com/repos/{owner}/{repo}',
            headers=headers
        )
        repo_data = repo_response.json()
        
        # Fetch recent commits
        commits_response = await self.http_client.get(
            f'https://api.github.com/repos/{owner}/{repo}/commits',
            headers=headers,
            params={'per_page': 30}
        )
        commits = commits_response.json()
        
        # Fetch issues
        issues_response = await self.http_client.get(
            f'https://api.github.com/repos/{owner}/{repo}/issues',
            headers=headers,
            params={'state': 'all', 'per_page': 100}
        )
        issues = issues_response.json()
        
        # Calculate metrics
        last_commit = datetime.fromisoformat(commits[0]['commit']['committer']['date'].replace('Z', '+00:00'))
        days_since_last_commit = (datetime.now(last_commit.tzinfo) - last_commit).days
        
        commit_frequency = len([
            c for c in commits 
            if (datetime.now(last_commit.tzinfo) - datetime.fromisoformat(
                c['commit']['committer']['date'].replace('Z', '+00:00')
            )).days <= 90
        ]) / 90.0
        
        issue_response_time = timedelta(days=0)
        responded_issues = 0
        
        for issue in issues:
            if issue.get('comments', 0) > 0:
                created = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
                first_response = datetime.fromisoformat(issue['updated_at'].replace('Z', '+00:00'))
                issue_response_time += first_response - created
                responded_issues += 1
                
        avg_response_time = (
            issue_response_time / responded_issues if responded_issues > 0 
            else timedelta(days=30)
        )
        
        return {
            'stars': repo_data['stargazers_count'],
            'forks': repo_data['forks_count'],
            'open_issues': repo_data['open_issues_count'],
            'days_since_last_commit': days_since_last_commit,
            'commit_frequency': commit_frequency,
            'avg_response_time_days': avg_response_time.days,
            'watchers': repo_data['subscribers_count'],
            'license': repo_data.get('license', {}).get('key', 'unknown')
        }
        
    def _calculate_maintenance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate maintenance score based on GitHub metrics."""
        # Scoring factors
        commit_recency = max(0, 1 - (metrics['days_since_last_commit'] / 180))
        commit_frequency = min(1, metrics['commit_frequency'] * 2)
        response_time = max(0, 1 - (metrics['avg_response_time_days'] / 14))
        
        # Weighted average
        weights = {'recency': 0.4, 'frequency': 0.4, 'response': 0.2}
        score = (
            commit_recency * weights['recency'] +
            commit_frequency * weights['frequency'] +
            response_time * weights['response']
        )
        
        return score
        
    def _calculate_community_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate community health score based on GitHub metrics."""
        # Normalize metrics to 0-1 scale
        stars_score = min(1, metrics['stars'] / 1000)
        forks_score = min(1, metrics['forks'] / 100)
        watchers_score = min(1, metrics['watchers'] / 50)
        
        # Weighted average
        weights = {'stars': 0.5, 'forks': 0.3, 'watchers': 0.2}
        score = (
            stars_score * weights['stars'] +
            forks_score * weights['forks'] +
            watchers_score * weights['watchers']
        )
        
        return score
        
    def _check_version_compatibility(
        self,
        candidate: SolutionCandidate
    ) -> tuple[bool, List[str]]:
        """Check version compatibility with project dependencies."""
        incompatibilities = []
        
        # Check Python version
        if 'requires_python' in candidate.metadata:
            try:
                required = candidate.metadata['requires_python']
                current = self.project_context.python_version
                if not self._is_python_compatible(required, current):
                    incompatibilities.append(
                        f"Python version {current} not compatible with requirement {required}"
                    )
            except Exception:
                pass
                
        # Check other dependencies
        for dep, ver in candidate.metadata.get('dependencies', {}).items():
            if dep in self.project_context.dependencies:
                project_ver = self.project_context.dependencies[dep]
                try:
                    if not self._is_dependency_compatible(ver, project_ver):
                        incompatibilities.append(
                            f"Dependency {dep} version {project_ver} not compatible with {ver}"
                        )
                except Exception:
                    pass
                    
        return len(incompatibilities) == 0, incompatibilities
        
    def _is_python_compatible(self, required: str, current: str) -> bool:
        """Check if Python version is compatible."""
        try:
            # Handle common version specifiers
            required = required.replace('>=', '').replace('<=', '').replace('~=', '')
            required_ver = version.parse(required)
            current_ver = version.parse(current)
            return current_ver >= required_ver
        except Exception:
            return True  # If we can't parse versions, assume compatible
            
    def _is_dependency_compatible(self, required: str, current: str) -> bool:
        """Check if dependency versions are compatible."""
        try:
            required_ver = version.parse(required)
            current_ver = version.parse(current)
            return current_ver >= required_ver
        except Exception:
            return True  # If we can't parse versions, assume compatible
            
    def _estimate_integration_time(
        self,
        candidate: SolutionCandidate,
        compatibility_issues: List[str]
    ) -> str:
        """Estimate time required for integration."""
        base_hours = 2  # Minimum time for basic integration
        
        # Add time for compatibility issues
        issue_hours = len(compatibility_issues) * 2
        
        # Add time based on solution type
        type_hours = {
            'AGENT': 4,
            'MCP_SERVER': 6,
            'API': 3,
            'LIBRARY': 2
        }.get(candidate.solution_type.name, 2)
        
        # Add time based on documentation quality
        if not candidate.metadata.get('has_documentation', True):
            base_hours += 4
            
        total_hours = base_hours + issue_hours + type_hours
        
        if total_hours <= 4:
            return "2-4 hours"
        elif total_hours <= 8:
            return "4-8 hours"
        else:
            return f"{total_hours-4}-{total_hours+4} hours"
            
    async def evaluate_candidate(
        self,
        candidate: SolutionCandidate
    ) -> EvaluationOutput:
        """Evaluate a single solution candidate."""
        # Fetch GitHub metrics
        github_metrics = await self._get_github_metrics(candidate.repository_url)
        
        # Calculate scores
        maintenance_score = self._calculate_maintenance_score(github_metrics)
        community_score = self._calculate_community_score(github_metrics)
        
        # Check documentation
        has_docs = candidate.metadata.get('has_documentation', False)
        doc_quality = candidate.metadata.get('documentation_quality', 0.5)
        documentation_score = 0.8 if has_docs else 0.2
        documentation_score *= doc_quality
        
        # Check compatibility
        is_compatible, issues = self._check_version_compatibility(candidate)
        compatibility_score = 1.0 if is_compatible else max(0.2, 1 - (len(issues) * 0.2))
        
        # Calculate integration complexity
        integration_time = self._estimate_integration_time(candidate, issues)
        estimated_hours = float(integration_time.split('-')[1].split()[0])
        integration_complexity = max(0, 1 - (estimated_hours / self.max_integration_time_hours))
        
        metrics = EvaluationMetrics(
            maintenance_score=maintenance_score,
            community_score=community_score,
            documentation_score=documentation_score,
            compatibility_score=compatibility_score,
            integration_complexity=integration_complexity
        )
        
        # Determine compatibility status
        if is_compatible:
            status = CompatibilityStatus.COMPATIBLE
        elif metrics.weighted_average() >= 0.8:
            status = CompatibilityStatus.NEEDS_ADAPTATION
        else:
            status = CompatibilityStatus.INCOMPATIBLE
            
        return EvaluationOutput(
            metrics=metrics,
            compatibility_status=status,
            integration_time=integration_time,
            issues=issues
        )
        
    async def run(self, input: EvaluationInput) -> EvaluationOutput:
        """
        Run the evaluation process for a single candidate.
        
        This is the main entry point required by the Agent base class.
        """
        return await self.evaluate_candidate(input.candidate)
        
    async def evaluate_solutions(
        self,
        candidates: List[SolutionCandidate]
    ) -> Union[BestSolution, NoViableSolution]:
        """
        Evaluate all solution candidates and select the best option.
        
        Args:
            candidates: List of solution candidates to evaluate
            
        Returns:
            Either BestSolution with the best candidate or NoViableSolution if no viable options
        """
        if not candidates:
            return NoViableSolution(
                reason="No solution candidates provided",
                suggestions=["Broaden search criteria", "Consider alternative approaches"],
                alternative_approaches=["Build custom solution", "Modify requirements"]
            )
            
        results = []
        for candidate in candidates:
            try:
                evaluation = await self.run(EvaluationInput(
                    candidate=candidate,
                    project_context=self.project_context
                ))
                score = evaluation.metrics.weighted_average()
                
                if score >= self.min_evaluation_score:
                    results.append((candidate, evaluation, score))
            except Exception as e:
                continue
                
        if not results:
            return NoViableSolution(
                reason="No solutions met minimum evaluation criteria",
                suggestions=[
                    "Lower minimum evaluation score",
                    "Consider newer or more actively maintained solutions",
                    "Check for compatibility issues with project requirements"
                ],
                alternative_approaches=[
                    "Build custom solution",
                    "Modify project requirements",
                    "Consider alternative solution types"
                ]
            )
            
        # Sort by score and select best
        results.sort(key=lambda x: x[2], reverse=True)
        best_candidate, evaluation, score = results[0]
            
        return BestSolution(
            candidate=best_candidate,
            compatibility_status=evaluation.compatibility_status,
            adaptation_required=evaluation.compatibility_status == CompatibilityStatus.NEEDS_ADAPTATION,
            adaptation_notes="\n".join(evaluation.issues) if evaluation.issues else None,
            estimated_integration_time=evaluation.integration_time,
            requirements=[
                f"Python {best_candidate.metadata.get('requires_python', 'any')}",
                *[f"{k}=={v}" for k, v in best_candidate.metadata.get('dependencies', {}).items()]
            ],
            risks=[
                "Integration complexity may be higher than estimated" if evaluation.metrics.integration_complexity < 0.5 else None,
                "Documentation may be incomplete" if evaluation.metrics.documentation_score < 0.6 else None,
                "Limited community support" if evaluation.metrics.community_score < 0.5 else None,
                "Maintenance status uncertain" if evaluation.metrics.maintenance_score < 0.6 else None
            ]
        ) 