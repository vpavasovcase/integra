#!/usr/bin/env python3
"""
Example script demonstrating the DocumentationAgent.
This script shows how to initialize and use the DocumentationAgent to retrieve
and process documentation for a solution.
"""

import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from integra.agents.documentation_agent import (
    DocumentationAgent, 
    DocumentationDependencies,
    DocumentationSection
)
from integra.models.solution import SolutionCandidate, SolutionType
from integra.models.project import ProjectContext, PydanticInfo, PydanticAIInfo, FrameworkDetails


async def main():
    # Initialize console for nice output
    console = Console()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # GitHub token for API access (optional)
    github_token = os.getenv("GITHUB_TOKEN")
    
    # Create dependencies for the Documentation Agent
    dependencies = DocumentationDependencies(
        github_token=github_token,
        request_timeout=30
    )
    
    # Initialize the Documentation Agent
    documentation_agent = DocumentationAgent(dependencies)
    
    # Example project context
    project_context = ProjectContext(
        name="PydanticAI Example Project",
        description="A project using PydanticAI for agent-based development",
        python_version="3.10",
        dependencies={
            "pydantic": "2.5.2",
            "pydantic_ai": "0.1.0",
            "aiohttp": "3.8.5"
        },
        pydantic=PydanticInfo(
            version="2.5.2",
            models=["UserProfile", "MessageHistory", "AgentConfig"],
            features_used=["Field validation", "JSON schema generation"]
        ),
        pydantic_ai=PydanticAIInfo(
            version="0.1.0",
            agents=["AnalysisAgent", "ResponseAgent"],
            models_used=["Gemini-2.0-Flash"],
            features_used=["Agent framework", "Tool usage"]
        ),
        frameworks=[
            FrameworkDetails(
                name="FastAPI",
                version="0.103.1",
                usage_level="core"
            )
        ]
    )
    
    # Example solution candidate (langchain in this example)
    solution = SolutionCandidate(
        name="langchain",
        description="Building applications with LLMs through composability",
        solution_type=SolutionType.FRAMEWORK,
        repository_url="https://github.com/langchain-ai/langchain",
        pypi_name="langchain",
        current_version="0.0.330",
        stars=75000,
        documentation_url="https://python.langchain.com/docs/get_started/introduction"
    )
    
    # Focus on these documentation sections
    focus_areas = [
        DocumentationSection.INSTALLATION,
        DocumentationSection.USAGE,
        DocumentationSection.CONFIGURATION,
        DocumentationSection.EXAMPLES,
        DocumentationSection.DEPENDENCIES
    ]
    
    # Process the documentation
    console.print(Panel.fit("Retrieving and processing documentation for langchain...", 
                          title="Documentation Agent Example"))
    
    documentation = await documentation_agent.get_documentation(
        solution=solution,
        project_context=project_context,
        focus_areas=focus_areas
    )
    
    if documentation:
        # Print the results
        console.print(Panel.fit(f"Source: {documentation.source} ({documentation.source_url})", 
                              title="Documentation Source"))
        
        if documentation.installation:
            console.print(Panel.fit(documentation.installation, title="Installation Instructions"))
        
        if documentation.usage_examples:
            console.print(Panel.fit("Found usage examples:", title="Usage Examples"))
            for i, example in enumerate(documentation.usage_examples[:3], 1):  # Show first 3 examples
                console.print(f"{i}. {example['title']}")
                console.print(f"```python\n{example['code']}\n```")
        
        if documentation.configuration_options:
            console.print(Panel.fit(str(documentation.configuration_options), title="Configuration Options"))
        
        if documentation.dependencies:
            console.print(Panel.fit("\n".join(documentation.dependencies), title="Dependencies"))
        
        if documentation.python_version:
            console.print(f"[bold]Python Version:[/bold] {documentation.python_version}")
    else:
        console.print("[bold red]Failed to retrieve documentation.[/bold red]")


if __name__ == "__main__":
    asyncio.run(main()) 