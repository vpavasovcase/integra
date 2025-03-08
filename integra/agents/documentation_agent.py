from typing import List, Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
from pydantic_ai import Agent, Tool, ToolCall
import os
from dotenv import load_dotenv

import aiohttp
import asyncio
import re
from bs4 import BeautifulSoup
import markdown
from urllib.parse import urljoin

from integra.models.solution import SolutionCandidate
from integra.models.project import ProjectContext

# Load environment variables
load_dotenv()

# Get API keys from environment
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
PYDANTIC_AI_MODEL = os.getenv("PYDANTIC_AI_MODEL", "google-gla:gemini-1.5-flash")


class DocumentationSource(str, Enum):
    """Source of documentation content."""
    GITHUB_README = "github_readme"
    PYPI = "pypi"
    READTHEDOCS = "readthedocs"
    OFFICIAL_DOCS = "official_docs"
    GITHUB_WIKI = "github_wiki"
    GITHUB_DOCS = "github_docs"


class DocumentationSection(str, Enum):
    """Types of documentation sections."""
    INSTALLATION = "installation"
    USAGE = "usage"
    CONFIGURATION = "configuration"
    API_REFERENCE = "api_reference"
    DEPENDENCIES = "dependencies"
    EXAMPLES = "examples"
    QUICKSTART = "quickstart"
    INTEGRATION = "integration"
    TROUBLESHOOTING = "troubleshooting"


class DocumentationContent(BaseModel):
    """Structured documentation content extracted from a solution."""
    solution_name: str = Field(..., description="Name of the solution")
    source: DocumentationSource = Field(..., description="Source of the documentation")
    source_url: str = Field(..., description="URL where the documentation was found")
    
    # Key documentation sections
    installation: Optional[str] = Field(None, description="Installation instructions")
    usage_examples: List[Dict[str, str]] = Field(
        default_factory=list, 
        description="Usage examples with title and code/description"
    )
    configuration_options: Optional[Dict[str, Any]] = Field(
        None, 
        description="Configuration options and their descriptions"
    )
    dependencies: Optional[List[str]] = Field(
        None, 
        description="Dependencies required by the solution"
    )
    api_reference: Optional[Dict[str, str]] = Field(
        None, 
        description="API reference documentation"
    )
    
    # Additional useful information
    python_version: Optional[str] = Field(None, description="Supported Python versions")
    integration_notes: Optional[str] = Field(None, description="Notes specific to integration")
    
    # Raw content for further processing if needed
    raw_content: Optional[str] = Field(None, description="Raw documentation content")


class DocumentationDependencies(BaseModel):
    """Dependencies required by the Documentation Agent."""
    github_token: Optional[str] = Field(default=GITHUB_TOKEN, description="GitHub API token for private repositories")
    max_content_length: int = Field(default=100000, description="Maximum length of content to process")
    request_timeout: int = Field(default=30, description="Timeout for HTTP requests in seconds")
    user_agent: str = Field(
        default="Integra-Documentation-Agent/1.0",
        description="User agent string for HTTP requests"
    )


class DocumentationInput(BaseModel):
    """Input for documentation requests."""
    solution: SolutionCandidate = Field(..., description="Solution candidate to retrieve documentation for")
    project_context: ProjectContext = Field(..., description="Context of the user's project")
    focus_areas: Optional[List[DocumentationSection]] = Field(
        None, 
        description="Specific documentation areas to focus on"
    )


class DocumentationOutput(BaseModel):
    """Output from documentation process."""
    content: DocumentationContent = Field(..., description="Structured documentation content")
    success: bool = Field(..., description="Whether documentation retrieval was successful")
    missing_sections: List[DocumentationSection] = Field(
        default_factory=list, 
        description="Documentation sections that could not be found"
    )


class DocumentationAgent(Agent[DocumentationInput, DocumentationOutput]):
    """Agent responsible for retrieving and processing documentation for solutions."""
    
    system_prompt = """You are the Documentation Agent for Integra, an expert at retrieving, processing, and structuring documentation for open-source solutions. Your role is to:

1. Retrieve documentation from various sources (GitHub READMEs, PyPI, ReadTheDocs, official documentation)
2. Extract key information needed for integration, including:
   - Installation instructions
   - Usage examples
   - Configuration options
   - Dependencies and requirements
   - API reference details
   - Integration patterns and best practices

3. Organize and structure the information in a consistent format
4. Identify missing information that might be critical for integration
5. Summarize complex documentation into clear, actionable points

When processing documentation:
- Focus on finding practical integration information rather than theoretical explanations
- Extract complete code examples that demonstrate actual usage
- Identify any compatibility issues or special requirements
- Look for explicit integration examples with Pydantic or similar frameworks
- Pay special attention to parameters, configuration options, and customization points
- Capture environment setup requirements including Python version constraints

Ensure the documentation is structured in a way that will be most useful for the Integration Agent to create step-by-step integration instructions."""

    fetch_github_readme = Tool(
        name="fetch_github_readme",
        description="Fetch README content from a GitHub repository",
        parameters={
            "repo_url": {
                "type": "string",
                "description": "URL of the GitHub repository"
            }
        },
        function=None  # Will be implemented in the constructor
    )

    fetch_pypi_documentation = Tool(
        name="fetch_pypi_documentation",
        description="Fetch documentation from PyPI for a package",
        parameters={
            "package_name": {
                "type": "string",
                "description": "Name of the PyPI package"
            }
        },
        function=None  # Will be implemented in the constructor
    )
    
    fetch_web_documentation = Tool(
        name="fetch_web_documentation",
        description="Fetch documentation from a web URL",
        parameters={
            "url": {
                "type": "string",
                "description": "URL of the documentation page"
            }
        },
        function=None  # Will be implemented in the constructor
    )

    extract_section = Tool(
        name="extract_section",
        description="Extract a specific section from documentation text",
        parameters={
            "content": {
                "type": "string",
                "description": "Documentation content to extract from"
            },
            "section_type": {
                "type": "string",
                "description": "Type of section to extract (installation, usage, configuration, etc.)"
            }
        },
        function=None  # Will be implemented in the constructor
    )

    def __init__(self, dependencies: DocumentationDependencies):
        """Initialize the Documentation Agent."""
        super().__init__(
            model=PYDANTIC_AI_MODEL,  # Use the model from environment
            deps_type=DocumentationInput,
            result_type=DocumentationOutput,
        )
        
        self.github_token = dependencies.github_token
        self.request_timeout = dependencies.request_timeout
        self.max_content_length = dependencies.max_content_length
        self.user_agent = dependencies.user_agent
        self.session = None
        
        # Implement tool functions
        self.fetch_github_readme.function = self._fetch_github_readme
        self.fetch_pypi_documentation.function = self._fetch_pypi_documentation
        self.fetch_web_documentation.function = self._fetch_web_documentation
        self.extract_section.function = self._extract_section
    
    async def _ensure_session(self):
        """Ensure HTTP session is initialized."""
        if self.session is None:
            self.session = aiohttp.ClientSession(headers=self.headers)
    
    async def _fetch_github_readme(self, repo_url: str) -> Dict[str, Any]:
        """Fetch README content from a GitHub repository."""
        await self._ensure_session()
        
        # Extract owner and repo from URL
        match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
        if not match:
            return {"success": False, "error": "Invalid GitHub repository URL"}
        
        owner, repo = match.groups()
        repo = repo.split("/")[0]  # Remove any trailing path
        
        # Try to fetch README through GitHub API
        api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
        
        try:
            async with self.session.get(api_url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    if "content" in data:
                        import base64
                        content = base64.b64decode(data["content"]).decode("utf-8")
                        
                        # Convert markdown to HTML for easier parsing
                        html_content = markdown.markdown(content)
                        
                        return {
                            "success": True,
                            "content": content,
                            "html_content": html_content,
                            "source": DocumentationSource.GITHUB_README,
                            "url": repo_url
                        }
                
                # Fallback: try to fetch directly
                readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md"
                async with self.session.get(readme_url, timeout=self.request_timeout) as response:
                    if response.status == 200:
                        content = await response.text()
                        html_content = markdown.markdown(content)
                        return {
                            "success": True,
                            "content": content,
                            "html_content": html_content,
                            "source": DocumentationSource.GITHUB_README,
                            "url": readme_url
                        }
                
                # Try master branch if main fails
                readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md"
                async with self.session.get(readme_url, timeout=self.request_timeout) as response:
                    if response.status == 200:
                        content = await response.text()
                        html_content = markdown.markdown(content)
                        return {
                            "success": True,
                            "content": content,
                            "html_content": html_content,
                            "source": DocumentationSource.GITHUB_README,
                            "url": readme_url
                        }
            
            return {"success": False, "error": "README not found"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _fetch_pypi_documentation(self, package_name: str) -> Dict[str, Any]:
        """Fetch documentation from PyPI for a package."""
        await self._ensure_session()
        
        pypi_url = f"https://pypi.org/project/{package_name}/"
        
        try:
            async with self.session.get(pypi_url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")
                    
                    # Extract project description
                    description_div = soup.find("div", class_="project-description")
                    if description_div:
                        description = description_div.get_text(strip=True)
                    else:
                        description = "No description available"
                    
                    # Extract package metadata
                    metadata = {}
                    metadata_table = soup.find("table", class_="table--hoverableRows")
                    if metadata_table:
                        for row in metadata_table.find_all("tr"):
                            cells = row.find_all("td")
                            if len(cells) >= 2:
                                key = cells[0].get_text(strip=True)
                                value = cells[1].get_text(strip=True)
                                metadata[key] = value
                    
                    # Find documentation links
                    doc_links = []
                    sidebar = soup.find("div", class_="sidebar-section")
                    if sidebar:
                        for link in sidebar.find_all("a"):
                            href = link.get("href")
                            text = link.get_text(strip=True)
                            if href and text:
                                doc_links.append({"text": text, "url": href})
                    
                    return {
                        "success": True,
                        "content": html,
                        "description": description,
                        "metadata": metadata,
                        "doc_links": doc_links,
                        "source": DocumentationSource.PYPI,
                        "url": pypi_url
                    }
                
                return {"success": False, "error": f"Failed to fetch PyPI documentation: {response.status}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _fetch_web_documentation(self, url: str) -> Dict[str, Any]:
        """Fetch documentation from a web URL."""
        await self._ensure_session()
        
        try:
            async with self.session.get(url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Determine documentation source type
                    source_type = DocumentationSource.OFFICIAL_DOCS
                    if "readthedocs.io" in url:
                        source_type = DocumentationSource.READTHEDOCS
                    elif "github.com" in url and "/wiki/" in url:
                        source_type = DocumentationSource.GITHUB_WIKI
                    elif "github.com" in url and "/docs/" in url:
                        source_type = DocumentationSource.GITHUB_DOCS
                    
                    # Parse HTML
                    soup = BeautifulSoup(content, "html.parser")
                    
                    # Remove scripts and styles to clean the content
                    for script in soup(["script", "style"]):
                        script.extract()
                    
                    # Get text
                    text = soup.get_text(separator="\n", strip=True)
                    
                    return {
                        "success": True,
                        "content": content,
                        "text_content": text,
                        "source": source_type,
                        "url": url
                    }
                
                return {"success": False, "error": f"Failed to fetch documentation: {response.status}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _extract_section(self, content: str, section_type: str) -> Dict[str, Any]:
        """Extract a specific section from documentation text."""
        # Convert section_type to enum if string is provided
        if isinstance(section_type, str):
            try:
                section_type = DocumentationSection(section_type.lower())
            except ValueError:
                return {"success": False, "error": f"Invalid section type: {section_type}"}
        
        # Define patterns to identify different sections
        section_patterns = {
            DocumentationSection.INSTALLATION: [
                r"(?i)# *install(ation|ing)?", 
                r"(?i)## *install(ation|ing)?",
                r"(?i)how to install",
                r"(?i)installation instructions"
            ],
            DocumentationSection.USAGE: [
                r"(?i)# *usage", 
                r"(?i)## *usage",
                r"(?i)# *how to use",
                r"(?i)## *how to use"
            ],
            DocumentationSection.CONFIGURATION: [
                r"(?i)# *config(uration)?", 
                r"(?i)## *config(uration)?",
                r"(?i)# *settings",
                r"(?i)## *settings"
            ],
            DocumentationSection.API_REFERENCE: [
                r"(?i)# *api", 
                r"(?i)## *api",
                r"(?i)# *api reference",
                r"(?i)## *api reference"
            ],
            DocumentationSection.DEPENDENCIES: [
                r"(?i)# *dependencies", 
                r"(?i)## *dependencies",
                r"(?i)# *requirements",
                r"(?i)## *requirements"
            ],
            DocumentationSection.EXAMPLES: [
                r"(?i)# *examples?", 
                r"(?i)## *examples?",
                r"(?i)# *sample(s)?",
                r"(?i)## *sample(s)?"
            ],
            DocumentationSection.QUICKSTART: [
                r"(?i)# *quick ?start", 
                r"(?i)## *quick ?start",
                r"(?i)# *getting started",
                r"(?i)## *getting started"
            ],
            DocumentationSection.INTEGRATION: [
                r"(?i)# *integration", 
                r"(?i)## *integration",
                r"(?i)# *integrating with",
                r"(?i)## *integrating with"
            ],
            DocumentationSection.TROUBLESHOOTING: [
                r"(?i)# *troubleshoot(ing)?", 
                r"(?i)## *troubleshoot(ing)?",
                r"(?i)# *common issues",
                r"(?i)## *common issues"
            ]
        }
        
        # Find the section
        patterns = section_patterns.get(section_type, [])
        if not patterns:
            return {"success": False, "error": f"No patterns defined for section type: {section_type}"}
        
        # Try to find the section using various patterns
        extracted_content = None
        for pattern in patterns:
            # Find all matches
            matches = list(re.finditer(pattern, content))
            if matches:
                for i, match in enumerate(matches):
                    start_pos = match.end()
                    
                    # Find the next heading or end of content
                    next_heading_match = re.search(r"(?m)^#+ ", content[start_pos:])
                    end_pos = start_pos + next_heading_match.start() if next_heading_match else len(content)
                    
                    # Extract section content
                    section_content = content[start_pos:end_pos].strip()
                    
                    if section_content:
                        extracted_content = section_content
                        break
                
                if extracted_content:
                    break
        
        # Return results
        if extracted_content:
            return {
                "success": True,
                "section_type": section_type,
                "content": extracted_content
            }
        else:
            return {
                "success": False,
                "error": f"Section '{section_type}' not found in documentation"
            }
    
    async def _extract_code_examples(self, content: str) -> List[Dict[str, str]]:
        """Extract code examples from documentation content."""
        examples = []
        
        # Find markdown code blocks
        code_blocks = re.findall(r"```(?:python)?(.*?)```", content, re.DOTALL)
        for i, code in enumerate(code_blocks):
            # Try to find a title for this example
            code = code.strip()
            if code:
                # Look for preceding heading
                preceding_text = content.split(f"```{code}```")[0].strip()
                heading_match = re.search(r"(?m)^(#+)\s*(.*?)$", preceding_text[::-1])
                title = f"Example {i+1}"
                if heading_match:
                    title = heading_match.group(2)[::-1].strip()
                
                examples.append({
                    "title": title,
                    "code": code
                })
        
        return examples
    
    async def _extract_installation_instructions(self, content: str) -> str:
        """Extract installation instructions from documentation."""
        result = self._extract_section(content, DocumentationSection.INSTALLATION)
        if result["success"]:
            return result["content"]
        
        # If explicit installation section not found, look for pip install commands
        pip_installs = re.findall(r"(?m)^(pip install .*?)$", content)
        if pip_installs:
            return "\n".join(pip_installs)
        
        # Look for code blocks that might contain installation instructions
        install_blocks = re.findall(r"```(?:bash|shell)?\s*(pip install .*?)```", content, re.DOTALL)
        if install_blocks:
            return "\n".join(install_blocks)
        
        return None
    
    async def _extract_configuration_options(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract configuration options from documentation."""
        result = self._extract_section(content, DocumentationSection.CONFIGURATION)
        if result["success"]:
            config_content = result["content"]
            
            # Look for configuration parameters in the content
            options = {}
            
            # Try to find parameter tables in markdown
            table_pattern = r"\|([^|]+)\|([^|]+)\|"
            table_matches = re.findall(table_pattern, config_content)
            if table_matches:
                for param, desc in table_matches:
                    param = param.strip()
                    desc = desc.strip()
                    if param and param != "Parameter" and param != "---":
                        options[param] = desc
            
            # Look for parameters defined in bullet points or numbered lists
            list_pattern = r"(?m)^[-*]\s*`([^`]+)`[:\s]+(.+)$"
            list_matches = re.findall(list_pattern, config_content)
            for param, desc in list_matches:
                options[param.strip()] = desc.strip()
            
            # If we found options, return them
            if options:
                return options
        
        return None
    
    async def _extract_dependencies(self, content: str) -> Optional[List[str]]:
        """Extract dependencies from documentation."""
        result = self._extract_section(content, DocumentationSection.DEPENDENCIES)
        if result["success"]:
            deps_content = result["content"]
            
            # Look for requirements.txt style dependencies
            deps = []
            lines = deps_content.split("\n")
            for line in lines:
                line = line.strip()
                if re.match(r"^[a-zA-Z0-9_-]+[>=<~!]?=?[\d\.]+$", line):
                    deps.append(line)
                elif re.match(r"^[a-zA-Z0-9_-]+$", line) and "=" not in line and ":" not in line:
                    deps.append(line)
            
            # Look for pip install commands
            pip_deps = re.findall(r"pip install ([a-zA-Z0-9_-]+(?:[>=<~!]=[\d\.]+)?)", deps_content)
            deps.extend(pip_deps)
            
            # Remove duplicates
            deps = list(set(deps))
            
            if deps:
                return deps
        
        # Check if there's a requirements.txt or setup.py content in the documentation
        req_pattern = r"```(?:python|text)?\s*((?:[a-zA-Z0-9_-]+[>=<~!]?=?[\d\.]+\s*)+)```"
        req_blocks = re.findall(req_pattern, content, re.DOTALL)
        if req_blocks:
            deps = []
            for block in req_blocks:
                lines = block.split("\n")
                for line in lines:
                    line = line.strip()
                    if re.match(r"^[a-zA-Z0-9_-]+[>=<~!]?=?[\d\.]+$", line):
                        deps.append(line)
            
            if deps:
                return list(set(deps))
        
        return None
    
    async def _extract_python_version(self, content: str) -> Optional[str]:
        """Extract Python version requirements from documentation."""
        # Look for explicit Python version mentions
        version_patterns = [
            r"Python\s*([0-9\.]+)\+",
            r"Python\s*>=\s*([0-9\.]+)",
            r"Python\s*version\s*([0-9\.]+)\s*or higher",
            r"requires\s*Python\s*([0-9\.]+)\+",
            r"supports\s*Python\s*([0-9\.]+)\s*and\s*above"
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)
        
        # Look in classifiers if they're in the content
        classifier_pattern = r"Programming Language :: Python :: ([0-9\.]+)"
        classifiers = re.findall(classifier_pattern, content)
        if classifiers:
            # Return the range
            versions = sorted(classifiers)
            if versions:
                return f"{versions[0]}+" if len(versions) == 1 else f"{versions[0]}-{versions[-1]}"
        
        return None
    
    async def _process_documentation(
        self, 
        solution: SolutionCandidate,
        focus_areas: Optional[List[DocumentationSection]] = None
    ) -> DocumentationOutput:
        """Process documentation for a solution candidate."""
        # If no focus areas specified, include all sections
        if not focus_areas:
            focus_areas = list(DocumentationSection)
        
        # Initialize documentation content
        doc_content = DocumentationContent(
            solution_name=solution.name,
            source=DocumentationSource.GITHUB_README,  # Default, will be updated
            source_url=solution.repository_url or solution.homepage_url or ""
        )
        
        # Track missing sections
        missing_sections = []
        
        # Try to fetch documentation from different sources
        documentation_data = None
        
        # 1. Try GitHub README if repository URL is available
        if solution.repository_url and "github.com" in solution.repository_url:
            github_result = await self._fetch_github_readme(solution.repository_url)
            if github_result["success"]:
                documentation_data = github_result
                doc_content.source = DocumentationSource.GITHUB_README
                doc_content.source_url = solution.repository_url
        
        # 2. Try PyPI if package name is available
        if not documentation_data and solution.pypi_name:
            pypi_result = await self._fetch_pypi_documentation(solution.pypi_name)
            if pypi_result["success"]:
                documentation_data = pypi_result
                doc_content.source = DocumentationSource.PYPI
                doc_content.source_url = f"https://pypi.org/project/{solution.pypi_name}/"
        
        # 3. Try documentation URL if available
        if not documentation_data and solution.documentation_url:
            doc_result = await self._fetch_web_documentation(solution.documentation_url)
            if doc_result["success"]:
                documentation_data = doc_result
                doc_content.source = doc_result["source"]
                doc_content.source_url = solution.documentation_url
        
        # 4. Try homepage URL as last resort
        if not documentation_data and solution.homepage_url:
            home_result = await self._fetch_web_documentation(solution.homepage_url)
            if home_result["success"]:
                documentation_data = home_result
                doc_content.source = home_result["source"]
                doc_content.source_url = solution.homepage_url
        
        # Check if we got any documentation
        if not documentation_data:
            return DocumentationOutput(
                content=doc_content,
                success=False,
                missing_sections=list(focus_areas)
            )
        
        # Store raw content
        if "content" in documentation_data:
            doc_content.raw_content = documentation_data["content"]
        elif "text_content" in documentation_data:
            doc_content.raw_content = documentation_data["text_content"]
        
        # Process content to extract relevant information
        content = doc_content.raw_content
        
        # Extract information based on focus areas
        if DocumentationSection.INSTALLATION in focus_areas:
            doc_content.installation = await self._extract_installation_instructions(content)
            if not doc_content.installation:
                missing_sections.append(DocumentationSection.INSTALLATION)
        
        if DocumentationSection.USAGE in focus_areas or DocumentationSection.EXAMPLES in focus_areas:
            doc_content.usage_examples = await self._extract_code_examples(content)
            if not doc_content.usage_examples:
                if DocumentationSection.USAGE in focus_areas:
                    missing_sections.append(DocumentationSection.USAGE)
                if DocumentationSection.EXAMPLES in focus_areas:
                    missing_sections.append(DocumentationSection.EXAMPLES)
        
        if DocumentationSection.CONFIGURATION in focus_areas:
            doc_content.configuration_options = await self._extract_configuration_options(content)
            if not doc_content.configuration_options:
                missing_sections.append(DocumentationSection.CONFIGURATION)
        
        if DocumentationSection.DEPENDENCIES in focus_areas:
            doc_content.dependencies = await self._extract_dependencies(content)
            if not doc_content.dependencies:
                missing_sections.append(DocumentationSection.DEPENDENCIES)
        
        # Extract Python version constraints
        doc_content.python_version = await self._extract_python_version(content)
        
        # Look for integration notes
        if DocumentationSection.INTEGRATION in focus_areas:
            integration_result = self._extract_section(content, DocumentationSection.INTEGRATION)
            if integration_result["success"]:
                doc_content.integration_notes = integration_result["content"]
            else:
                missing_sections.append(DocumentationSection.INTEGRATION)
        
        # Create output
        return DocumentationOutput(
            content=doc_content,
            success=len(missing_sections) < len(focus_areas),  # Success if we got at least some sections
            missing_sections=missing_sections
        )
    
    async def run(self, input: DocumentationInput) -> DocumentationOutput:
        """Process documentation for a solution candidate."""
        try:
            # Process documentation
            result = await self._process_documentation(input.solution, input.focus_areas)
            
            # Clean up
            if self.session:
                await self.session.close()
                self.session = None
            
            return result
        except Exception as e:
            # Create minimal output for failures
            doc_content = DocumentationContent(
                solution_name=input.solution.name,
                source=DocumentationSource.GITHUB_README,
                source_url=input.solution.repository_url or input.solution.homepage_url or ""
            )
            
            # Clean up
            if self.session:
                await self.session.close()
                self.session = None
            
            return DocumentationOutput(
                content=doc_content,
                success=False,
                missing_sections=input.focus_areas or list(DocumentationSection)
            )
    
    async def get_documentation(
        self, 
        solution: SolutionCandidate,
        project_context: ProjectContext,
        focus_areas: Optional[List[DocumentationSection]] = None
    ) -> DocumentationContent:
        """Convenience method to get documentation for a solution."""
        input_data = DocumentationInput(
            solution=solution,
            project_context=project_context,
            focus_areas=focus_areas
        )
        
        result = await self.run(input_data)
        return result.content if result.success else None 