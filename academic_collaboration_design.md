# Multi-Agent Academic Collaboration System Design

## 1. Project Overview
**Project Name**: `academic-collaboration-agents`
**Description**: A multi-agent system simulating the academic research and publication process. Agents collaborate to generate research ideas, write papers, perform peer reviews, and make editorial decisions.

## 2. Configuration (`pyproject.toml`)
Based on the `werewolf` project structure, here is the recommended configuration:

```toml
[project]
name = "academic-collaboration-agents"
version = "0.1.0"
description = "Multi-agent system for simulating academic research and review process"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "openai>=1.0.0",
    "python-dotenv>=1.0.0",
    "colorama>=0.4.6",
    "langchain>=0.1.0",  # Added for better chain management
    "chromadb>=0.4.0",   # For literature retrieval (RAG)
    "pandas>=2.0.0",     # For data analysis simulation
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "jupyter>=1.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.black]
line-length = 100
target-version = ['py39']

[tool.ruff]
line-length = 100
target-version = "py39"
```

## 3. Project Structure
```
academic_agents/
├── .env                    # API keys and config
├── pyproject.toml          # Project configuration
├── README.md              # Documentation
├── main.py                # Entry point
├── src/
│   ├── __init__.py
│   ├── config.py          # Global configuration
│   ├── agents/            # Agent definitions
│   │   ├── __init__.py
│   │   ├── base_agent.py  # Base class for all agents
│   │   ├── researcher.py  # Writes papers, runs experiments
│   │   ├── reviewer.py    # Critiques papers
│   │   └── editor.py      # Manages process, makes final decisions
│   ├── environment/       # Simulation environment
│   │   ├── __init__.py
│   │   ├── journal.py     # Manages submissions and issues
│   │   └── library.py     # Knowledge base (RAG system)
│   └── utils/
│       ├── llm_client.py  # Wrapper for API calls
│       └── logger.py      # Logging system
└── experiments/           # Output directory for generated papers/reviews
```

## 4. Agent Roles & Workflow

### Roles
1.  **Researcher (Author)**:
    *   **Goal**: Publish high-quality papers.
    *   **Actions**: Propose topics, conduct literature review (querying `library`), draft content, revise based on feedback.
    *   **Memory**: Keeps track of own research history and feedback received.

2.  **Reviewer**:
    *   **Goal**: Ensure scientific rigor and quality.
    *   **Actions**: Read submissions, identify flaws, rate novelty/methodology, write review reports.
    *   **Personality**: Can be configured as "Constructive", "Critical", or "Lazy".

3.  **Editor (Chair)**:
    *   **Goal**: Curate a high-impact journal issue.
    *   **Actions**: Assign reviewers, weigh reviews, make Accept/Reject decisions, write editorial notes.

### Workflow (Simulation Loop)
1.  **Initialization**: Load config, initialize agents with specific domains (e.g., "NLP", "Computer Vision").
2.  **Phase 1: Drafting**: Researcher agents generate abstracts and drafts.
3.  **Phase 2: Submission**: Drafts are submitted to the `Journal` environment.
4.  **Phase 3: Review**: Editor assigns 2-3 Reviewers to each paper. Reviewers generate reports.
5.  **Phase 4: Rebuttal (Optional)**: Researchers read reviews and provide a rebuttal or revision plan.
6.  **Phase 5: Decision**: Editor makes final decision based on reviews and rebuttals.
7.  **Phase 6: Publication**: Accepted papers are added to the "Proceedings".

## 5. Key Features to Implement
*   **RAG Integration**: Use `chromadb` to allow Researchers to cite real papers or previous simulation outputs.
*   **Iterative Refinement**: Allow multi-round review process (Revise & Resubmit).
*   **Citation Network**: Track how agents cite each other's work over multiple simulation epochs.
