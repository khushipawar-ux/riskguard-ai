## Repository Development Instructions

Build this project using a **professional, production-ready repository structure**. Do not put the entire implementation into one or two large files.

### 1. Repository Structure

Before writing substantial code:

* Analyze the project requirements and identify the major components.
* Design an appropriate folder and file structure.
* Create separate folders for logically different responsibilities.
* Keep files small, focused, modular, and maintainable.
* Follow separation of concerns.
* Do not create unnecessary files or folders just for the sake of structure.

Use a structure appropriate to the technology stack, for example:

```text
project-root/
├── src/
│   ├── api/
│   ├── core/
│   ├── services/
│   ├── models/
│   ├── repositories/
│   ├── utils/
│   └── config/
├── tests/
├── docs/
├── scripts/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt / pyproject.toml
└── ...
```

Adapt this structure to the actual project instead of blindly copying it.

### 2. MCP Protocol

Follow the **MCP (Model Context Protocol) architecture/specification** wherever MCP is part of the project.

* Clearly separate MCP server/client components.
* Keep tools, resources, prompts, transports, configuration, and business logic modular.
* Follow MCP naming and architectural conventions.
* Do not mix MCP protocol handling with core business logic.
* Use appropriate MCP SDKs rather than manually recreating protocol functionality when an official/standard SDK is available.
* Keep MCP-specific code isolated so the application can be maintained or extended easily.

### 3. Professional Code Quality

Write code as if this repository will be reviewed by a professional engineering team.

Follow:

* Clean Code principles
* SOLID principles where appropriate
* DRY
* Separation of concerns
* Meaningful naming
* Type hints/types where applicable
* Proper error handling
* Logging instead of unnecessary print statements
* Configuration through environment variables
* No hardcoded secrets, API keys, passwords, or credentials
* Reusable functions/classes
* Minimal duplication
* Clear interfaces between components

### 4. File Responsibility

Every file should have a clear responsibility.

Before adding code, ask:

> "Does this code belong in an existing module, or does it represent a genuinely separate responsibility?"

Avoid:

* `main.py` containing the entire application
* 1000+ line files when functionality can reasonably be separated
* duplicated utility functions
* business logic inside API routes/controllers
* database logic scattered throughout the application
* hardcoded configuration
* unnecessary abstractions

### 5. Development Workflow

Work incrementally:

1. Understand the requirements.
2. Inspect the existing repository.
3. Identify the required architecture.
4. Propose the folder/file structure.
5. Create the required directories and files.
6. Implement the core components.
7. Integrate the components.
8. Add error handling and validation.
9. Add tests for important functionality.
10. Update documentation.
11. Review the entire repository for consistency.
12. Remove unnecessary/duplicate/dead code.

Do not randomly create files while coding without considering the overall architecture.

### 6. Existing Repository

If files already exist:

* Inspect them before creating replacements.
* Reuse existing components when appropriate.
* Do not duplicate functionality.
* Refactor only when necessary.
* Preserve working functionality unless the requirement explicitly requires changing it.
* Maintain backward compatibility where practical.

### 7. Configuration & Secrets

Use environment variables for configuration.

Provide:

```text
.env.example
```

but NEVER commit:

```text
.env
API keys
tokens
passwords
credentials
private keys
```

Ensure `.gitignore` is properly configured.

### 8. Documentation

Maintain a professional `README.md` containing:

* Project overview
* Architecture
* Features
* Repository structure
* Installation
* Environment configuration
* Running the project
* Testing
* API/MCP usage where applicable
* Example usage
* Troubleshooting
* Contribution/development guidelines

Add documentation under `docs/` when the architecture or setup becomes sufficiently complex.

### 9. Testing

Create a `tests/` structure appropriate to the project.

Test:

* Core business logic
* Important utilities
* API endpoints
* MCP tools/resources where applicable
* Error cases
* Integration points

Do not create meaningless tests simply to increase coverage.

### 10. Dependency Management

Use the appropriate dependency-management system for the chosen stack.

* Avoid unnecessary dependencies.
* Prefer stable, well-maintained libraries.
* Do not introduce a dependency when the functionality can be implemented cleanly without one.
* Keep dependency versions reproducible where appropriate.

### 11. Before Finishing

Perform a repository-wide review.

Check for:

* Duplicate code
* Dead code
* Unused imports
* Incorrect paths
* Circular dependencies
* Hardcoded secrets
* Poor naming
* Oversized files
* Missing error handling
* Missing configuration
* Missing tests
* Incorrect MCP implementation
* Inconsistent architecture
* Documentation that no longer matches the code

The final repository should look like something that could be handed to another engineering team and maintained without needing the original developer to explain the entire codebase.

### Core Principle

**Think like a senior software architect before coding, and like a code reviewer after coding.**

Prioritize:

**Clean architecture → modularity → maintainability → security → testability → documentation → implementation speed.**

Do not sacrifice repository quality merely to produce code quickly.
