# Talon-AI-Tools Development Guide

## Commands

- **Linting**: `black .` and `isort .` (configured in pyproject.toml)
- **Type checking**: `pyright .`
- **Testing**: No test suite found, use manual testing with Talon

## Code Style Guidelines

- **Imports**: Follow `isort` with Black profile (group imports, stdlib first)
- **Formatting**: Black compatible (line length 88, double quotes)
- **Types**: Use type hints for all functions, specify Optional types
- **Naming**:
  - Functions: snake_case (e.g., `send_request`)
  - Classes: PascalCase (e.g., `GPTState`)
  - Constants: UPPER_SNAKE_CASE
  - Variables: snake_case
- **Error handling**: Use try/except with specific error types, notify user with `notify()`
- **Documentation**: Docstrings for all functions and classes
- **Structure**:
  - Separate core functionality into lib/ modules
  - Each tool in its own directory (GPT/, Images/, copilot/)
  - Use .talon files for voice commands
  - Avoid side effects when possible

## Project Organization

This repository integrates Talon Voice with AI tools like OpenAI API and GitHub Copilot.
Use module-specific directories for implementation and corresponding .talon files for commands.
