# Talon-AI-Tools Development Guide

## Commands

- **Linting**: `ruff format && ruff check --fix && pre-commit run --files` followed by a list of changed files.
- **Type checking**: `pyright .`
- **Testing** (this should be done after linting and type checking):
  - Push changes to local Talon user directory with `sync_talon_repo`.
  - Wait a couple seconds for Talon to load the changes.
  - Use `tail /mnt/c/Users/james/AppData/Roaming/talon/talon.log` to view recent logs (adding flags as needed to view more logs).
  - Changed files will show up in logs as `DEBUG [~] c:\path\to\file`, with possible `WARNING` or `ERROR` lines shown afterwards.
  - The user will need to manually test any changed functionality.
- **Committing**: Always run `sync_talon_repo` after committing.

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

- This repository integrates Talon Voice with AI tools like OpenAI API and GitHub Copilot
- Use module-specific directories for implementation and corresponding .talon files for voice commands
- .talon files refer to Talon lists using braces and Talon captures using angle brackets. Both should be declared in lib/talonSettings.py. Talon lists should be populated via .talon-list files using `key: value` format.
- Settings should be declared in lib/talonSettings.py, with an example per setting in talon-ai-settings.talon.example

## Talon documentation

@~/.claude/talon.md
