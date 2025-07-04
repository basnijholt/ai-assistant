# Contributing to Agent CLI

Thank you for contributing to Agent CLI! Please follow these development guidelines to ensure consistency and quality across the codebase.

## Development Rules

### 1. Package Management
- **Always use UV with --all-extras** for installation:
  ```bash
  uv pip install -e ".[dev]" --all-extras
  ```
- This ensures all optional dependencies are installed for development.

### 2. Testing & Commits
- **Commit frequently** but ensure tests pass before each commit
- **Always run the test suite** before committing:
  ```bash
  uv run pytest
  ```
- If tests fail, fix them before committing
- Use descriptive commit messages

### 3. Code Style
- **Prefer functional programming** over class-based inheritance
  - Use functions and pure functions when possible
  - Avoid complex class hierarchies
  - Favor composition over inheritance
  - Use dataclasses or Pydantic models for data structures

### 4. Code Quality
- **Keep it DRY** (Don't Repeat Yourself)
  - Extract common functionality into utility functions
  - Reuse code across modules
  - Create shared utilities for common patterns

### 5. Pre-commit Hooks
- **Always run pre-commit** before committing:
  ```bash
  uv run pre-commit run --all-files
  ```
- Pre-commit is already configured with:
  - Ruff for linting and formatting
  - MyPy for type checking
  - Basic file checks

## Development Workflow

1. **Setup environment:**
   ```bash
   uv pip install -e ".[dev]" --all-extras
   uv run pre-commit install
   ```

2. **Before making changes:**
   ```bash
   # Run tests to ensure everything is working
   uv run pytest
   ```

3. **During development:**
   - Write functional code where possible
   - Add tests for new functionality
   - Keep functions small and focused
   - Extract common patterns into utilities

4. **Before committing:**
   ```bash
   # Run pre-commit hooks
   uv run pre-commit run --all-files

   # Run tests
   uv run pytest

   # If everything passes, commit
   git commit -m "descriptive commit message"
   ```

## Code Examples

### ✅ Preferred Functional Style
```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class ProcessingResult:
    processed_items: List[str]
    errors: List[str]

def process_items(items: List[str]) -> ProcessingResult:
    """Process items functionally."""
    processed = []
    errors = []

    for item in items:
        try:
            processed.append(transform_item(item))
        except Exception as e:
            errors.append(str(e))

    return ProcessingResult(processed, errors)

def transform_item(item: str) -> str:
    """Transform a single item."""
    return item.upper().strip()
```

### ❌ Avoid Complex Inheritance
```python
# Avoid this pattern
class BaseProcessor:
    def process(self): pass

class TextProcessor(BaseProcessor):
    def process(self): pass

class AudioProcessor(BaseProcessor):
    def process(self): pass
```

## Testing Guidelines

- Write tests for all new functionality
- Use pytest fixtures for common setup
- Keep tests focused and isolated
- Use descriptive test names
- Test both success and failure cases

## Questions?

If you have questions about these guidelines or need clarification, please open an issue or discussion on the repository.
