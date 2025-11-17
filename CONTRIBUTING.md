# Contributing to Hermes

Thank you for your interest in contributing to Hermes, part of Project LOGOS!

## Development Setup

1. Fork and clone the repository
2. Set up your development environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Code Standards

### Style Guide

- Follow PEP 8 style guidelines
- Use Black for code formatting (line length: 88)
- Use Ruff for linting
- Use type hints for all functions
- Write docstrings for all public APIs

### Testing

- Write tests for all new features
- Maintain or improve code coverage
- Run the test suite before submitting:
```bash
pytest --cov=hermes
```

### Type Checking

All code should pass mypy type checking:
```bash
mypy src
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add/update tests as needed
4. Update documentation if needed
5. Run tests and linting:
```bash
pytest
black src tests
ruff check src tests
mypy src
```
6. Commit your changes with clear, descriptive messages
7. Push to your fork and submit a pull request

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

- Address review comments promptly
- Keep pull requests focused and atomic
- Squash commits if requested

## Reporting Issues

Use GitHub Issues to report bugs or request features. Include:

- Clear description of the issue
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Your environment (Python version, OS, etc.)

## OpenAPI Contract

Hermes follows the canonical OpenAPI specification from the [LOGOS meta repository](https://github.com/c-daly/logos/blob/main/contracts/hermes.openapi.yaml). Any API changes must:

1. Be backwards compatible, or
2. Include a version bump and migration path
3. Update the contract in the logos repo

## Questions?

- Check the [Project LOGOS documentation](https://github.com/c-daly/logos)
- Open a discussion in GitHub Discussions
- Reach out to the maintainers

Thank you for contributing!
