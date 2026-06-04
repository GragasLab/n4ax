# mypackage

TODO: one-line description.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
# CPU only
uv sync --extra cpu

# CUDA 12
uv sync --extra cuda12

# Development (linting, tests)
uv sync --extra dev

# Combine extras
uv sync --extra cuda12 --extra dev
```

## Running tests

```bash
uv run pytest
```

## Linting

```bash
uv run ruff check mypackage/ tests/
uv run ruff format mypackage/ tests/
```

Pre-commit hooks run ruff automatically on staged files (one-time setup, requires dev extras):

```bash
uv sync --extra dev
uv run pre-commit install
```

## Project conventions

### Single source of truth

- All config lives in `pyproject.toml`: project metadata, dependencies, ruff, pytest.
- No `setup.py`, no `ruff.toml`, no `setup.cfg`.

### Dependency management

- `uv` for everything: `uv sync`, `uv run`, `uv build`.
- Core dependencies in `[project.dependencies]`.
- Optional extras: `cpu`, `cuda12`, `dev`.
- No `uv.lock` in git (library pattern — consumers manage their own).

### Code style

- Ruff for both linting and formatting (configured in `pyproject.toml`).
- Line length: 119 characters.
- Double quotes, space indentation.
- Imports sorted by isort with 2 blank lines after import blocks.
- Pre-commit hooks enforce ruff on every commit.

### Testing

- Tests live in `tests/` with `test_*.py` naming.
- Run with `uv run pytest`.
- CI runs on every push/PR to `main` (GitHub Actions).

### CI

- GitHub Actions workflow in `.github/workflows/ci.yml`.
- Lint (ruff check + format check) then test, then build + twine check.
- Python version matrix matches `pyproject.toml` classifiers.
