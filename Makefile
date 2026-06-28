.PHONY: setup tests lint format

setup:
	uv sync --extra dev

tests:
	uv run pytest

lint:
	uv run prek run --all-files

format:
	uv run ruff format src tests examples
