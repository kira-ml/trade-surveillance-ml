"""
conftest.py — root-level pytest configuration.
Adds the project root to sys.path so `src.*` imports resolve without
installing the package.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (use -m 'not integration' to skip)",
    )
