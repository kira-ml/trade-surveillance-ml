"""
conftest.py — root-level pytest configuration.
Adds the project root to sys.path so `src.*` imports resolve without
installing the package.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
