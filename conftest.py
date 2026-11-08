"""
Pytest configuration for EcoSort-AI.

Sets up the project root on sys.path so tests can import
project modules without installation.
"""
import sys
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).parent))

