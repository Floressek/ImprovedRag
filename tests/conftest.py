"""Pytest configuration (placeholder)."""
import os
import sys

# Ensure src is on path for imports
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(ROOT), 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
