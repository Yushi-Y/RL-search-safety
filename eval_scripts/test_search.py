#!/usr/bin/env python3
"""
Test runner for search.py - runs only the test cases without main evaluation
"""

import sys
import os

# Add the current directory to path so we can import from search.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from search import test_search_evaluation

if __name__ == "__main__":
    print("Running search.py test cases only...")
    test_search_evaluation()
    print("\nTest completed!")