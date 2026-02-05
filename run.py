#!/usr/bin/env python3
"""
Run script for Pattern Recognition Engine.
"""

import sys
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function."""
    print("Pattern Recognition Engine - Run Script")
    print("Use 'python start.py' for simple startup")
    print("Or 'python main.py' for detailed logging")
    return 0

if __name__ == "__main__":
    sys.exit(main())

