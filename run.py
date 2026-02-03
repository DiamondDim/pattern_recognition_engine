"""
Альтернативный скрипт запуска Pattern Recognition Engine
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в путь Python
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from main import main

if __name__ == "__main__":
    main()

