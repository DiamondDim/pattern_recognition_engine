"""
Упрощенный скрипт для запуска Pattern Recognition Engine
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent))

from main import PatternRecognitionEngine
from utils.logger import logger


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Pattern Recognition Engine')

    parser.add_argument('--mode', choices=['console', 'file', 'socket'],
                        default='console', help='Режим работы')

    parser.add_argument('--symbol', type=str, default='EURUSD',
                        help='Символ для анализа')

    parser.add_argument('--timeframe', type=str, default='H1',
                        help='Таймфрейм для анализа')

    parser.add_argument('--input', type=str,
                        help='Входной файл с данными (CSV)')

    parser.add_argument('--output', type=str,
                        help='Выходной файл для результатов')

    parser.add_argument('--single', action='store_true',
                        help='Однократный анализ файла')

    parser.add_argument('--visualize', action='store_true',
                        help='Визуализация результатов')

    parser.add_argument('--debug', action='store_true',
                        help='Режим отладки')

    return parser.parse_args()


async def run_single_file_analysis(input_file: str, output_file: str = None):
    """
    Однократный анализ файла с данными

    Args:
        input_file: Входной файл с данными
        output_file: Выходной файл для результатов
    """
    logger.info(f"Анализ файла: {input_file}")

    # TODO: Реализовать однократный анализ файла
    # Здесь можно использовать компоненты движка для анализа
    # без запуска полного цикла

    logger.info("Однократный анализ завершен")


async def main():
    """Основная функция запуска"""
    args = parse_args()

    logger.info(f"Запуск Pattern Recognition Engine в режиме: {args.mode}")

    if args.single and args.input:
        # Однократный анализ файла
        await run_single_file_analysis(args.input, args.output)

    else:
        # Полный запуск движка
        engine = PatternRecognitionEngine()

        # Устанавливаем параметры из аргументов
        if args.symbol:
            engine.current_symbol = args.symbol
        if args.timeframe:
            engine.current_timeframe = args.timeframe

        try:
            await engine.run()
        except KeyboardInterrupt:
            logger.info("Работа прервана пользователем")
        except Exception as e:
            logger.critical(f"Критическая ошибка: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())

