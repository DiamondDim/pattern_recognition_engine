"""
Модуль для визуализации паттернов и финансовых данных
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Исправляем импорт: используем config.config вместо config
try:
    from config import config
    VISUALIZATION_CONFIG = config.VISUALIZATION
except ImportError:
    # Для обратной совместимости
    from config import VISUALIZATION_CONFIG


class PatternVisualizer:
    """Класс для визуализации паттернов на графиках"""

    def __init__(self):
        try:
            self.config = config.VISUALIZATION
        except:
            self.config = VISUALIZATION_CONFIG
        self.colors = self.config.COLORS

    def plot_patterns(self, data: pd.DataFrame, patterns: List[Dict],
                     symbol: str, timeframe: str, save_path: Optional[str] = None):
        """
        Построение графика с обнаруженными паттернами

        Args:
            data: DataFrame с данными OHLC
            patterns: Список обнаруженных паттернов
            symbol: Название символа
            timeframe: Таймфрейм
            save_path: Путь для сохранения графика
        """
        if not self.config.ENABLE_PLOTTING:
            print("Визуализация отключена в конфигурации")
            return

        # Создаем график
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(self.config.FIGURE_WIDTH, self.config.FIGURE_HEIGHT),
            gridspec_kw={'height_ratios': self.config.SUBPLOT_HEIGHT_RATIO}
        )

        # Верхний график: цены
        ax1.set_title(f'{symbol} - {timeframe} - Паттерны')

        # Рисуем свечи
        self._plot_candlesticks(ax1, data)

        # Отмечаем паттерны
        for pattern in patterns:
            self._plot_pattern(ax1, data, pattern)

        # Нижний график: объемы
        if 'volume' in data.columns:
            ax2.bar(data.index, data['volume'], color=self.colors['neutral'], alpha=0.5)
            ax2.set_ylabel('Объем')

        # Настройка осей
        ax1.set_ylabel('Цена')
        ax1.grid(True, linestyle='--', alpha=0.3, color=self.colors['grid'])
        ax2.grid(True, linestyle='--', alpha=0.3, color=self.colors['grid'])

        # Форматирование дат
        fig.autofmt_xdate()

        plt.tight_layout()

        # Сохранение или отображение
        if save_path:
            plt.savefig(save_path, dpi=self.config.PLOT_DPI, format=self.config.PLOT_FORMAT)
            print(f"График сохранен: {save_path}")

        if self.config.INTERACTIVE_PLOTS:
            plt.show()
        else:
            plt.close(fig)

    def _plot_candlesticks(self, ax, data: pd.DataFrame):
        """Рисование свечей"""
        # Простая реализация свечей
        for idx, row in data.iterrows():
            color = self.colors['bullish'] if row['close'] >= row['open'] else self.colors['bearish']

            # Тело свечи
            ax.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=0.5)

            # Тень
            ax.bar(idx, height=abs(row['close'] - row['open']),
                  bottom=min(row['open'], row['close']),
                  width=0.6, color=color)

    def _plot_pattern(self, ax, data: pd.DataFrame, pattern: Dict):
        """Отображение паттерна на графике"""
        pattern_type = pattern.get('type', 'unknown')
        pattern_name = pattern.get('name', 'Unknown')

        # Определяем цвет в зависимости от типа паттерна
        if 'bull' in pattern_name.lower() or pattern.get('direction') == 'bullish':
            color = self.colors['bullish']
        elif 'bear' in pattern_name.lower() or pattern.get('direction') == 'bearish':
            color = self.colors['bearish']
        else:
            color = self.colors['neutral']

        # Отмечаем точку паттерна
        if 'timestamp' in pattern:
            try:
                timestamp = pd.to_datetime(pattern['timestamp'])
                if timestamp in data.index:
                    price = data.loc[timestamp, 'close']
                    ax.scatter(timestamp, price, color=color, s=100,
                             label=f"{pattern_name} ({pattern.get('quality', 0):.0%})")
            except:
                pass

    def plot_interactive(self, data: pd.DataFrame, patterns: List[Dict],
                        symbol: str, timeframe: str):
        """Интерактивный график с Plotly"""
        if not self.config.INTERACTIVE_PLOTS:
            print("Интерактивные графики отключены в конфигурации")
            return

        # Создаем фигуру
        fig = make_subplots(rows=2, cols=1,
                           row_heights=[0.7, 0.3],
                           shared_xaxes=True,
                           vertical_spacing=0.03)

        # Добавляем свечи
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Цены'
        ), row=1, col=1)

        # Добавляем паттерны
        for pattern in patterns:
            if 'timestamp' in pattern:
                try:
                    timestamp = pd.to_datetime(pattern['timestamp'])
                    if timestamp in data.index:
                        price = data.loc[timestamp, 'close']

                        fig.add_trace(go.Scatter(
                            x=[timestamp],
                            y=[price],
                            mode='markers',
                            marker=dict(
                                size=12,
                                color=self.colors['bullish'] if 'bull' in pattern.get('name', '').lower() else self.colors['bearish']
                            ),
                            name=f"{pattern['name']} ({pattern.get('quality', 0):.0%})",
                            text=f"{pattern['name']}<br>Качество: {pattern.get('quality', 0):.2%}<br>Цена: {price:.5f}",
                            hoverinfo='text'
                        ), row=1, col=1)
                except:
                    pass

        # Добавляем объемы
        if 'volume' in data.columns:
            colors = ['green' if data['close'].iloc[i] >= data['open'].iloc[i]
                     else 'red' for i in range(len(data))]

            fig.add_trace(go.Bar(
                x=data.index,
                y=data['volume'],
                name='Объем',
                marker_color=colors
            ), row=2, col=1)

        # Настройка layout
        fig.update_layout(
            title=f"{symbol} - {timeframe}",
            yaxis_title="Цена",
            yaxis2_title="Объем",
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=True
        )

        fig.show()

    def plot_statistics(self, statistics: Dict, title: str = "Статистика"):
        """Визуализация статистических данных"""
        if not self.config.ENABLE_PLOTTING:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Распределение паттернов по типам
        if 'pattern_types' in statistics:
            pattern_types = statistics['pattern_types']
            axes[0, 0].pie(pattern_types.values(), labels=pattern_types.keys(),
                          autopct='%1.1f%%')
            axes[0, 0].set_title('Распределение паттернов по типам')

        # 2. Качество паттернов
        if 'qualities' in statistics:
            qualities = statistics['qualities']
            axes[0, 1].hist(qualities, bins=20, alpha=0.7, color=self.colors['neutral'])
            axes[0, 1].set_title('Распределение качества паттернов')
            axes[0, 1].set_xlabel('Качество')
            axes[0, 1].set_ylabel('Количество')

        # 3. Временное распределение
        if 'timestamps' in statistics:
            timestamps = statistics['timestamps']
            axes[1, 0].plot(timestamps, range(len(timestamps)),
                           marker='o', color=self.colors['bullish'])
            axes[1, 0].set_title('Обнаружение паттернов во времени')
            axes[1, 0].set_xlabel('Время')
            axes[1, 0].set_ylabel('Кумулятивное количество')

        # 4. По символам
        if 'symbols' in statistics:
            symbols = statistics['symbols']
            axes[1, 1].bar(symbols.keys(), symbols.values(),
                          color=self.colors['neutral'])
            axes[1, 1].set_title('Паттерны по символам')
            axes[1, 1].set_xlabel('Символ')
            axes[1, 1].set_ylabel('Количество')

        plt.suptitle(title)
        plt.tight_layout()

        if self.config.SAVE_PLOTS:
            save_path = f"{title.replace(' ', '_')}.{self.config.PLOT_FORMAT}"
            plt.savefig(save_path, dpi=self.config.PLOT_DPI,
                       format=self.config.PLOT_FORMAT)

        if self.config.INTERACTIVE_PLOTS:
            plt.show()
        else:
            plt.close(fig)


# Создаем глобальный экземпляр для обратной совместимости
visualizer = PatternVisualizer()


# Функции для обратной совместимости (если другие файлы импортируют их напрямую)
def plot_patterns(data: pd.DataFrame, patterns: List[Dict],
                 symbol: str, timeframe: str, save_path: Optional[str] = None):
    """Функция для быстрой визуализации паттернов"""
    return visualizer.plot_patterns(data, patterns, symbol, timeframe, save_path)


def plot_interactive(data: pd.DataFrame, patterns: List[Dict],
                    symbol: str, timeframe: str):
    """Функция для интерактивной визуализации"""
    return visualizer.plot_interactive(data, patterns, symbol, timeframe)


def plot_statistics(statistics: Dict, title: str = "Статистика"):
    """Функция для визуализации статистики"""
    return visualizer.plot_statistics(statistics, title)

