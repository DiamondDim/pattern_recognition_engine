"""
Визуализация паттернов и данных
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from config import VISUALIZATION_CONFIG
from utils.logger import logger


class PatternVisualizer:
    """Класс для визуализации паттернов"""

    def __init__(self, config: VISUALIZATION_CONFIG = None):
        self.config = config or VISUALIZATION_CONFIG
        self.logger = logger.bind(name="PatternVisualizer")

        # Цветовая схема
        self.colors = self.config.COLORS

        # Настройки matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.figsize'] = self.config.FIGURE_SIZE
        plt.rcParams['font.size'] = self.config.FONT_SIZE

    def plot_pattern(self,
                     pattern: Dict[str, Any],
                     ohlc_data: Dict[str, np.ndarray],
                     indicators: Optional[Dict[str, np.ndarray]] = None,
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Построение графика с паттерном

        Args:
            pattern: Данные паттерна
            ohlc_data: Данные OHLC
            indicators: Технические индикаторы
            save_path: Путь для сохранения

        Returns:
            Объект Figure matplotlib
        """
        try:
            # Извлекаем данные
            opens = ohlc_data.get('open', np.array([]))
            highs = ohlc_data.get('high', np.array([]))
            lows = ohlc_data.get('low', np.array([]))
            closes = ohlc_data.get('close', np.array([]))
            volumes = ohlc_data.get('volume', np.array([]))
            timestamps = ohlc_data.get('timestamp', np.arange(len(closes)))

            if len(closes) == 0:
                self.logger.error("Нет данных для построения графика")
                return None

            # Создаем график
            fig, axes = plt.subplots(3, 1, figsize=self.config.FIGURE_SIZE,
                                     gridspec_kw={'height_ratios': [3, 1, 1]})

            # Основной график цен
            ax_price = axes[0]
            ax_volume = axes[1]
            ax_indicators = axes[2]

            # Настройка осей
            self._setup_price_axis(ax_price, pattern, timestamps)

            # Рисуем свечи
            self._plot_candlesticks(ax_price, opens, highs, lows, closes, timestamps)

            # Рисуем паттерн
            self._plot_pattern_on_chart(ax_price, pattern, timestamps)

            # Рисуем объемы
            if len(volumes) > 0:
                self._plot_volume(ax_volume, closes, volumes, timestamps)

            # Рисуем индикаторы
            if indicators:
                self._plot_indicators(ax_indicators, closes, indicators, timestamps)

            # Добавляем заголовок и легенду
            self._add_chart_title(fig, pattern, ax_price)

            # Настраиваем layout
            plt.tight_layout()

            # Сохраняем если указан путь
            if save_path:
                plt.savefig(save_path, dpi=self.config.PLOT_DPI,
                            bbox_inches='tight')
                self.logger.debug(f"График сохранен: {save_path}")

            return fig

        except Exception as e:
            self.logger.error(f"Ошибка построения графика: {e}")
            return None

    def _setup_price_axis(self, ax, pattern, timestamps):
        """Настройка оси цен"""
        pattern_name = pattern.get('name', 'Unknown')
        direction = pattern.get('direction', 'neutral')
        symbol = pattern.get('metadata', {}).get('symbol', 'UNKNOWN')
        timeframe = pattern.get('metadata', {}).get('timeframe', 'UNKNOWN')

        # Заголовок
        title_color = self.colors['bullish'] if direction == 'bullish' else self.colors['bearish']
        ax.set_title(f"{symbol} - {timeframe} - {pattern_name} ({direction})",
                     color=title_color, fontweight='bold')

        # Метки осей
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)

        # Настройка временной оси
        if len(timestamps) > 0:
            if isinstance(timestamps[0], (datetime, np.datetime64)):
                # Для временных меток форматируем ось X
                ax.xaxis_date()
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_candlesticks(self, ax, opens, highs, lows, closes, timestamps):
        """Рисование свечей"""
        # Определяем цвета свечей
        colors = np.where(closes >= opens, self.colors['bullish'], self.colors['bearish'])

        # Ширина свечей (в единицах времени)
        width = 0.8

        # Рисуем тела свечей
        for i in range(len(opens)):
            # Тело свечи
            body_top = max(opens[i], closes[i])
            body_bottom = min(opens[i], closes[i])
            body_height = body_top - body_bottom

            if body_height > 0:
                rect = patches.Rectangle(
                    (i - width / 2, body_bottom),
                    width,
                    body_height,
                    facecolor=colors[i],
                    edgecolor=colors[i],
                    alpha=0.8
                )
                ax.add_patch(rect)

            # Тени
            ax.plot([i, i], [lows[i], body_bottom],
                    color=colors[i], linewidth=1, alpha=0.6)
            ax.plot([i, i], [body_top, highs[i]],
                    color=colors[i], linewidth=1, alpha=0.6)

        # Линия закрытий для наглядности
        ax.plot(closes, color='blue', linewidth=0.5, alpha=0.5, label='Close')

        # Устанавливаем лимиты оси Y
        price_min = np.min(lows) if len(lows) > 0 else 0
        price_max = np.max(highs) if len(highs) > 0 else 1
        margin = (price_max - price_min) * 0.05
        ax.set_ylim(price_min - margin, price_max + margin)

    def _plot_pattern_on_chart(self, ax, pattern, timestamps):
        """Рисование паттерна на графике"""
        points = pattern.get('points', [])
        targets = pattern.get('targets', {})
        direction = pattern.get('direction', 'neutral')

        if not points:
            return

        # Рисуем точки паттерна
        for point in points:
            idx = point.get('index', 0)
            price = point.get('price', 0)
            point_type = point.get('point_type', '')

            # Определяем цвет точки
            if 'shoulder' in point_type:
                color = 'orange'
                marker = '^'
                size = 80
            elif 'head' in point_type:
                color = 'red'
                marker = 'v'
                size = 100
            elif 'neckline' in point_type:
                color = 'blue'
                marker = 'o'
                size = 60
            elif 'top' in point_type or 'high' in point_type:
                color = self.colors['resistance']
                marker = 'v'
                size = 70
            elif 'bottom' in point_type or 'low' in point_type:
                color = self.colors['support']
                marker = '^'
                size = 70
            else:
                color = 'gray'
                marker = 'o'
                size = 50

            ax.scatter(idx, price, color=color, marker=marker,
                       s=size, zorder=5, label=f'{point_type}')

        # Рисуем линии паттерна (если есть достаточно точек)
        if len(points) >= 2:
            # Сортируем точки по индексу
            sorted_points = sorted(points, key=lambda p: p.get('index', 0))
            indices = [p.get('index', 0) for p in sorted_points]
            prices = [p.get('price', 0) for p in sorted_points]

            # Рисуем линии между точками
            ax.plot(indices, prices, color='purple', linewidth=2,
                    alpha=0.6, linestyle='--', label='Pattern lines')

        # Рисуем целевые уровни
        entry_price = targets.get('entry_price')
        stop_loss = targets.get('stop_loss')
        take_profit = targets.get('take_profit')

        if entry_price:
            ax.axhline(y=entry_price, color=self.colors['entry'],
                       linewidth=2, linestyle='-', alpha=0.7,
                       label=f'Entry: {entry_price:.4f}')

        if stop_loss:
            ax.axhline(y=stop_loss, color=self.colors['stop_loss'],
                       linewidth=2, linestyle='--', alpha=0.7,
                       label=f'Stop: {stop_loss:.4f}')

        if take_profit:
            ax.axhline(y=take_profit, color=self.colors['take_profit'],
                       linewidth=2, linestyle='-.', alpha=0.7,
                       label=f'TP: {take_profit:.4f}')

        # Добавляем зону риска/прибыли
        if entry_price and stop_loss and take_profit:
            if direction == 'bullish':
                # Зона риска (ниже входа)
                ax.axhspan(stop_loss, entry_price,
                           facecolor=self.colors['bearish'], alpha=0.1)
                # Зона прибыли (выше входа)
                ax.axhspan(entry_price, take_profit,
                           facecolor=self.colors['bullish'], alpha=0.1)
            else:
                # Зона риска (выше входа)
                ax.axhspan(entry_price, stop_loss,
                           facecolor=self.colors['bearish'], alpha=0.1)
                # Зона прибыли (ниже входа)
                ax.axhspan(take_profit, entry_price,
                           facecolor=self.colors['bullish'], alpha=0.1)

    def _plot_volume(self, ax, closes, volumes, timestamps):
        """Рисование объемов"""
        # Цвета объемов (зеленый для бычьих свечей, красный для медвежьих)
        volume_colors = np.where(
            np.diff(np.concatenate(([closes[0]], closes))) >= 0,
            self.colors['bullish'],
            self.colors['bearish']
        )

        # Рисуем объемы
        ax.bar(range(len(volumes)), volumes, color=volume_colors, alpha=0.7)

        # Настройка оси объемов
        ax.set_ylabel('Volume')
        ax.grid(True, alpha=0.3)

        # Скрываем метки оси X для объемов
        ax.set_xticklabels([])

    def _plot_indicators(self, ax, closes, indicators, timestamps):
        """Рисование индикаторов"""
        # RSI
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            ax.plot(rsi, color='purple', linewidth=1.5, label='RSI')
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax.set_ylim(0, 100)
            ax.set_ylabel('RSI')

        # MACD
        elif 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd']
            macd_signal = indicators['macd_signal']

            ax.plot(macd, color='blue', linewidth=1.5, label='MACD')
            ax.plot(macd_signal, color='red', linewidth=1.5, label='Signal')

            # Гистограмма MACD
            if 'macd_hist' in indicators:
                macd_hist = indicators['macd_hist']
                colors_hist = np.where(macd_hist >= 0,
                                       self.colors['bullish'],
                                       self.colors['bearish'])
                ax.bar(range(len(macd_hist)), macd_hist,
                       color=colors_hist, alpha=0.5, width=0.8)

            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_ylabel('MACD')

        # Просто цена закрытия, если нет индикаторов
        else:
            ax.plot(closes, color='blue', linewidth=1, label='Close')
            ax.set_ylabel('Price')

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

        # Скрываем метки оси X для индикаторов
        ax.set_xticklabels([])

    def _add_chart_title(self, fig, pattern, ax_price):
        """Добавление заголовка и информации"""
        metadata = pattern.get('metadata', {})
        targets = pattern.get('targets', {})
        statistics = pattern.get('statistics', {})
        strength = pattern.get('strength_analysis', {})

        # Информация о качестве
        quality = metadata.get('quality_score', 0)
        confidence = metadata.get('confidence', 0)
        market_context = metadata.get('market_context', 'neutral')

        # Целевые уровни
        entry = targets.get('entry_price')
        stop_loss = targets.get('stop_loss')
        take_profit = targets.get('take_profit')
        risk_reward = targets.get('profit_risk_ratio')

        # Статистика
        hist_matches = statistics.get('historical_matches', 0)
        hist_success = statistics.get('historical_success_rate', 0)

        # Текст с информацией
        info_text = (
            f"Quality: {quality:.2f} | Confidence: {confidence:.2f} | "
            f"Market: {market_context}\n"
        )

        if entry and stop_loss and take_profit:
            info_text += (
                f"Entry: {entry:.4f} | Stop: {stop_loss:.4f} | "
                f"TP: {take_profit:.4f} | R/R: {risk_reward:.2f}\n"
            )

        if hist_matches > 0:
            info_text += (
                f"Historical: {hist_matches} matches, "
                f"{hist_success:.1%} success rate\n"
            )

        if strength:
            total_score = strength.get('total_score', 0)
            info_text += f"Strength score: {total_score:.2f}"

        # Добавляем текст на график
        fig.text(0.02, 0.02, info_text, fontsize=9,
                 verticalalignment='bottom',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    def create_interactive_chart(self,
                                pattern: Dict[str, Any],
                                ohlc_data: Dict[str, np.ndarray],
                                indicators: Optional[Dict[str, np.ndarray]] = None) -> go.Figure:
        """
        Создание интерактивного графика с использованием Plotly

        Args:
            pattern: Данные паттерна
            ohlc_data: Данные OHLC
            indicators: Технические индикаторы

        Returns:
            Интерактивный график Plotly
        """
        try:
            # Извлекаем данные
            opens = ohlc_data.get('open', np.array([]))
            highs = ohlc_data.get('high', np.array([]))
            lows = ohlc_data.get('low', np.array([]))
            closes = ohlc_data.get('close', np.array([]))
            volumes = ohlc_data.get('volume', np.array([]))
            timestamps = ohlc_data.get('timestamp', np.arange(len(closes)))

            if len(closes) == 0:
                self.logger.error("Нет данных для построения интерактивного графика")
                return None

            # Создаем подграфики
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('Цены', 'Объемы', 'Индикаторы'),
                row_width=[0.6, 0.2, 0.2]
            )

            # Добавляем свечи
            fig.add_trace(
                go.Candlestick(
                    x=timestamps,
                    open=opens,
                    high=highs,
                    low=lows,
                    close=closes,
                    name='OHLC'
                ),
                row=1, col=1
            )

            # Добавляем точки паттерна
            points = pattern.get('points', [])
            for point in points:
                idx = point.get('index', 0)
                price = point.get('price', 0)
                point_type = point.get('point_type', '')

                fig.add_trace(
                    go.Scatter(
                        x=[timestamps[idx] if idx < len(timestamps) else timestamps[-1]],
                        y=[price],
                        mode='markers+text',
                        name=point_type,
                        marker=dict(size=10, symbol='diamond'),
                        text=[point_type],
                        textposition="top center"
                    ),
                    row=1, col=1
                )

            # Добавляем объемы
            if len(volumes) > 0:
                colors_volume = ['green' if closes[i] >= opens[i] else 'red'
                               for i in range(len(closes))]

                fig.add_trace(
                    go.Bar(
                        x=timestamps,
                        y=volumes,
                        name='Volume',
                        marker_color=colors_volume,
                        opacity=0.7
                    ),
                    row=2, col=1
                )

            # Добавляем индикаторы
            if indicators:
                if 'rsi' in indicators:
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=indicators['rsi'],
                            name='RSI',
                            line=dict(color='purple', width=2)
                        ),
                        row=3, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            # Настраиваем layout
            pattern_name = pattern.get('name', 'Unknown')
            direction = pattern.get('direction', 'neutral')
            symbol = pattern.get('metadata', {}).get('symbol', 'UNKNOWN')
            timeframe = pattern.get('metadata', {}).get('timeframe', 'UNKNOWN')

            fig.update_layout(
                title=f"{symbol} - {timeframe} - {pattern_name} ({direction})",
                xaxis_title="Дата",
                yaxis_title="Цена",
                template="plotly_dark",
                height=800,
                showlegend=True
            )

            # Отключаем режим масштабирования по диапазону
            fig.update_xaxes(rangeslider_visible=False)

            return fig

        except Exception as e:
            self.logger.error(f"Ошибка создания интерактивного графика: {e}")
            return None

