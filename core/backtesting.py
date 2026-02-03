"""
Модуль бэктестинга паттернов
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')

from config import config
from utils.logger import logger
from .pattern_detector import PatternDetector
from .pattern_analyzer import PatternAnalyzer


@dataclass
class BacktestResult:
    """Результат бэктестинга"""

    # Статистика
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Прибыль/убыток
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0

    # Показатели эффективности
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Риск/прибыль
    max_drawdown: float = 0.0
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None

    # Время
    avg_holding_period: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Детали сделок
    trades: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def expectancy(self) -> float:
        """Ожидаемая прибыль на сделку"""
        if self.total_trades == 0:
            return 0.0
        return (self.win_rate * self.avg_winning_trade) - ((1 - self.win_rate) * self.avg_losing_trade)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'statistics': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': self.win_rate,
                'total_profit': self.total_profit,
                'total_loss': self.total_loss,
                'net_profit': self.net_profit,
                'profit_factor': self.profit_factor,
                'avg_profit': self.avg_profit,
                'avg_loss': self.avg_loss,
                'avg_winning_trade': self.avg_winning_trade,
                'avg_losing_trade': self.avg_losing_trade,
                'largest_win': self.largest_win,
                'largest_loss': self.largest_loss,
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio,
                'expectancy': self.expectancy,
                'avg_holding_period': self.avg_holding_period,
                'max_consecutive_wins': self.max_consecutive_wins,
                'max_consecutive_losses': self.max_consecutive_losses
            },
            'trades': self.trades
        }


class PatternBacktester:
    """Класс для бэктестинга паттернов"""

    def __init__(self):
        self.logger = logger.bind(module="backtesting")
        self.detector = PatternDetector()
        self.analyzer = PatternAnalyzer()

        # Параметры бэктестинга
        self.initial_balance = config.BACKTESTING.INITIAL_BALANCE
        self.risk_per_trade = config.BACKTESTING.RISK_PER_TRADE
        self.commission = config.BACKTESTING.COMMISSION
        self.slippage = config.BACKTESTING.SLIPPAGE
        self.max_holding_period = config.BACKTESTING.MAX_HOLDING_PERIOD

    async def run_backtest(self,
                           data: Dict[str, np.ndarray],
                           symbol: str = "TEST",
                           timeframe: str = "H1",
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> BacktestResult:
        """
        Запуск бэктестинга на исторических данных

        Args:
            data: Исторические данные OHLC
            symbol: Символ инструмента
            timeframe: Таймфрейм
            start_date: Начальная дата для тестирования
            end_date: Конечная дата для тестирования

        Returns:
            Результат бэктестинга
        """
        result = BacktestResult()
        trades = []

        # Извлекаем данные
        timestamps = data.get('timestamp', np.arange(len(data.get('close', []))))
        closes = data.get('close', np.array([]))

        if len(closes) == 0:
            self.logger.error("Нет данных для бэктестинга")
            return result

        # Фильтрация по дате
        if start_date or end_date:
            filtered_indices = await self._filter_by_date(timestamps, start_date, end_date)
        else:
            filtered_indices = np.arange(len(closes))

        if len(filtered_indices) == 0:
            self.logger.error("Нет данных в указанном диапазоне дат")
            return result

        # Запускаем детектирование паттернов
        self.logger.info(f"Запуск бэктестинга на {len(filtered_indices)} свечах")

        # Для каждой точки данных (скользящее окно)
        window_size = 100
        for i in range(window_size, len(filtered_indices)):
            current_idx = filtered_indices[i]

            # Берем окно данных
            window_start = max(0, current_idx - window_size)
            window_end = current_idx

            window_data = {
                'open': data['open'][window_start:window_end],
                'high': data['high'][window_start:window_end],
                'low': data['low'][window_start:window_end],
                'close': data['close'][window_start:window_end],
                'volume': data.get('volume', np.ones(window_end - window_start))[window_start:window_end]
            }

            # Детектируем паттерны
            detection_result = await self.detector.detect_all_patterns(
                symbol=symbol,
                timeframe=timeframe,
                data=window_data
            )

            # Анализируем найденные паттерны
            for pattern in detection_result.patterns:
                # Проверяем, нужно ли входить в сделку
                if self._should_enter_trade(pattern, current_idx, closes):
                    # Создаем сделку
                    trade = await self._create_trade(
                        pattern=pattern,
                        entry_index=current_idx,
                        entry_price=closes[current_idx],
                        closes=closes
                    )

                    if trade:
                        trades.append(trade)

        # Анализируем сделки
        if trades:
            result = self._analyze_trades(trades)

        self.logger.info(f"Бэктестинг завершен. Сделок: {len(trades)}")
        return result

    async def _filter_by_date(self,
                              timestamps: np.ndarray,
                              start_date: Optional[datetime],
                              end_date: Optional[datetime]) -> np.ndarray:
        """Фильтрация данных по дате"""
        indices = []

        for i, ts in enumerate(timestamps):
            # Конвертируем timestamp в datetime если нужно
            if isinstance(ts, (datetime, pd.Timestamp)):
                dt = ts
            elif isinstance(ts, np.datetime64):
                dt = pd.Timestamp(ts)
            else:
                # Предполагаем, что это числовой индекс
                dt = datetime.fromtimestamp(ts)

            # Проверяем диапазон
            if start_date and dt < start_date:
                continue
            if end_date and dt > end_date:
                continue

            indices.append(i)

        return np.array(indices)

    def _should_enter_trade(self,
                            pattern: Dict[str, Any],
                            current_idx: int,
                            closes: np.ndarray) -> bool:
        """Определение, нужно ли входить в сделку на основе паттерна"""
        # Проверяем качество паттерна
        quality = pattern.get('metadata', {}).get('quality_score', 0)
        if quality < config.DETECTION.MIN_PATTERN_QUALITY:
            return False

        # Проверяем, что паттерн свежий (последние N свечей)
        points = pattern.get('points', [])
        if points:
            last_point_idx = max(p['index'] for p in points)
            if current_idx - last_point_idx > 10:  # Паттерн старше 10 свечей
                return False

        # Проверяем, что цена достигла точки входа
        entry_price = pattern.get('targets', {}).get('entry_price')
        if entry_price is None:
            return False

        current_price = closes[current_idx]

        # Для бычьего паттерна: цена выше точки входа
        if pattern.get('direction') == 'bullish':
            if current_price >= entry_price * 0.995:  # 0.5% ниже точки входа
                return True

        # Для медвежьего паттерна: цена ниже точки входа
        elif pattern.get('direction') == 'bearish':
            if current_price <= entry_price * 1.005:  # 0.5% выше точки входа
                return True

        return False

    async def _create_trade(self,
                            pattern: Dict[str, Any],
                            entry_index: int,
                            entry_price: float,
                            closes: np.ndarray) -> Optional[Dict[str, Any]]:
        """Создание сделки на основе паттерна"""
        try:
            targets = pattern.get('targets', {})
            stop_loss = targets.get('stop_loss')
            take_profit = targets.get('take_profit')

            if stop_loss is None or take_profit is None:
                return None

            # Определяем направление
            direction = pattern.get('direction', 'bullish')

            # Ищем выход из сделки
            exit_index, exit_price, exit_reason = await self._find_exit(
                entry_index=entry_index,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                direction=direction,
                closes=closes
            )

            if exit_index is None:
                return None

            # Рассчитываем P&L
            if direction == 'bullish':
                pnl_pips = exit_price - entry_price
            else:
                pnl_pips = entry_price - exit_price

            # Учитываем комиссию и проскальзывание
            pnl = pnl_pips - (entry_price * self.commission) - (entry_price * self.slippage)

            # Создаем запись о сделке
            trade = {
                'pattern_id': pattern.get('id'),
                'pattern_name': pattern.get('name'),
                'direction': direction,
                'entry_index': entry_index,
                'entry_price': entry_price,
                'exit_index': exit_index,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'pnl': pnl,
                'pnl_pips': pnl_pips,
                'quality': pattern.get('metadata', {}).get('quality_score', 0),
                'holding_period': exit_index - entry_index
            }

            return trade

        except Exception as e:
            self.logger.error(f"Ошибка создания сделки: {e}")
            return None

    async def _find_exit(self,
                         entry_index: int,
                         entry_price: float,
                         stop_loss: float,
                         take_profit: float,
                         direction: str,
                         closes: np.ndarray) -> Tuple[Optional[int], Optional[float], str]:
        """Поиск выхода из сделки"""
        max_lookahead = self.max_holding_period

        for i in range(1, min(max_lookahead, len(closes) - entry_index)):
            current_idx = entry_index + i
            current_price = closes[current_idx]

            # Проверяем стоп-лосс
            if direction == 'bullish':
                if current_price <= stop_loss:
                    return current_idx, current_price, 'stop_loss'
                elif current_price >= take_profit:
                    return current_idx, current_price, 'take_profit'
            else:
                if current_price >= stop_loss:
                    return current_idx, current_price, 'stop_loss'
                elif current_price <= take_profit:
                    return current_idx, current_price, 'take_profit'

        # Выход по времени
        last_idx = entry_index + max_lookahead
        if last_idx < len(closes):
            return last_idx, closes[last_idx], 'time_exit'

        return None, None, 'no_exit'

    def _analyze_trades(self, trades: List[Dict[str, Any]]) -> BacktestResult:
        """Анализ сделок и расчет статистики"""
        result = BacktestResult()
        result.trades = trades

        if not trades:
            return result

        # Базовая статистика
        result.total_trades = len(trades)
        result.winning_trades = len([t for t in trades if t['pnl'] > 0])
        result.losing_trades = len([t for t in trades if t['pnl'] <= 0])
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0

        # Прибыль/убыток
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]

        result.total_profit = sum(t['pnl'] for t in winning_trades)
        result.total_loss = abs(sum(t['pnl'] for t in losing_trades))
        result.net_profit = result.total_profit - result.total_loss

        if result.total_loss > 0:
            result.profit_factor = result.total_profit / result.total_loss

        # Средние значения
        if winning_trades:
            result.avg_winning_trade = result.total_profit / len(winning_trades)
            result.avg_profit = result.avg_winning_trade
            result.largest_win = max(t['pnl'] for t in winning_trades)

        if losing_trades:
            result.avg_losing_trade = result.total_loss / len(losing_trades)
            result.avg_loss = result.avg_losing_trade
            result.largest_loss = min(t['pnl'] for t in losing_trades)

        # Максимальная просадка
        equity_curve = []
        balance = self.initial_balance

        for trade in trades:
            balance += trade['pnl'] * balance * self.risk_per_trade
            equity_curve.append(balance)

        if equity_curve:
            result.max_drawdown = self._calculate_max_drawdown(equity_curve)

        # Период удержания
        holding_periods = [t['holding_period'] for t in trades]
        if holding_periods:
            result.avg_holding_period = np.mean(holding_periods)

        # Серии побед/поражений
        result.max_consecutive_wins = self._calculate_max_consecutive(trades, 'win')
        result.max_consecutive_losses = self._calculate_max_consecutive(trades, 'loss')

        # Рассчитываем коэффициенты Шарпа, Сортино и Калмара
        returns = [t['pnl'] for t in trades]
        if returns:
            result.sharpe_ratio = self._calculate_sharpe_ratio(returns)
            result.sortino_ratio = self._calculate_sortino_ratio(returns)
            if result.max_drawdown > 0:
                result.calmar_ratio = result.net_profit / result.max_drawdown

        return result

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Расчет максимальной просадки"""
        peak = equity_curve[0]
        max_dd = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value

            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _calculate_max_consecutive(self, trades: List[Dict[str, Any]], trade_type: str) -> int:
        """Расчет максимальной серии побед или поражений"""
        max_streak = 0
        current_streak = 0

        for trade in trades:
            is_win = trade['pnl'] > 0

            if (trade_type == 'win' and is_win) or (trade_type == 'loss' and not is_win):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Расчет коэффициента Шарпа"""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252  # Дневная безрисковая ставка

        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return float(sharpe)

    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Расчет коэффициента Сортино"""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252

        # Только отрицательные возвраты
        negative_returns = excess_returns[excess_returns < 0]

        if len(negative_returns) == 0 or np.std(negative_returns) == 0:
            return 0.0

        sortino = np.mean(excess_returns) / np.std(negative_returns) * np.sqrt(252)
        return float(sortino)

    def generate_report(self, result: BacktestResult, save_path: Optional[str] = None) -> str:
        """Генерация отчета о бэктестинге"""
        report = "=" * 80 + "\n"
        report += "BACKTESTING REPORT\n"
        report += "=" * 80 + "\n\n"

        # Основная статистика
        report += "📊 PERFORMANCE SUMMARY\n"
        report += "-" * 40 + "\n"
        report += f"Total Trades: {result.total_trades}\n"
        report += f"Winning Trades: {result.winning_trades} ({result.win_rate:.1%})\n"
        report += f"Losing Trades: {result.losing_trades} ({1 - result.win_rate:.1%})\n"
        report += f"Net Profit: ${result.net_profit:.2f}\n"
        report += f"Profit Factor: {result.profit_factor:.2f}\n"
        report += f"Expectancy: ${result.expectancy:.2f}\n\n"

        # Прибыль/убыток
        report += "💰 PROFIT/LOSS ANALYSIS\n"
        report += "-" * 40 + "\n"
        report += f"Total Profit: ${result.total_profit:.2f}\n"
        report += f"Total Loss: ${result.total_loss:.2f}\n"
        report += f"Average Winning Trade: ${result.avg_winning_trade:.2f}\n"
        report += f"Average Losing Trade: ${result.avg_losing_trade:.2f}\n"
        report += f"Largest Win: ${result.largest_win:.2f}\n"
        report += f"Largest Loss: ${result.largest_loss:.2f}\n\n"

        # Риск
        report += "⚠️ RISK METRICS\n"
        report += "-" * 40 + "\n"
        report += f"Max Drawdown: {result.max_drawdown:.1%}\n"
        report += f"Sharpe Ratio: {result.sharpe_ratio or 0:.2f}\n"
        report += f"Sortino Ratio: {result.sortino_ratio or 0:.2f}\n"
        report += f"Calmar Ratio: {result.calmar_ratio or 0:.2f}\n\n"

        # Время
        report += "⏰ TIME ANALYSIS\n"
        report += "-" * 40 + "\n"
        report += f"Average Holding Period: {result.avg_holding_period:.1f} periods\n"
        report += f"Max Consecutive Wins: {result.max_consecutive_wins}\n"
        report += f"Max Consecutive Losses: {result.max_consecutive_losses}\n\n"

        # Последние сделки
        report += "📈 RECENT TRADES\n"
        report += "-" * 40 + "\n"

        if result.trades:
            last_trades = result.trades[-5:]  # Последние 5 сделок

            for i, trade in enumerate(last_trades, 1):
                report += f"{i}. {trade['pattern_name']} ({trade['direction']}): "
                report += f"P&L: ${trade['pnl']:.2f}, "
                report += f"Exit: {trade['exit_reason']}\n"

        report += "\n" + "=" * 80 + "\n"

        # Сохранение отчета
        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)

            self.logger.info(f"Отчет сохранен в {save_path}")

        return report


def run_backtest_cli():
    """CLI для запуска бэктестинга"""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Backtesting CLI for Pattern Recognition Engine")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default="H1", help="Timeframe")
    parser.add_argument("--bars", type=int, default=1000, help="Number of bars")
    parser.add_argument("--output", type=str, help="Output file for report")

    args = parser.parse_args()

    async def main():
        # Здесь должен быть код загрузки данных и запуска бэктестинга
        # Пока заглушка
        print(f"Backtesting for {args.symbol} {args.timeframe}")

    asyncio.run(main())


if __name__ == "__main__":
    run_backtest_cli()

