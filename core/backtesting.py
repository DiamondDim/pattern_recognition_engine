"""
Модуль для бэктестинга торговых стратегий
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import warnings

# Исправляем импорты для обратной совместимости
try:
    from config import config, BACKTESTING_CONFIG, BACKTEST_CONFIG, TRADING_CONFIG
except ImportError:
    # Создаем fallback конфиг
    class BacktestConfig:
        INITIAL_BALANCE = 10000.0
        RISK_PER_TRADE = 0.02
        COMMISSION = 0.0001
        SLIPPAGE = 0.0001
        MAX_HOLDING_PERIOD = 100
        MIN_TRADES_FOR_VALIDATION = 20

    BACKTESTING_CONFIG = BacktestConfig()
    BACKTEST_CONFIG = BACKTESTING_CONFIG
    TRADING_CONFIG = BACKTESTING_CONFIG
    print("Используется fallback конфиг для backtesting")


class OrderType(Enum):
    """Типы ордеров"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderDirection(Enum):
    """Направление ордера"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Торговый ордер"""
    order_id: str
    symbol: str
    order_type: OrderType
    direction: OrderDirection
    quantity: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    commission: float = 0.0
    profit_loss: Optional[float] = None
    closed: bool = False

    def calculate_pnl(self, current_price: float) -> float:
        """Расчет прибыли/убытка"""
        if self.closed and self.profit_loss is not None:
            return self.profit_loss

        if self.direction == OrderDirection.BUY:
            pnl = (current_price - self.entry_price) * self.quantity
        else:
            pnl = (self.entry_price - current_price) * self.quantity

        return pnl - self.commission


class BacktestEngine:
    """Движок для бэктестинга торговых стратегий"""

    def __init__(self, initial_balance: float = 10000.0, commission: float = 0.0001,
                 slippage: float = 0.0001, risk_per_trade: float = 0.02):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.risk_per_trade = risk_per_trade

        self.orders: List[Order] = []
        self.closed_orders: List[Order] = []
        self.portfolio: Dict[str, float] = {}

        self.equity_curve = []
        self.drawdown_curve = []

        self.current_date = None

    def run_backtest(self, data: pd.DataFrame, strategy, **kwargs) -> Dict[str, Any]:
        """
        Запуск бэктеста

        Args:
            data: DataFrame с историческими данными
            strategy: Функция стратегии, возвращающая торговые сигналы
            **kwargs: Дополнительные параметры стратегии

        Returns:
            Словарь с результатами бэктеста
        """
        print(f"Запуск бэктеста на {len(data)} барах")

        # Сбрасываем состояние
        self.orders = []
        self.closed_orders = []
        self.portfolio = {}
        self.balance = self.initial_balance
        self.equity_curve = []
        self.drawdown_curve = []

        # Проходим по каждому бару
        for idx, row in data.iterrows():
            self.current_date = idx

            # Закрываем ордера по стоп-лоссам и тейк-профитам
            self._check_orders(row)

            # Получаем сигналы от стратегии
            signals = strategy(row, self.portfolio, self.balance, **kwargs)

            # Исполняем сигналы
            if signals:
                self._execute_signals(signals, row)

            # Рассчитываем текущий эквити
            current_equity = self._calculate_equity(row)
            self.equity_curve.append(current_equity)

            # Рассчитываем просадку
            if self.equity_curve:
                peak = max(self.equity_curve)
                drawdown = (peak - current_equity) / peak if peak > 0 else 0
                self.drawdown_curve.append(drawdown)

        # Закрываем все открытые ордера по последней цене
        if len(data) > 0:
            last_row = data.iloc[-1]
            for order in self.orders[:]:
                self._close_order(order, last_row['close'])

        # Генерируем отчет
        report = self._generate_report(data)

        return report

    def _check_orders(self, current_bar):
        """Проверка ордеров на срабатывание стоп-лоссов и тейк-профитов"""
        for order in self.orders[:]:
            if order.closed:
                continue

            current_price = current_bar['close']

            # Проверяем стоп-лосс
            if order.stop_loss is not None:
                if order.direction == OrderDirection.BUY and current_price <= order.stop_loss:
                    self._close_order(order, order.stop_loss, reason="stop_loss")
                    continue
                elif order.direction == OrderDirection.SELL and current_price >= order.stop_loss:
                    self._close_order(order, order.stop_loss, reason="stop_loss")
                    continue

            # Проверяем тейк-профит
            if order.take_profit is not None:
                if order.direction == OrderDirection.BUY and current_price >= order.take_profit:
                    self._close_order(order, order.take_profit, reason="take_profit")
                    continue
                elif order.direction == OrderDirection.SELL and current_price <= order.take_profit:
                    self._close_order(order, order.take_profit, reason="take_profit")
                    continue

    def _close_order(self, order: Order, exit_price: float, reason: str = "manual"):
        """Закрытие ордера"""
        order.exit_price = exit_price
        order.exit_time = self.current_date
        order.profit_loss = order.calculate_pnl(exit_price)
        order.closed = True

        # Обновляем баланс
        self.balance += order.profit_loss

        # Перемещаем ордер в список закрытых
        self.orders.remove(order)
        self.closed_orders.append(order)

        print(f"Закрыт ордер {order.order_id} по цене {exit_price:.5f}, PnL: {order.profit_loss:.2f}, Причина: {reason}")

    def _execute_signals(self, signals: List[Dict], current_bar):
        """Исполнение торговых сигналов"""
        for signal in signals:
            symbol = signal.get('symbol', 'EURUSD')
            direction = OrderDirection.BUY if signal.get('direction') == 'buy' else OrderDirection.SELL
            order_type = OrderType.MARKET

            # Рассчитываем объем на основе риск-менеджмента
            quantity = self._calculate_position_size(current_bar['close'], signal.get('risk', 0.02))

            if quantity <= 0:
                continue

            # Создаем ордер
            order_id = f"{symbol}_{direction.value}_{self.current_date.timestamp()}"
            order = Order(
                order_id=order_id,
                symbol=symbol,
                order_type=order_type,
                direction=direction,
                quantity=quantity,
                entry_price=current_bar['close'],
                stop_loss=signal.get('stop_loss'),
                take_profit=signal.get('take_profit')
            )

            # Учитываем комиссию
            order.commission = self.commission * quantity * current_bar['close']
            self.balance -= order.commission

            # Добавляем ордер в список
            self.orders.append(order)

            print(f"Открыт ордер {order_id}: {direction.value} {quantity:.2f} {symbol} по цене {current_bar['close']:.5f}")

    def _calculate_position_size(self, price: float, risk_percent: float) -> float:
        """Расчет размера позиции на основе риск-менеджмента"""
        risk_amount = self.balance * risk_percent
        position_size = risk_amount / price

        # Ограничиваем размер позиции доступным балансом
        max_position = self.balance * 0.1 / price  # Не более 10% баланса на сделку
        position_size = min(position_size, max_position)

        return position_size

    def _calculate_equity(self, current_bar) -> float:
        """Расчет текущего эквити (баланс + незакрытые позиции)"""
        equity = self.balance

        for order in self.orders:
            if not order.closed:
                unrealized_pnl = order.calculate_pnl(current_bar['close'])
                equity += unrealized_pnl

        return equity

    def _generate_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Генерация отчета о бэктесте"""
        if not self.closed_orders:
            return {
                'initial_balance': self.initial_balance,
                'final_balance': self.balance,
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }

        # Рассчитываем прибыли и убытки
        profits = [o.profit_loss for o in self.closed_orders if o.profit_loss > 0]
        losses = [o.profit_loss for o in self.closed_orders if o.profit_loss <= 0]

        total_profit = sum(profits) if profits else 0
        total_loss = abs(sum(losses)) if losses else 0

        # Profit Factor
        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        # Win Rate
        win_rate = len(profits) / len(self.closed_orders)

        # Максимальная просадка
        max_drawdown = max(self.drawdown_curve) if self.drawdown_curve else 0

        # Коэффициент Шарпа (упрощенный)
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1] if len(self.equity_curve) > 1 else [0]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0

        report = {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return': (self.balance - self.initial_balance) / self.initial_balance,
            'total_trades': len(self.closed_orders),
            'profitable_trades': len(profits),
            'losing_trades': len(losses),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'equity_curve': self.equity_curve,
            'drawdown_curve': self.drawdown_curve
        }

        return report


def simple_moving_average_strategy(data_point: pd.Series, portfolio: Dict, balance: float,
                                  short_window: int = 10, long_window: int = 30) -> List[Dict]:
    """
    Простая стратегия на основе скользящих средних

    Args:
        data_point: Текущая строка данных
        portfolio: Текущий портфель
        balance: Текущий баланс
        short_window: Период короткой SMA
        long_window: Период длинной SMA

    Returns:
        Список торговых сигналов
    """
    # Эта стратегия требует исторических данных, поэтому в чистом виде не может работать на одном баре
    # В реальном бэктесте нужно передавать весь DataFrame или рассчитанные индикаторы
    return []

