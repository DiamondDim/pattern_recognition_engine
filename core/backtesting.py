"""
Модуль для бэктестинга торговых стратегий
"""

from enum import Enum
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
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
    BUY = 'buy'
    SELL = 'sell'
    BUY_LIMIT = 'buy_limit'
    SELL_LIMIT = 'sell_limit'
    BUY_STOP = 'buy_stop'
    SELL_STOP = 'sell_stop'


class OrderStatus(Enum):
    """Статусы ордеров"""
    PENDING = 'pending'
    OPEN = 'open'
    CLOSED = 'closed'
    CANCELLED = 'cancelled'


@dataclass
class Order:
    """Торговый ордер"""
    id: str
    symbol: str
    order_type: OrderType
    volume: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: datetime = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    commission: float = 0.0
    swap: float = 0.0
    profit: float = 0.0
    comment: str = ""

    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = datetime.now()


class BacktestEngine:
    """Движок бэктестинга"""

    def __init__(self, config: BACKTEST_CONFIG = None):
        self.config = config or BACKTEST_CONFIG
        self.logger = logger.bind(module="BacktestEngine")
        self.initial_balance = self.config.INITIAL_BALANCE
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.margin = 0.0
        self.free_margin = self.initial_balance
        self.leverage = self.config.LEVERAGE
        self.orders = []
        self.closed_orders = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = self.initial_balance
        self.position_count = 0
        self.winning_trades = 0
        self.losing_trades = 0

    def calculate_position_size(self, entry_price: float, stop_loss: float, risk_percent: float = 1.0) -> float:
        """
        Расчет размера позиции на основе риска

        Args:
            entry_price: Цена входа
            stop_loss: Цена стоп-лосса
            risk_percent: Процент риска от баланса

        Returns:
            Размер позиции в лотах
        """
        if entry_price <= 0 or stop_loss <= 0:
            return 0.0

        # Расчет риска в деньгах
        risk_amount = self.balance * (risk_percent / 100)

        # Расчет риска в пунктах
        if stop_loss < entry_price:  # BUY позиция
            risk_points = entry_price - stop_loss
        else:  # SELL позиция
            risk_points = stop_loss - entry_price

        # Размер позиции (в единицах базовой валюты)
        position_size = risk_amount / risk_points

        # Конвертация в лоты
        lot_size = 100000  # Стандартный лот
        lots = position_size / lot_size

        # Ограничение максимальным размером позиции
        max_lots = self.config.MAX_POSITION_SIZE
        lots = min(lots, max_lots)

        # Проверка на достаточность маржи
        required_margin = (lots * lot_size * entry_price) / self.leverage
        if required_margin > self.free_margin:
            # Автоматическое уменьшение размера позиции
            max_lots_by_margin = (self.free_margin * self.leverage) / (lot_size * entry_price)
            lots = min(lots, max_lots_by_margin)

        return round(lots, 2)

    def place_order(self, order: Order) -> bool:
        """
        Размещение ордера

        Args:
            order: Объект ордера

        Returns:
            Успешность размещения
        """
        try:
            # Проверка маржи
            required_margin = self._calculate_required_margin(order)
            if required_margin > self.free_margin:
                self.logger.warning(f"Недостаточно маржи для ордера {order.id}")
                return False

            # Расчет комиссии
            order.commission = self._calculate_commission(order)

            # Обновление баланса
            self.margin += required_margin
            self.free_margin = self.equity - self.margin

            # Добавление ордера
            order.status = OrderStatus.OPEN
            self.orders.append(order)
            self.position_count += 1

            self.logger.info(f"Ордер размещен: {order.id} {order.order_type.value} {order.volume} @ {order.entry_price}")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка размещения ордера: {e}")
            return False

    def _calculate_required_margin(self, order: Order) -> float:
        """Расчет требуемой маржи"""
        lot_size = 100000  # Стандартный лот
        margin = (order.volume * lot_size * order.entry_price) / self.leverage
        return margin

    def _calculate_commission(self, order: Order) -> float:
        """Расчет комиссии"""
        # Комиссия в процентах от объема
        commission_rate = self.config.COMMISSION_RATE
        lot_size = 100000
        commission = order.volume * lot_size * order.entry_price * commission_rate
        return commission

    def update_prices(self, current_prices: Dict[str, Dict[str, float]]):
        """
        Обновление цен и проверка условий ордеров

        Args:
            current_prices: Текущие цены по символам
        """
        orders_to_close = []

        for order in self.orders:
            if order.status != OrderStatus.OPEN:
                continue

            symbol = order.symbol
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            bid = current_price.get('bid', order.entry_price)
            ask = current_price.get('ask', order.entry_price)

            # Проверка стоп-лосса и тейк-профита
            should_close = False
            exit_price = 0.0
            profit = 0.0

            if order.order_type == OrderType.BUY:
                # Для BUY позиции
                current_value = bid

                if order.stop_loss and current_value <= order.stop_loss:
                    should_close = True
                    exit_price = order.stop_loss
                    profit = (exit_price - order.entry_price) * order.volume * 100000 - order.commission
                elif order.take_profit and current_value >= order.take_profit:
                    should_close = True
                    exit_price = order.take_profit
                    profit = (exit_price - order.entry_price) * order.volume * 100000 - order.commission

            elif order.order_type == OrderType.SELL:
                # Для SELL позиции
                current_value = ask

                if order.stop_loss and current_value >= order.stop_loss:
                    should_close = True
                    exit_price = order.stop_loss
                    profit = (order.entry_price - exit_price) * order.volume * 100000 - order.commission
                elif order.take_profit and current_value <= order.take_profit:
                    should_close = True
                    exit_price = order.take_profit
                    profit = (order.entry_price - exit_price) * order.volume * 100000 - order.commission

            if should_close:
                order.exit_price = exit_price
                order.exit_time = datetime.now()
                order.status = OrderStatus.CLOSED
                order.profit = profit

                # Обновление статистики
                if profit > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1

                orders_to_close.append(order)

        # Закрытие ордеров
        for order in orders_to_close:
            self.close_order(order)

        # Обновление эквити
        self._update_equity(current_prices)

    def close_order(self, order: Order):
        """
        Закрытие ордера

        Args:
            order: Ордер для закрытия
        """
        try:
            # Освобождение маржи
            required_margin = self._calculate_required_margin(order)
            self.margin -= required_margin

            # Обновление баланса
            self.balance += order.profit
            self.equity = self.balance

            # Обновление свободной маржи
            self.free_margin = self.equity - self.margin

            # Перемещение в закрытые ордера
            self.orders.remove(order)
            self.closed_orders.append(order)

            self.logger.info(f"Ордер закрыт: {order.id} Profit: {order.profit:.2f}")

        except Exception as e:
            self.logger.error(f"Ошибка закрытия ордера: {e}")

    def _update_equity(self, current_prices: Dict[str, Dict[str, float]]):
        """Обновление эквити и просадки"""
        # Расчет текущей прибыли/убытка по открытым позициям
        floating_profit = 0.0

        for order in self.orders:
            if order.status != OrderStatus.OPEN:
                continue

            symbol = order.symbol
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]

            if order.order_type == OrderType.BUY:
                bid = current_price.get('bid', order.entry_price)
                floating_profit += (bid - order.entry_price) * order.volume * 100000
            elif order.order_type == OrderType.SELL:
                ask = current_price.get('ask', order.entry_price)
                floating_profit += (order.entry_price - ask) * order.volume * 100000

        # Обновление эквити
        self.equity = self.balance + floating_profit
        self.equity_curve.append(self.equity)

        # Расчет просадки
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self.current_drawdown = (self.peak_equity - self.equity) / self.peak_equity * 100
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        self.drawdown_curve.append(self.current_drawdown)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Расчет метрик производительности

        Returns:
            Словарь с метриками
        """
        if not self.closed_orders:
            return {}

        # Базовые метрики
        total_trades = len(self.closed_orders)
        winning_trades = self.winning_trades
        losing_trades = self.losing_trades

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Прибыль/убыток
        total_profit = sum(order.profit for order in self.closed_orders if order.profit > 0)
        total_loss = abs(sum(order.profit for order in self.closed_orders if order.profit < 0))
        net_profit = total_profit - total_loss

        # Средние значения
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Возвраты
        total_return = (self.equity - self.initial_balance) / self.initial_balance * 100
        annual_return = self._calculate_annual_return()

        # Волатильность
        returns = self._calculate_returns_series()
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

        # Коэффициент Шарпа
        risk_free_rate = 0.02  # 2% безрисковая ставка
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

        # Максимальная просадка
        max_dd = self.max_drawdown

        # Восстановление после просадки
        recovery_factor = net_profit / (self.initial_balance * max_dd / 100) if max_dd > 0 else 0

        # Статистика по сделкам
        profits = [order.profit for order in self.closed_orders]
        largest_win = max(profits) if profits else 0
        largest_loss = min(profits) if profits else 0

        # Серии прибылей/убытков
        consecutive_wins = self._calculate_consecutive_wins()
        consecutive_losses = self._calculate_consecutive_losses()

        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.equity,
            'net_profit': net_profit,
            'total_return_pct': total_return,
            'annual_return_pct': annual_return,

            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate * 100,

            'total_profit': total_profit,
            'total_loss': total_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,

            'largest_win': largest_win,
            'largest_loss': largest_loss,

            'max_drawdown_pct': max_dd,
            'recovery_factor': recovery_factor,

            'sharpe_ratio': sharpe_ratio,
            'volatility_pct': volatility * 100,

            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,

            'position_count': self.position_count,
            'avg_position_hold_time': self._calculate_avg_hold_time()
        }

    def _calculate_annual_return(self) -> float:
        """Расчет годовой доходности"""
        if not self.closed_orders:
            return 0.0

        # Время первой и последней сделки
        first_trade = min(order.entry_time for order in self.closed_orders)
        last_trade = max(order.exit_time for order in self.closed_orders)

        total_days = (last_trade - first_trade).days
        if total_days == 0:
            return 0.0

        total_return = (self.equity - self.initial_balance) / self.initial_balance
        annual_return = (1 + total_return) ** (365 / total_days) - 1

        return annual_return * 100

    def _calculate_returns_series(self) -> np.ndarray:
        """Расчет серии доходностей"""
        if len(self.equity_curve) < 2:
            return np.array([])

        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        return returns

    def _calculate_consecutive_wins(self) -> int:
        """Расчет максимальной серии прибыльных сделок"""
        if not self.closed_orders:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for order in self.closed_orders:
            if order.profit > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_consecutive_losses(self) -> int:
        """Расчет максимальной серии убыточных сделок"""
        if not self.closed_orders:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for order in self.closed_orders:
            if order.profit < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_avg_hold_time(self) -> float:
        """Расчет среднего времени удержания позиции"""
        if not self.closed_orders:
            return 0.0

        total_seconds = sum(
            (order.exit_time - order.entry_time).total_seconds()
            for order in self.closed_orders
            if order.exit_time and order.entry_time
        )

        avg_seconds = total_seconds / len(self.closed_orders)
        avg_hours = avg_seconds / 3600

        return avg_hours

    def generate_report(self) -> Dict[str, Any]:
        """
        Генерация полного отчета

        Returns:
            Полный отчет о бэктесте
        """
        metrics = self.get_performance_metrics()

        report = {
            'summary': metrics,
            'orders': [
                {
                    'id': order.id,
                    'symbol': order.symbol,
                    'type': order.order_type.value,
                    'volume': order.volume,
                    'entry_price': order.entry_price,
                    'exit_price': order.exit_price,
                    'entry_time': order.entry_time.isoformat() if order.entry_time else None,
                    'exit_time': order.exit_time.isoformat() if order.exit_time else None,
                    'profit': order.profit,
                    'status': order.status.value,
                    'comment': order.comment
                }
                for order in self.closed_orders
            ],
            'equity_curve': self.equity_curve,
            'drawdown_curve': self.drawdown_curve,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'initial_balance': self.config.INITIAL_BALANCE,
                'leverage': self.config.LEVERAGE,
                'commission_rate': self.config.COMMISSION_RATE,
                'max_position_size': self.config.MAX_POSITION_SIZE
            }
        }

        return report

    def reset(self):
        """Сброс движка бэктестинга"""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.margin = 0.0
        self.free_margin = self.initial_balance
        self.orders = []
        self.closed_orders = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = self.initial_balance
        self.position_count = 0
        self.winning_trades = 0
        self.losing_trades = 0

        self.logger.info("Движок бэктестинга сброшен")

