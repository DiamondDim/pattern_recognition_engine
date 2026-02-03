import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from .pattern_detector import PatternDetector

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, data_feeder, pattern_detector: PatternDetector, 
                 initial_balance: float = 10000.0, commission: float = 0.001):
        self.data_feeder = data_feeder
        self.pattern_detector = pattern_detector
        self.initial_balance = initial_balance
        self.commission = commission
        self.positions = []
        self.trades = []
        self.balance = initial_balance
        self.equity_curve = []
        self.current_bar_index = 0
        
    def _place_order(self, signal: str, price: float, timestamp: datetime, 
                    symbol: str, stop_loss: Optional[float] = None, 
                    take_profit: Optional[float] = None) -> Dict:
        """Размещение ордера в бэктесте"""
        if self.balance <= 0:
            logger.warning("Недостаточно баланса для открытия позиции")
            return {}
        
        # Рассчитываем объем позиции (1% от баланса на сделку)
        risk_percentage = 0.01
        position_value = self.balance * risk_percentage
        volume = position_value / price
        
        order = {
            'timestamp': timestamp,
            'signal': signal,
            'price': price,
            'symbol': symbol,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'volume': volume,
            'status': 'open',
            'commission': price * volume * self.commission
        }
        
        # Уменьшаем баланс на комиссию
        self.balance -= order['commission']
        
        # Добавляем в список позиций
        self.positions.append(order)
        
        logger.info(f"Открыта позиция: {signal} {symbol} по цене {price:.5f}, объем: {volume:.2f}")
        return order
    
    def _close_position(self, position_idx: int, close_price: float, 
                       close_time: datetime, reason: str = "manual") -> Optional[Dict]:
        """Закрытие позиции"""
        if position_idx >= len(self.positions) or position_idx < 0:
            return None
            
        position = self.positions[position_idx]
        
        if position['status'] != 'open':
            return None
        
        # Расчет P&L
        if position['signal'] == 'buy':
            profit = (close_price - position['price']) * position['volume']
            profit_pct = (close_price - position['price']) / position['price'] * 100
        else:  # sell
            profit = (position['price'] - close_price) * position['volume']
            profit_pct = (position['price'] - close_price) / position['price'] * 100
        
        # Вычитаем комиссию за закрытие
        close_commission = close_price * position['volume'] * self.commission
        profit -= close_commission
        
        trade = {
            'open_time': position['timestamp'],
            'close_time': close_time,
            'symbol': position['symbol'],
            'signal': position['signal'],
            'open_price': position['price'],
            'close_price': close_price,
            'volume': position['volume'],
            'profit': profit,
            'profit_pct': profit_pct,
            'commission': position['commission'] + close_commission,
            'close_reason': reason
        }
        
        # Обновляем баланс
        self.balance += profit
        
        # Добавляем в историю сделок
        self.trades.append(trade)
        
        # Удаляем из открытых позиций
        self.positions.pop(position_idx)
        
        logger.info(f"Закрыта позиция: {position['signal']} с прибылью {profit_pct:.2f}% ({profit:.2f})")
        return trade
    
    def _check_stop_levels(self, current_bar: pd.Series) -> List[Dict]:
        """Проверка стоп-лоссов и тейк-профитов"""
        closed_trades = []
        
        for i in range(len(self.positions) - 1, -1, -1):
            pos = self.positions[i]
            
            if pos['signal'] == 'buy':
                # Проверка стоп-лосса (цена опустилась ниже стопа)
                if pos['stop_loss'] and current_bar['low'] <= pos['stop_loss']:
                    close_price = min(pos['stop_loss'], current_bar['open'])
                    trade = self._close_position(i, close_price, current_bar.name, "stop_loss")
                    if trade:
                        closed_trades.append(trade)
                # Проверка тейк-профита (цена поднялась выше профита)
                elif pos['take_profit'] and current_bar['high'] >= pos['take_profit']:
                    close_price = min(pos['take_profit'], current_bar['open'])
                    trade = self._close_position(i, close_price, current_bar.name, "take_profit")
                    if trade:
                        closed_trades.append(trade)
            else:  # sell
                if pos['stop_loss'] and current_bar['high'] >= pos['stop_loss']:
                    close_price = max(pos['stop_loss'], current_bar['open'])
                    trade = self._close_position(i, close_price, current_bar.name, "stop_loss")
                    if trade:
                        closed_trades.append(trade)
                elif pos['take_profit'] and current_bar['low'] <= pos['take_profit']:
                    close_price = max(pos['take_profit'], current_bar['open'])
                    trade = self._close_position(i, close_price, current_bar.name, "take_profit")
                    if trade:
                        closed_trades.append(trade)
        
        return closed_trades
    
    def _log_trade(self, trade: Dict):
        """Логирование сделки"""
        self.trades.append(trade)
        logger.debug(f"Сделка залогирована: {trade}")
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Расчет метрик бэктестинга"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'total_profit_pct': 0,
                'max_drawdown': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0,
                'avg_trade': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        
        # Расчет базовых метрик
        profits = [t['profit'] for t in self.trades]
        profits_pct = [t['profit_pct'] for t in self.trades]
        
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        total_profit = sum(profits)
        total_profit_pct = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        # Расчет максимальной просадки
        equity_points = [self.initial_balance]
        for trade in self.trades:
            equity_points.append(equity_points[-1] + trade['profit'])
        
        max_equity = equity_points[0]
        max_drawdown = 0
        max_drawdown_pct = 0
        
        for equity in equity_points:
            if equity > max_equity:
                max_equity = equity
            
            drawdown = max_equity - equity
            drawdown_pct = (max_equity - equity) / max_equity * 100 if max_equity > 0 else 0
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
        
        # Расчет Profit Factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Расчет Sharpe Ratio (упрощенный)
        if len(profits) > 1:
            returns = np.array(profits_pct) / 100
            if np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Средние значения
        avg_trade = np.mean(profits) if profits else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_profit_pct': total_profit_pct,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'final_balance': self.balance,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def run(self, start_date: Optional[str] = None, end_date: Optional[str] = None, 
            initial_balance: Optional[float] = None) -> Dict[str, Any]:
        """Запуск бэктестинга"""
        if initial_balance:
            self.initial_balance = initial_balance
            self.balance = initial_balance
        
        # Получение данных
        logger.info("Загрузка данных для бэктестинга...")
        data = self.data_feeder.get_data()
        
        if data.empty:
            logger.error("Нет данных для бэктестинга")
            return self._calculate_metrics()
        
        # Фильтрация по датам если указаны
        if start_date:
            data = data[data.index >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data.index <= pd.to_datetime(end_date)]
        
        if len(data) == 0:
            logger.error("Нет данных после фильтрации по датам")
            return self._calculate_metrics()
        
        logger.info(f"Начало бэктестинга с {len(data)} барами")
        logger.info(f"Период: {data.index[0]} - {data.index[-1]}")
        
        # Основной цикл бэктестинга
        for i in range(20, len(data)):  # Начинаем с 20 для расчета индикаторов
            current_bar = data.iloc[i]
            previous_bars = data.iloc[:i+1].copy()  # Включая текущий бар
            
            # Проверка стоп-уровней для открытых позиций
            self._check_stop_levels(current_bar)
            
            # Детектирование паттернов (на предыдущих данных)
            if i > 0:
                patterns = self.pattern_detector.detect_patterns(previous_bars.iloc[:-1])
            else:
                patterns = []
            
            # Получение сигналов
            signals = []
            for pattern in patterns:
                if pattern.timestamp == previous_bars.index[-2]:  # Сигнал на предыдущем баре
                    signal = pattern.get_signal()
                    if signal != 'hold':
                        signals.append({
                            'pattern': pattern.__class__.__name__,
                            'signal': signal,
                            'confidence': pattern.confidence,
                            'timestamp': pattern.timestamp
                        })
            
            # Обработка сигналов (открываем новую позицию только если нет открытых)
            if signals and not self.positions:
                # Используем самый уверенный сигнал
                best_signal = max(signals, key=lambda x: x['confidence'])
                
                # Расчет уровней стоп-лосса и тейк-профита на основе ATR
                if len(previous_bars) >= 14:
                    high_low = previous_bars['high'].tail(14) - previous_bars['low'].tail(14)
                    atr = high_low.mean()
                else:
                    atr = previous_bars['high'].iloc[-1] - previous_bars['low'].iloc[-1]
                
                current_price = current_bar['close']
                
                if best_signal['signal'] == 'buy':
                    stop_loss = current_price - atr * 2
                    take_profit = current_price + atr * 3
                else:  # sell
                    stop_loss = current_price + atr * 2
                    take_profit = current_price - atr * 3
                
                # Открытие позиции
                self._place_order(
                    signal=best_signal['signal'],
                    price=current_price,
                    timestamp=current_bar.name,
                    symbol=current_bar.get('symbol', 'Unknown'),
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
            
            # Запись точки equity curve
            total_value = self.balance
            
            # Добавляем незакрытую прибыль/убыток
            for pos in self.positions:
                if pos['signal'] == 'buy':
                    unrealized = (current_bar['close'] - pos['price']) * pos['volume']
                else:
                    unrealized = (pos['price'] - current_bar['close']) * pos['volume']
                total_value += unrealized
            
            self.equity_curve.append({
                'timestamp': current_bar.name,
                'equity': total_value,
                'balance': self.balance,
                'open_positions': len(self.positions)
            })
        
        # Закрытие всех открытых позиций в конце теста
        for i in range(len(self.positions) - 1, -1, -1):
            self._close_position(i, data.iloc[-1]['close'], data.iloc[-1].name, "end_of_test")
        
        # Расчет финальных метрик
        metrics = self._calculate_metrics()
        
        # Дополнительная информация
        metrics['initial_balance'] = self.initial_balance
        metrics['final_balance'] = self.balance
        metrics['equity_curve'] = self.equity_curve
        metrics['trades'] = self.trades
        
        logger.info("=" * 50)
        logger.info("РЕЗУЛЬТАТЫ БЭКТЕСТИНГА")
        logger.info("=" * 50)
        logger.info(f"Начальный баланс: {self.initial_balance:.2f}")
        logger.info(f"Конечный баланс: {self.balance:.2f}")
        logger.info(f"Общая прибыль: {metrics['total_profit']:.2f} ({metrics['total_profit_pct']:.2f}%)")
        logger.info(f"Всего сделок: {metrics['total_trades']}")
        logger.info(f"Процент прибыльных: {metrics['win_rate']:.2f}%")
        logger.info(f"Максимальная просадка: {metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)")
        logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        
        return metrics
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Получение сделок в виде DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def get_equity_curve_dataframe(self) -> pd.DataFrame:
        """Получение кривой эквити в виде DataFrame"""
        if not self.equity_curve:
            return pd.DataFrame()
        
        return pd.DataFrame(self.equity_curve).set_index('timestamp')

