import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
import json

logger = logging.getLogger(__name__)


class BacktestingEngine:
    """
    Движок для бэктестинга торговых стратегий
    """

    def __init__(self, initial_capital: float = 10000.0,
                 commission: float = 0.0005,
                 slippage: float = 0.0001):
        """
        Инициализация движка бэктестинга

        Args:
            initial_capital (float): Начальный капитал
            commission (float): Комиссия за сделку (в процентах)
            slippage (float): Проскальзывание (в процентах)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        self.reset()
        logger.info(f"Инициализирован движок бэктестинга с капиталом ${initial_capital:.2f}")

    def reset(self):
        """Сброс состояния бэктестинга"""
        self.capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.current_position = None
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.max_drawdown = 0
        self.max_drawdown_percent = 0
        self.peak_equity = self.initial_capital

        logger.debug("Состояние бэктестинга сброшено")

    def run(self, data: pd.DataFrame, signals: List[Dict[str, Any]],
            stop_loss_pct: float = 0.02,
            take_profit_pct: float = 0.04,
            position_size_pct: float = 0.1) -> Dict[str, Any]:
        """
        Запуск бэктестинга

        Args:
            data (pd.DataFrame): Исторические данные
            signals (list): Список торговых сигналов
            stop_loss_pct (float): Стоп-лосс в процентах
            take_profit_pct (float): Тейк-профит в процентах
            position_size_pct (float): Размер позиции в процентах от капитала

        Returns:
            dict: Результаты бэктестинга
        """
        self.reset()

        if data.empty:
            logger.error("Нет данных для бэктестинга")
            return self._generate_empty_report()

        if not signals:
            logger.warning("Нет сигналов для бэктестинга")
            return self._generate_empty_report()

        logger.info(f"Начало бэктестинга: {len(data)} баров, {len(signals)} сигналов")
        logger.info(
            f"Параметры: SL={stop_loss_pct * 100}%, TP={take_profit_pct * 100}%, Position={position_size_pct * 100}%")

        try:
            # Сортируем сигналы по времени
            sorted_signals = sorted(signals, key=lambda x: x.get('index', 0))

            # Основной цикл бэктестинга
            for i in range(len(data)):
                if i >= len(data):
                    break

                current_time = data.index[i]
                current_price = data.iloc[i]['Close']

                # Проверяем условия выхода для текущей позиции
                if self.current_position:
                    self._check_exit_conditions(i, current_price, current_time, data)

                # Ищем сигналы для текущего бара
                current_signals = []
                for signal in sorted_signals:
                    signal_index = signal.get('index', -1)
                    if signal_index == i:
                        current_signals.append(signal)
                    elif signal_index > i:
                        break

                # Обрабатываем сигналы
                for signal in current_signals:
                    if not self.current_position:  # Открываем новую позицию только если нет текущей
                        self._process_signal(
                            signal, i, current_price, current_time,
                            position_size_pct, stop_loss_pct, take_profit_pct
                        )

                # Обновляем кривую капитала
                self._update_equity(current_price, current_time)

            # Закрываем последнюю позицию в конце данных
            if self.current_position:
                last_price = data.iloc[-1]['Close']
                last_time = data.index[-1]
                self._close_position(
                    len(data) - 1, last_price, last_time, 'end_of_data'
                )
                self._update_equity(last_price, last_time)

            # Генерируем отчет
            report = self._generate_report()

            logger.info(f"Бэктестинг завершен. Сделок: {self.trade_count}, "
                        f"Финансовый результат: ${report.get('total_pnl', 0):.2f}")

            return report

        except Exception as e:
            logger.error(f"Ошибка при выполнении бэктестинга: {e}")
            return self._generate_error_report(str(e))

    def _process_signal(self, signal: Dict[str, Any], index: int, price: float,
                        timestamp: datetime, position_size_pct: float,
                        stop_loss_pct: float, take_profit_pct: float):
        """
        Обработка торгового сигнала

        Args:
            signal (dict): Торговый сигнал
            index (int): Индекс бара
            price (float): Текущая цена
            timestamp (datetime): Временная метка
            position_size_pct (float): Размер позиции в процентах
            stop_loss_pct (float): Стоп-лосс в процентах
            take_profit_pct (float): Тейк-профит в процентах
        """
        try:
            signal_type = signal.get('type', '').lower()
            signal_strength = signal.get('strength', 1.0)

            if signal_type not in ['buy', 'sell']:
                logger.warning(f"Неизвестный тип сигнала: {signal_type}")
                return

            # Корректировка размера позиции на основе силы сигнала
            adjusted_position_size = position_size_pct * signal_strength
            position_size = self.capital * adjusted_position_size

            if position_size < 10:  # Минимальный размер позиции
                logger.debug(f"Слишком маленький размер позиции: ${position_size:.2f}")
                return

            # Корректировка цены на проскальзывание
            if signal_type == 'buy':
                entry_price = price * (1 + self.slippage)
                direction = 'long'
                stop_loss_price = entry_price * (1 - stop_loss_pct)
                take_profit_price = entry_price * (1 + take_profit_pct)
            else:  # sell
                entry_price = price * (1 - self.slippage)
                direction = 'short'
                stop_loss_price = entry_price * (1 + stop_loss_pct)
                take_profit_price = entry_price * (1 - take_profit_pct)

            # Расчет количества единиц
            units = position_size / entry_price

            # Комиссия за вход
            entry_commission = position_size * self.commission

            # Проверяем, достаточно ли капитала
            if self.capital < entry_commission:
                logger.warning(f"Недостаточно капитала для комиссии: {self.capital:.2f} < {entry_commission:.2f}")
                return

            # Создаем позицию
            self.current_position = {
                'id': self.trade_count + 1,
                'direction': direction,
                'entry_time': timestamp,
                'entry_price': entry_price,
                'entry_index': index,
                'position_size': position_size,
                'units': units,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'entry_commission': entry_commission,
                'exit_commission': 0,  # Будет рассчитана при выходе
                'signal_info': signal,
                'status': 'open'
            }

            # Списание комиссии
            self.capital -= entry_commission

            logger.info(
                f"Открыта позиция #{self.current_position['id']}: "
                f"{direction.upper()} {units:.4f} единиц по ${entry_price:.5f}, "
                f"SL: ${stop_loss_price:.5f}, TP: ${take_profit_price:.5f}"
            )

        except Exception as e:
            logger.error(f"Ошибка обработки сигнала: {e}")

    def _check_exit_conditions(self, index: int, current_price: float,
                               timestamp: datetime, data: pd.DataFrame):
        """
        Проверка условий выхода из позиции

        Args:
            index (int): Индекс текущего бара
            current_price (float): Текущая цена
            timestamp (datetime): Текущее время
            data (pd.DataFrame): Исторические данные
        """
        if not self.current_position:
            return

        position = self.current_position
        exit_reason = None
        exit_price = current_price

        # Проверка стоп-лосса и тейк-профита
        if position['direction'] == 'long':
            if current_price <= position['stop_loss']:
                exit_reason = 'stop_loss'
                exit_price = position['stop_loss']
            elif current_price >= position['take_profit']:
                exit_reason = 'take_profit'
                exit_price = position['take_profit']
        else:  # short
            if current_price >= position['stop_loss']:
                exit_reason = 'stop_loss'
                exit_price = position['stop_loss']
            elif current_price <= position['take_profit']:
                exit_reason = 'take_profit'
                exit_price = position['take_profit']

        # Выход по времени (максимальное удержание)
        max_hold_bars = 50
        if index - position['entry_index'] >= max_hold_bars:
            exit_reason = 'time_exit'

        # Выход по сигналу (если есть противоположный сигнал)
        # Эта логика может быть расширена

        if exit_reason:
            self._close_position(index, exit_price, timestamp, exit_reason)

    def _close_position(self, index: int, exit_price: float,
                        timestamp: datetime, reason: str):
        """
        Закрытие позиции

        Args:
            index (int): Индекс бара закрытия
            exit_price (float): Цена закрытия
            timestamp (datetime): Время закрытия
            reason (str): Причина закрытия
        """
        if not self.current_position:
            return

        position = self.current_position

        # Корректировка цены на проскальзывание
        if position['direction'] == 'long':
            adjusted_exit = exit_price * (1 - self.slippage)
        else:
            adjusted_exit = exit_price * (1 + self.slippage)

        # Расчет прибыли/убытка
        if position['direction'] == 'long':
            pnl = (adjusted_exit - position['entry_price']) * position['units']
        else:
            pnl = (position['entry_price'] - adjusted_exit) * position['units']

        # Комиссия за выход
        exit_commission = position['position_size'] * self.commission
        total_commission = position['entry_commission'] + exit_commission

        # Чистая прибыль/убыток
        net_pnl = pnl - total_commission

        # Обновление капитала
        self.capital += net_pnl

        # Расчет процентов
        pnl_percent = (net_pnl / position['position_size']) * 100

        # Время удержания
        holding_time = timestamp - position['entry_time']
        holding_bars = index - position['entry_index']

        # Сохранение информации о сделке
        trade = {
            'id': position['id'],
            'direction': position['direction'],
            'entry_time': position['entry_time'],
            'entry_price': position['entry_price'],
            'exit_time': timestamp,
            'exit_price': adjusted_exit,
            'exit_index': index,
            'position_size': position['position_size'],
            'units': position['units'],
            'pnl': net_pnl,
            'pnl_percent': pnl_percent,
            'gross_pnl': pnl,
            'commission': total_commission,
            'exit_reason': reason,
            'holding_time': holding_time,
            'holding_bars': holding_bars,
            'stop_loss': position['stop_loss'],
            'take_profit': position['take_profit'],
            'signal_info': position.get('signal_info', {})
        }

        self.trades.append(trade)
        self.trade_count += 1

        # Обновление счетчиков побед/поражений
        if net_pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1

        logger.info(
            f"Закрыта позиция #{position['id']}: "
            f"PNL ${net_pnl:.2f} ({pnl_percent:.2f}%), "
            f"Причина: {reason}, "
            f"Удержание: {holding_bars} баров"
        )

        self.current_position = None

    def _update_equity(self, current_price: float, timestamp: datetime):
        """
        Обновление кривой капитала

        Args:
            current_price (float): Текущая цена
            timestamp (datetime): Текущее время
        """
        equity = self.capital

        # Добавляем незакрытую прибыль/убыток
        if self.current_position:
            position = self.current_position
            if position['direction'] == 'long':
                unrealized_pnl = (current_price - position['entry_price']) * position['units']
            else:
                unrealized_pnl = (position['entry_price'] - current_price) * position['units']

            # Вычитаем будущую комиссию за выход
            future_commission = position['position_size'] * self.commission
            equity += unrealized_pnl - future_commission

        # Обновляем максимальную просадку
        if equity > self.peak_equity:
            self.peak_equity = equity

        drawdown = self.peak_equity - equity
        drawdown_percent = (drawdown / self.peak_equity) * 100 if self.peak_equity > 0 else 0

        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            self.max_drawdown_percent = drawdown_percent

        # Сохраняем точку на кривых
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'capital': self.capital,
            'unrealized_pnl': equity - self.capital
        })

        self.drawdown_curve.append({
            'timestamp': timestamp,
            'drawdown': drawdown,
            'drawdown_percent': drawdown_percent
        })

    def _generate_report(self) -> Dict[str, Any]:
        """
        Генерация отчета о бэктестинге

        Returns:
            dict: Полный отчет
        """
        report = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_trades': self.trade_count,
            'winning_trades': self.win_count,
            'losing_trades': self.loss_count,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_percent': self.max_drawdown_percent,
            'commission': self.commission,
            'slippage': self.slippage,
            'report_time': datetime.now()
        }

        if self.trade_count == 0:
            report['error'] = 'No trades executed'
            return report

        # Расчет основных метрик
        total_pnl = sum(t['pnl'] for t in self.trades)
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100

        report['total_pnl'] = total_pnl
        report['total_return_percent'] = total_return

        # Винрейт
        win_rate = (self.win_count / self.trade_count) * 100
        report['win_rate_percent'] = win_rate

        # Прибыльные и убыточные сделки
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]

        # Средняя прибыль/убыток
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        avg_win_percent = np.mean([t['pnl_percent'] for t in winning_trades]) if winning_trades else 0
        avg_loss_percent = np.mean([t['pnl_percent'] for t in losing_trades]) if losing_trades else 0

        report['avg_win'] = avg_win
        report['avg_loss'] = avg_loss
        report['avg_win_percent'] = avg_win_percent
        report['avg_loss_percent'] = avg_loss_percent

        # Фактор прибыли
        total_win = sum(t['pnl'] for t in winning_trades)
        total_loss = abs(sum(t['pnl'] for t in losing_trades))

        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        report['profit_factor'] = profit_factor

        # Соотношение риск/вознаграждение
        if avg_loss != 0:
            risk_reward_ratio = abs(avg_win / avg_loss)
            report['risk_reward_ratio'] = risk_reward_ratio

        # Коэффициент Шарпа (упрощенный)
        if len(self.equity_curve) > 1:
            equities = [e['equity'] for e in self.equity_curve]
            returns = np.diff(equities) / equities[:-1]

            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
                report['sharpe_ratio'] = sharpe_ratio

        # Коэффициент Сортино
        if len(self.equity_curve) > 1:
            downside_returns = [r for r in returns if r < 0]
            if downside_returns and np.std(downside_returns) > 0:
                sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
                report['sortino_ratio'] = sortino_ratio

        # Общая статистика по сделкам
        report['trades'] = self.trades[:100]  # Сохраняем только первые 100 сделок для отчета
        report['equity_curve'] = self.equity_curve[-1000:]  # Последние 1000 точек
        report['drawdown_curve'] = self.drawdown_curve[-1000:]  # Последние 1000 точек

        # Статистика по времени удержания
        holding_times = [t['holding_bars'] for t in self.trades]
        if holding_times:
            report['avg_holding_bars'] = np.mean(holding_times)
            report['median_holding_bars'] = np.median(holding_times)

        # Общая комиссия
        total_commission = sum(t['commission'] for t in self.trades)
        report['total_commission'] = total_commission

        logger.info(f"Отчет сгенерирован: {self.trade_count} сделок, "
                    f"Возврат: {total_return:.2f}%, "
                    f"Винрейт: {win_rate:.1f}%")

        return report

    def _generate_empty_report(self) -> Dict[str, Any]:
        """Генерация пустого отчета"""
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_trades': 0,
            'error': 'No data or signals provided',
            'report_time': datetime.now()
        }

    def _generate_error_report(self, error_msg: str) -> Dict[str, Any]:
        """Генерация отчета об ошибке"""
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_trades': self.trade_count,
            'error': error_msg,
            'report_time': datetime.now()
        }

    def save_report(self, filename: str):
        """
        Сохранение отчета в файл

        Args:
            filename (str): Имя файла
        """
        try:
            report = self._generate_report()

            # Преобразуем datetime в строки
            def datetime_converter(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                if isinstance(obj, timedelta):
                    return str(obj)
                raise TypeError(f"Type {type(obj)} not serializable")

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, default=datetime_converter, indent=2)

            logger.info(f"Отчет сохранен в {filename}")

        except Exception as e:
            logger.error(f"Ошибка сохранения отчета: {e}")

    def get_summary(self) -> str:
        """
        Получение краткого текстового summary

        Returns:
            str: Текстовый summary
        """
        if self.trade_count == 0:
            return "No trades executed"

        report = self._generate_report()

        summary_lines = [
            "=== BACKTESTING SUMMARY ===",
            f"Initial Capital: ${report['initial_capital']:.2f}",
            f"Final Capital: ${report['final_capital']:.2f}",
            f"Total Return: {report.get('total_return_percent', 0):.2f}%",
            f"Total PnL: ${report.get('total_pnl', 0):.2f}",
            f"Total Trades: {report['total_trades']}",
            f"Winning Trades: {report['winning_trades']} ({report.get('win_rate_percent', 0):.1f}%)",
            f"Max Drawdown: {report.get('max_drawdown_percent', 0):.2f}%",
            f"Profit Factor: {report.get('profit_factor', 0):.2f}",
            f"Sharpe Ratio: {report.get('sharpe_ratio', 0):.3f}" if 'sharpe_ratio' in report else "Sharpe Ratio: N/A",
            f"Avg Win: ${report.get('avg_win', 0):.2f} ({report.get('avg_win_percent', 0):.2f}%)",
            f"Avg Loss: ${report.get('avg_loss', 0):.2f} ({report.get('avg_loss_percent', 0):.2f}%)"
        ]

        return "\n".join(summary_lines)


# Создаем глобальный экземпляр для удобства
backtesting_engine = BacktestingEngine()

