"""
Графический интерфейс для Pattern Recognition Engine
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import threading
import websockets
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from config import CONFIG, MT5_CONFIG
from utils.logger import logger


class PatternRecognitionGUI:
    """Главное окно приложения"""

    def __init__(self, root):
        self.root = root
        self.root.title(f"Pattern Recognition Engine v{CONFIG.VERSION}")
        self.root.geometry("1400x900")

        # WebSocket клиент
        self.ws_client = None
        self.ws_connected = False
        self.ws_url = f"ws://{MT5_CONFIG.SOCKET_HOST}:{MT5_CONFIG.SOCKET_PORT + 1}/gui"

        # Данные
        self.patterns_data = []
        self.current_symbol = "EURUSD"
        self.current_timeframe = "H1"

        # Графики
        self.figures = []

        # Настройка стилей
        self.setup_styles()

        # Создание интерфейса
        self.create_widgets()

        # Запуск WebSocket клиента
        self.start_websocket_client()

        # Обновление статистики
        self.update_stats()

    def setup_styles(self):
        """Настройка стилей интерфейса"""
        style = ttk.Style()

        # Темы
        style.theme_use('clam')

        # Цвета
        self.colors = {
            'bg_dark': '#2b2b2b',
            'bg_medium': '#3c3f41',
            'bg_light': '#4e5254',
            'fg_light': '#ffffff',
            'fg_dark': '#cccccc',
            'accent': '#4CAF50',
            'warning': '#FF9800',
            'error': '#F44336',
            'success': '#4CAF50'
        }

        # Настройка виджетов
        style.configure('TFrame', background=self.colors['bg_dark'])
        style.configure('TLabel', background=self.colors['bg_dark'],
                        foreground=self.colors['fg_light'])
        style.configure('TButton', background=self.colors['bg_medium'],
                        foreground=self.colors['fg_light'])
        style.configure('TEntry', fieldbackground=self.colors['bg_light'],
                        foreground=self.colors['fg_light'])
        style.configure('TCombobox', fieldbackground=self.colors['bg_light'],
                        foreground=self.colors['fg_light'])

        # Настройка окна
        self.root.configure(bg=self.colors['bg_dark'])

    def create_widgets(self):
        """Создание виджетов интерфейса"""
        # Главный контейнер
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Верхняя панель (статус и управление)
        self.create_top_panel(main_container)

        # Центральная область (графики и информация)
        center_container = ttk.Frame(main_container)
        center_container.pack(fill=tk.BOTH, expand=True, pady=10)

        # Левая панель (графики)
        self.create_left_panel(center_container)

        # Правая панель (информация и управление)
        self.create_right_panel(center_container)

        # Нижняя панель (логи и статистика)
        self.create_bottom_panel(main_container)

    def create_top_panel(self, parent):
        """Создание верхней панели"""
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        # Статус подключения
        self.status_label = ttk.Label(top_frame, text="❌ Не подключено",
                                      font=('Arial', 10, 'bold'))
        self.status_label.pack(side=tk.LEFT, padx=5)

        # Символ и таймфрейм
        symbol_frame = ttk.Frame(top_frame)
        symbol_frame.pack(side=tk.LEFT, padx=20)

        ttk.Label(symbol_frame, text="Символ:").pack(side=tk.LEFT)
        self.symbol_combo = ttk.Combobox(symbol_frame, values=['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'],
                                         width=10, state='readonly')
        self.symbol_combo.set(self.current_symbol)
        self.symbol_combo.pack(side=tk.LEFT, padx=5)
        self.symbol_combo.bind('<<ComboboxSelected>>', self.on_symbol_changed)

        ttk.Label(symbol_frame, text="Таймфрейм:").pack(side=tk.LEFT, padx=(10, 0))
        self.timeframe_combo = ttk.Combobox(symbol_frame,
                                            values=['M1', 'M5', 'M15', 'H1', 'H4', 'D1'],
                                            width=6, state='readonly')
        self.timeframe_combo.set(self.current_timeframe)
        self.timeframe_combo.pack(side=tk.LEFT, padx=5)
        self.timeframe_combo.bind('<<ComboboxSelected>>', self.on_timeframe_changed)

        # Кнопки управления
        button_frame = ttk.Frame(top_frame)
        button_frame.pack(side=tk.RIGHT)

        self.connect_btn = ttk.Button(button_frame, text="Подключиться",
                                      command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=5)

        self.analyze_btn = ttk.Button(button_frame, text="Анализировать",
                                      command=self.start_analysis, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)

        self.export_btn = ttk.Button(button_frame, text="Экспорт",
                                     command=self.export_data)
        self.export_btn.pack(side=tk.LEFT, padx=5)

        self.settings_btn = ttk.Button(button_frame, text="Настройки",
                                       command=self.open_settings)
        self.settings_btn.pack(side=tk.LEFT, padx=5)

    def create_left_panel(self, parent):
        """Создание левой панели (графики)"""
        left_frame = ttk.Frame(parent)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Заголовок графиков
        ttk.Label(left_frame, text="Графики паттернов",
                  font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))

        # Notebook для вкладок с графиками
        self.graph_notebook = ttk.Notebook(left_frame)
        self.graph_notebook.pack(fill=tk.BOTH, expand=True)

        # Вкладка с основным графиком
        main_tab = ttk.Frame(self.graph_notebook)
        self.graph_notebook.add(main_tab, text="Основной график")

        # Создание графика
        self.create_main_chart(main_tab)

        # Вкладка со статистикой
        stats_tab = ttk.Frame(self.graph_notebook)
        self.graph_notebook.add(stats_tab, text="Статистика")

        self.create_stats_chart(stats_tab)

        # Вкладка с ML аналитикой
        ml_tab = ttk.Frame(self.graph_notebook)
        self.graph_notebook.add(ml_tab, text="ML Аналитика")

        self.create_ml_chart(ml_tab)

    def create_main_chart(self, parent):
        """Создание основного графика"""
        # Создаем фигуру matplotlib
        self.main_fig = Figure(figsize=(10, 6), dpi=100)
        self.main_ax = self.main_fig.add_subplot(111)

        # Настройка графика
        self.main_ax.set_title(f"{self.current_symbol} - {self.current_timeframe}")
        self.main_ax.set_xlabel("Время")
        self.main_ax.set_ylabel("Цена")
        self.main_ax.grid(True, alpha=0.3)

        # Создаем canvas для Tkinter
        self.main_canvas = FigureCanvasTkAgg(self.main_fig, parent)
        self.main_canvas.draw()
        self.main_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Панель инструментов
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(toolbar_frame, text="Обновить",
                   command=self.update_chart).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar_frame, text="Сохранить",
                   command=self.save_chart).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar_frame, text="Очистить",
                   command=self.clear_chart).pack(side=tk.LEFT, padx=5)

    def create_stats_chart(self, parent):
        """Создание графика статистики"""
        self.stats_fig = Figure(figsize=(10, 6), dpi=100)
        self.stats_ax = self.stats_fig.add_subplot(111)

        self.stats_ax.set_title("Статистика паттернов")
        self.stats_ax.set_xlabel("Тип паттерна")
        self.stats_ax.set_ylabel("Количество")

        self.stats_canvas = FigureCanvasTkAgg(self.stats_fig, parent)
        self.stats_canvas.draw()
        self.stats_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_ml_chart(self, parent):
        """Создание графика ML аналитики"""
        self.ml_fig = Figure(figsize=(10, 6), dpi=100)
        self.ml_ax = self.ml_fig.add_subplot(111)

        self.ml_ax.set_title("ML Предсказания")
        self.ml_ax.set_xlabel("Паттерны")
        self.ml_ax.set_ylabel("Вероятность успеха")

        self.ml_canvas = FigureCanvasTkAgg(self.ml_fig, parent)
        self.ml_canvas.draw()
        self.ml_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_right_panel(self, parent):
        """Создание правой панели (информация и управление)"""
        right_frame = ttk.Frame(parent, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        # Заголовок
        ttk.Label(right_frame, text="Обнаруженные паттерны",
                  font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))

        # Список паттернов
        self.create_patterns_list(right_frame)

        # Информация о паттерне
        self.create_pattern_info(right_frame)

        # Управление
        self.create_control_panel(right_frame)

    def create_patterns_list(self, parent):
        """Создание списка паттернов"""
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Заголовки
        header_frame = ttk.Frame(list_frame)
        header_frame.pack(fill=tk.X)

        ttk.Label(header_frame, text="Паттерн", width=15).pack(side=tk.LEFT)
        ttk.Label(header_frame, text="Напр.", width=5).pack(side=tk.LEFT)
        ttk.Label(header_frame, text="Кач.", width=5).pack(side=tk.LEFT)
        ttk.Label(header_frame, text="Вер.", width=5).pack(side=tk.LEFT)

        # Список с прокруткой
        self.patterns_listbox = tk.Listbox(list_frame, height=15,
                                           bg=self.colors['bg_light'],
                                           fg=self.colors['fg_light'],
                                           selectbackground=self.colors['accent'])
        self.patterns_listbox.pack(fill=tk.BOTH, expand=True)
        self.patterns_listbox.bind('<<ListboxSelect>>', self.on_pattern_selected)

        # Полоса прокрутки
        scrollbar = ttk.Scrollbar(self.patterns_listbox)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.patterns_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.patterns_listbox.yview)

    def create_pattern_info(self, parent):
        """Создание панели информации о паттерне"""
        info_frame = ttk.LabelFrame(parent, text="Информация о паттерне")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        # Текстовая область
        self.pattern_info_text = scrolledtext.ScrolledText(info_frame,
                                                           height=10,
                                                           bg=self.colors['bg_light'],
                                                           fg=self.colors['fg_light'])
        self.pattern_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.pattern_info_text.config(state=tk.DISABLED)

    def create_control_panel(self, parent):
        """Создание панели управления"""
        control_frame = ttk.LabelFrame(parent, text="Управление")
        control_frame.pack(fill=tk.X)

        # Фильтры
        filter_frame = ttk.Frame(control_frame)
        filter_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(filter_frame, text="Тип:").pack(side=tk.LEFT)
        self.type_filter = ttk.Combobox(filter_frame,
                                        values=['Все', 'Геометрические', 'Свечные', 'Гармонические'],
                                        width=15, state='readonly')
        self.type_filter.set('Все')
        self.type_filter.pack(side=tk.LEFT, padx=5)
        self.type_filter.bind('<<ComboboxSelected>>', self.apply_filters)

        ttk.Label(filter_frame, text="Напр.:").pack(side=tk.LEFT, padx=(10, 0))
        self.direction_filter = ttk.Combobox(filter_frame,
                                             values=['Все', 'Бычьи', 'Медвежьи'],
                                             width=10, state='readonly')
        self.direction_filter.set('Все')
        self.direction_filter.pack(side=tk.LEFT, padx=5)
        self.direction_filter.bind('<<ComboboxSelected>>', self.apply_filters)

        # Кнопки управления
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="Обновить список",
                   command=self.update_patterns_list).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Очистить фильтры",
                   command=self.clear_filters).pack(side=tk.LEFT, padx=2)

        # ML обучение
        ml_frame = ttk.Frame(control_frame)
        ml_frame.pack(fill=tk.X, pady=5)

        ttk.Button(ml_frame, text="Обучить ML модель",
                   command=self.train_ml_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(ml_frame, text="ML предсказания",
                   command=self.show_ml_predictions).pack(side=tk.LEFT, padx=2)

    def create_bottom_panel(self, parent):
        """Создание нижней панели (логи и статистика)"""
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))

        # Notebook для логов и статистики
        self.bottom_notebook = ttk.Notebook(bottom_frame, height=200)
        self.bottom_notebook.pack(fill=tk.BOTH, expand=True)

        # Вкладка с логами
        log_tab = ttk.Frame(self.bottom_notebook)
        self.bottom_notebook.add(log_tab, text="Логи")

        self.log_text = scrolledtext.ScrolledText(log_tab,
                                                  bg=self.colors['bg_light'],
                                                  fg=self.colors['fg_light'])
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)

        # Вкладка со статистикой
        stats_tab = ttk.Frame(self.bottom_notebook)
        self.bottom_notebook.add(stats_tab, text="Статистика")

        self.stats_text = scrolledtext.ScrolledText(stats_tab,
                                                    bg=self.colors['bg_light'],
                                                    fg=self.colors['fg_light'])
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.stats_text.config(state=tk.DISABLED)

        # Вкладка с настройками
        settings_tab = ttk.Frame(self.bottom_notebook)
        self.bottom_notebook.add(settings_tab, text="Настройки")

        self.create_settings_tab(settings_tab)

    def create_settings_tab(self, parent):
        """Создание вкладки настроек"""
        settings_frame = ttk.Frame(parent)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Настройки подключения
        conn_frame = ttk.LabelFrame(settings_frame, text="Настройки подключения")
        conn_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(conn_frame, text="WebSocket URL:").grid(row=0, column=0,
                                                          sticky=tk.W, padx=5, pady=5)
        self.ws_url_entry = ttk.Entry(conn_frame, width=40)
        self.ws_url_entry.insert(0, self.ws_url)
        self.ws_url_entry.grid(row=0, column=1, padx=5, pady=5)

        # Настройки анализа
        analysis_frame = ttk.LabelFrame(settings_frame, text="Настройки анализа")
        analysis_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(analysis_frame, text="Минимальное качество:").grid(row=0, column=0,
                                                                     sticky=tk.W, padx=5, pady=5)
        self.min_quality_var = tk.DoubleVar(value=0.6)
        ttk.Scale(analysis_frame, from_=0.1, to=1.0, variable=self.min_quality_var,
                  orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(analysis_frame, textvariable=self.min_quality_var).grid(row=0, column=2, padx=5)

        ttk.Label(analysis_frame, text="Доверительный порог:").grid(row=1, column=0,
                                                                    sticky=tk.W, padx=5, pady=5)
        self.confidence_var = tk.DoubleVar(value=0.7)
        ttk.Scale(analysis_frame, from_=0.1, to=1.0, variable=self.confidence_var,
                  orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(analysis_frame, textvariable=self.confidence_var).grid(row=1, column=2, padx=5)

        # Кнопки
        button_frame = ttk.Frame(settings_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Сохранить настройки",
                   command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Сбросить",
                   command=self.reset_settings).pack(side=tk.LEFT, padx=5)

    def start_websocket_client(self):
        """Запуск WebSocket клиента в отдельном потоке"""

        def run_websocket():
            asyncio.run(self.websocket_loop())

        self.ws_thread = threading.Thread(target=run_websocket, daemon=True)
        self.ws_thread.start()

    async def websocket_loop(self):
        """Цикл WebSocket клиента"""
        while True:
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    self.ws_client = websocket
                    self.ws_connected = True

                    # Подписка на обновления
                    subscribe_msg = {
                        'type': 'subscribe',
                        'subscription_type': 'all'
                    }
                    await websocket.send(json.dumps(subscribe_msg))

                    # Обработка сообщений
                    async for message in websocket:
                        await self.handle_websocket_message(message)

            except Exception as e:
                self.ws_connected = False
                self.log_message(f"WebSocket ошибка: {str(e)}", "error")
                await asyncio.sleep(5)  # Пауза перед повторным подключением

    async def handle_websocket_message(self, message: str):
        """Обработка сообщений от WebSocket сервера"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            if msg_type == 'welcome':
                self.update_connection_status(True)
                self.log_message(f"Подключено к серверу: {data.get('client_id')}", "success")

            elif msg_type == 'new_patterns':
                await self.handle_new_patterns(data)

            elif msg_type == 'analysis_result':
                await self.handle_analysis_result(data)

            elif msg_type == 'heartbeat':
                pass  # Игнорируем heartbeat

            elif msg_type == 'error':
                self.log_message(f"Ошибка сервера: {data.get('error')}", "error")

            else:
                self.log_message(f"Неизвестное сообщение: {msg_type}", "warning")

        except Exception as e:
            self.log_message(f"Ошибка обработки сообщения: {str(e)}", "error")

    async def handle_new_patterns(self, data: Dict[str, Any]):
        """Обработка новых паттернов"""
        patterns_count = data.get('patterns_count', 0)
        symbol = data.get('symbol', '')
        timeframe = data.get('timeframe', '')

        if patterns_count > 0:
            self.log_message(f"Новые паттерны: {patterns_count} для {symbol} {timeframe}", "info")

            # Запрос полного списка паттернов
            if self.ws_client and self.ws_connected:
                get_patterns_msg = {
                    'type': 'get_patterns',
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'limit': 50
                }
                await self.ws_client.send(json.dumps(get_patterns_msg))

    async def handle_analysis_result(self, data: Dict[str, Any]):
        """Обработка результатов анализа"""
        patterns = data.get('patterns', [])

        if patterns:
            # Обновление списка паттернов
            self.patterns_data.extend(patterns)
            self.update_patterns_list()

            # Обновление графика
            self.update_chart_with_patterns(patterns)

            # Логирование
            self.log_message(f"Анализ завершен: найдено {len(patterns)} паттернов", "success")

    def update_connection_status(self, connected: bool):
        """Обновление статуса подключения"""
        self.ws_connected = connected

        if connected:
            self.status_label.config(text="✅ Подключено", foreground=self.colors['success'])
            self.connect_btn.config(text="Отключиться")
            self.analyze_btn.config(state=tk.NORMAL)
        else:
            self.status_label.config(text="❌ Не подключено", foreground=self.colors['error'])
            self.connect_btn.config(text="Подключиться")
            self.analyze_btn.config(state=tk.DISABLED)

        # Обновление в UI потоке
        self.root.update()

    def toggle_connection(self):
        """Переключение подключения"""
        if self.ws_connected:
            # TODO: Реализовать отключение
            pass
        else:
            # Переподключение
            self.start_websocket_client()

    def start_analysis(self):
        """Запуск анализа"""
        if not self.ws_connected or not self.ws_client:
            messagebox.showerror("Ошибка", "Нет подключения к серверу")
            return

        # В реальном приложении здесь нужно получить данные из MT5
        # Для демонстрации используем тестовые данные
        test_data = self.generate_test_data()

        # Отправка запроса на анализ
        async def send_analysis_request():
            if self.ws_client:
                analysis_msg = {
                    'type': 'analyze',
                    'symbol': self.current_symbol,
                    'timeframe': self.current_timeframe,
                    'data': test_data
                }
                await self.ws_client.send(json.dumps(analysis_msg))

        # Запуск в отдельном потоке
        asyncio.run_coroutine_threadsafe(send_analysis_request(), asyncio.get_event_loop())

        self.log_message(f"Запущен анализ {self.current_symbol} {self.current_timeframe}", "info")

    def generate_test_data(self) -> Dict[str, List[float]]:
        """Генерация тестовых данных"""
        np.random.seed(42)
        n = 100

        base_price = 1.1000
        trend = np.linspace(0, 0.02, n)
        noise = np.random.normal(0, 0.001, n)

        prices = base_price + trend + noise

        return {
            'open': (prices - np.random.uniform(0.0001, 0.0005, n)).tolist(),
            'high': (prices + np.random.uniform(0.0001, 0.0005, n)).tolist(),
            'low': (prices - np.random.uniform(0.0002, 0.0006, n)).tolist(),
            'close': prices.tolist(),
            'volume': np.random.uniform(1000, 10000, n).tolist()
        }

    def update_patterns_list(self):
        """Обновление списка паттернов"""
        self.patterns_listbox.delete(0, tk.END)

        filtered_patterns = self.filter_patterns(self.patterns_data)

        for pattern in filtered_pattern

