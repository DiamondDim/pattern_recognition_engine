"""
Главное окно GUI для Pattern Recognition Engine
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from config import config
from utils.logger import logger
from core.pattern_recognition_engine import PatternRecognitionEngine

class MainWindow:
    """Главное окно приложения"""

    def __init__(self, engine: Optional[PatternRecognitionEngine] = None):
        self.engine = engine or PatternRecognitionEngine()
        self.root = tk.Tk()
        self.setup_window()
        self.create_widgets()
        self.running = False

        # Событийный цикл для асинхронных операций
        self.event_loop = asyncio.new_event_loop()

    def setup_window(self):
        """Настройка главного окна"""
        self.root.title("Pattern Recognition Engine")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # Иконка (если есть)
        icon_path = Path(__file__).parent / "icon.ico"
        if icon_path.exists():
            try:
                self.root.iconbitmap(str(icon_path))
            except:
                pass

        # Стили
        self.setup_styles()

        # Протокол закрытия окна
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_styles(self):
        """Настройка стилей"""
        style = ttk.Style()

        # Темы
        style.theme_use('clam')

        # Цвета
        colors = config.VISUALIZATION.COLORS

        # Настраиваем цвета
        style.configure('Title.TLabel',
                       font=('Arial', 16, 'bold'),
                       foreground=colors.get('bullish', '#000000'))

        style.configure('Subtitle.TLabel',
                       font=('Arial', 12),
                       foreground=colors.get('neutral', '#666666'))

        style.configure('Status.TLabel',
                       font=('Arial', 10),
                       foreground=colors.get('neutral', '#666666'))

        style.configure('Bullish.TLabel',
                       foreground=colors.get('bullish', '#26A69A'))

        style.configure('Bearish.TLabel',
                       foreground=colors.get('bearish', '#EF5350'))

        style.configure('Action.TButton',
                       font=('Arial', 10, 'bold'),
                       padding=5)

        style.map('Action.TButton',
                 foreground=[('active', 'white')],
                 background=[('active', colors.get('bullish', '#26A69A'))])

    def create_widgets(self):
        """Создание виджетов"""
        # Главный контейнер
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Настраиваем расширение
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(1, weight=1)

        # Верхняя панель
        self.create_top_panel(main_container)

        # Основная область
        self.create_main_area(main_container)

        # Нижняя панель
        self.create_bottom_panel(main_container)

    def create_top_panel(self, parent):
        """Создание верхней панели"""
        top_frame = ttk.Frame(parent)
        top_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        top_frame.columnconfigure(1, weight=1)

        # Заголовок
        title_label = ttk.Label(top_frame,
                               text="Pattern Recognition Engine",
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, sticky=tk.W)

        # Версия
        version_label = ttk.Label(top_frame,
                                 text="v1.0.0",
                                 style='Subtitle.TLabel')
        version_label.grid(row=1, column=0, sticky=tk.W)

        # Статус
        self.status_label = ttk.Label(top_frame,
                                     text="Готов к работе",
                                     style='Status.TLabel')
        self.status_label.grid(row=0, column=1, sticky=tk.E, rowspan=2)

        # Кнопки управления
        control_frame = ttk.Frame(top_frame)
        control_frame.grid(row=0, column=2, rowspan=2, sticky=tk.E, padx=(20, 0))

        self.start_button = ttk.Button(control_frame,
                                      text="Запуск",
                                      command=self.start_engine,
                                      style='Action.TButton')
        self.start_button.grid(row=0, column=0, padx=(0, 5))

        self.stop_button = ttk.Button(control_frame,
                                     text="Остановка",
                                     command=self.stop_engine,
                                     state=tk.DISABLED,
                                     style='Action.TButton')
        self.stop_button.grid(row=0, column=1)

    def create_main_area(self, parent):
        """Создание основной области"""
        # Notebook для вкладок
        notebook = ttk.Notebook(parent)
        notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Вкладка "Мониторинг"
        self.create_monitoring_tab(notebook)

        # Вкладка "Настройки"
        self.create_settings_tab(notebook)

        # Вкладка "Логи"
        self.create_logs_tab(notebook)

        # Вкладка "Статистика"
        self.create_statistics_tab(notebook)

    def create_monitoring_tab(self, notebook):
        """Создание вкладки мониторинга"""
        monitoring_frame = ttk.Frame(notebook)
        notebook.add(monitoring_frame, text="Мониторинг")

        # Разделение на левую и правую части
        paned_window = ttk.PanedWindow(monitoring_frame, orient=tk.HORIZONTAL)
        paned_window.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        monitoring_frame.columnconfigure(0, weight=1)
        monitoring_frame.rowconfigure(0, weight=1)

        # Левая часть - список паттернов
        left_frame = ttk.Frame(paned_window)
        paned_window.add(left_frame, weight=1)

        # Заголовок
        patterns_label = ttk.Label(left_frame, text="Обнаруженные паттерны", font=('Arial', 12, 'bold'))
        patterns_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        # Дерево паттернов
        self.create_patterns_tree(left_frame)

        # Правая часть - детали
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=1)

        # Детали паттерна
        self.create_pattern_details(right_frame)

        # График (заглушка)
        chart_frame = ttk.LabelFrame(right_frame, text="График", padding="5")
        chart_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)

        chart_label = ttk.Label(chart_frame, text="График будет отображен здесь")
        chart_label.grid(row=0, column=0)

    def create_patterns_tree(self, parent):
        """Создание дерева паттернов"""
        # Фрейм для дерева
        tree_frame = ttk.Frame(parent)
        tree_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        # Дерево
        columns = ('symbol', 'pattern', 'direction', 'quality', 'timeframe', 'time')
        self.patterns_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)

        # Заголовки
        self.patterns_tree.heading('symbol', text='Символ')
        self.patterns_tree.heading('pattern', text='Паттерн')
        self.patterns_tree.heading('direction', text='Направление')
        self.patterns_tree.heading('quality', text='Качество')
        self.patterns_tree.heading('timeframe', text='Таймфрейм')
        self.patterns_tree.heading('time', text='Время')

        # Колонки
        self.patterns_tree.column('symbol', width=80)
        self.patterns_tree.column('pattern', width=120)
        self.patterns_tree.column('direction', width=80)
        self.patterns_tree.column('quality', width=80)
        self.patterns_tree.column('timeframe', width=80)
        self.patterns_tree.column('time', width=120)

        # Скроллбар
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.patterns_tree.yview)
        self.patterns_tree.configure(yscrollcommand=scrollbar.set)

        # Размещение
        self.patterns_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Бинд события выбора
        self.patterns_tree.bind('<<TreeviewSelect>>', self.on_pattern_select)

        # Кнопки управления
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=2, column=0, sticky=tk.W, pady=(5, 0))

        ttk.Button(button_frame, text="Обновить", command=self.refresh_patterns).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="Экспорт", command=self.export_patterns).grid(row=0, column=1)

    def create_pattern_details(self, parent):
        """Создание области деталей паттерна"""
        details_frame = ttk.LabelFrame(parent, text="Детали паттерна", padding="10")
        details_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N))
        details_frame.columnconfigure(1, weight=1)

        # Поля деталей
        row = 0

        # Символ
        ttk.Label(details_frame, text="Символ:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.detail_symbol = ttk.Label(details_frame, text="-")
        self.detail_symbol.grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        # Паттерн
        ttk.Label(details_frame, text="Паттерн:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.detail_pattern = ttk.Label(details_frame, text="-")
        self.detail_pattern.grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        # Направление
        ttk.Label(details_frame, text="Направление:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.detail_direction = ttk.Label(details_frame, text="-")
        self.detail_direction.grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        # Качество
        ttk.Label(details_frame, text="Качество:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.detail_quality = ttk.Label(details_frame, text="-")
        self.detail_quality.grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        # Таймфрейм
        ttk.Label(details_frame, text="Таймфрейм:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.detail_timeframe = ttk.Label(details_frame, text="-")
        self.detail_timeframe.grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        # Время обнаружения
        ttk.Label(details_frame, text="Обнаружен:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.detail_time = ttk.Label(details_frame, text="-")
        self.detail_time.grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        # Точки паттерна
        ttk.Label(details_frame, text="Точки:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.detail_points = ttk.Label(details_frame, text="-")
        self.detail_points.grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        # Цели
        ttk.Label(details_frame, text="Цели:").grid(row=row, column=0, sticky=tk.W, pady=2)

        targets_frame = ttk.Frame(details_frame)
        targets_frame.grid(row=row, column=1, sticky=tk.W, pady=2)

        ttk.Label(targets_frame, text="Вход:").grid(row=0, column=0, sticky=tk.W)
        self.detail_entry = ttk.Label(targets_frame, text="-")
        self.detail_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 15))

        ttk.Label(targets_frame, text="Стоп:").grid(row=0, column=2, sticky=tk.W)
        self.detail_stop = ttk.Label(targets_frame, text="-")
        self.detail_stop.grid(row=0, column=3, sticky=tk.W, padx=(5, 15))

        ttk.Label(targets_frame, text="Цель:").grid(row=0, column=4, sticky=tk.W)
        self.detail_target = ttk.Label(targets_frame, text="-")
        self.detail_target.grid(row=0, column=5, sticky=tk.W, padx=(5, 0))

        row += 1

        # Действия
        action_frame = ttk.Frame(details_frame)
        action_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))

        ttk.Button(action_frame, text="Торговать", command=self.trade_pattern, width=15).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(action_frame, text="Игнорировать", command=self.ignore_pattern, width=15).grid(row=0, column=1)

    def create_settings_tab(self, notebook):
        """Создание вкладки настроек"""
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="Настройки")

        # Прокручиваемая область
        canvas = tk.Canvas(settings_frame)
        scrollbar = ttk.Scrollbar(settings_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Настройки
        self.create_settings_controls(scrollable_frame)

    def create_settings_controls(self, parent):
        """Создание контролов настроек"""
        row = 0

        # Режим работы
        mode_frame = ttk.LabelFrame(parent, text="Режим работы", padding="10")
        mode_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        mode_frame.columnconfigure(0, weight=1)
        row += 1

        self.mode_var = tk.StringVar(value=config.MODE)

        modes = [('Файловый', 'file'),
                ('WebSocket', 'websocket'),
                ('API', 'api'),
                ('GUI', 'gui')]

        for i, (text, value) in enumerate(modes):
            rb = ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var, value=value)
            rb.grid(row=i//2, column=i%2, sticky=tk.W, padx=5, pady=2)

        # Детектирование паттернов
        detection_frame = ttk.LabelFrame(parent, text="Детектирование паттернов", padding="10")
        detection_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        row += 1

        self.enable_geometric = tk.BooleanVar(value=config.DETECTION.ENABLE_GEOMETRIC)
        self.enable_candlestick = tk.BooleanVar(value=config.DETECTION.ENABLE_CANDLESTICK)
        self.enable_harmonic = tk.BooleanVar(value=config.DETECTION.ENABLE_HARMONIC)

        ttk.Checkbutton(detection_frame, text="Геометрические", variable=self.enable_geometric).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(detection_frame, text="Свечные", variable=self.enable_candlestick).grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(detection_frame, text="Гармонические", variable=self.enable_harmonic).grid(row=2, column=0, sticky=tk.W, pady=2)

        # Параметры качества
        quality_frame = ttk.LabelFrame(parent, text="Параметры качества", padding="10")
        quality_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        row += 1

        ttk.Label(quality_frame, text="Минимальное качество:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.min_quality = tk.DoubleVar(value=config.DETECTION.MIN_PATTERN_QUALITY)
        ttk.Scale(quality_frame, from_=0.0, to=1.0, variable=self.min_quality, orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=5)
        ttk.Label(quality_frame, textvariable=self.min_quality).grid(row=0, column=2, padx=5)

        ttk.Label(quality_frame, text="Порог уверенности:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.confidence_threshold = tk.DoubleVar(value=config.DETECTION.CONFIDENCE_THRESHOLD)
        ttk.Scale(quality_frame, from_=0.0, to=1.0, variable=self.confidence_threshold, orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, padx=5)
        ttk.Label(quality_frame, textvariable=self.confidence_threshold).grid(row=1, column=2, padx=5)

        # MT5 настройки
        mt5_frame = ttk.LabelFrame(parent, text="MetaTrader 5", padding="10")
        mt5_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        row += 1

        self.mt5_enabled = tk.BooleanVar(value=config.MT5.ENABLED)
        ttk.Checkbutton(mt5_frame, text="Включить MT5", variable=self.mt5_enabled).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(mt5_frame, text="Путь к MT5:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.mt5_path = tk.StringVar(value=config.MT5.PATH)
        ttk.Entry(mt5_frame, textvariable=self.mt5_path, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(mt5_frame, text="Обзор...", command=self.browse_mt5_path).grid(row=1, column=2)

        ttk.Label(mt5_frame, text="Логин:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.mt5_login = tk.StringVar(value=str(config.MT5.LOGIN) if config.MT5.LOGIN else "")
        ttk.Entry(mt5_frame, textvariable=self.mt5_login, width=20).grid(row=2, column=1, sticky=tk.W, padx=5)

        ttk.Label(mt5_frame, text="Символы:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.mt5_symbols = tk.StringVar(value=", ".join(config.MT5.SYMBOLS))
        ttk.Entry(mt5_frame, textvariable=self.mt5_symbols, width=40).grid(row=3, column=1, padx=5)

        # Кнопки сохранения
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, column=0, sticky=tk.E, padx=10, pady=10)
        row += 1

        ttk.Button(button_frame, text="Сохранить", command=self.save_settings, width=15).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="Сбросить", command=self.reset_settings, width=15).grid(row=0, column=1)

    def create_logs_tab(self, notebook):
        """Создание вкладки логов"""
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="Логи")

        logs_frame.columnconfigure(0, weight=1)
        logs_frame.rowconfigure(0, weight=1)

        # Текстовое поле для логов
        self.log_text = scrolledtext.ScrolledText(logs_frame, wrap=tk.WORD, width=80, height=30)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        # Кнопки управления логами
        button_frame = ttk.Frame(logs_frame)
        button_frame.grid(row=1, column=0, sticky=tk.W, padx=10, pady=(0, 10))

        ttk.Button(button_frame, text="Очистить", command=self.clear_logs).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="Сохранить", command=self.save_logs).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(button_frame, text="Обновить", command=self.refresh_logs).grid(row=0, column=2)

    def create_statistics_tab(self, notebook):
        """Создание вкладки статистики"""
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Статистика")

        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)

        # Текстовое поле для статистики
        self.stats_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD, width=80, height=30)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        # Кнопки
        button_frame = ttk.Frame(stats_frame)
        button_frame.grid(row=1, column=0, sticky=tk.W, padx=10, pady=(0, 10))

        ttk.Button(button_frame, text="Обновить", command=self.refresh_statistics).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="Экспорт", command=self.export_statistics).grid(row=0, column=1)

    def create_bottom_panel(self, parent):
        """Создание нижней панели"""
        bottom_frame = ttk.Frame(parent)
        bottom_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        # Прогресс-бар
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(bottom_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        bottom_frame.columnconfigure(0, weight=1)

        # Статус
        self.status_text = tk.StringVar(value="Готов")
        status_label = ttk.Label(bottom_frame, textvariable=self.status_text)
        status_label.grid(row=0, column=1)

    def browse_mt5_path(self):
        """Выбор пути к MT5"""
        path = filedialog.askopenfilename(
            title="Выберите MetaTrader 5 terminal64.exe",
            filetypes=[("Executable files", "*.exe"), ("All files", "*.*")]
        )
        if path:
            self.mt5_path.set(path)

    def save_settings(self):
        """Сохранение настроек"""
        try:
            # Обновляем конфигурацию
            config.MODE = self.mode_var.get()

            config.DETECTION.ENABLE_GEOMETRIC = self.enable_geometric.get()
            config.DETECTION.ENABLE_CANDLESTICK = self.enable_candlestick.get()
            config.DETECTION.ENABLE_HARMONIC = self.enable_harmonic.get()

            config.DETECTION.MIN_PATTERN_QUALITY = self.min_quality.get()
            config.DETECTION.CONFIDENCE_THRESHOLD = self.confidence_threshold.get()

            config.MT5.ENABLED = self.mt5_enabled.get()
            config.MT5.PATH = self.mt5_path.get()

            if self.mt5_login.get():
                config.MT5.LOGIN = int(self.mt5_login.get())

            symbols = [s.strip() for s in self.mt5_symbols.get().split(',')]
            config.MT5.SYMBOLS = symbols

            # Сохраняем в файл
            config.save()

            messagebox.showinfo("Настройки", "Настройки успешно сохранены")
            self.log("Настройки сохранены")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка сохранения настроек: {e}")
            self.log(f"Ошибка сохранения настроек: {e}", level="error")

    def reset_settings(self):
        """Сброс настроек к значениям по умолчанию"""
        if messagebox.askyesno("Сброс", "Сбросить настройки к значениям по умолчанию?"):
            # Сбрасываем переменные
            self.mode_var.set('file')

            self.enable_geometric.set(True)
            self.enable_candlestick.set(True)
            self.enable_harmonic.set(True)

            self.min_quality.set(0.6)
            self.confidence_threshold.set(0.7)

            self.mt5_enabled.set(True)
            self.mt5_path.set("C:/Program Files/MetaTrader 5/terminal64.exe")
            self.mt5_login.set("")
            self.mt5_symbols.set("EURUSD, GBPUSD, USDJPY")

            messagebox.showinfo("Сброс", "Настройки сброшены")
            self.log("Настройки сброшены")

    def start_engine(self):
        """Запуск движка"""
        if self.running:
            messagebox.showwarning("Внимание", "Движок уже запущен")
            return

        try:
            # Сохраняем настройки перед запуском
            self.save_settings()

            # Запускаем в отдельном потоке
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_text.set("Запуск...")

            thread = threading.Thread(target=self._run_engine, daemon=True)
            thread.start()

            self.log("Движок запущен")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка запуска движка: {e}")
            self.log(f"Ошибка запуска движка: {e}", level="error")
            self.running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def _run_engine(self):
        """Запуск движка в отдельном потоке"""
        try:
            asyncio.set_event_loop(self.event_loop)
            self.event_loop.run_until_complete(self.engine.run())
        except Exception as e:
            self.log(f"Ошибка в работе движка: {e}", level="error")
        finally:
            self.running = False
            self.root.after(0, self._on_engine_stopped)

    def _on_engine_stopped(self):
        """Обработка остановки движка"""
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_text.set("Остановлен")
        self.log("Движок остановлен")

    def stop_engine(self):
        """Остановка движка"""
        if not self.running:
            messagebox.showwarning("Внимание", "Движок не запущен")
            return

        self.running = False
        self.status_text.set("Остановка...")

        # Останавливаем асинхронный цикл
        if self.event_loop.is_running():
            self.event_loop.call_soon_threadsafe(self.event_loop.stop)

        self.log("Запрошена остановка движка")

    def refresh_patterns(self):
        """Обновление списка паттернов"""
        # Здесь должна быть логика загрузки паттернов
        # Пока что очищаем и добавляем тестовые данные
        for item in self.patterns_tree.get_children():
            self.patterns_tree.delete(item)

        # Тестовые данные
        test_patterns = [
            ('EURUSD', 'Head & Shoulders', 'Bearish', 0.85, 'H1', '2024-01-15 14:30'),
            ('GBPUSD', 'Double Top', 'Bearish', 0.78, 'H4', '2024-01-15 12:00'),
            ('USDJPY', 'Morning Star', 'Bullish', 0.92, 'D1', '2024-01-14 00:00'),
            ('EURUSD', 'Triangle', 'Bullish', 0.67, 'H1', '2024-01-15 10:15'),
            ('XAUUSD', 'Hammer', 'Bullish', 0.88, 'H4', '2024-01-15 08:45')
        ]

        for pattern in test_patterns:
            self.patterns_tree.insert('', tk.END, values=pattern)

        self.log("Список паттернов обновлен")

    def on_pattern_select(self, event):
        """Обработка выбора паттерна"""
        selection = self.patterns_tree.selection()
        if not selection:
            return

        item = self.patterns_tree.item(selection[0])
        values = item['values']

        # Обновляем детали
        self.detail_symbol.config(text=values[0])
        self.detail_pattern.config(text=values[1])

        # Направление с цветом
        direction = values[2]
        if direction == 'Bullish':
            self.detail_direction.config(text=direction, style='Bullish.TLabel')
        else:
            self.detail_direction.config(text=direction, style='Bearish.TLabel')

        self.detail_quality.config(text=f"{values[3]:.2f}")
        self.detail_timeframe.config(text=values[4])
        self.detail_time.config(text=values[5])

        # Тестовые данные для точек и целей
        self.detail_points.config(text="5 точек")
        self.detail_entry.config(text="1.0950")
        self.detail_stop.config(text="1.0900")
        self.detail_target.config(text="1.1050")

    def trade_pattern(self):
        """Торговля по выбранному паттерну"""
        selection = self.patterns_tree.selection()
        if not selection:
            messagebox.showwarning("Внимание", "Выберите паттерн для торговли")
            return

        if messagebox.askyesno("Подтверждение", "Открыть сделку по выбранному паттерну?"):
            # Здесь должна быть логика открытия сделки
            self.log(f"Сделка открыта по паттерну")
            messagebox.showinfo("Торговля", "Сделка открыта")

    def ignore_pattern(self):
        """Игнорирование выбранного паттерна"""
        selection = self.patterns_tree.selection()
        if not selection:
            messagebox.showwarning("Внимание", "Выберите паттерн для игнорирования")
            return

        item = self.patterns_tree.item(selection[0])
        values = item['values']

        if messagebox.askyesno("Подтверждение", f"Игнорировать паттерн {values[1]} на {values[0]}?"):
            self.patterns_tree.delete(selection[0])
            self.log(f"Паттерн {values[1]} на {values[0]} игнорирован")

    def export_patterns(self):
        """Экспорт паттернов"""
        filepath = filedialog.asksaveasfilename(
            title="Экспорт паттернов",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filepath:
            # Здесь должна быть логика экспорта
            self.log(f"Паттерны экспортированы в {filepath}")
            messagebox.showinfo("Экспорт", f"Паттерны экспортированы в {filepath}")

    def clear_logs(self):
        """Очистка логов"""
        self.log_text.delete(1.0, tk.END)
        self.log("Логи очищены")

    def save_logs(self):
        """Сохранение логов в файл"""
        filepath = filedialog.asksaveasfilename(
            title="Сохранить логи",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get(1.0, tk.END))

                self.log(f"Логи сохранены в {filepath}")
                messagebox.showinfo("Сохранение", f"Логи сохранены в {filepath}")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка сохранения логов: {e}")

    def refresh_logs(self):
        """Обновление логов"""
        # Здесь должна быть логика загрузки логов из файла
        # Пока что просто добавляем тестовое сообщение
        self.log("Логи обновлены")

    def refresh_statistics(self):
        """Обновление статистики"""
        # Здесь должна быть логика загрузки статистики
        # Пока что тестовые данные
        stats = f"""Статистика Pattern Recognition Engine
Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Общая статистика:
- Всего паттернов обнаружено: 127
- Успешных сделок: 45
- Неуспешных сделок: 32
- В процессе: 5

По типам паттернов:
- Геометрические: 67
- Свечные: 42
- Гармонические: 18

Эффективность:
- Win Rate: 58.4%
- Средняя прибыль: 1.8%
- Максимальная просадка: 4.2%
- Profit Factor: 1.75

Последние 5 сделок:
1. EURUSD H1 Head & Shoulders: +2.1%
2. GBPUSD H4 Double Top: -1.2%
3. USDJPY D1 Morning Star: +3.4%
4. EURUSD H1 Triangle: +0.8%
5. XAUUSD H4 Hammer: +1.5%
"""

        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)

        self.log("Статистика обновлена")

    def export_statistics(self):
        """Экспорт статистики"""
        filepath = filedialog.asksaveasfilename(
            title="Экспорт статистики",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self.stats_text.get(1.0, tk.END))

                self.log(f"Статистика экспортирована в {filepath}")
                messagebox.showinfo("Экспорт", f"Статистика экспортирована в {filepath}")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка экспорта статистики: {e}")

    def log(self, message: str, level: str = "info"):
        """Добавление сообщения в лог"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Определяем цвет в зависимости от уровня
        if level == "error":
            tag = "error"
            color = "red"
        elif level == "warning":
            tag = "warning"
            color = "orange"
        else:
            tag = "info"
            color = "black"

        # Добавляем в текстовое поле
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n", tag)

        # Прокручиваем вниз
        self.log_text.see(tk.END)

        # Обновляем статус
        if level == "info":
            self.status_text.set(message[:50])

    def on_closing(self):
        """Обработка закрытия окна"""
        if self.running:
            if messagebox.askyesno("Выход", "Движок все еще работает. Завершить работу?"):
                self.stop_engine()
                self.root.after(1000, self.root.destroy)  # Даем время на остановку
            else:
                return
        else:
            self.root.destroy()

    def run(self):
        """Запуск главного цикла"""
        # Настраиваем теги для цветного текста в логах
        self.log_text.tag_config("error", foreground="red")
        self.log_text.tag_config("warning", foreground="orange")
        self.log_text.tag_config("info", foreground="black")

        # Обновляем начальные данные
        self.refresh_patterns()
        self.refresh_statistics()
        self.log("GUI инициализировано")

        # Запускаем главный цикл
        self.root.mainloop()

def main():
    """Точка входа для GUI"""
    engine = PatternRecognitionEngine(mode="gui")
    app = MainWindow(engine)
    app.run()

if __name__ == "__main__":
    main()

