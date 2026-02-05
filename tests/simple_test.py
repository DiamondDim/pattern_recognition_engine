import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

# Инициализация MT5
if not mt5.initialize():
    print(f"Ошибка инициализации MT5: {mt5.last_error()}")
    quit()

# Подключение к демо-счету
account = 2000067543  # Ваш логин
password = "9@b8X8C4eC"  # Ваш пароль
server = "AlfaForexRU-Real"  # Сервер

if not mt5.login(account, password, server):
    print(f"Ошибка входа: {mt5.last_error()}")
    mt5.shutdown()
    quit()

print("✅ Успешно подключено к MT5!")
print(f"Логин: {account}")
print(f"Сервер: {server}")

# Получение информации о счете
account_info = mt5.account_info()
if account_info:
    print(f"\n📊 Информация о счете:")
    print(f"   Логин: {account_info.login}")
    print(f"   Баланс: {account_info.balance}")
    print(f"   Валюта: {account_info.currency}")
    print(f"   Компания: {account_info.company}")

# Получение данных
symbol = "EURUSDrfd"
timeframe = mt5.TIMEFRAME_H1
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)

if rates is not None:
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"\n📈 Данные по {symbol}:")
    print(f"   Баров: {len(df)}")
    print(f"   Диапазон: {df['time'].iloc[0]} - {df['time'].iloc[-1]}")
    print(f"   Текущая цена: {df['close'].iloc[-1]:.5f}")

    # Сохраняем для теста
    df.to_csv("test_data.csv", index=False)
    print("   💾 Данные сохранены в test_data.csv")
else:
    print(f"❌ Не удалось получить данные по {symbol}")

# Отключение
mt5.shutdown()
print("\n✅ Тест завершен успешно!")

