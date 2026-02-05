import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Генерация тестовых данных
def generate_test_data(days=30):
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    np.random.seed(42)

    data = []
    price = 1.1000

    for i in range(days):
        open_price = price
        high = open_price + np.random.uniform(0.0001, 0.0050)
        low = open_price - np.random.uniform(0.0001, 0.0050)
        close = np.random.uniform(low, high)
        volume = np.random.randint(100, 1000)

        data.append({
            'time': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

        price = close

    return pd.DataFrame(data)


# Тестирование на сгенерированных данных
df = generate_test_data(100)
print(f"Сгенерировано данных: {len(df)}")
df.to_csv("test_historical_data.csv", index=False)
print("✅ Тестовые данные сохранены в test_historical_data.csv")

