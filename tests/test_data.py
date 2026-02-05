# test_data.py
from core.data_feeder import DataFeeder

feeder = DataFeeder()
data = feeder.get_data("EURUSD", "H1", bars=200)
print(f"Загружено баров: {len(data)}")
print(f"Колонки: {data.columns.tolist()}")
print(data.tail())

