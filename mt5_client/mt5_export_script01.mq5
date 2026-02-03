//+------------------------------------------------------------------+
//|                                                  MT5_Export.mq5 |
//+------------------------------------------------------------------+
#property copyright "Pattern Recognition Team"
#property link      "https://pattern-recognition.com"
#property version   "1.00"
#property script_show_inputs

//--- input parameters
input string   InpFileName    = "mt5_data.csv";   // Имя файла для экспорта
input int      InpBars        = 1000;             // Количество баров для экспорта
input bool     InpIncludeOHLC = true;             // Включать OHLC данные
input bool     InpIncludeVolume = true;           // Включать объем
input bool     InpIncludeSpread = false;          // Включать спред
input bool     InpIncludeTime = true;             // Включать время
string   InpSymbol      = _Symbol;          // Символ для экспорта
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_CURRENT; // Таймфрейм

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
   // Проверяем входные параметры - этот код был ранее
    if(InpBars <= 0 || InpFileName == "") {
       Print("Ошибка: Неверные входные параметры");
       return;
    }

   // Получаем данные
    MqlRates rates[];
    ArraySetAsSeries(rates,true);

    int copied = CopyRates(InpSymbol, InpTimeframe, 0, InpBars, rates);
    if(copied <= 0){
       Print("Ошибка получения данных: ", copied);
       return;
    }
    Print("Получено баров: ", copied);
   // Создаем файл
   int file_handle = FileOpen(InpFileName, FILE_WRITE|FILE_CSV|FILE_ANSI, ",");

   if(file_handle == INVALID_HANDLE)
   {
      Print("Ошибка: Не удалось создать файл ", InpFileName, ". Ошибка: ", GetLastError());
      return;
   }
   // Записываем заголовок
      FileWrite(file_handle, "time,open,high,low,close,volume,spread,symbol,timeframe");

   // Записываем данные построчно
   int digits = SymbolInfoInteger(InpSymbol, SYMBOL_DIGITS);
   for(int i = copied - 1; i >= 0; i--)
   {
       string row = "";
        if(InpIncludeTime)
            row += TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES) + ",";

        row += DoubleToString(rates[i].open, digits) + "," + DoubleToString(rates[i].high, digits) + "," + DoubleToString(rates[i].low, digits) + "," + DoubleToString(rates[i].close, digits);

        if(InpIncludeVolume)
            row += "," + DoubleToString(rates[i].tick_volume, 0);
        if(InpIncludeSpread)
           row += "," + IntegerToString(rates[i].spread);

        row += "," + InpSymbol + "," + TimeframeToString(InpTimeframe);
        FileWrite(file_handle, row);
   }
   FileClose(file_handle);
   Print("Экспорт завершен. Файл: ", InpFileName);
}

// Строковое представление таймфрейма
string TimeframeToString(ENUM_TIMEFRAMES tf){
    return EnumToString(tf);  //Наиболее простой способ
}

#import "shell32.dll"
   int ShellExecuteW(int hwnd, string lpOperation, string lpFile, string lpParameters, string lpDirectory, int nShowCmd);
#import

