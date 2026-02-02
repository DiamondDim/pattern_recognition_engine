
## 6. mt5_client/mt5_socket_client.mq5 (новый файл)

```mql5
//+------------------------------------------------------------------+
//|                                                      MT5_Socket.mq5 |
//|                        Pattern Recognition Engine Socket Client   |
//|                                              https://your-domain.com |
//+------------------------------------------------------------------+
#property copyright "Pattern Recognition Engine"
#property link      "https://your-domain.com"
#property version   "1.00"
#property description "Socket клиент для Pattern Recognition Engine"
#property script_show_inputs

//--- Входные параметры
input string   InpHost = "localhost";    // Хост сервера PRE
input int      InpPort = 5555;           // Порт сервера PRE
input string   InpSymbol = "EURUSD";     // Символ
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_H1; // Таймфрейм
input int      InpBarsCount = 1000;      // Количество баров

//--- Глобальные переменные
int socketHandle;

//+------------------------------------------------------------------+
//| Функция инициализации                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("Инициализация Socket клиента...");
   Print("Подключение к ", InpHost, ":", InpPort);
   
   // Пытаемся подключиться к серверу
   socketHandle = SocketCreate();
   
   if(socketHandle == INVALID_HANDLE)
   {
      Print("Ошибка: Не удалось создать сокет");
      return(INIT_FAILED);
   }
   
   // Подключаемся к серверу
   if(!SocketConnect(socketHandle, InpHost, InpPort, 10000))
   {
      Print("Ошибка: Не удалось подключиться к серверу ", InpHost, ":", InpPort);
      SocketClose(socketHandle);
      return(INIT_FAILED);
   }
   
   Print("Успешно подключено к серверу PRE");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Основная функция скрипта                                         |
//+------------------------------------------------------------------+
void OnStart()
{
   // 1. Получаем исторические данные
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   
   int copied = CopyRates(InpSymbol, InpTimeframe, 0, InpBarsCount, rates);
   
   if(copied <= 0)
   {
      Print("Ошибка: Не удалось получить исторические данные");
      return;
   }
   
   Print("Получено ", copied, " баров данных");
   
   // 2. Подготавливаем данные для отправки
   string jsonData = PrepareJSONData(rates, copied);
   
   // 3. Отправляем данные на сервер
   if(!SendDataToServer(jsonData))
   {
      Print("Ошибка отправки данных на сервер");
      return;
   }
   
   Print("Данные успешно отправлены на сервер PRE");
   
   // 4. Ждем ответа от сервера (опционально)
   string response = ReceiveDataFromServer();
   if(response != "")
   {
      Print("Ответ от сервера: ", response);
      ProcessServerResponse(response);
   }
   
   // 5. Закрываем соединение
   SocketClose(socketHandle);
   Print("Соединение закрыто");
}

//+------------------------------------------------------------------+
//| Подготовка данных в формате JSON                                 |
//+------------------------------------------------------------------+
string PrepareJSONData(MqlRates &rates[], int count)
{
   string json = "{";
   json += "\"command\":\"send_data\",";
   json += "\"symbol\":\"" + InpSymbol + "\",";
   json += "\"timeframe\":\"" + EnumToString(InpTimeframe) + "\",";
   json += "\"bars_count\":" + IntegerToString(count) + ",";
   json += "\"data\":[";
   
   for(int i = 0; i < count; i++)
   {
      if(i > 0) json += ",";
      json += "{";
      json += "\"time\":\"" + TimeToString(rates[i].time) + "\",";
      json += "\"open\":" + DoubleToString(rates[i].open, 5) + ",";
      json += "\"high\":" + DoubleToString(rates[i].high, 5) + ",";
      json += "\"low\":" + DoubleToString(rates[i].low, 5) + ",";
      json += "\"close\":" + DoubleToString(rates[i].close, 5) + ",";
      json += "\"volume\":" + DoubleToString(rates[i].tick_volume, 0);
      json += "}";
   }
   
   json += "]}";
   return json;
}

//+------------------------------------------------------------------+
//| Отправка данных на сервер                                        |
//+------------------------------------------------------------------+
bool SendDataToServer(string data)
{
   // Конвертируем строку в массив байт
   uchar bytes[];
   StringToCharArray(data, bytes);
   
   // Отправляем данные
   int sent = SocketSend(socketHandle, bytes);
   
   if(sent <= 0)
   {
      Print("Ошибка отправки данных");
      return false;
   }
   
   Print("Отправлено ", sent, " байт данных");
   return true;
}

//+------------------------------------------------------------------+
//| Получение данных от сервера                                      |
//+------------------------------------------------------------------+
string ReceiveDataFromServer()
{
   uchar buffer[4096];
   string response = "";
   
   // Ждем данные (таймаут 5 секунд)
   uint timeout = GetTickCount() + 5000;
   
   while(GetTickCount() < timeout)
   {
      // Проверяем, есть ли данные
      if(SocketIsReadable(socketHandle))
      {
         // Читаем данные
         int received = SocketRead(socketHandle, buffer, 4096, 1000);
         
         if(received > 0)
         {
            response = CharArrayToString(buffer, 0, received);
            break;
         }
      }
      
      Sleep(100);
   }
   
   return response;
}

//+------------------------------------------------------------------+
//| Обработка ответа сервера                                         |
//+------------------------------------------------------------------+
void ProcessServerResponse(string response)
{
   // Здесь можно разобрать JSON ответ от сервера
   // и выполнить соответствующие действия (например, отрисовать паттерны)
   
   Print("Обработка ответа сервера...");
   
   // Пример простой обработки
   if(StringFind(response, "patterns") >= 0)
   {
      Print("Сервер обнаружил паттерны");
      
      // Можно разобрать JSON и отрисовать паттерны на графике
      // DrawPatternsFromJSON(response);
   }
}

//+------------------------------------------------------------------+
//| Функция деинициализации                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Закрываем сокет при завершении
   if(socketHandle != INVALID_HANDLE)
   {
      SocketClose(socketHandle);
   }
   
   Print("Скрипт завершил работу. Причина: ", reason);
}

