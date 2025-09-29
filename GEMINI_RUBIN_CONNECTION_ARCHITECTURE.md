# 🤖 АРХИТЕКТУРА ПОДКЛЮЧЕНИЯ GEMINI К RUBIN AI

## 📊 Схема подключения

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GEMINI AI     │    │  GEMINI BRIDGE   │    │   RUBIN AI      │
│                 │    │   (Порт 8082)    │    │                 │
│ • Обучение      │───▶│ • Мост           │───▶│ • Smart         │
│ • Анализ        │    │ • Сессии         │    │   Dispatcher    │
│ • Обратная связь│    │ • Логирование    │    │   (Порт 8080)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │  СПЕЦИАЛИЗИРОВАННЫЕ │
                                                │     МОДУЛИ       │
                                                │                 │
                                                │ • General (8085) │
                                                │ • Math (8086)    │
                                                │ • Electrical    │
                                                │   (8087)         │
                                                │ • Programming   │
                                                │   (8088)         │
                                                │ • Neuro (8090)   │
                                                │ • Controllers   │
                                                │   (9000)         │
                                                │ • GAI (8104)    │
                                                └─────────────────┘
```

## 🔗 Точки подключения

### 1. **Gemini Bridge (Порт 8082)**
- **Файл**: `gemini_rubin_bridge.py`
- **Роль**: Мост между Gemini и Rubin AI
- **Эндпоинты**:
  - `POST /api/gemini/teach` - Обучение Rubin
  - `POST /api/gemini/analyze` - Анализ системы
  - `POST /api/gemini/feedback` - Обратная связь
  - `GET /api/gemini/status` - Статус моста

### 2. **Smart Dispatcher (Порт 8080)**
- **Файл**: `smart_dispatcher.py` (Docker)
- **Роль**: Центральный диспетчер Rubin AI
- **Эндпоинт**: `POST /api/chat`

### 3. **Специализированные модули**
- **General Server** (8085) - Docker
- **Math Server** (8086) - Docker  
- **Electrical Server** (8087) - Docker
- **Programming Server** (8088) - Docker
- **Neuro Server** (8090) - Docker
- **Controllers Server** (9000) - Docker
- **GAI Server** (8104) - Локальный

## 🔄 Поток данных

### **Обучение от Gemini:**
1. Gemini отправляет инструкцию → Gemini Bridge
2. Bridge форматирует запрос → Smart Dispatcher
3. Dispatcher маршрутизирует → Специализированный модуль
4. Модуль обрабатывает → Возвращает ответ
5. Bridge логирует взаимодействие

### **Анализ системы:**
1. Gemini запрашивает анализ → Gemini Bridge
2. Bridge формирует аналитический запрос → Smart Dispatcher
3. Dispatcher анализирует систему → Возвращает данные
4. Bridge предоставляет результат → Gemini

## ⚙️ Конфигурация

### **URL-адреса в gemini_rubin_bridge.py:**
```python
RUBIN_SMART_DISPATCHER_URL = "http://localhost:8080"
RUBIN_MODULES = {
    'general': 'http://localhost:8085',
    'mathematics': 'http://localhost:8086', 
    'electrical': 'http://localhost:8087',
    'programming': 'http://localhost:8088',
    'neuro': 'http://localhost:8090',
    'controllers': 'http://localhost:9000',
    'gai': 'http://localhost:8104'
}
```

## 📈 Статистика подключения

- **Активные сессии**: 0
- **Всего взаимодействий**: 0+
- **Статус Rubin AI**: ✅ healthy
- **Статус Bridge**: ✅ success

## 🚀 Запуск системы

1. **Docker серверы**: `start_docker_system.ps1`
2. **Gemini Bridge**: `python gemini_rubin_bridge.py`
3. **Telegram Bot**: `python telegram_bot.py`

## 🔧 Тестирование

```bash
# Статус Bridge
curl http://localhost:8082/api/gemini/status

# Обучение Rubin
curl -X POST http://localhost:8082/api/gemini/teach \
  -H "Content-Type: application/json" \
  -d '{"topic": "математика", "instruction": "Объясни алгебру", "category": "mathematics"}'

# Анализ системы
curl -X POST http://localhost:8082/api/gemini/analyze \
  -H "Content-Type: application/json" \
  -d '{"type": "performance", "query": "Как работает система?"}'
```

## ✅ Статус подключения

**Gemini Bridge**: ✅ Работает (порт 8082)
**Rubin AI**: ✅ Работает (Docker + локальные серверы)
**Подключение**: ✅ Установлено
**Тестирование**: ✅ Пройдено



