"""
Rubin AI v2.0 - Стабильный основной API сервер
Исправлены проблемы с кодировкой и стабильностью
"""

import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import pickle

# Добавляем пути к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'providers'))

from config_stable import StableConfig
from providers.smart_provider_selector import SmartProviderSelector
from providers.huggingface_provider import HuggingFaceProvider
from providers.base_provider import TaskType

# Инициализация Flask приложения
app = Flask(__name__)
CORS(app, 
     origins=StableConfig.ALLOWED_ORIGINS,
     allow_headers=['Content-Type', 'Authorization'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     supports_credentials=True)

# Настройка логирования (без эмодзи)
logging.basicConfig(
    level=getattr(logging, StableConfig.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(StableConfig.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("rubin_ai_v2_stable")

# Глобальные переменные
provider_selector = None
documents_storage = []

def initialize_system():
    """Инициализация системы"""
    global provider_selector, documents_storage
    
    try:
        logger.info("Инициализация Rubin AI v2.0...")
        
        # Инициализация провайдеров
        provider_selector = SmartProviderSelector()
        
        # Регистрация провайдеров
        try:
            huggingface_provider = HuggingFaceProvider()
            if huggingface_provider.initialize():
                provider_selector.register_provider("huggingface", huggingface_provider)
                logger.info("Hugging Face провайдер инициализирован")
            else:
                logger.warning("Hugging Face провайдер недоступен")
        except Exception as e:
            logger.warning(f"Hugging Face провайдер недоступен: {e}")
        
        # Загрузка документов
        load_documents()
        
        logger.info("Rubin AI v2.0 успешно инициализирован!")
        
    except Exception as e:
        logger.error(f"Ошибка инициализации: {e}")

def load_documents():
    """Загрузка документов из хранилища"""
    global documents_storage
    
    try:
        if os.path.exists(StableConfig.DOCUMENTS_STORAGE):
            with open(StableConfig.DOCUMENTS_STORAGE, 'rb') as f:
                documents_storage = pickle.load(f)
            logger.info(f"Загружено {len(documents_storage)} документов")
        else:
            documents_storage = []
            logger.info("Документы не найдены, создаем новое хранилище")
    except Exception as e:
        logger.error(f"Ошибка загрузки документов: {e}")
        documents_storage = []

def save_documents():
    """Сохранение документов в хранилище"""
    try:
        with open(StableConfig.DOCUMENTS_STORAGE, 'wb') as f:
            pickle.dump(documents_storage, f)
        logger.info(f"Сохранено {len(documents_storage)} документов")
    except Exception as e:
        logger.error(f"Ошибка сохранения документов: {e}")

# API Endpoints

@app.route('/')
def index():
    """Главная страница"""
    return jsonify({
        'name': 'Rubin AI v2.0 Stable',
        'version': '2.0.0-stable',
        'description': 'Стабильная версия Rubin AI',
        'status': 'running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health')
@app.route('/health')
def health_check():
    """Проверка здоровья системы"""
    provider_status = provider_selector.get_provider_status() if provider_selector else {}
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'providers': provider_status,
        'documents_count': len(documents_storage),
        'config': {
            'debug': StableConfig.DEBUG,
            'port': StableConfig.PORT,
            'available_providers': StableConfig.get_available_providers()
        }
    })

@app.route('/api/stats')
def get_stats():
    """Получить статистику системы"""
    return jsonify({
        'system': {
            'name': 'Rubin AI v2.0 Stable',
            'version': '2.0.0-stable',
            'uptime': 'running',
            'timestamp': datetime.now().isoformat()
        },
        'providers': provider_selector.get_provider_status() if provider_selector else {},
        'documents': {
            'total': len(documents_storage),
            'storage_size': os.path.getsize(StableConfig.DOCUMENTS_STORAGE) if os.path.exists(StableConfig.DOCUMENTS_STORAGE) else 0
        }
    })

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def ai_chat():
    """Основной endpoint для AI чата"""
    # Обработка OPTIONS запроса для CORS
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        message = data.get('message', '')
        context = data.get('context', {})
        category = data.get('category', '')
        
        if not message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
        
        logger.info(f"Получен запрос: {message[:100]}...")
        logger.info(f"Категория: {category}")
        
        # Простые быстрые ответы
        simple_responses = {
            'привет': "Привет! Я Rubin AI - ваш помощник по программированию, электротехнике и автоматизации. Чем могу помочь?",
            'как дела': "У меня все отлично! Готов помочь с техническими вопросами. Что вас интересует?",
            'статус': "Система Rubin AI работает нормально. Все модули активны и готовы к работе."
        }
        
        message_lower = message.lower().strip()
        if message_lower in simple_responses:
            logger.info("Быстрый ответ")
            return jsonify({
                'response': simple_responses[message_lower],
                'provider': 'fast_cache',
                'category': 'general_chat',
                'cached': True,
                'response_time': 0.001
            })
        
        # Определение категории
        detected_category = None
        
        if category:
            detected_category = category
            logger.info(f"Используется явно указанная категория: {category}")
        else:
            if provider_selector:
                detected_category = provider_selector.detect_task_type(message, context)
                logger.info(f"Автоматически определена категория: {detected_category}")
            else:
                # Fallback к простому определению
                message_lower = message.lower()
                if any(word in message_lower for word in ['пид', 'pid', 'scada', 'plc', 'pmac', 'контроллер', 'автоматизация']):
                    detected_category = 'controllers'
                elif any(word in message_lower for word in ['c++', 'python', 'программирование', 'код', 'алгоритм']):
                    detected_category = 'programming'
                else:
                    detected_category = 'general'
        
        # Получение ответа
        response = get_specialized_response(message, detected_category, context)
        
        # Добавляем timestamp
        response['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Ответ получен от {response.get('provider', 'unknown')}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Ошибка в AI чате: {e}")
        return jsonify({
            'error': f'Ошибка обработки запроса: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

def get_specialized_response(message, category, context):
    """Получение специализированного ответа"""
    
    # Специализированные ответы для контроллеров
    if category in ['controllers', 'plc_analysis', 'pmac_analysis']:
        logger.info("Используем специализированный ответ для контроллеров")
        return get_controllers_response(message)
    
    # Специализированные ответы для программирования
    elif category in ['programming', 'code_analysis', 'code_generation']:
        logger.info("Используем специализированный ответ для программирования")
        return get_programming_response(message)
    
    # Специализированные ответы для электротехники
    elif category == 'electrical_analysis':
        logger.info("Используем специализированный ответ для электротехники")
        return get_electrical_response(message)
    
    # Специализированные ответы для радиомеханики
    elif category == 'radiomechanics_analysis':
        logger.info("Используем специализированный ответ для радиомеханики")
        return get_radiomechanics_response(message)
    
    # Общие ответы через провайдер
    else:
        if provider_selector:
            enhanced_context = context.copy() if context else {}
            enhanced_context['task_type'] = category
            return provider_selector.get_response(message, enhanced_context)
        else:
            return {
                'response': 'Извините, система временно недоступна. Попробуйте позже.',
                'provider': 'System'
            }

def get_controllers_response(message):
    """Специализированные ответы по контроллерам"""
    message_lower = message.lower()
    
    # ПИД-регулятор
    if 'пид' in message_lower or 'pid' in message_lower:
        return {
            'response': '''ПИД-регулятор (Proportional-Integral-Derivative) - это устройство управления с обратной связью, широко используемое в системах автоматического управления.

**Принцип работы:**
- **P (Пропорциональная)** - реагирует на текущую ошибку
- **I (Интегральная)** - устраняет статическую ошибку
- **D (Дифференциальная)** - улучшает динамику системы

**Формула ПИД-регулятора:**
u(t) = Kp * e(t) + Ki * ∫e(t)dt + Kd * de(t)/dt

**Настройка параметров:**
- Kp - коэффициент пропорциональности
- Ki - коэффициент интегральной составляющей  
- Kd - коэффициент дифференциальной составляющей

**Применение в PLC:**
```ladder
LD I0.0          // Вход датчика
SUB SP1, I0.0    // Вычисление ошибки
MUL Kp, SP1      // Пропорциональная часть
// ... остальная логика
```''',
            'provider': 'PLC Specialist',
            'category': 'controllers'
        }
    
    # PLC программирование
    elif 'plc' in message_lower or 'контроллер' in message_lower:
        return {
            'response': '''PLC (Programmable Logic Controller) - программируемый логический контроллер для автоматизации промышленных процессов.

**Основные языки программирования PLC:**
1. **Ladder Logic (LD)** - релейная логика
2. **Function Block Diagram (FBD)** - функциональные блоки
3. **Structured Text (ST)** - структурированный текст
4. **Instruction List (IL)** - список инструкций
5. **Sequential Function Chart (SFC)** - последовательные функциональные схемы

**Пример программы на Ladder Logic:**
```
|--[ ]--[ ]--( )--|  // AND логика
|--[ ]--+--( )--|   // OR логика
|--[ ]--+--( )--|
```

**Основные функции:**
- Цифровые входы/выходы (DI/DO)
- Аналоговые входы/выходы (AI/AO)
- Таймеры (TON, TOF, RTO)
- Счетчики (CTU, CTD, CTUD)
- Сравнение (CMP, LIM)
- Арифметические операции (ADD, SUB, MUL, DIV)''',
            'provider': 'PLC Specialist',
            'category': 'controllers'
        }
    
    # Общий ответ по контроллерам
    else:
        return {
            'response': '''Я специализируюсь на промышленной автоматизации и контроллерах. Могу помочь с:

**PLC программированием:**
- Ladder Logic, FBD, ST, IL, SFC
- Настройка таймеров и счетчиков
- Работа с аналоговыми сигналами

**ПИД-регуляторами:**
- Настройка параметров Kp, Ki, Kd
- Реализация в PLC
- Оптимизация процессов

**SCADA системами:**
- Создание HMI интерфейсов
- Мониторинг процессов
- Сбор данных

**Сетевые протоколы:**
- Modbus RTU/TCP
- Profibus, Profinet
- Ethernet/IP

Задайте конкретный вопрос, и я дам подробный ответ с примерами!''',
            'provider': 'Industrial Automation Specialist',
            'category': 'controllers'
        }

def get_programming_response(message):
    """Специализированные ответы по программированию"""
    message_lower = message.lower()
    
    # Python
    if 'python' in message_lower:
        return {
            'response': '''Python - высокоуровневый язык программирования, отлично подходящий для автоматизации и анализа данных.

**Основные возможности:**
- Простой и читаемый синтаксис
- Богатая стандартная библиотека
- Множество специализированных библиотек
- Кроссплатформенность

**Для автоматизации:**
```python
import time
import serial

# Подключение к PLC через Serial
ser = serial.Serial('COM1', 9600)

# Отправка команды
ser.write(b'READ D100\n')
response = ser.readline()
print(f"Ответ PLC: {response.decode()}")

ser.close()
```

**Популярные библиотеки:**
- `pyserial` - работа с COM портами
- `requests` - HTTP запросы
- `pandas` - анализ данных
- `numpy` - численные вычисления
- `matplotlib` - построение графиков

**Пример алгоритма управления:**
```python
def pid_controller(setpoint, current_value, kp, ki, kd):
    error = setpoint - current_value
    integral += error * dt
    derivative = (error - last_error) / dt
    
    output = kp * error + ki * integral + kd * derivative
    last_error = error
    
    return output
```''',
            'provider': 'Programming Specialist',
            'category': 'programming'
        }
    
    # C++
    elif 'c++' in message_lower or 'cpp' in message_lower:
        return {
            'response': '''C++ - мощный язык программирования для системного программирования и встраиваемых систем.

**Преимущества для автоматизации:**
- Высокая производительность
- Прямой доступ к железу
- Богатые возможности ООП
- Широкая поддержка библиотек

**Пример работы с PLC:**
```cpp
#include <iostream>
#include <windows.h>

class PLCController {
private:
    HANDLE hSerial;
    
public:
    bool connect(const char* port) {
        hSerial = CreateFile(port, GENERIC_READ | GENERIC_WRITE,
                           0, NULL, OPEN_EXISTING, 0, NULL);
        return hSerial != INVALID_HANDLE_VALUE;
    }
    
    void writeCommand(const char* cmd) {
        DWORD bytesWritten;
        WriteFile(hSerial, cmd, strlen(cmd), &bytesWritten, NULL);
    }
    
    std::string readResponse() {
        char buffer[256];
        DWORD bytesRead;
        ReadFile(hSerial, buffer, sizeof(buffer), &bytesRead, NULL);
        return std::string(buffer, bytesRead);
    }
};
```

**Для встраиваемых систем:**
- Минимальное потребление памяти
- Прямой доступ к регистрам
- Оптимизация компилятора
- Real-time возможности''',
            'provider': 'Programming Specialist',
            'category': 'programming'
        }
    
    # Общий ответ по программированию
    else:
        return {
            'response': '''Я специализируюсь на программировании для автоматизации и промышленных систем. Могу помочь с:

**Языки программирования:**
- Python - для анализа данных и автоматизации
- C/C++ - для встраиваемых систем
- JavaScript - для веб-интерфейсов
- Ladder Logic - для PLC

**Алгоритмы:**
- ПИД-регуляторы
- Фильтрация сигналов
- Управление процессами
- Обработка данных

**Интеграция с оборудованием:**
- Работа с COM портами
- Modbus протокол
- OPC UA
- REST API

**Примеры кода:**
- Управление конвейерами
- Мониторинг датчиков
- Обработка аварийных ситуаций
- Логирование данных

Задайте конкретный вопрос, и я покажу примеры кода!''',
            'provider': 'Programming Specialist',
            'category': 'programming'
        }

def get_electrical_response(message):
    """Специализированные ответы по электротехнике"""
    return {
        'response': '''Я специализируюсь на электротехнике и могу помочь с:

**Основы электротехники:**
- Закон Ома: U = I × R
- Мощность: P = U × I
- Последовательное и параллельное соединение

**Электрические схемы:**
- Чтение принципиальных схем
- Расчет токов и напряжений
- Выбор компонентов

**Автоматизация:**
- Реле и контакторы
- Датчики и исполнительные механизмы
- Системы управления

**Безопасность:**
- Защита от коротких замыканий
- Заземление и зануление
- Нормы и стандарты

Задайте конкретный вопрос по электротехнике!''',
        'provider': 'Electrical Specialist',
        'category': 'electrical'
    }

def get_radiomechanics_response(message):
    """Специализированные ответы по радиомеханике"""
    return {
        'response': '''Я специализируюсь на радиомеханике и могу помочь с:

**Радиотехника:**
- Антенны и их характеристики
- Распространение радиоволн
- Модуляция сигналов

**Электроника:**
- Усилители и генераторы
- Фильтры и резонаторы
- Цифровая обработка сигналов

**Измерения:**
- Осциллографы и анализаторы
- Генераторы сигналов
- Измерительные антенны

**Связь:**
- Радиосвязь
- Спутниковая связь
- Оптоволоконная связь

Задайте конкретный вопрос по радиомеханике!''',
        'provider': 'Radiomechanics Specialist',
        'category': 'radiomechanics'
    }

# Статические файлы
@app.route('/<path:filename>')
def serve_static(filename):
    """Обслуживание статических файлов"""
    try:
        return send_from_directory('../matrix', filename)
    except:
        return "Файл не найден", 404

if __name__ == '__main__':
    initialize_system()
    
    logger.info(f"Запуск Rubin AI v2.0 Stable сервера на {StableConfig.HOST}:{StableConfig.PORT}")
    logger.info(f"Веб-интерфейс: http://localhost:{StableConfig.PORT}/RubinIDE.html")
    logger.info(f"API документация: http://localhost:{StableConfig.PORT}/api/health")
    
    app.run(
        host=StableConfig.HOST,
        port=StableConfig.PORT,
        debug=StableConfig.DEBUG,
        threaded=True
    )

















