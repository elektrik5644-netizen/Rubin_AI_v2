"""
Rubin AI v2.0 - Упрощенная стабильная версия
Без внешних зависимостей, только встроенные ответы
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

# Импортируем новые модули самоисправления
try:
    from error_logger import error_logger
    from auto_healer import auto_healer
    from backup_manager import backup_manager
    from performance_optimizer import performance_optimizer
    SELF_HEALING_ENABLED = True
except ImportError as e:
    print(f"⚠️ Модули самоисправления недоступны: {e}")
    SELF_HEALING_ENABLED = False

# Инициализация Flask приложения
app = Flask(__name__)
CORS(app, 
     origins=['http://localhost:8084', 'http://127.0.0.1:8084', 
              'http://localhost:8085', 'http://127.0.0.1:8085',
              'file://', '*'],
     allow_headers=['Content-Type', 'Authorization', 'X-Requested-With'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     supports_credentials=True)

# Настройка логирования (без эмодзи)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rubin_ai_v2_simple.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("rubin_ai_v2_simple")

# Глобальные переменные
documents_storage = []
hybrid_search_engine = None
intelligent_dispatcher = None

def initialize_system():
    """Инициализация системы"""
    global documents_storage, hybrid_search_engine, intelligent_dispatcher
    
    try:
        logger.info("Инициализация Rubin AI v2.0 Simple...")
        
        # Загрузка документов
        load_documents()
        
        # Инициализация гибридного поиска (опционально)
        logger.info("Попытка инициализации гибридного поиска...")
        try:
            # Добавляем корневую папку в путь
            import sys
            import os
            root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if root_path not in sys.path:
                sys.path.insert(0, root_path)
            
            from hybrid_search import HybridSearchEngine
            logger.info("Модуль hybrid_search импортирован")
            hybrid_search_engine = HybridSearchEngine()
            logger.info("Гибридный поиск инициализирован")
        except Exception as e:
            logger.warning(f"Гибридный поиск недоступен: {e}")
            hybrid_search_engine = None
        
        # Инициализация интеллектуального диспетчера
        logger.info("Попытка инициализации интеллектуального диспетчера...")
        try:
            # Добавляем корневую папку в путь (если еще не добавлена)
            import sys
            import os
            root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if root_path not in sys.path:
                sys.path.insert(0, root_path)
            
            from intelligent_dispatcher import get_intelligent_dispatcher
            logger.info("Модуль intelligent_dispatcher импортирован")
            intelligent_dispatcher = get_intelligent_dispatcher()
            logger.info("Интеллектуальный диспетчер инициализирован")
        except Exception as e:
            logger.warning(f"Интеллектуальный диспетчер недоступен: {e}")
            intelligent_dispatcher = None
        
        logger.info("Rubin AI v2.0 Simple успешно инициализирован!")
        
    except Exception as e:
        logger.error(f"Ошибка инициализации: {e}")

def load_documents():
    """Загрузка документов из базы данных"""
    global documents_storage
    
    try:
        # Проверяем наличие базы данных
        if os.path.exists('rubin_ai_documents.db'):
            import sqlite3
            conn = sqlite3.connect('rubin_ai_documents.db')
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM documents')
            count = cursor.fetchone()[0]
            conn.close()
            
            if count > 0:
                documents_storage = []  # Используем базу данных вместо pickle
                logger.info(f"База данных найдена с {count} документами")
            else:
                documents_storage = []
                logger.info("База данных пуста, создаем новое хранилище")
        else:
            documents_storage = []
            logger.info("База данных не найдена, создаем новое хранилище")
    except Exception as e:
        logger.error(f"Ошибка загрузки документов: {e}")
        documents_storage = []

# API Endpoints

@app.route('/')
def index():
    """Главная страница"""
    return jsonify({
        'name': 'Rubin AI v2.0 Simple',
        'version': '2.0.0-simple',
        'description': 'Упрощенная стабильная версия Rubin AI',
        'status': 'running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health')
@app.route('/health')
def health_check():
    """Проверка здоровья системы"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'providers': {
            'specialized': True,
            'huggingface': False,
            'openai': False
        },
        'documents_count': len(documents_storage),
        'config': {
            'debug': True,
            'port': 8084,
            'available_providers': ['specialized']
        }
    })

@app.route('/api/stats')
def get_stats():
    """Получить статистику системы"""
    return jsonify({
        'system': {
            'name': 'Rubin AI v2.0 Simple',
            'version': '2.0.0-simple',
            'uptime': 'running',
            'timestamp': datetime.now().isoformat()
        },
        'providers': {
            'specialized': True
        },
        'documents': {
            'total': len(documents_storage),
            'storage_size': os.path.getsize('documents_storage_v2.pkl') if os.path.exists('documents_storage_v2.pkl') else 0
        }
    })

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def ai_chat():
    """Основной endpoint для AI чата с интеллектуальной маршрутизацией"""
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
        
        # Использование интеллектуального диспетчера
        if intelligent_dispatcher:
            logger.info("Используется интеллектуальная маршрутизация")
            response, selected_module = intelligent_dispatcher.route_request(message, data)
            
            if response:
                response['timestamp'] = datetime.now().isoformat()
                response['dispatcher_used'] = True
                response['selected_module'] = selected_module
                logger.info(f"Ответ получен от модуля {selected_module}")
                return jsonify(response)
            else:
                logger.warning("Интеллектуальный диспетчер не смог обработать запрос, используется fallback")
        
        # Fallback на старую логику
        logger.info("Используется fallback маршрутизация")
        
        # Простые быстрые ответы
        simple_responses = {
            'привет': "Привет! Я Rubin AI - ваш помощник по программированию, электротехнике и автоматизации. Чем могу помочь?",
            'как дела': "У меня все отлично! Готов помочь с техническими вопросами. Что вас интересует?",
            'статус': "Система Rubin AI работает нормально. Все модули активны и готовы к работе.",
            'помощь': "Я могу помочь с программированием, электротехникой, радиомеханикой и контроллерами. Задайте конкретный вопрос!"
        }
        
        message_lower = message.lower().strip()
        if message_lower in simple_responses:
            logger.info("Быстрый ответ")
            return jsonify({
                'response': simple_responses[message_lower],
                'provider': 'fast_cache',
                'category': 'general_chat',
                'cached': True,
                'response_time': 0.001,
                'dispatcher_used': False
            })
        
        # Определение категории
        detected_category = detect_category(message, category)
        logger.info(f"Определена категория: {detected_category}")
        
        # Получение ответа
        response = get_specialized_response(message, detected_category, context)
        
        # Добавляем timestamp
        response['timestamp'] = datetime.now().isoformat()
        response['dispatcher_used'] = False
        
        logger.info(f"Ответ получен от {response.get('provider', 'unknown')}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Ошибка в AI чате: {e}")
        return jsonify({
            'error': f'Ошибка обработки запроса: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

def detect_category(message, category):
    """Определение категории запроса"""
    if category:
        return category
    
    message_lower = message.lower()
    
    # Контроллеры и автоматизация
    if any(word in message_lower for word in ['пид', 'pid', 'scada', 'plc', 'pmac', 'контроллер', 'автоматизация', 'регулятор']):
        return 'controllers'
    
    # Программирование
    elif any(word in message_lower for word in ['c++', 'cpp', 'python', 'программирование', 'код', 'алгоритм', 'функция', 'класс']):
        return 'programming'
    
    # Электротехника
    elif any(word in message_lower for word in ['электричество', 'схема', 'резистор', 'конденсатор', 'индуктивность', 'напряжение', 'ток']):
        return 'electrical'
    
    # Радиомеханика
    elif any(word in message_lower for word in ['радио', 'антенна', 'сигнал', 'модуляция', 'частота', 'волна']):
        return 'radiomechanics'
    
    # Общие вопросы
    else:
        return 'general'

def get_specialized_response(message, category, context):
    """Получение специализированного ответа"""
    
    if category == 'controllers':
        return get_controllers_response(message)
    elif category == 'programming':
        return get_programming_response(message)
    elif category == 'electrical':
        return get_electrical_response(message)
    elif category == 'radiomechanics':
        return get_radiomechanics_response(message)
    else:
        return get_general_response(message)

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
u(t) = Kp × e(t) + Ki × ∫e(t)dt + Kd × de(t)/dt

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
ser.write(b'READ D100\\n')
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

def get_general_response(message):
    """Общие ответы с использованием гибридного поиска и локального LLM"""
    
    message_lower = message.lower()
    
    # Проверяем на общие вопросы и приветствия - НЕ ищем в базе данных
    if any(phrase in message_lower for phrase in [
        "какой хороший день", "хороший день", "как дела", "как поживаешь",
        "что нового", "как настроение", "как жизнь", "привет", "hello", "hi"
    ]):
        logger.info("Обнаружен общий вопрос/приветствие, используем fallback ответ")
        # Импортируем локальный LLM провайдер для fallback
        try:
            from providers.local_llm_provider import LocalLLMProvider
            local_llm = LocalLLMProvider()
            local_llm.initialize()
            
            # Генерируем fallback ответ
            llm_response = local_llm.get_response(message, {'task_type': 'general_chat', 'documents': []})
            
            if llm_response.get('success', True):
                return {
                    'response': llm_response['content'],
                    'provider': 'Local LLM (Fallback)',
                    'category': 'general',
                    'search_results_count': 0
                }
        except Exception as e:
            logger.error(f"Ошибка fallback ответа: {e}")
    
    # Импортируем локальный LLM провайдер
    try:
        from providers.local_llm_provider import LocalLLMProvider
        local_llm = LocalLLMProvider()
        local_llm.initialize()
    except ImportError as e:
        logger.error(f"Не удалось импортировать LocalLLMProvider: {e}")
        local_llm = None
    
    # Сначала пробуем последовательный поиск
    try:
        # Импортируем последовательный поиск
        from sequential_search_engine import SequentialSearchEngine
        
        # Создаем экземпляр последовательного поиска
        sequential_search_engine = SequentialSearchEngine()
        
        logger.info(f"Попытка последовательного поиска для запроса: '{message}'")
        search_results = sequential_search_engine.search(message, limit=3)
        logger.info(f"Последовательный поиск вернул {len(search_results) if search_results else 0} результатов")
        
        if search_results and len(search_results) > 0:
            # Используем локальный LLM для генерации ответа
            if local_llm:
                try:
                    logger.info("Генерируем ответ с помощью локального LLM")
                    context = {
                        'task_type': 'general_chat',
                        'documents': search_results
                    }
                    llm_response = local_llm.get_response(message, context)
                    
                    if llm_response.get('success', True):
                        logger.info("Локальный LLM успешно сгенерировал ответ")
                        return {
                            'response': llm_response['content'],
                            'provider': 'Local LLM + Sequential Search',
                            'category': 'general',
                            'search_results_count': len(search_results),
                            'documents_used': len(search_results)
                        }
                    else:
                        logger.warning(f"Ошибка локального LLM: {llm_response.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"Ошибка локального LLM: {e}")
            
            # Fallback: показываем найденные документы как раньше
            logger.info("Используем fallback форматирование документов")
            response_text = "**Найденная информация:**\n\n"
            
            for i, result in enumerate(search_results[:2], 1):
                response_text += f"**{i}. {result.get('file_name', result.get('title', 'Документ'))}**\n"
                content = result.get('content_preview', result.get('content', ''))
                # Показываем больше текста, но не более 1000 символов
                if len(content) > 1000:
                    response_text += f"{content[:1000]}...\n\n"
                else:
                    response_text += f"{content}\n\n"
            
            response_text += "\n*Ответ основан на документах из базы знаний Rubin AI*"
            
            logger.info(f"Последовательный поиск нашел {len(search_results)} результатов, возвращаем ответ")
            return {
                'response': response_text,
                'provider': 'Sequential Search',
                'category': 'general',
                'search_results_count': len(search_results)
            }
        else:
            logger.warning("Последовательный поиск не нашел результатов")
    except Exception as e:
        logger.error(f"Ошибка последовательного поиска: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Используем локальный LLM для fallback ответа
    if local_llm:
        try:
            logger.info("Генерируем fallback ответ с помощью локального LLM")
            context = {
                'task_type': 'general_chat',
                'documents': []
            }
            llm_response = local_llm.get_response(message, context)
            
            if llm_response.get('success', True):
                logger.info("Локальный LLM успешно сгенерировал fallback ответ")
                return {
                    'response': llm_response['content'],
                    'provider': 'Local LLM',
                    'category': 'general',
                    'search_results_count': 0
                }
        except Exception as e:
            logger.error(f"Ошибка локального LLM fallback: {e}")
    
    # Последний fallback на шаблонный ответ
    return {
        'response': '''Привет! Я Rubin AI - ваш помощник по техническим вопросам.

Я специализируюсь на:
- **Программировании** (Python, C++, алгоритмы)
- **Электротехнике** (схемы, расчеты, компоненты)
- **Радиомеханике** (антенны, сигналы, измерения)
- **Контроллерах** (PLC, ПИД-регуляторы, автоматизация)

Задайте конкретный вопрос, и я дам подробный ответ с примерами!

**Примеры вопросов:**
- "Как написать ПИД-регулятор на Python?"
- "Объясни принцип работы PLC"
- "Как рассчитать ток в цепи?"
- "Что такое модуляция сигнала?"''',
        'provider': 'General Assistant',
        'category': 'general'
    }

# Гибридный поиск
@app.route('/api/hybrid/search', methods=['POST'])
def hybrid_search():
    """Гибридный поиск документов"""
    try:
        if not hybrid_search_engine:
            return jsonify({
                'error': 'Hybrid search engine not available'
            }), 503
            
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Query parameter is required'
            }), 400
            
        query = data['query']
        limit = data.get('limit', 10)
        search_type = data.get('search_type', 'hybrid')
        
        logger.info(f"Гибридный поиск: '{query}' (limit={limit}, type={search_type})")
        
        # Выполнение поиска
        results = hybrid_search_engine.search(query, limit, search_type)
        
        return jsonify({
            'query': query,
            'results': results,
            'total_found': len(results),
            'search_type': search_type
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка гибридного поиска: {e}")
        return jsonify({
            'error': str(e)
        }), 500
    
# Статистика диспетчера
@app.route('/api/dispatcher/stats', methods=['GET'])
def dispatcher_stats():
    """Статистика интеллектуального диспетчера"""
    try:
        if not intelligent_dispatcher:
            return jsonify({
                'error': 'Intelligent dispatcher not available'
            }), 503
        
        stats = intelligent_dispatcher.get_statistics()
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики диспетчера: {e}")
        return jsonify({
            'error': str(e)
        }), 500

# Управление диспетчером
@app.route('/api/dispatcher/config', methods=['POST'])
def dispatcher_config():
    """Конфигурация диспетчера"""
    try:
        if not intelligent_dispatcher:
            return jsonify({
                'error': 'Intelligent dispatcher not available'
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Configuration data is required'
            }), 400
        
        # Обновление конфигурации
        if 'load_balancing' in data:
            intelligent_dispatcher.set_load_balancing(data['load_balancing'])
            logger.info(f"Балансировка нагрузки: {data['load_balancing']}")
        
        return jsonify({
            'message': 'Configuration updated successfully',
            'load_balancing': intelligent_dispatcher.load_balanced
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка конфигурации диспетчера: {e}")
        return jsonify({
            'error': str(e)
        }), 500

# Управление системой
@app.route('/api/system/restart', methods=['POST'])
def restart_system():
    """Перезагрузка системы"""
    try:
        logger.info("🔄 Запрос на перезагрузку системы получен")
        
        # Логируем ошибку перезапуска для анализа
        if SELF_HEALING_ENABLED:
            error_logger.log_error(
                error_type="system_restart",
                error_message="Manual system restart requested",
                module="system",
                severity=2,
                context="User requested restart"
            )
        
        # Здесь можно добавить логику для перезагрузки компонентов
        # Пока что просто возвращаем успешный ответ
        
        return jsonify({
            'success': True,
            'message': 'Система будет перезагружена',
            'timestamp': datetime.now().isoformat(),
            'note': 'Для полной перезагрузки необходимо перезапустить сервисы вручную'
        })
        
    except Exception as e:
        logger.error(f"Ошибка при перезагрузке системы: {e}")
        
        # Логируем ошибку
        if SELF_HEALING_ENABLED:
            error_logger.log_error(
                error_type="restart_failed",
                error_message=str(e),
                module="system",
                severity=3,
                context="System restart failed"
            )
        
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Получить статус системы"""
    try:
        import requests
        
        # Проверяем статус всех модулей
        modules_status = {
            'ai_chat': True,  # Основной сервер работает
            'electrical': False,
            'radiomechanics': False,
            'controllers': False,
            'documents': False
        }
        
        # Реальные проверки статуса других сервисов
        try:
            # Проверяем сервис электротехники (8087)
            response = requests.get('http://localhost:8087/health', timeout=2)
            if response.status_code == 200:
                modules_status['electrical'] = True
        except:
            pass
        
        try:
            # Проверяем сервис радиомеханики (8089)
            response = requests.get('http://localhost:8089/health', timeout=2)
            if response.status_code == 200:
                modules_status['radiomechanics'] = True
        except:
            pass
        
        try:
            # Проверяем сервис контроллеров (8090)
            response = requests.get('http://localhost:8090/health', timeout=2)
            if response.status_code == 200:
                modules_status['controllers'] = True
        except:
            pass
        
        try:
            # Проверяем сервис документов (8088)
            response = requests.get('http://localhost:8088/health', timeout=2)
            if response.status_code == 200:
                modules_status['documents'] = True
        except:
            pass
        
        return jsonify({
            'success': True,
            'modules': modules_status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка при получении статуса системы: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# API для самоисправления системы
if SELF_HEALING_ENABLED:
    
    @app.route('/api/self-healing/diagnose', methods=['GET'])
    def diagnose_system():
        """Диагностика системы"""
        try:
            diagnosis = auto_healer.diagnose_system()
            return jsonify(diagnosis)
        except Exception as e:
            logger.error(f"Ошибка диагностики: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/self-healing/auto-heal', methods=['POST'])
    def auto_heal_system():
        """Автоматическое восстановление системы"""
        try:
            healing_result = auto_healer.auto_heal()
            return jsonify(healing_result)
        except Exception as e:
            logger.error(f"Ошибка автоматического восстановления: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/self-healing/error-stats', methods=['GET'])
    def get_error_statistics():
        """Получение статистики ошибок"""
        try:
            hours = request.args.get('hours', 24, type=int)
            stats = error_logger.get_error_statistics(hours)
            return jsonify(stats)
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/self-healing/predictions', methods=['GET'])
    def get_predictions():
        """Получение предсказаний сбоев"""
        try:
            predictions = error_logger.get_predictions()
            return jsonify(predictions)
        except Exception as e:
            logger.error(f"Ошибка получения предсказаний: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/self-healing/backup', methods=['POST'])
    def create_backup():
        """Создание резервной копии"""
        try:
            data = request.get_json() or {}
            backup_name = data.get('name')
            include_data = data.get('include_data', True)
            include_config = data.get('include_config', True)
            include_logs = data.get('include_logs', False)
            
            backup_result = backup_manager.create_backup(
                backup_name, include_data, include_config, include_logs
            )
            return jsonify(backup_result)
        except Exception as e:
            logger.error(f"Ошибка создания резервной копии: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/self-healing/backup/list', methods=['GET'])
    def list_backups():
        """Список резервных копий"""
        try:
            backups = backup_manager.list_backups()
            return jsonify(backups)
        except Exception as e:
            logger.error(f"Ошибка получения списка резервных копий: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/self-healing/backup/restore', methods=['POST'])
    def restore_backup():
        """Восстановление из резервной копии"""
        try:
            data = request.get_json() or {}
            backup_name = data.get('name')
            restore_data = data.get('restore_data', True)
            restore_config = data.get('restore_config', True)
            restore_logs = data.get('restore_logs', False)
            
            restore_result = backup_manager.restore_backup(
                backup_name, restore_data, restore_config, restore_logs
            )
            return jsonify(restore_result)
        except Exception as e:
            logger.error(f"Ошибка восстановления: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/self-healing/backup/delete', methods=['DELETE'])
    def delete_backup():
        """Удаление резервной копии"""
        try:
            data = request.get_json() or {}
            backup_name = data.get('name')
            
            delete_result = backup_manager.delete_backup(backup_name)
            return jsonify(delete_result)
        except Exception as e:
            logger.error(f"Ошибка удаления резервной копии: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/performance/analyze', methods=['GET'])
    def analyze_performance():
        """Анализ производительности системы"""
        try:
            analysis = performance_optimizer.analyze_performance()
            return jsonify(analysis)
        except Exception as e:
            logger.error(f"Ошибка анализа производительности: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/performance/optimize', methods=['POST'])
    def optimize_performance():
        """Оптимизация производительности"""
        try:
            data = request.get_json() or {}
            module_name = data.get('module')
            
            if module_name:
                result = performance_optimizer.optimize_module(module_name)
            else:
                result = performance_optimizer.optimize_all_modules()
            
            return jsonify(result)
        except Exception as e:
            logger.error(f"Ошибка оптимизации производительности: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/performance/history', methods=['GET'])
    def get_optimization_history():
        """Получение истории оптимизации"""
        try:
            limit = request.args.get('limit', 10, type=int)
            history = performance_optimizer.get_optimization_history(limit)
            return jsonify(history)
        except Exception as e:
            logger.error(f"Ошибка получения истории оптимизации: {e}")
            return jsonify({'error': str(e)}), 500

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
    
    logger.info("Запуск Rubin AI v2.0 Simple сервера на 0.0.0.0:8084")
    logger.info("Веб-интерфейс: http://localhost:8084/RubinIDE.html")
    logger.info("API документация: http://localhost:8084/api/health")
    
    app.run(
        host='0.0.0.0',
        port=8084,
        debug=False,
        threaded=True
    )
