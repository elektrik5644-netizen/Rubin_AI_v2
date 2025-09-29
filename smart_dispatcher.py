#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой диспетчер для правильной маршрутизации запросов
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import logging
import os
import json
from datetime import datetime
from ethical_core import ActionType, assess_action

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация Flask приложения
app = Flask(__name__)
CORS(app)

# Импорт менеджера директив
try:
    from directives_manager import check_and_apply_directives, process_directives_command
    DIRECTIVES_AVAILABLE = True
except ImportError:
    DIRECTIVES_AVAILABLE = False

def get_base_url():
    """Определяет базовый URL в зависимости от окружения"""
    return "host.docker.internal" if os.getenv("DOCKER_ENV") else "localhost"

def check_ethical_core_availability():
    """Динамическая проверка доступности Ethical Core"""
    try:
        response = requests.get(f"http://{get_base_url()}:8105/api/health", timeout=1)
        return response.status_code == 200
    except:
        return False

def handle_arduino_nano_query(query):
    """Встроенная обработка запросов Arduino Nano"""
    query_lower = query.lower()
    
    # База знаний Arduino Nano
    arduino_knowledge = {
        'pins': {
            'digital': ['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13'],
            'analog': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7'],
            'pwm': ['D3', 'D5', 'D6', 'D9', 'D10', 'D11'],
            'power': ['5V', '3.3V', 'GND', 'VIN']
        },
        'functions': {
            'digitalwrite': 'digitalWrite(pin, value) - устанавливает HIGH или LOW на цифровой пин',
            'digitalread': 'digitalRead(pin) - читает состояние цифрового пина',
            'analogread': 'analogRead(pin) - читает аналоговое значение (0-1023)',
            'analogwrite': 'analogWrite(pin, value) - PWM сигнал (0-255)',
            'pinmode': 'pinMode(pin, mode) - настраивает пин как INPUT или OUTPUT',
            'delay': 'delay(ms) - пауза в миллисекундах',
            'millis': 'millis() - возвращает время с запуска в миллисекундах'
        },
        'libraries': {
            'servo': 'Servo.h - управление сервоприводами',
            'wire': 'Wire.h - I2C коммуникация',
            'spi': 'SPI.h - SPI коммуникация',
            'eeprom': 'EEPROM.h - работа с энергонезависимой памятью',
            'wifi': 'WiFi.h - подключение к WiFi (ESP32/ESP8266)',
            'bluetooth': 'BluetoothSerial.h - Bluetooth коммуникация'
        },
        'projects': {
            'led_blink': 'Мигание светодиода - базовый проект',
            'button_led': 'Управление светодиодом кнопкой',
            'servo_control': 'Управление сервоприводом',
            'sensor_read': 'Чтение данных с датчиков',
            'motor_control': 'Управление моторами',
            'lcd_display': 'Вывод информации на LCD дисплей'
        }
    }
    
    # Анализ запроса и формирование ответа
    response_parts = []
    
    if any(word in query_lower for word in ['пин', 'pin', 'пины', 'pins']):
        response_parts.append("**Пины Arduino Nano:**")
        for pin_type, pins in arduino_knowledge['pins'].items():
            response_parts.append(f"- {pin_type.upper()}: {', '.join(pins)}")
    
    if any(word in query_lower for word in ['функция', 'function', 'функции', 'functions']):
        response_parts.append("\n**Основные функции:**")
        for func, desc in arduino_knowledge['functions'].items():
            response_parts.append(f"- {func}: {desc}")
    
    if any(word in query_lower for word in ['библиотека', 'library', 'библиотеки', 'libraries']):
        response_parts.append("\n**Популярные библиотеки:**")
        for lib, desc in arduino_knowledge['libraries'].items():
            response_parts.append(f"- {lib}: {desc}")
    
    if any(word in query_lower for word in ['проект', 'project', 'проекты', 'projects']):
        response_parts.append("\n**Примеры проектов:**")
        for proj, desc in arduino_knowledge['projects'].items():
            response_parts.append(f"- {proj}: {desc}")
    
    if any(word in query_lower for word in ['ошибка', 'error', 'проблема', 'problem', 'не работает']):
        response_parts.append("\n**Частые проблемы и решения:**")
        response_parts.append("- Проверьте подключение USB кабеля")
        response_parts.append("- Убедитесь, что выбран правильный порт в IDE")
        response_parts.append("- Проверьте правильность подключения компонентов")
        response_parts.append("- Убедитесь в корректности синтаксиса кода")
    
    if not response_parts:
        response_parts.append("**Arduino Nano - микроконтроллер:**")
        response_parts.append("- 14 цифровых пинов (6 с PWM)")
        response_parts.append("- 8 аналоговых пинов")
        response_parts.append("- Рабочее напряжение: 5V")
        response_parts.append("- Питание: USB или внешний источник")
        response_parts.append("- Процессор: ATmega328P")
    
    return "\n".join(response_parts)

# Проверка доступности Ethical Core
ETHICAL_CORE_AVAILABLE = False

# Контекстная память для диалогов
CONVERSATION_HISTORY = {
    "sessions": {},
    "global_context": {
        "session_start": datetime.now().isoformat(),
        "total_interactions": 0,
        "last_topics": []
    }
}

def get_session_id():
    """Получает ID сессии из заголовков или создает новый"""
    session_id = request.headers.get('X-Session-ID', 'default')
    if session_id not in CONVERSATION_HISTORY["sessions"]:
        CONVERSATION_HISTORY["sessions"][session_id] = {
            "start_time": datetime.now().isoformat(),
            "messages": [],
            "context": {
                "current_topic": None,
                "last_module": None,
                "user_preferences": {}
            }
        }
    return session_id

def add_to_history(session_id, message, category, response):
    """Добавляет сообщение в историю диалога"""
    CONVERSATION_HISTORY["sessions"][session_id]["messages"].append({
        "timestamp": datetime.now().isoformat(),
        "message": message,
        "category": category,
        "response": response[:200] + "..." if len(response) > 200 else response
    })
    
    # Обновляем контекст
    CONVERSATION_HISTORY["sessions"][session_id]["context"]["last_module"] = category
    CONVERSATION_HISTORY["global_context"]["total_interactions"] += 1
    
    # Сохраняем последние темы
    if category not in CONVERSATION_HISTORY["global_context"]["last_topics"]:
        CONVERSATION_HISTORY["global_context"]["last_topics"].append(category)
        if len(CONVERSATION_HISTORY["global_context"]["last_topics"]) > 5:
            CONVERSATION_HISTORY["global_context"]["last_topics"].pop(0)

def get_context_for_message(session_id, message):
    """Получает контекст для текущего сообщения"""
    session = CONVERSATION_HISTORY["sessions"][session_id]
    context = {
        "recent_messages": session["messages"][-3:] if len(session["messages"]) > 0 else [],
        "last_module": session["context"]["last_module"],
        "session_duration": (datetime.now() - datetime.fromisoformat(session["start_time"])).seconds,
        "total_interactions": CONVERSATION_HISTORY["global_context"]["total_interactions"]
    }
    
    # Добавляем контекст к сообщению
    if context["recent_messages"]:
        context_hint = f"[Контекст: предыдущие темы: {', '.join([m['category'] for m in context['recent_messages']])}] "
        return context_hint + message
    return message

# Конфигурация серверов
SERVERS = {
    'electrical': {
        'port': 8087,
        'endpoint': '/api/electrical/solve',
        'keywords': ['закон', 'кирхгофа', 'резистор', 'резисторы', 'транзистор', 'транзисторы', 'диод', 'диоды', 'конденсатор', 'конденсаторы', 'контактор', 'реле', 'мощность', 'ток', 'напряжение', 'схема', 'схемы', 'электрические', 'электричество', 'цепи', 'тиристор', 'симистр', 'ом', 'закон ома', 'электрическая цепь', 'сопротивление', 'катушка', 'индуктивность', 'емкость', 'коэффициент мощности', 'power factor', 'cos φ', 'cosφ', 'реактивная мощность', 'как работает', 'как устроен', 'принцип работы', 'электротехника', 'электроника', 'электронные компоненты']
    },
    'radiomechanics': {
        'port': 8089,
        'endpoint': '/api/chat',
        'keywords': ['антенна', 'сигнал', 'радио', 'модуляция', 'частота', 'передатчик', 'приемник', 'радиоволны', 'электромагнитные', 'передача данных']
    },
    'controllers': {
        'port': 9000,
        'endpoint': '/api/controllers/topic/general',
        'keywords': ['пид', 'регулятор', 'plc', 'ПЛК', 'плк', 'контроллер', 'автоматизация', 'логика', 'события', 'прерывания', 'events', 'interrupts', 'ascii', 'команды', 'протокол', 'отправка', 'получение', 'ответы', 'чпу', 'cnc', 'числовое', 'программное', 'управление', 'передача', 'данные', 'g-коды', 'координаты', 'pmac', 'многоосевой', 'движение', 'позиционирование', 'траектория', 'ось', 'оси', 'серводвигатель', 'шаговый', 'энкодер', 'обратная связь', 'сервопривод', 'настроить', 'настройка', 'конфигурация', 'параметры', 'scada', 'скада', 'мониторинг', 'диспетчеризация', 'i130', 'i130a', 'i130b', 'i130c', 'i130d', 'i130e', 'i130f', 'i130g', 'i130h', 'i130i', 'i130j', 'i130k', 'i130l', 'i130m', 'i130n', 'i130o', 'i130p', 'i130q', 'i130r', 'i130s', 'i130t', 'i130u', 'i130v', 'i130w', 'i130x', 'i130y', 'i130z']
    },
    'mathematics': {
        'port': 8086,
        'endpoint': '/api/chat',
        'keywords': ['уравнение', 'квадратное', 'математика', 'алгебра', 'геометрия', 'арифметика', '+', '-', '*', '/', '=', 'вычислить', 'посчитать', 'сложить', 'вычесть', 'умножить', 'разделить', 'число', 'цифра', 'результат', 'ответ', 'интеграл', 'производная', 'функция', 'график', 'система уравнений']
    },
    'programming': {
        'port': 8088,
        'endpoint': '/api/programming/explain',
        'keywords': ['продвинутые', 'специфические', 'функции', 'алгоритмы', 'алгоритм', 'программирование', 'програмировать', 'программировать', 'код', 'разработка', 'python', 'javascript', 'c++', 'java', 'автоматизация', 'промышленная', 'конвейер', 'управление', 'сортировка', 'ошибки', 'error', 'xml', 'обработка', 'debug', 'отладка', 'синтаксис', 'переменные', 'переменная', 'логика', 'управления', 'if', 'endif', 'условия', 'циклы', 'функции', 'методы', 'классы', 'объекты', 'как писать', 'как создать', 'как сделать', 'написать код', 'создать программу', 'разработать', 'программист', 'разработчик']
    },
    'physics': {
        'port': 8110,
        'endpoint': '/api/physics/explain',
        'keywords': ['фотон', 'электрон', 'протон', 'нейтрон', 'атом', 'молекула', 'квант', 'квантовая', 'физика', 'механика', 'термодинамика', 'оптика', 'электродинамика', 'ядерная физика', 'релятивистская', 'эйнштейн', 'ньютон', 'законы ньютона', 'гравитация', 'магнетизм', 'электромагнитное поле', 'волна', 'частица', 'энергия', 'масса', 'скорость света', 'планк', 'бозон', 'фермион', 'спин', 'орбиталь', 'изотоп', 'радиоактивность', 'ядерная реакция', 'синтез', 'деление', 'плазма', 'сверхпроводимость', 'криогеника', 'лазер', 'полупроводник', 'диэлектрик', 'проводник', 'изолятор', 'что такое', 'что такой', 'объясни', 'расскажи']
    },
    'general': {
        'port': 8085,
        'endpoint': '/api/chat',
        'keywords': ['привет', 'hello', 'hi', 'здравствуй', 'помощь', 'help', 'справка', 'статус', 'status', 'работает', 'онлайн', 'как', 'объясни', 'расскажи']
    },
    'neuro': {
        'port': 8090,
        'endpoint': '/api/neuro/chat',
        'keywords': ['нейросеть', 'нейронная сеть', 'neural network', 'машинное обучение', 'обучение', 'обучается', 'обучать', 'тренировка', 'тренировать', 'ml', 'ai', 'искусственный интеллект', 'торговля', 'финансы', 'акции', 'валюты', 'криптовалюты', 'прогноз', 'анализ', 'кредит', 'риск', 'заемщик', 'кредитоспособность', 'scoring', 'симуляция', 'торговый эмулятор', 'алгоритмическая торговля', 'портфель', 'инвестиции']
    },
    # НОВЫЕ ПРИОРИТЕТНЫЕ ФУНКЦИИ
    'plc_analysis': {
        'port': 8099,
        'endpoint': '/api/plc/analyze',
        'keywords': ['plc', 'анализ plc', 'программа plc', 'логика plc', 'программирование plc', 'диагностика plc', 'отладка plc', 'проверка plc', 'тестирование plc', 'симуляция plc']
    },
    'advanced_math': {
        'port': 8100,
        'endpoint': '/api/math/advanced',
        'keywords': ['квадратное уравнение', 'решить уравнение', 'система уравнений', 'интеграл', 'производная', 'дифференциальное уравнение', 'линейная алгебра', 'матрицы', 'векторы', 'комплексные числа', 'тригонометрия', 'логарифмы']
    },
    'data_processing': {
        'port': 8101,
        'endpoint': '/api/data/process',
        'keywords': ['обработка данных', 'анализ данных', 'предобработка', 'фильтрация', 'нормализация', 'стандартизация', 'временные ряды', 'статистика', 'корреляция', 'регрессия', 'кластеризация']
    },
    'search_engine': {
        'port': 8102,
        'endpoint': '/api/search/hybrid',
        'keywords': ['поиск', 'найти', 'искать', 'гибридный поиск', 'векторный поиск', 'семантический поиск', 'полнотекстовый поиск', 'индексация', 'релевантность']
    },
    'system_utils': {
        'port': 8103,
        'endpoint': '/api/system/utils',
        'keywords': ['статус системы', 'проверка системы', 'диагностика', 'мониторинг', 'логи', 'производительность', 'оптимизация', 'очистка', 'резервное копирование', 'миграция']
    },
        'gai': {
            'port': 8104,
            'endpoint': '/api/gai/generate_text',
            'keywords': ['сгенерировать', 'сгенерируй', 'создать', 'написать', 'придумать', 'разработать', 'составить', 'построить', 'сформировать', 'выработать', 'произвести', 'generate', 'create', 'write', 'develop', 'build', 'compose']
        },
        'ethical_core': {
            'port': 8105,
            'endpoint': '/api/ethical/assess',
            'keywords': ['этическое ядро', 'безопасность', 'риск', 'оценка', 'этика', 'контроль', 'veto', 'заблокировать', 'разрешить', 'проверить безопасность']
        },
        'arduino_nano': {
            'port': None,  # Встроенный модуль
            'endpoint': None,  # Встроенный модуль
            'keywords': ['arduino', 'ардуино', 'nano', 'нано', 'микроконтроллер', 'пин', 'pin', 'digitalwrite', 'analogread', 'servo', 'серво', 'светодиод', 'led', 'кнопка', 'button', 'датчик', 'sensor', 'мотор', 'motor', 'библиотека', 'library', 'функция', 'function', 'код', 'code', 'скетч', 'sketch', 'проект', 'project', 'подключение', 'connection', 'схема', 'circuit', 'программирование', 'programming', 'troubleshooting', 'ошибка', 'error', 'проблема', 'problem', 'не работает', 'не определяется', 'не загружается', 'pwm', 'шим', 'аналоговый', 'analog', 'цифровой', 'digital', 'встроенный', 'builtin', 'led_builtin']
        },
        'mcsetup': {
            'port': 8096,
            'endpoint': '/api/mcsetup/integrate/rubin',
            'keywords': ['mcsetup', 'приводы', 'моторы', 'графики', 'настройки приводов', 'анализ графиков', 'производительность моторов', 'конфигурация моторов', 'мониторинг приводов', 'диагностика моторов', 'оптимизация приводов', 'сервоприводы', 'шаговые двигатели', 'частотные преобразователи', 'pmac', 'двигатели', 'приводная система', 'mc setup', 'mc-setup', 'график моторов', 'анализ моторов', 'производительность', 'настройка моторов', 'график привода', 'анализ привода', 'настройка параметров', 'kp', 'ki', 'kd', 'pid настройка', 'рекомендации по графику', 'параметры настройки', 'настройка контроллера', 'график производительности', 'анализ производительности']
        },
        'graph_analyzer': {
            'port': 8097,
            'endpoint': '/api/graph/integrate/rubin',
            'keywords': ['анализ графиков', 'графики моторов', 'тренды', 'визуализация', 'отчеты производительности', 'статистика моторов', 'анализ трендов', 'графический анализ', 'диаграммы', 'чарты', 'график', 'диаграмма', 'график привода', 'анализ привода', 'график производительности', 'анализ производительности', 'визуализация данных', 'графический отчет', 'анализ данных', 'статистический анализ']
        }
}

# Helper to normalize provider responses to plain text for frontend display
def _extract_text_from_result(result):
    if isinstance(result, dict):
        # Direct text fields
        for key in ("response", "content", "text", "message"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
            # Nested structure like { response: { explanation: "..." } }
            if isinstance(value, dict):
                for inner in ("explanation", "response", "content", "text", "message"):
                    inner_val = value.get(inner)
                    if isinstance(inner_val, str) and inner_val.strip():
                        return inner_val
        # Fallback to JSON string if no obvious text field
        try:
            return json.dumps(result, ensure_ascii=False)
        except Exception:
            return str(result)
    if isinstance(result, list):
        return "\n".join(str(item) for item in result)
    return str(result)

def _is_greeting(message: str) -> bool:
    msg = (message or "").strip().lower()
    if not msg:
        return False
    greetings = [
        "привет", "здравствуй", "здравствуйте", "hi", "hello", "добрый день",
        "как дела", "как ты", "как поживаешь", "хай", "ку", "йо"
    ]
    return any(msg == g or msg.startswith(g) for g in greetings)

def categorize_message(message):
    """Определяет категорию сообщения"""
    message_lower = message.lower()
    
    # Подсчитываем совпадения для каждой категории
    scores = {}
    for category, config in SERVERS.items():
        score = sum(1 for keyword in config['keywords'] if keyword in message_lower)
        scores[category] = score
    
    # Приоритет для технических терминов - если есть специфические ключевые слова,
    # они имеют приоритет над общими словами
    technical_categories = ['neuro', 'electrical', 'mathematics', 'controllers', 'programming', 'plc_analysis', 'advanced_math', 'data_processing', 'gai', 'arduino_nano', 'mcsetup', 'graph_analyzer', 'physics']
    technical_scores = {cat: scores.get(cat, 0) for cat in technical_categories if scores.get(cat, 0) > 0}
    
    # Специальная логика для физики - приоритет над electrical
    if 'фотон' in message_lower or 'электрон' in message_lower or 'атом' in message_lower or 'квант' in message_lower:
        if 'physics' in technical_scores and technical_scores['physics'] > 0:
            logger.info(f"📊 Категоризация: '{message[:50]}...' → physics (приоритет физических терминов)")
            return 'physics'
    
    if technical_scores:
        # Если есть технические совпадения, выбираем лучший технический
        best_technical = max(technical_scores, key=technical_scores.get)
        logger.info(f"📊 Категоризация: '{message[:50]}...' → {best_technical} (technical score: {technical_scores[best_technical]})")
        return best_technical
    
    # Возвращаем категорию с наибольшим количеством совпадений
    if scores and max(scores.values()) > 0:
        best_category = max(scores, key=scores.get)
        logger.info(f"📊 Категоризация: '{message[:50]}...' → {best_category} (score: {scores[best_category]})")
        return best_category
    
    # Если нет совпадений, возвращаем general как fallback
    logger.info(f"❓ Неопределенная категория: '{message[:50]}...' → general (fallback)")
    return 'general'

def ethical_check(message, category):
    """Проверка сообщения через Этическое Ядро"""
    global ETHICAL_CORE_AVAILABLE
    ETHICAL_CORE_AVAILABLE = check_ethical_core_availability()
    if not ETHICAL_CORE_AVAILABLE:
        return True, "Этическое Ядро недоступно"
    
    try:
        # Определяем тип действия на основе категории
        action_type_mapping = {
            'system_utils': ActionType.SYSTEM_CONTROL,
            'gai': ActionType.GENERATION,
            'data_processing': ActionType.ANALYSIS,
            'search_engine': ActionType.NETWORK_ACCESS,
            'plc_analysis': ActionType.FILE_ACCESS,
            'advanced_math': ActionType.CALCULATION,
            'electrical': ActionType.CALCULATION,
            'mathematics': ActionType.CALCULATION,
            'programming': ActionType.GENERATION,
            'radiomechanics': ActionType.INFORMATION,
            'controllers': ActionType.INFORMATION,
            'neuro': ActionType.ANALYSIS,
            'arduino_nano': ActionType.INFORMATION,
            'general': ActionType.INFORMATION
        }
        
        action_type = action_type_mapping.get(category, ActionType.INFORMATION)
        
        # Оценка действия
        assessment = assess_action(message, action_type, {'category': category})
        
        if assessment.approved:
            return True, f"✅ Этическое Ядро: Действие безопасно (риск: {assessment.risk_score:.2f})"
        else:
            return False, f"❌ Этическое Ядро: {assessment.veto_reason}\n" \
                         f"⚠️ Проблемы: {', '.join(assessment.concerns)}\n" \
                         f"💡 Рекомендации: {', '.join(assessment.recommendations)}"
    
    except Exception as e:
        logger.error(f"Ошибка этической проверки: {e}")
        return True, f"Ошибка этической проверки: {e}"

def forward_request(category, message):
    """Пересылает запрос к соответствующему серверу"""
    if category not in SERVERS:
        return None, "Неизвестная категория"
    
    # Получаем контекст для сообщения
    session_id = get_session_id()
    contextual_message = get_context_for_message(session_id, message)
    
    # Проверяем, является ли это встроенным модулем Arduino Nano
    if category == 'arduino_nano':
        logger.info("🔧 Обработка запроса Arduino Nano как встроенного модуля")
        result = handle_arduino_nano_query(contextual_message)
        return result, None  # Возвращаем кортеж (result, error)
    
    # Получаем конфигурацию для внешнего сервера
    config = SERVERS[category]
    if not config.get('port'):
        logger.warning(f"⚠️ Модуль {category} не настроен или не имеет порта")
        return None, f'Модуль {category} недоступен'
    
    url = f"http://{get_base_url()}:{config['port']}{config['endpoint']}"

    # Предотвращаем самозацикливание: никогда не шлём запросы на 8080 (сам Smart Dispatcher)
    if config.get('port') == 8080:
        # Перенаправляем в general как безопасный дефолт
        fallback_cfg = SERVERS.get('general')
        if fallback_cfg:
            category = 'general'
            config = fallback_cfg
            url = f"http://{get_base_url()}:{config['port']}{config['endpoint']}"
    
    # Подготавливаем данные в зависимости от сервера
    if category in ['radiomechanics']:
        payload = {'concept': contextual_message}
    elif category in ['physics']:
        # Извлекаем ключевое понятие из сообщения для Physics Server
        physics_keywords = ['фотон', 'электрон', 'протон', 'нейтрон', 'атом', 'молекула', 'квант', 'квантовая', 'физика', 'механика', 'термодинамика', 'оптика', 'электродинамика', 'ядерная физика', 'релятивистская', 'эйнштейн', 'ньютон', 'законы ньютона', 'гравитация', 'магнетизм', 'электромагнитное поле', 'волна', 'частица', 'энергия', 'масса', 'скорость света', 'планк', 'бозон', 'фермион', 'спин', 'орбиталь', 'изотоп', 'радиоактивность', 'ядерная реакция', 'синтез', 'деление', 'плазма', 'сверхпроводимость', 'криогеника', 'лазер', 'полупроводник', 'диэлектрик', 'проводник', 'изолятор']
        concept = None
        for keyword in physics_keywords:
            if keyword in contextual_message.lower():
                concept = keyword
                break
        if not concept:
            concept = contextual_message  # Fallback к полному сообщению
        payload = {'concept': concept}
        logger.info(f"🔍 Physics payload: {payload}")
    elif category in ['electrical', 'controllers']:
        payload = {'message': contextual_message}
    elif category in ['programming']:
        payload = {'concept': contextual_message}
    elif category in ['plc_analysis']:
        payload = {'file_path': contextual_message, 'action': 'analyze'}
    elif category in ['advanced_math']:
        payload = {'equation': contextual_message, 'type': 'advanced'}
    elif category in ['data_processing']:
        payload = {'data': contextual_message, 'operation': 'process'}
    elif category in ['search_engine']:
        payload = {'query': contextual_message, 'type': 'hybrid'}
    elif category in ['system_utils']:
        payload = {'command': contextual_message, 'action': 'execute'}
    elif category in ['gai']:
        payload = {'prompt': contextual_message, 'max_tokens': 200, 'temperature': 0.7}
    elif category in ['mcsetup']:
        payload = {'query': contextual_message, 'analysis_type': 'general'}
    elif category in ['graph_analyzer']:
        payload = {'query': contextual_message, 'analysis_type': 'graph_analysis'}
    else:  # mathematics, general, neuro
        payload = {'message': contextual_message}
    
    try:
        logger.info(f"🌐 Отправляем запрос к {category} на {url}")
        response = requests.post(url, json=payload, timeout=15)
        
        if response.status_code == 200:
            logger.info(f"✅ Сервер {category} ответил успешно")
            result = response.json()
            
            # Проверяем и применяем директивы
            if DIRECTIVES_AVAILABLE:
                context = {
                    "category": category,
                    "message": message,
                    "response": result,
                    "session_id": session_id
                }
                directive_results = check_and_apply_directives(context)
                if directive_results:
                    logger.info(f"📋 Применены директивы: {len(directive_results)}")
                    # Добавляем результаты директив к ответу
                    if isinstance(result, dict):
                        result["directives_applied"] = directive_results
            
            # Добавляем в историю диалога
            response_text = _extract_text_from_result(result)
            add_to_history(session_id, message, category, response_text)
            
            return result, None
        else:
            return None, f"HTTP {response.status_code}: {response.text}"
            
    except Exception as e:
        logger.error(f"❌ Ошибка соединения с {category}: {e}")
        
        # Fallback для физических задач - отправляем к mathematics
        if category == 'electrical' and any(word in message.lower() for word in ['напряжение', 'ток', 'мощность', 'энергия', 'кинетическая', 'потенциальная']):
            logger.info(f"🔄 Fallback: отправляем физическую задачу к mathematics")
            result, error = forward_request('mathematics', message)
            return result, error
        
        # Fallback для математических задач - отправляем к advanced_math
        if category == 'mathematics' and any(word in message.lower() for word in ['уравнение', 'интеграл', 'производная', 'система']):
            logger.info(f"🔄 Fallback: отправляем сложную математическую задачу к advanced_math")
            result, error = forward_request('advanced_math', message)
            return result, error
        
        return None, str(e)

@app.route('/')
def index():
    """Главная страница - веб-интерфейс Rubin IDE"""
    try:
        return send_from_directory('.', 'RubinIDE.html')
    except FileNotFoundError:
        return jsonify({
            'name': 'Smart Dispatcher',
            'version': '1.0',
            'status': 'online',
            'servers': {name: f"{get_base_url()}:{config['port']}" for name, config in SERVERS.items()},
            'note': 'RubinIDE.html not found'
        })

@app.route('/matrix/RubinDeveloper.html')
def rubin_developer():
    """Rubin Developer интерфейс"""
    return send_from_directory('matrix', 'RubinDeveloper.html')

@app.route('/test-rubin')
def test_rubin():
    """Тестовая страница для RubinDeveloper"""
    return send_from_directory('.', 'test_rubin_developer.html')

@app.route('/api/dispatcher/info')
def dispatcher_info():
    """Информация о диспетчере"""
    return jsonify({
        'name': 'Smart Dispatcher',
        'version': '1.0',
        'status': 'online',
        'servers': {name: f"localhost:{config['port']}" for name, config in SERVERS.items()}
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Основной endpoint для чата"""
    global ETHICAL_CORE_AVAILABLE
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
        
        # Проверяем команды директив
        if DIRECTIVES_AVAILABLE and any(cmd in message.lower() for cmd in [
            'прими директиву', 'список директив', 'удали директиву', 
            'статистика директив', 'помощь по директивам'
        ]):
            user_id = data.get('user_id', 'default')
            directive_result = process_directives_command(message, user_id)
            return jsonify(directive_result)
        
        # Определяем категорию
        category = categorize_message(message)
        
        # Этическая проверка
        ethical_approved, ethical_message = ethical_check(message, category)
        
        if not ethical_approved:
            return jsonify({
                'success': False,
                'error': ethical_message,
                'category': category,
                'ethical_core_blocked': True
            }), 403  # Forbidden
        
        # Короткие дружелюбные ответы на приветствия — без шаблонов
        if _is_greeting(message):
            return jsonify({
                'success': True,
                'response': 'Привет! Готов помочь по программированию, электротехнике, автоматизации и математике. Чем заняться? ',
                'category': 'general',
                'server': f"{get_base_url()}:{SERVERS['general']['port']}",
                'ethical_core': {
                    'active': ETHICAL_CORE_AVAILABLE,
                    'message': ethical_message
                }
            })

        # Пересылаем запрос
        result, error = forward_request(category, message)

        # Если основной модуль не ответил — мягкий фолбэк на general
        if not result and category != 'general':
            logger.warning(f"Fallback → general из-за ошибки категории {category}: {error}")
            result_fallback, error_fallback = forward_request('general', message)
            if result_fallback:
                response_text = _extract_text_from_result(result_fallback)
                return jsonify({
                    'success': True,
                    'response': response_text,
                    'category': 'general',
                    'server': f"{get_base_url()}:{SERVERS['general']['port']}",
                    'fallback_from': category,
                    'ethical_core': {
                        'active': ETHICAL_CORE_AVAILABLE,
                        'message': ethical_message
                    }
                })
            else:
                error = error_fallback or error
        
        if result:
            response_text = _extract_text_from_result(result)
            
            # Определяем сервер для ответа
            if category == 'arduino_nano':
                server_info = f"{get_base_url()}:8080 (встроенный модуль)"
            else:
                server_info = f"{get_base_url()}:{SERVERS[category]['port']}"
            
            return jsonify({
                'success': True,
                'response': response_text,
                'category': category,
                'server': server_info,
                'ethical_core': {
                    'active': ETHICAL_CORE_AVAILABLE,
                    'message': ethical_message
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': f"Ошибка сервера {category}: {error}",
                'category': category
            }), 500
        
    except Exception as e:
        import traceback
        logger.error(f"Ошибка в диспетчере: {e}")
        logger.error(f"Трейс ошибки: {traceback.format_exc()}")
        return jsonify({'error': f'Внутренняя ошибка: {str(e)}'}), 500

@app.route('/api/ethical/status')
def ethical_status():
    """Статус Этического Ядра"""
    global ETHICAL_CORE_AVAILABLE
    ETHICAL_CORE_AVAILABLE = check_ethical_core_availability()
    if ETHICAL_CORE_AVAILABLE:
        try:
            from ethical_core import ethical_core
            report = ethical_core.get_safety_report()
            return jsonify({
                'success': True,
                'ethical_core': 'active',
                'report': report
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'ethical_core': 'error',
                'error': str(e)
            })
    else:
        return jsonify({
            'success': False,
            'ethical_core': 'inactive',
            'error': 'Этическое Ядро недоступно'
        })

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Получает историю диалога для текущей сессии"""
    try:
        session_id = get_session_id()
        session = CONVERSATION_HISTORY["sessions"][session_id]
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'session_start': session['start_time'],
            'messages': session['messages'],
            'context': session['context'],
            'total_interactions': CONVERSATION_HISTORY["global_context"]["total_interactions"]
        })
    except Exception as e:
        logger.error(f"❌ Ошибка получения истории: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat/context', methods=['GET'])
def get_context():
    """Получает текущий контекст диалога"""
    try:
        session_id = get_session_id()
        session = CONVERSATION_HISTORY["sessions"][session_id]
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'context': session['context'],
            'recent_topics': CONVERSATION_HISTORY["global_context"]["last_topics"],
            'session_duration': (datetime.now() - datetime.fromisoformat(session['start_time'])).seconds
        })
    except Exception as e:
        logger.error(f"❌ Ошибка получения контекста: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    return jsonify({
        'dispatcher': 'online',
        'status': 'healthy',
        'message': 'Smart Dispatcher is running'
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Проверка здоровья Smart Dispatcher"""
    # Проверяем состояние всех модулей
    module_status = {}
    for name, config in SERVERS.items():
        try:
            response = requests.get(f"http://{get_base_url()}:{config['port']}/api/health", timeout=5)
            module_status[name] = {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'port': config['port'],
                'response_time': response.elapsed.total_seconds()
            }
        except Exception as e:
            module_status[name] = {
                'status': 'unreachable',
                'port': config['port'],
                'error': str(e)
            }
    
    # Определяем общий статус
    unhealthy_count = sum(1 for status in module_status.values() if status['status'] != 'healthy')
    overall_status = 'healthy' if unhealthy_count == 0 else 'degraded' if unhealthy_count < len(SERVERS) else 'critical'
    
    return jsonify({
        'service': 'Smart Dispatcher',
        'status': overall_status,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0',
        'modules': module_status,
        'total_modules': len(SERVERS),
        'healthy_modules': len(SERVERS) - unhealthy_count,
        'unhealthy_modules': unhealthy_count
    })

@app.route('/api/openrouter/setup', methods=['POST'])
def setup_openrouter():
    """Установка API ключа OpenRouter"""
    if not OPENROUTER_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Модуль OpenRouter недоступен'
        }), 500
    
    try:
        data = request.get_json()
        api_key = data.get('api_key')
        
        if not api_key:
            return jsonify({
                'success': False,
                'error': 'API ключ не предоставлен'
            }), 400
        
        result = setup_openrouter_api_key(api_key)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/openrouter/learn', methods=['POST'])
def learn_from_chat():
    """Обучение Rubin на основе диалога"""
    if not OPENROUTER_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Модуль OpenRouter недоступен'
        }), 500
    
    try:
        data = request.get_json()
        user_query = data.get('user_query', '')
        rubin_response = data.get('rubin_response', '')
        category = data.get('category', 'general')
        feedback = data.get('feedback', '')
        
        if not user_query or not rubin_response:
            return jsonify({
                'success': False,
                'error': 'Необходимы user_query и rubin_response'
            }), 400
        
        result = learn_from_interaction(user_query, rubin_response, category, feedback)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/openrouter/enhance', methods=['POST'])
def enhance_response():
    """Улучшение ответа с помощью OpenRouter"""
    if not OPENROUTER_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Модуль OpenRouter недоступен'
        }), 500
    
    try:
        data = request.get_json()
        user_query = data.get('message', '')
        category = data.get('category', 'general')
        
        if not user_query:
            return jsonify({
                'success': False,
                'error': 'Сообщение не может быть пустым'
            }), 400
        
        # Получаем контекст сессии
        session_id = get_session_id()
        context = {
            'recent_topics': [m['category'] for m in CONVERSATION_HISTORY["sessions"].get(session_id, {}).get("messages", [])[-5:]],
            'history': CONVERSATION_HISTORY["sessions"].get(session_id, {}).get("messages", [])
        }
        
        result = generate_enhanced_response(user_query, category, context)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/openrouter/analyze', methods=['GET'])
def analyze_patterns():
    """Анализ паттернов диалогов"""
    if not OPENROUTER_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Модуль OpenRouter недоступен'
        }), 500
    
    try:
        result = analyze_conversation_patterns()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/openrouter/stats', methods=['GET'])
def openrouter_stats():
    """Статистика обучения"""
    if not OPENROUTER_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Модуль OpenRouter недоступен'
        }), 500
    
    try:
        result = get_learning_stats()
        return jsonify({
            'success': True,
            'stats': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats')
def stats():
    """Статистика системы"""
    return jsonify({
        'system': 'Rubin AI v2',
        'version': '1.0',
        'status': 'online',
        'modules': {
            'electrical': {'port': 8087, 'status': 'online'},
            'radiomechanics': {'port': 8089, 'status': 'online'},
            'controllers': {'port': 9000, 'status': 'online'},
            'mathematics': {'port': 8086, 'status': 'online'},
            'programming': {'port': 8088, 'status': 'online'},
            'general': {'port': 8085, 'status': 'online'},
            'localai': {'port': 11434, 'status': 'online'}
        },
        'uptime': 'running',
        'requests_processed': 'active'
    })

if __name__ == '__main__':
    print("🚀 Запуск Smart Dispatcher...")
    print("📡 Порт: 8080")
    print("🔗 URL: http://localhost:8080")
    print("📋 Эндпоинты:")
    print("  - POST /api/chat - Основной чат")
    print("  - GET /api/health - Проверка здоровья")
    print("  - GET /api/chat/history - История диалогов")
    print("  - GET /api/chat/context - Контекст диалога")
    print("=" * 50)
    
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
