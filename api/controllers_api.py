#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rubin AI v2 - Controllers API Server
Сервер для обработки вопросов по контроллерам и промышленной автоматизации
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

# Обработка CORS preflight запросов
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization")
        response.headers.add('Access-Control-Allow-Methods', "GET,POST,OPTIONS")
        response.headers.add('Content-Type', 'application/json; charset=utf-8')
        return response

# Установка правильных заголовков для всех ответов
@app.after_request
def after_request(response):
    if response.content_type == 'application/json':
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

# База знаний по контроллерам
CONTROLLERS_KNOWLEDGE = {
    "pmac": {
        "title": "PMAC контроллеры",
        "description": "Программируемые многоосевые контроллеры движения",
        "explanation": """
**PMAC (Programmable Multi-Axis Controller):**

**Основные возможности:**
• Управление до 32 осями одновременно
• Высокая точность позиционирования (до 0.1 мкм)
• Встроенная математика для траекторий
• Работа в реальном времени (1 мс цикл)

**Архитектура:**
• **DSP** - цифровой сигнальный процессор
• **Память программ** - хранение алгоритмов
• **Память данных** - переменные и параметры
• **Интерфейсы** - энкодеры, аналоговые входы

**Языки программирования:**
• **Motion Programs** - программы движения
• **PLC Programs** - логика управления
• **Background Programs** - фоновые задачи

**Основные функции:**
• **Позиционирование** - точное позиционирование осей
• **Интерполяция** - плавные траектории движения
• **Синхронизация** - координация нескольких осей
• **Обратная связь** - контроль положения и скорости

**Применение:**
• Станки с ЧПУ
• Робототехника
• Измерительные системы
• Промышленная автоматизация
        """,
        "examples": [
            "3-осевой фрезерный станок",
            "6-осевой робот-манипулятор",
            "Координатно-измерительная машина"
        ]
    },
    
    "plc": {
        "title": "PLC контроллеры",
        "description": "Программируемые логические контроллеры",
        "explanation": """
**PLC (Programmable Logic Controller):**

**Основные функции:**
• Логическое управление процессами
• Обработка дискретных и аналоговых сигналов
• Работа в реальном времени
• Высокая надежность

**Языки программирования:**
• **Ladder Logic (LD)** - релейная логика
• **Structured Text (ST)** - текстовый язык
• **Function Block Diagram (FBD)** - функциональные блоки
• **Instruction List (IL)** - низкоуровневые инструкции

**Популярные производители:**
• **Siemens** - S7-1200, S7-1500, TIA Portal
• **Allen-Bradley** - CompactLogix, ControlLogix, Studio 5000
• **Schneider Electric** - Modicon, Unity Pro
• **Omron** - CP1, CJ2, CX-Programmer

**Применение:**
• Промышленная автоматизация
• Управление процессами
• Системы безопасности
• Мониторинг оборудования
        """
    },
    
    "энкодер": {
        "title": "Энкодеры",
        "description": "Датчики обратной связи для измерения положения и скорости",
        "explanation": """
**Энкодеры - датчики обратной связи:**

**Типы энкодеров:**
• **Инкрементальные** - относительное положение
• **Абсолютные** - абсолютное положение
• **Линейные** - прямолинейное движение
• **Ротационные** - вращательное движение

**Принцип работы:**
• **Оптические** - светодиод + фотодиод
• **Магнитные** - магнитное поле + датчик Холла
• **Индуктивные** - электромагнитная индукция
• **Емкостные** - изменение емкости

**Разрешение:**
• **PPR** - импульсы на оборот
• **CPR** - циклы на оборот
• **Биты** - для абсолютных энкодеров

**Применение:**
• Серводвигатели
• Шаговые двигатели
• Позиционирование осей
• Измерение скорости
        """
    },
    
    "сервопривод": {
        "title": "Сервоприводы",
        "description": "Точные приводы с обратной связью",
        "explanation": """
**Сервоприводы - точные приводы:**

**Компоненты:**
• **Двигатель** - AC или DC
• **Энкодер** - датчик обратной связи
• **Контроллер** - управление движением
• **Усилитель** - питание двигателя

**Типы управления:**
• **Позиционное** - точное позиционирование
• **Скоростное** - управление скоростью
• **Моментное** - управление усилием
• **Комбинированное** - несколько режимов

**Характеристики:**
• Высокая точность позиционирования
• Быстрый отклик
• Широкий диапазон скоростей
• Плавное движение

**Применение:**
• Станки с ЧПУ
• Робототехника
• Промышленные манипуляторы
• Измерительные системы
        """
    },
    "i130": {
        "title": "I130 контроллер",
        "description": "Программируемый контроллер движения I130",
        "explanation": """
**I130 - программируемый контроллер движения:**

**Основные характеристики:**
• **Процессор** - высокопроизводительный DSP
• **Память** - 512KB программ + 256KB данных
• **Оси** - до 8 осей управления
• **Частота** - до 1 МГц для энкодеров
• **Интерфейсы** - RS-232, RS-485, Ethernet

**Настройка I130:**

**1. Подключение:**
• Подключите питание 24V DC
• Подключите энкодеры к разъемам ENC1-8
• Подключите двигатели к разъемам MOT1-8
• Подключите коммуникационный кабель

**2. Базовая конфигурация:**
• Загрузите конфигурационный файл
• Установите параметры осей
• Настройте энкодеры (PPR, тип)
• Конфигурируйте двигатели

**3. Параметры настройки:**
• **Kp, Ki, Kd** - коэффициенты ПИД-регулятора
• **MaxVel** - максимальная скорость
• **MaxAcc** - максимальное ускорение
• **HomeVel** - скорость поиска дома
• **DeadBand** - мертвая зона

**4. Программирование:**
• **Motion Programs** - программы движения
• **PLC Programs** - логика управления
• **Background Programs** - фоновые задачи

**5. Тестирование:**
• Проверьте работу каждой оси
• Тестируйте позиционирование
• Проверьте точность движения
• Калибруйте систему

**Применение:**
• Станки с ЧПУ
• Робототехника
• Промышленная автоматизация
• Измерительные системы
        """,
        "examples": [
            "3-осевой фрезерный станок",
            "6-осевой робот-манипулятор",
            "Координатно-измерительная машина"
        ]
    },
    
    "события": {
        "title": "События и прерывания",
        "description": "Система событий и прерываний в контроллерах",
        "explanation": """
**События и прерывания (Events & Interrupts):**

**Типы событий:**
• **Внешние прерывания** - сигналы от датчиков
• **Внутренние прерывания** - таймеры, счетчики
• **Программные прерывания** - вызовы функций
• **Исключения** - ошибки выполнения

**Приоритеты прерываний:**
• **Высокий** - критические события (аварии)
• **Средний** - обычные задачи
• **Низкий** - фоновые процессы

**Обработка событий:**
• **ISR (Interrupt Service Routine)** - обработчики
• **Event Handlers** - обработчики событий
• **Callbacks** - функции обратного вызова
• **State Machines** - конечные автоматы

**Примеры применения:**
• **Аварийные остановы** - мгновенная реакция
• **Синхронизация** - координация процессов
• **Мониторинг** - контроль состояния
• **Уведомления** - информирование оператора

**Программирование:**
• **Регистрация обработчиков** - привязка функций
• **Управление приоритетами** - настройка важности
• **Маскирование** - временное отключение
• **Очереди событий** - буферизация
        """
    },
    "прерывания": {
        "title": "Прерывания в контроллерах",
        "description": "Система прерываний для обработки событий",
        "explanation": """
**Прерывания в контроллерах:**

**Механизм работы:**
• **Прерывание** - принудительная передача управления
• **Контекст** - сохранение состояния процессора
• **Обработчик** - функция обработки события
• **Возврат** - восстановление прерванной программы

**Типы прерываний:**
• **Маскируемые** - можно отключить программно
• **Немаскируемые** - критически важные
• **Программные** - вызванные программой
• **Аппаратные** - от внешних устройств

**Векторная таблица:**
• **Адреса обработчиков** - указатели на функции
• **Приоритеты** - порядок обработки
• **Маски** - управление активностью
• **Флаги** - состояние прерываний

**Применение в автоматизации:**
• **Аварийные остановы** - мгновенная реакция
• **Синхронизация процессов** - координация
• **Мониторинг состояния** - контроль
• **Обработка ошибок** - диагностика
        """
    },
    "чпу": {
        "title": "ЧПУ (Числовое Программное Управление)",
        "description": "Передача данных с ЧПУ в контроллер",
        "explanation": """
**Передача данных с ЧПУ в контроллер:**

**Основные протоколы передачи:**
• **RS-232/RS-485** - последовательная передача
• **Ethernet** - сетевая передача
• **USB** - универсальная шина
• **CAN Bus** - промышленная сеть

**Типы данных ЧПУ:**
• **G-коды** - управляющие команды
• **Координаты** - позиции осей
• **Скорости** - параметры движения
• **Статус** - состояние системы

**Методы передачи:**
• **Прямое подключение** - кабель RS-232
• **Сетевое подключение** - Ethernet TCP/IP
• **Промышленные сети** - Modbus, Profinet
• **Беспроводная передача** - WiFi, Bluetooth

**Примеры протоколов:**
• **Modbus RTU** - для RS-485
• **Modbus TCP** - для Ethernet
• **Profinet** - промышленный Ethernet
• **EtherCAT** - высокоскоростная передача

**Программирование передачи:**
```python
# Пример передачи данных ЧПУ
import serial
import socket

# RS-232 подключение
ser = serial.Serial('COM1', 9600)
ser.write(b'G01 X10 Y20 Z5\\n')

# Ethernet подключение
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('192.168.1.100', 502))
sock.send(b'G01 X10 Y20 Z5\\n')
```

**Обработка данных:**
• **Парсинг команд** - разбор G-кодов
• **Валидация** - проверка корректности
• **Буферизация** - временное хранение
• **Обработка ошибок** - коррекция данных
        """
    },
    "ascii": {
        "title": "ASCII-команды и протоколы",
        "description": "Отправка ASCII-команд и получение ответов",
        "explanation": """
**ASCII-команды и протоколы:**

**Основные принципы:**
• **ASCII** - стандарт кодирования символов
• **Команды** - текстовые инструкции
• **Ответы** - подтверждения выполнения
• **Протокол** - правила обмена данными

**Типы команд:**
• **Чтение** - получение данных (READ, GET)
• **Запись** - изменение параметров (WRITE, SET)
• **Управление** - запуск/остановка (START, STOP)
• **Статус** - проверка состояния (STATUS, INFO)

**Форматы команд:**
• **Простой** - команда без параметров
• **С параметрами** - команда + данные
• **С адресом** - команда + адрес устройства
• **С контрольной суммой** - команда + CRC

**Примеры протоколов:**
• **Modbus ASCII** - промышленный стандарт
• **SCPI** - стандарт для измерительных приборов
• **AT-команды** - для модемов и GSM
• **Custom ASCII** - собственные протоколы

**Обработка ответов:**
• **Подтверждение** - ACK/NAK
• **Данные** - запрошенная информация
• **Ошибки** - коды ошибок
• **Таймауты** - обработка задержек

**Программирование:**
• **Последовательный порт** - RS232/RS485
• **TCP/IP** - сетевые соединения
• **USB** - универсальная шина
• **Ethernet** - локальные сети
        """
    },
    "пид": {
        "title": "ПИД-регуляторы",
        "description": "Пропорционально-интегрально-дифференциальные регуляторы",
        "explanation": """
**ПИД-регуляторы (PID Controllers):**

**Принцип работы:**
• **P (Пропорциональная)** - реакция на текущую ошибку
• **I (Интегральная)** - устранение накопленной ошибки
• **D (Дифференциальная)** - предсказание будущих изменений

**Формула ПИД-регулятора:**
```
u(t) = Kp × e(t) + Ki × ∫e(t)dt + Kd × de(t)/dt
```

**Параметры настройки:**
• **Kp (Пропорциональный коэффициент)** - скорость реакции
• **Ki (Интегральный коэффициент)** - устранение статической ошибки
• **Kd (Дифференциальный коэффициент)** - стабилизация системы

**Методы настройки:**
• **Метод Зиглера-Николса** - классический подход
• **Метод Коэна-Куна** - для процессов с запаздыванием
• **Автонастройка** - автоматический подбор параметров

**Типы ПИД-регуляторов:**
• **Позиционный** - выход как абсолютное значение
• **Скоростной** - выход как изменение управляющего воздействия
• **Каскадный** - несколько контуров регулирования

**Применение в промышленности:**
• **Температурное регулирование** - печи, реакторы
• **Позиционирование** - станки, роботы
• **Регулирование давления** - компрессоры, насосы
• **Контроль уровня** - резервуары, баки

**Настройка для точного позиционирования:**
1. **Начальные параметры:** Kp=1, Ki=0, Kd=0
2. **Увеличение Kp** до появления колебаний
3. **Добавление Kd** для стабилизации
4. **Корректировка Ki** для устранения ошибки

**Типичные проблемы:**
• **Перерегулирование** - слишком большой Kp
• **Медленная реакция** - малый Kp
• **Статическая ошибка** - нужен Ki
• **Нестабильность** - нужен Kd

**Современные решения:**
• **Адаптивные ПИД** - самонастройка параметров
• **Нечеткие ПИД** - использование нечеткой логики
• **Нейросетевые ПИД** - обучение на данных
• **Предиктивные ПИД** - учет будущих изменений
        """
    },
    "pid": {
        "title": "PID Controllers",
        "description": "Proportional-Integral-Derivative controllers",
        "explanation": """
**PID Controllers:**

**Working Principle:**
• **P (Proportional)** - response to current error
• **I (Integral)** - elimination of accumulated error
• **D (Derivative)** - prediction of future changes

**PID Formula:**
```
u(t) = Kp × e(t) + Ki × ∫e(t)dt + Kd × de(t)/dt
```

**Tuning Parameters:**
• **Kp (Proportional gain)** - response speed
• **Ki (Integral gain)** - static error elimination
• **Kd (Derivative gain)** - system stabilization

**Tuning Methods:**
• **Ziegler-Nichols** - classical approach
• **Cohen-Coon** - for processes with delay
• **Auto-tuning** - automatic parameter selection

**PID Types:**
• **Positional** - output as absolute value
• **Velocity** - output as control change
• **Cascade** - multiple control loops

**Industrial Applications:**
• **Temperature control** - furnaces, reactors
• **Positioning** - machines, robots
• **Pressure control** - compressors, pumps
• **Level control** - tanks, vessels

**Tuning for Precise Positioning:**
1. **Initial parameters:** Kp=1, Ki=0, Kd=0
2. **Increase Kp** until oscillations appear
3. **Add Kd** for stabilization
4. **Adjust Ki** to eliminate error

**Common Issues:**
• **Overshoot** - Kp too large
• **Slow response** - Kp too small
• **Steady-state error** - need Ki
• **Instability** - need Kd

**Modern Solutions:**
• **Adaptive PID** - self-tuning parameters
• **Fuzzy PID** - fuzzy logic application
• **Neural PID** - learning from data
• **Predictive PID** - future change consideration
        """
    }
}

def find_best_match(query):
    """Поиск наиболее подходящего ответа по запросу"""
    query_lower = query.lower()
    
    # Прямое совпадение
    for key, data in CONTROLLERS_KNOWLEDGE.items():
        if key in query_lower:
            return data
    
    # Поиск по ключевым словам
    keywords = {
        "pmac": "pmac",
        "plc": "plc",
        "энкодер": "энкодер",
        "сервопривод": "сервопривод",
        "контроллер": "plc",
        "автоматизация": "plc",
        "события": "события",
        "прерывания": "прерывания",
        "events": "события",
        "interrupts": "прерывания",
        "isr": "прерывания",
        "обработчик": "прерывания",
        "ascii": "ascii",
        "команды": "ascii",
        "протокол": "ascii",
        "отправка": "ascii",
        "получение": "ascii",
        "ответы": "ascii",
        "пид": "пид",
        "pid": "pid",
        "регулятор": "пид",
        "регулирование": "пид",
        "настройка": "пид",
        "позиционирование": "пид",
        "точность": "пид",
        "коэффициент": "пид",
        "пропорциональный": "пид",
        "интегральный": "пид",
        "дифференциальный": "пид",
        "чпу": "чпу",
        "cnc": "чпу",
        "числовое": "чпу",
        "программное": "чпу",
        "управление": "чпу",
        "передача": "чпу",
        "данные": "чпу",
        "g-коды": "чпу",
        "координаты": "чпу"
    }
    
    for keyword, topic in keywords.items():
        if keyword in query_lower:
            return CONTROLLERS_KNOWLEDGE[topic]
    
    return None

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка состояния сервера"""
    return jsonify({
        "status": "healthy",
        "service": "Controllers API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/controllers/status', methods=['GET'])
def get_status():
    """Получение статуса модуля контроллеров"""
    return jsonify({
        "status": "online",
        "module": "Контроллеры",
        "port": 8090,
        "description": "PMAC, PLC, микроконтроллеры, промышленная автоматизация",
        "topics_available": list(CONTROLLERS_KNOWLEDGE.keys()),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/controllers/topic/<topic>', methods=['GET', 'POST'])
def explain_topic(topic):
    """Объяснение тем по контроллерам"""
    try:
        # Обработка как GET, так и POST запросов
        if request.method == 'POST':
            data = request.get_json()
            concept = data.get('concept', topic) if data else topic
        else:  # GET запрос
            concept = request.args.get('concept', topic)
        
        # Поиск подходящего ответа
        knowledge = find_best_match(concept)
        
        if knowledge:
            response = {
                "success": True,
                "concept": concept,
                "title": knowledge["title"],
                "description": knowledge["description"],
                "explanation": knowledge["explanation"]
            }
            
            if "examples" in knowledge:
                response["examples"] = knowledge["examples"]
            
            return jsonify(response)
        else:
            return jsonify({
                "success": False,
                "message": f"Тема '{concept}' не найдена в базе знаний контроллеров",
                "available_topics": list(CONTROLLERS_KNOWLEDGE.keys())
            })
    
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        return jsonify({
            "error": "Внутренняя ошибка сервера",
            "details": str(e)
        }), 500

@app.route('/api/controllers/topics', methods=['GET'])
def get_topics():
    """Получение списка доступных тем"""
    return jsonify({
        "success": True,
        "topics": list(CONTROLLERS_KNOWLEDGE.keys()),
        "count": len(CONTROLLERS_KNOWLEDGE)
    })

@app.route('/api/chat', methods=['GET', 'POST'])
def chat():
    """Универсальный endpoint для чата"""
    try:
        # Получаем параметры из GET или POST запроса
        if request.method == 'GET':
            concept = request.args.get('concept', '')
            message = request.args.get('message', '')
        else:
            data = request.get_json() or {}
            concept = data.get('concept', '')
            message = data.get('message', '')
        
        # Если есть concept, используем его для поиска темы
        if concept:
            # Ищем подходящую тему
            topic = None
            for key, value in CONTROLLERS_KNOWLEDGE.items():
                if concept.lower() in key.lower() or concept.lower() in value.get('title', '').lower():
                    topic = key
                    break
            
            if topic:
                return jsonify({
                    "success": True,
                    "response": CONTROLLERS_KNOWLEDGE[topic],
                    "topic": topic
                })
            else:
                return jsonify({
                    "success": False,
                    "message": f"Тема '{concept}' не найдена в базе знаний контроллеров",
                    "available_topics": list(CONTROLLERS_KNOWLEDGE.keys())
                })
        
        # Если есть message, обрабатываем его
        if message:
            # Простой поиск по ключевым словам
            message_lower = message.lower()
            for key, value in CONTROLLERS_KNOWLEDGE.items():
                if any(keyword in message_lower for keyword in [key, value.get('title', '').lower()]):
                    return jsonify({
                        "success": True,
                        "response": value,
                        "topic": key
                    })
            
            return jsonify({
                "success": False,
                "message": "Не удалось найти подходящую тему для вашего вопроса",
                "available_topics": list(CONTROLLERS_KNOWLEDGE.keys())
            })
        
        # Если нет параметров, возвращаем общую информацию
        return jsonify({
            "success": True,
            "message": "Сервер контроллеров готов к работе",
            "available_topics": list(CONTROLLERS_KNOWLEDGE.keys()),
            "endpoints": {
                "/api/chat": "Универсальный endpoint для чата",
                "/api/controllers/topic/<topic>": "Получение информации по конкретной теме",
                "/api/controllers/topics": "Список доступных тем",
                "/health": "Проверка состояния сервера"
            }
        })
        
    except Exception as e:
        logger.error(f"Ошибка в chat endpoint: {e}")
        return jsonify({
            "success": False,
            "error": "Внутренняя ошибка сервера",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    logger.info("Запуск сервера контроллеров на порту 9000...")
    app.run(host='0.0.0.0', port=9000, debug=True)

