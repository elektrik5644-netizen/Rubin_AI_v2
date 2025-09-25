#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сервер программирования для Rubin AI v2
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Programming API',
        'port': 8088,
        'timestamp': time.time()
    })

# База знаний по программированию
PROGRAMMING_KNOWLEDGE = {
    "продвинутые": {
        "title": "Продвинутые и специфические функции",
        "description": "Продвинутые возможности программирования",
        "explanation": """
**Продвинутые и специфические функции:**

**Функциональное программирование:**
• **Лямбда-функции** - анонимные функции
• **Замыкания** - функции с доступом к внешним переменным
• **Каррирование** - преобразование функций
• **Композиция** - объединение функций

**Метапрограммирование:**
• **Рефлексия** - анализ кода во время выполнения
• **Декораторы** - модификация функций
• **Генераторы** - создание итераторов
• **Метаклассы** - создание классов

**Асинхронное программирование:**
• **Async/Await** - асинхронные функции
• **Промисы** - отложенные вычисления
• **Корутины** - кооперативная многозадачность
• **Event Loop** - цикл событий

**Специализированные функции:**
• **Мемоизация** - кэширование результатов
• **Ленивые вычисления** - отложенные вычисления
• **Паттерны проектирования** - архитектурные решения
• **Оптимизация** - улучшение производительности

**Примеры применения:**
• **Веб-разработка** - современные фреймворки
• **Data Science** - обработка больших данных
• **Machine Learning** - алгоритмы машинного обучения
• **Системное программирование** - низкоуровневые задачи
        """
    },
    "функции": {
        "title": "Функции в программировании",
        "description": "Основы работы с функциями",
        "explanation": """
**Функции в программировании:**

**Основные концепции:**
• **Определение** - создание функции
• **Вызов** - использование функции
• **Параметры** - входные данные
• **Возвращаемое значение** - результат работы

**Типы функций:**
• **Встроенные** - часть языка
• **Пользовательские** - созданные программистом
• **Методы** - функции объектов
• **Конструкторы** - создание объектов

**Область видимости:**
• **Локальные переменные** - внутри функции
• **Глобальные переменные** - доступны везде
• **Параметры** - входные данные
• **Замыкания** - доступ к внешним переменным

**Продвинутые возможности:**
• **Рекурсия** - вызов функции самой себя
• **Перегрузка** - разные версии функции
• **Полиморфизм** - разные реализации
• **Инкапсуляция** - скрытие деталей
        """
    },
    "автоматизация": {
        "title": "Промышленная автоматизация",
        "description": "Программирование для промышленной автоматизации",
        "explanation": """
**Промышленная автоматизация:**

**C++ для промышленной автоматизации:**
• **Производительность** - высокая скорость выполнения
• **Память** - точный контроль ресурсов
• **Время выполнения** - детерминированное поведение
• **Интеграция** - прямая работа с оборудованием
• **Применение** - ПЛК, роботы, системы управления

**Python для промышленной автоматизации:**
• **Разработка** - быстрая разработка и прототипирование
• **Библиотеки** - богатая экосистема (NumPy, Pandas)
• **Интеграция** - легкое подключение к базам данных
• **Анализ данных** - обработка больших объемов данных
• **Применение** - SCADA, HMI, аналитика

**Сравнение для промышленности:**
• **C++** - критичные системы, реальное время
• **Python** - аналитика, мониторинг, отчетность
• **Гибридный подход** - C++ для управления, Python для анализа
• **Выбор зависит от требований** к производительности и времени разработки
        """
    },
    "конвейер": {
        "title": "Управление конвейером",
        "description": "Алгоритмы управления конвейерными системами",
        "explanation": """
**Управление конвейером на Python:**

**Основные компоненты:**
• **Датчики** - позиция, скорость, вес
• **Приводы** - моторы, сервоприводы
• **Контроллер** - логика управления
• **Интерфейс** - мониторинг и настройка

**Алгоритм управления:**
```python
class ConveyorController:
    def __init__(self):
        self.speed = 0
        self.position = 0
        self.target_position = 0
        
    def update(self, sensor_data):
        # Получение данных с датчиков
        current_pos = sensor_data['position']
        current_speed = sensor_data['speed']
        
        # Расчет ошибки
        error = self.target_position - current_pos
        
        # ПИД-регулятор
        self.speed = self.pid_controller.calculate(error)
        
        # Ограничение скорости
        self.speed = max(-100, min(100, self.speed))
        
        # Отправка команды на привод
        self.send_command(self.speed)
        
    def set_target(self, position):
        self.target_position = position
```

**Функции безопасности:**
• **Аварийная остановка** - при обнаружении препятствий
• **Ограничение скорости** - предотвращение перегрузок
• **Мониторинг** - контроль состояния системы
• **Логирование** - запись событий и ошибок
        """
    },
    "сортировка": {
        "title": "Алгоритмы сортировки",
        "description": "Основные алгоритмы сортировки данных",
        "explanation": """
**Алгоритмы сортировки:**

**Быстрая сортировка (Quick Sort):**
• **Принцип** - разделяй и властвуй
• **Сложность** - O(n log n) в среднем
• **Применение** - универсальная сортировка
• **Особенности** - нестабильная, на месте

**Сортировка пузырьком (Bubble Sort):**
• **Принцип** - сравнение соседних элементов
• **Сложность** - O(n²)
• **Применение** - простые случаи
• **Особенности** - стабильная, простая реализация

**Сортировка вставками (Insertion Sort):**
• **Принцип** - вставка элемента в отсортированную часть
• **Сложность** - O(n²)
• **Применение** - малые массивы
• **Особенности** - стабильная, эффективна для почти отсортированных данных

**Сортировка слиянием (Merge Sort):**
• **Принцип** - разделение на части и слияние
• **Сложность** - O(n log n)
• **Применение** - внешняя сортировка
• **Особенности** - стабильная, требует дополнительной памяти

**Выбор алгоритма:**
• **Малые данные** - сортировка вставками
• **Большие данные** - быстрая сортировка
• **Стабильность важна** - сортировка слиянием
• **Ограниченная память** - быстрая сортировка
        """
    },
    "ошибки": {
        "title": "Обработка ошибок",
        "description": "Системы обработки ошибок в программировании",
        "explanation": """
**Обработка ошибок:**

**Типы ошибок:**
• **Синтаксические** - ошибки в коде
• **Логические** - неправильная логика
• **Время выполнения** - ошибки во время выполнения
• **Системные** - ошибки операционной системы

**Механизмы обработки:**
• **Try-Catch** - перехват исключений
• **Assertions** - проверка условий
• **Logging** - запись ошибок
• **Graceful degradation** - плавное снижение функциональности

**XML для ошибок (user_error.xml):**
```xml
<errors>
    <error id="001" level="critical">
        <message>Критическая ошибка системы</message>
        <action>Перезапуск системы</action>
    </error>
    <error id="002" level="warning">
        <message>Предупреждение о состоянии</message>
        <action>Логирование</action>
    </error>
</errors>
```

**Лучшие практики:**
• **Предотвращение** - валидация входных данных
• **Обработка** - корректная обработка исключений
• **Логирование** - детальная запись ошибок
• **Восстановление** - автоматическое восстановление
        """
    },
    "сценарии": {
        "title": "Сценарии и Решение Проблем",
        "description": "Разработка сценариев и решение технических проблем",
        "explanation": """
**Сценарии и Решение Проблем:**

**Типы сценариев:**
• **Пользовательские сценарии** - Use Cases
• **Тестовые сценарии** - Test Cases
• **Сценарии автоматизации** - Automation Scripts
• **Сценарии развертывания** - Deployment Scripts

**Методологии решения проблем:**
• **Анализ проблемы** - определение корня
• **Постановка задачи** - четкое формулирование
• **Разработка решения** - создание алгоритма
• **Тестирование** - проверка работоспособности
• **Документирование** - описание решения

**Инструменты для сценариев:**
• **Python Scripts** - автоматизация задач
• **Shell Scripts** - системные операции
• **PowerShell** - Windows автоматизация
• **Bash** - Linux/Unix скрипты

**Паттерны решения проблем:**
• **Divide and Conquer** - разделяй и властвуй
• **Backtracking** - возврат с проверкой
• **Dynamic Programming** - динамическое программирование
• **Greedy Algorithms** - жадные алгоритмы

**Примеры сценариев:**
• **Автоматизация тестирования** - CI/CD пайплайны
• **Мониторинг системы** - проверка состояния
• **Резервное копирование** - автоматические бэкапы
• **Развертывание** - деплой приложений

**Лучшие практики:**
• Модульность и переиспользование
• Обработка ошибок и исключений
• Логирование и мониторинг
• Документирование и комментарии
        """
    }
}

def find_best_match(query):
    """Поиск наиболее подходящего ответа по запросу"""
    query_lower = query.lower()
    
    # Прямое совпадение
    for key, data in PROGRAMMING_KNOWLEDGE.items():
        if key in query_lower:
            return data
    
    # Поиск по ключевым словам
    keywords = {
        "продвинутые": "продвинутые",
        "специфические": "продвинутые",
        "функции": "функции",
        "алгоритмы": "алгоритмы",
        "программирование": "продвинутые",
        "код": "функции",
        "разработка": "продвинутые",
        "c++": "автоматизация",
        "python": "автоматизация",
        "автоматизация": "автоматизация",
        "промышленная": "автоматизация",
        "конвейер": "конвейер",
        "управление": "конвейер",
        "сортировка": "сортировка",
        "алгоритм": "сортировка",
        "ошибки": "ошибки",
        "error": "ошибки",
        "xml": "ошибки",
        "обработка": "ошибки",
        "сценарии": "сценарии",
        "сценарий": "сценарии",
        "решение": "сценарии",
        "проблем": "сценарии",
        "проблемы": "сценарии"
    }
    
    for keyword, topic in keywords.items():
        if keyword in query_lower:
            return PROGRAMMING_KNOWLEDGE[topic]
    
    return None

@app.route('/')
def index():
    return jsonify({
        'name': 'Programming Server',
        'version': '1.0',
        'status': 'online',
        'features': ['programming', 'algorithms', 'functions']
    })

@app.route('/api/programming/explain', methods=['GET', 'POST'])
def explain_programming():
    """Объяснение тем по программированию"""
    try:
        # Обработка как GET, так и POST запросов
        if request.method == 'POST':
            data = request.get_json()
            concept = data.get('concept', '') if data else ''
        else:  # GET запрос
            concept = request.args.get('concept', '')
        
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
            
            return jsonify(response)
        else:
            return jsonify({
                "success": False,
                "message": f"Тема '{concept}' не найдена в базе знаний программирования",
                "available_topics": list(PROGRAMMING_KNOWLEDGE.keys())
            })
    
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        return jsonify({
            "error": "Внутренняя ошибка сервера",
            "details": str(e)
        }), 500

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': '2025-09-20'})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Универсальный endpoint для чата"""
    try:
        data = request.get_json() or {}
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({
                "success": False,
                "error": "Сообщение не может быть пустым"
            }), 400
        
        # Точечный ответ по условному оператору IF
        ml = message.lower()
        if any(k in ml for k in [' if ', 'if(', 'if ', 'условие', 'условный оператор', 'endif']):
            return jsonify({
                'success': True,
                'category': 'programming',
                'response': (
                    'Условный оператор if — ветвление выполнения по логическому условию.\n\n'
                    'Python:\n'
                    '    if x > 0:\n'
                    '        print("positive")\n'
                    '    elif x == 0:\n'
                    '        print("zero")\n'
                    '    else:\n'
                    '        print("negative")\n\n'
                    'C/C++:\n'
                    '    if (x > 0) { /*...*/ } else if (x == 0) { /*...*/ } else { /*...*/ }\n\n'
                    'PLC ST (IEC 61131-3):\n'
                    '    IF Speed > 1000 THEN\n'
                    '        Conveyor := TRUE;\n'
                    '    ELSIF Speed = 0 THEN\n'
                    '        Alarm := TRUE;\n'
                    '    ELSE\n'
                    '        Conveyor := FALSE;\n'
                    '    END_IF;\n\n'
                    'Советы: группируйте сложные условия скобками, избегайте глубокой вложенности,\n'
                    'выносите проверки в именованные функции/предикаты.'
                )
            })

        # Поиск подходящего ответа в базе знаний программирования
        knowledge = find_programming_knowledge(message)
        
        if knowledge:
            response = {
                "success": True,
                "response": knowledge["explanation"],
                "concept": knowledge["title"],
                "server_type": "programming",
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(response)
        else:
            return jsonify({
                "success": False,
                "error": f"Не найдена информация по запросу: {message}",
                "suggestions": ["Python", "JavaScript", "алгоритмы", "структуры данных"]
            }), 404
            
    except Exception as e:
        logger.error(f"Ошибка в chat endpoint: {e}")
        return jsonify({
            "success": False,
            "error": f"Внутренняя ошибка сервера: {str(e)}"
        }), 500

def find_programming_knowledge(query):
    """Поиск знаний по программированию"""
    query_lower = query.lower()
    
    # Простая база знаний по программированию
    programming_knowledge = {
        "python": {
            "title": "Python программирование",
            "explanation": """
**Python - высокоуровневый язык программирования:**

**Основные особенности:**
• Простой и читаемый синтаксис
• Динамическая типизация
• Интерпретируемый язык
• Большая стандартная библиотека

**Примеры использования:**
• Веб-разработка (Django, Flask)
• Наука о данных (NumPy, Pandas)
• Машинное обучение (TensorFlow, PyTorch)
• Автоматизация задач

**Базовый синтаксис:**
```python
# Переменные
name = "Python"
age = 30

# Функции
def greet(name):
    return f"Привет, {name}!"

# Циклы
for i in range(5):
    print(i)
```
            """
        },
        "алгоритмы": {
            "title": "Алгоритмы и структуры данных",
            "explanation": """
**Алгоритмы - пошаговые инструкции для решения задач:**

**Основные типы алгоритмов:**
• Сортировка (быстрая, пузырьковая, слияние)
• Поиск (линейный, бинарный)
• Графовые алгоритмы (DFS, BFS)
• Динамическое программирование

**Структуры данных:**
• Массивы и списки
• Стеки и очереди
• Деревья и графы
• Хеш-таблицы

**Пример сортировки пузырьком:**
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
```
            """
        }
    }
    
    # Поиск по ключевым словам
    for key, knowledge in programming_knowledge.items():
        if key in query_lower or any(word in query_lower for word in key.split()):
            return knowledge
    
    return None

if __name__ == '__main__':
    logger.info("Запуск сервера программирования на порту 8088...")
    app.run(host='0.0.0.0', port=8088, debug=False)
