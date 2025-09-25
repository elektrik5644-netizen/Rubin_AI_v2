#!/usr/bin/env python3
"""
Скрипт для начала обучения Rubin AI
"""

import os
import json
import time
import requests
from datetime import datetime

class RubinTrainer:
    def __init__(self):
        self.server_url = "http://localhost:8083"
        self.training_data = []
        self.knowledge_categories = {
            "industrial_automation": {
                "weight": 1.0,
                "priority": "high",
                "subcategories": ["plc", "scada", "hmi", "networks", "servo", "pid"]
            },
            "programming": {
                "weight": 0.8,
                "priority": "medium",
                "subcategories": ["python", "c++", "javascript", "plc_languages"]
            },
            "electronics": {
                "weight": 0.9,
                "priority": "high",
                "subcategories": ["sensors", "actuators", "power_electronics"]
            }
        }
        
    def check_server_status(self):
        """Проверка статуса сервера"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Сервер Rubin AI запущен и готов к обучению")
                return True
            else:
                print("❌ Сервер недоступен")
                return False
        except requests.exceptions.RequestException:
            print("❌ Не удалось подключиться к серверу")
            return False
    
    def load_basic_training_data(self):
        """Загрузка базовых обучающих данных"""
        
        basic_knowledge = [
            {
                "filename": "servo_drives_basics.txt",
                "content": """
СЕРВОПРИВОДЫ - ОСНОВЫ

Сервопривод - это система управления, состоящая из:
1. Двигателя (синхронного или асинхронного)
2. Датчика обратной связи (энкодера или резолвера)
3. Контроллера (устройства управления)

Принцип работы:
- Контроллер сравнивает заданное положение с текущим
- Вычисляет ошибку позиционирования
- Подает управляющий сигнал на двигатель
- Двигатель поворачивается для устранения ошибки

Технические характеристики:
- Номинальная скорость: 1000-6000 об/мин
- Точность позиционирования: ±0.01°
- Время разгона: 0.1-1.0 секунды
- Динамическая точность: ±0.005°

Применение:
- ЧПУ станки
- Робототехника
- Печатные машины
- Текстильное оборудование
- Медицинское оборудование

Производители:
- Siemens (Sinamics S)
- ABB (ACS)
- Schneider Electric (Lexium)
- Delta Tau (PMAC)
- Yaskawa (Sigma)
                """,
                "category": "industrial_automation",
                "tags": "servo, drive, motor, positioning, control"
            },
            {
                "filename": "pid_controllers_basics.txt",
                "content": """
ПИД РЕГУЛЯТОРЫ - ОСНОВЫ

ПИД регулятор - устройство автоматического управления с тремя компонентами:

1. ПРОПОРЦИОНАЛЬНЫЙ (P) компонент:
   - Реагирует на текущую ошибку
   - Выход = Kp × Error
   - Быстрая реакция, но может быть статическая ошибка

2. ИНТЕГРАЛЬНЫЙ (I) компонент:
   - Накапливает ошибку во времени
   - Устраняет статическую ошибку
   - Выход = Ki × ∫Error dt
   - Может вызвать перерегулирование

3. ДИФФЕРЕНЦИАЛЬНЫЙ (D) компонент:
   - Предсказывает будущую ошибку
   - Улучшает стабильность
   - Выход = Kd × dError/dt
   - Чувствителен к шумам

Общая формула ПИД:
Output = Kp×Error + Ki×∫Error dt + Kd×dError/dt

Настройка параметров:
- Kp: увеличить для быстрой реакции
- Ki: увеличить для устранения статической ошибки
- Kd: увеличить для улучшения стабильности

Методы настройки:
1. Метод Зиглера-Николса
2. Метод Коэна-Куна
3. Автонастройка
4. Эмпирическая настройка

Применение:
- Управление температурой
- Управление давлением
- Управление уровнем
- Управление скоростью
- Управление положением
                """,
                "category": "industrial_automation",
                "tags": "pid, controller, control, automation, tuning"
            },
            {
                "filename": "plc_programming_basics.txt",
                "content": """
ПРОГРАММИРОВАНИЕ PLC - ОСНОВЫ

PLC (Programmable Logic Controller) - программируемый логический контроллер

Языки программирования по стандарту IEC 61131-3:

1. LADDER DIAGRAM (LD) - Релейная логика:
   - Графический язык
   - Основан на релейно-контактных схемах
   - Легок для понимания электриками
   - Пример: |--[ ]--[ ]--( )--|

2. STRUCTURED TEXT (ST) - Структурированный текст:
   - Текстовый язык высокого уровня
   - Похож на Pascal/C
   - Подходит для сложных алгоритмов
   - Пример: IF Enable THEN Output := TRUE; END_IF;

3. FUNCTION BLOCK DIAGRAM (FBD) - Функциональные блоки:
   - Графический язык
   - Блоки функций соединены линиями
   - Хорош для аналоговых сигналов

4. SEQUENTIAL FUNCTION CHART (SFC) - Последовательные функциональные схемы:
   - Для описания последовательностей
   - Шаги и переходы
   - Подходит для технологических процессов

5. INSTRUCTION LIST (IL) - Список инструкций:
   - Низкоуровневый язык
   - Похож на ассемблер
   - Редко используется

Основные функции:
- Цифровые входы/выходы (DI/DO)
- Аналоговые входы/выходы (AI/AO)
- Таймеры (TON, TOF, TP)
- Счетчики (CTU, CTD, CTUD)
- Сравнение (>, <, =, <>)
- Арифметические операции (+, -, *, /)

Производители:
- Siemens (S7-1200, S7-1500)
- Allen-Bradley (CompactLogix, ControlLogix)
- Schneider Electric (Modicon M580)
- ABB (AC500)
- Omron (CP1, CJ2)
                """,
                "category": "programming",
                "tags": "plc, programming, ladder, structured_text, automation"
            },
            {
                "filename": "pmac_controllers_basics.txt",
                "content": """
PMAC КОНТРОЛЛЕРЫ - ОСНОВЫ

PMAC (Programmable Multi-Axis Controller) - программируемый многоосевой контроллер от Delta Tau

Основные возможности:
- Управление до 32 осями одновременно
- Высокая точность позиционирования
- Быстрая обработка (1 мкс цикл)
- Поддержка различных типов двигателей

Типы двигателей:
1. СЕРВОПРИВОДЫ:
   - С обратной связью по положению
   - Высокая точность
   - Быстрая динамика
   - Применение: ЧПУ, роботы

2. ШАГОВЫЕ ДВИГАТЕЛИ:
   - Открытый контур
   - Простота управления
   - Низкая стоимость
   - Применение: 3D принтеры, станки

3. АСИНХРОННЫЕ ДВИГАТЕЛИ:
   - С векторным управлением
   - Высокая мощность
   - Применение: насосы, вентиляторы

Языки программирования:
1. PMAC Script:
   - Высокоуровневый язык
   - Похож на C
   - Для сложных алгоритмов

2. Motion Programs:
   - Для траекторного движения
   - Команды PVT, SPLINE
   - Синхронизация осей

3. PLC Programs:
   - Для логического управления
   - Быстрая обработка
   - Цикл 1 мкс

Основные команды:
- #1j+ - включить ось 1
- #1j- - выключить ось 1
- #1p1000 - установить позицию 1000
- #1v100 - установить скорость 100
- #1a1000 - установить ускорение 1000

Применение:
- ЧПУ станки
- Робототехника
- Полупроводниковое оборудование
- Медицинские устройства
- Измерительные машины

Преимущества:
- Высокая производительность
- Гибкость программирования
- Надежность
- Поддержка различных интерфейсов
                """,
                "category": "industrial_automation",
                "tags": "pmac, controller, delta_tau, motion_control, cnc"
            },
            {
                "filename": "python_programming_basics.txt",
                "content": """
PYTHON ПРОГРАММИРОВАНИЕ - ОСНОВЫ

Python - высокоуровневый язык программирования общего назначения

Основные особенности:
- Простой и читаемый синтаксис
- Динамическая типизация
- Интерпретируемый язык
- Кроссплатформенность
- Большая стандартная библиотека

Переменные и типы данных:
- Числа: int, float, complex
- Строки: str
- Списки: list
- Словари: dict
- Кортежи: tuple
- Множества: set

Примеры:
```python
# Числа
age = 25
height = 1.75
complex_num = 3 + 4j

# Строки
name = "Rubin AI"
message = 'Привет, мир!'

# Списки
numbers = [1, 2, 3, 4, 5]
fruits = ['яблоко', 'банан', 'апельсин']

# Словари
person = {
    'name': 'Иван',
    'age': 30,
    'city': 'Москва'
}
```

Управляющие структуры:
```python
# Условные операторы
if age >= 18:
    print("Совершеннолетний")
elif age >= 13:
    print("Подросток")
else:
    print("Ребенок")

# Циклы
for i in range(5):
    print(i)

while condition:
    # код
    pass
```

Функции:
```python
def greet(name):
    return f"Привет, {name}!"

def calculate_area(length, width):
    return length * width

# Лямбда функции
square = lambda x: x ** 2
```

Классы и объекты:
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Привет, я {self.name}"

person = Person("Анна", 25)
print(person.greet())
```

Модули и пакеты:
```python
import math
from datetime import datetime
import numpy as np

# Создание модуля
# mymodule.py
def my_function():
    return "Hello from module"

# Использование
import mymodule
result = mymodule.my_function()
```

Обработка исключений:
```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Деление на ноль!")
except Exception as e:
    print(f"Ошибка: {e}")
finally:
    print("Блок finally выполняется всегда")
```

Применение в промышленности:
- Автоматизация процессов
- Анализ данных
- Машинное обучение
- Веб-разработка
- Научные вычисления
                """,
                "category": "programming",
                "tags": "python, programming, variables, functions, classes"
            }
        ]
        
        self.training_data = basic_knowledge
        print(f"📚 Загружено {len(basic_knowledge)} базовых знаний")
        
    def upload_training_data(self):
        """Загрузка обучающих данных в систему"""
        
        print("🚀 Начинаем загрузку обучающих данных...")
        
        for i, data in enumerate(self.training_data, 1):
            try:
                # Отправка данных на сервер
                response = requests.post(
                    f"{self.server_url}/api/documents/upload-content",
                    json={
                        "filename": data["filename"],
                        "content": data["content"],
                        "category": data["category"],
                        "tags": data["tags"].split(", ")
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "success":
                        print(f"✅ {i}/{len(self.training_data)} Загружено: {data['filename']}")
                    else:
                        print(f"❌ {i}/{len(self.training_data)} Ошибка загрузки: {data['filename']}")
                else:
                    print(f"❌ {i}/{len(self.training_data)} HTTP ошибка: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ {i}/{len(self.training_data)} Ошибка соединения: {e}")
                
            # Небольшая пауза между загрузками
            time.sleep(0.5)
            
        print("🎉 Загрузка обучающих данных завершена!")
        
    def test_learning(self):
        """Тестирование обученной системы"""
        
        print("🧪 Начинаем тестирование обученной системы...")
        
        test_questions = [
            "Как работает сервопривод?",
            "Что такое ПИД регулятор?",
            "Как программировать PLC?",
            "Что такое PMAC контроллер?",
            "Как работают переменные в Python?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            try:
                print(f"\n📝 Тест {i}/{len(test_questions)}: {question}")
                
                response = requests.post(
                    f"{self.server_url}/api/chat",
                    json={"message": question},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("response", "Нет ответа")
                    print(f"🤖 Ответ: {answer[:100]}...")
                else:
                    print(f"❌ Ошибка HTTP: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Ошибка соединения: {e}")
                
            time.sleep(1)
            
        print("\n🎯 Тестирование завершено!")
        
    def show_training_stats(self):
        """Показать статистику обучения"""
        
        print("\n📊 СТАТИСТИКА ОБУЧЕНИЯ")
        print("=" * 50)
        
        # Статистика по категориям
        category_stats = {}
        for data in self.training_data:
            category = data["category"]
            category_stats[category] = category_stats.get(category, 0) + 1
            
        print("📚 Загруженные категории:")
        for category, count in category_stats.items():
            print(f"  • {category}: {count} документов")
            
        print(f"\n📈 Общая статистика:")
        print(f"  • Всего документов: {len(self.training_data)}")
        print(f"  • Категорий знаний: {len(category_stats)}")
        print(f"  • Время загрузки: {datetime.now().strftime('%H:%M:%S')}")
        
    def run_training(self):
        """Запуск полного процесса обучения"""
        
        print("🎓 RUBIN AI - СИСТЕМА ОБУЧЕНИЯ")
        print("=" * 50)
        
        # Проверка сервера
        if not self.check_server_status():
            print("❌ Не удалось запустить обучение. Проверьте сервер.")
            return
            
        # Загрузка базовых данных
        self.load_basic_training_data()
        
        # Загрузка в систему
        self.upload_training_data()
        
        # Показать статистику
        self.show_training_stats()
        
        # Тестирование
        self.test_learning()
        
        print("\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("Rubin AI готов к работе с расширенными знаниями.")
        print("\n💡 Следующие шаги:")
        print("  1. Откройте RubinIDE.html в браузере")
        print("  2. Протестируйте новые знания")
        print("  3. Добавьте обратную связь")
        print("  4. Продолжайте обучение новыми данными")

def main():
    """Главная функция"""
    
    trainer = RubinTrainer()
    trainer.run_training()

if __name__ == "__main__":
    main()
