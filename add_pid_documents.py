#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Добавление документов по ПИД-регуляторам в базу данных Rubin AI
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any

class PIDDocumentAdder:
    """Класс для добавления документов по ПИД-регуляторам"""
    
    def __init__(self, db_path: str = "rubin_ai_v2.db"):
        self.db_path = db_path
        self.connection = None
        self.connect_database()
    
    def connect_database(self):
        """Подключение к базе данных"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.execute("PRAGMA foreign_keys = ON")
            print("✅ Подключение к базе данных установлено")
        except Exception as e:
            print(f"❌ Ошибка подключения к БД: {e}")
            raise
    
    def add_pid_documents(self):
        """Добавление документов по ПИД-регуляторам"""
        try:
            cursor = self.connection.cursor()
            
            # Создаем таблицу документов если не существует
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT,
                    file_name TEXT,
                    file_size INTEGER,
                    file_type TEXT,
                    file_hash TEXT,
                    content TEXT,
                    metadata TEXT,
                    category TEXT,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    difficulty_level TEXT,
                    last_updated TEXT
                )
            """)
            
            # Документы по ПИД-регуляторам
            pid_documents = [
                {
                    "file_name": "Основы_ПИД_регулирования.pdf",
                    "content": """
# Основы ПИД-регулирования

## Введение
ПИД-регулятор (Пропорционально-Интегрально-Дифференциальный регулятор) - это устройство управления с обратной связью, широко используемое в системах автоматического управления.

## Принцип работы
ПИД-регулятор вычисляет ошибку как разность между заданным значением (уставкой) и текущим значением процесса, и применяет корректирующее воздействие на основе пропорционального, интегрального и дифференциального членов.

### Формула ПИД-регулятора:
u(t) = Kp * e(t) + Ki * ∫e(t)dt + Kd * de(t)/dt

Где:
- u(t) - управляющее воздействие
- e(t) - ошибка регулирования
- Kp - коэффициент пропорционального усиления
- Ki - коэффициент интегрального усиления  
- Kd - коэффициент дифференциального усиления

## Настройка параметров

### Пропорциональный член (P)
- Увеличивает скорость отклика системы
- Слишком большое значение может вызвать колебания
- Слишком малое значение приводит к медленному отклику

### Интегральный член (I)
- Устраняет статическую ошибку
- Может вызвать перерегулирование
- Важен для точного поддержания заданного значения

### Дифференциальный член (D)
- Предсказывает будущее поведение системы
- Уменьшает перерегулирование
- Может усилить шум в системе

## Методы настройки

### Метод Циглера-Николса
1. Установить Ki = 0, Kd = 0
2. Увеличивать Kp до появления устойчивых колебаний
3. Записать критический период колебаний
4. Рассчитать параметры по формулам

### Метод проб и ошибок
1. Начать с малых значений всех параметров
2. Постепенно увеличивать Kp до получения быстрого отклика
3. Добавить Ki для устранения статической ошибки
4. Добавить Kd для уменьшения перерегулирования

## Практические рекомендации
- Всегда начинайте настройку с пропорционального члена
- Интегральный член добавляйте только при необходимости
- Дифференциальный член используйте осторожно
- Тестируйте систему в различных условиях
- Документируйте найденные параметры
                    """,
                    "category": "автоматизация",
                    "tags": "ПИД, регулятор, автоматизация, настройка, параметры",
                    "difficulty_level": "средний"
                },
                {
                    "file_name": "Настройка_ПИД_регуляторов_практическое_руководство.pdf",
                    "content": """
# Настройка ПИД-регуляторов: Практическое руководство

## Пошаговая настройка

### Шаг 1: Анализ объекта управления
Перед настройкой ПИД-регулятора необходимо:
- Определить тип объекта управления
- Измерить временные характеристики
- Оценить инерционность системы
- Учесть внешние возмущения

### Шаг 2: Выбор метода настройки
- **Метод Циглера-Николса** - для простых объектов
- **Метод проб и ошибок** - для сложных систем
- **Адаптивная настройка** - для изменяющихся условий

### Шаг 3: Начальная настройка
1. Установить все параметры в ноль
2. Постепенно увеличивать Kp
3. Найти критическое значение
4. Определить период колебаний

### Шаг 4: Тонкая настройка
- Корректировка пропорционального коэффициента
- Добавление интегрального члена
- Настройка дифференциального члена
- Проверка качества регулирования

## Критерии качества регулирования

### Временные характеристики
- Время переходного процесса
- Перерегулирование
- Время установления
- Количество колебаний

### Статические характеристики
- Статическая ошибка
- Коэффициент статизма
- Зона нечувствительности

### Динамические характеристики
- Запас устойчивости
- Быстродействие
- Сглаживание возмущений

## Типичные проблемы и решения

### Проблема: Система нестабильна
**Причины:**
- Слишком большой Kp
- Неправильная настройка Ki
- Высокий уровень шума

**Решения:**
- Уменьшить Kp
- Проверить настройки Ki и Kd
- Добавить фильтрацию сигнала

### Проблема: Медленный отклик
**Причины:**
- Слишком малый Kp
- Отсутствие дифференциального члена
- Большая инерционность объекта

**Решения:**
- Увеличить Kp
- Добавить Kd
- Оптимизировать объект управления

### Проблема: Статическая ошибка
**Причины:**
- Отсутствие интегрального члена
- Слишком малый Ki
- Насыщение исполнительного устройства

**Решения:**
- Добавить интегральный член
- Увеличить Ki
- Проверить диапазон работы исполнительного устройства

## Программные инструменты

### Симуляторы ПИД-регуляторов
- MATLAB Simulink
- LabVIEW Control Design
- Python Control Systems Library
- Scilab Xcos

### Автоматическая настройка
- Адаптивные алгоритмы
- Нейронные сети
- Генетические алгоритмы
- Нечеткая логика

## Практические примеры

### Пример 1: Регулирование температуры
**Объект:** Печь с нагревателем
**Задача:** Поддержание температуры 200°C
**Параметры:** Kp=2.5, Ki=0.1, Kd=0.05
**Результат:** Время установления 3 мин, перерегулирование 5%

### Пример 2: Регулирование уровня
**Объект:** Емкость с жидкостью
**Задача:** Поддержание уровня 1.5 м
**Параметры:** Kp=1.8, Ki=0.05, Kd=0.02
**Результат:** Статическая ошибка < 1%, время отклика 2 мин

### Пример 3: Регулирование давления
**Объект:** Пневматическая система
**Задача:** Поддержание давления 5 бар
**Параметры:** Kp=3.0, Ki=0.2, Kd=0.1
**Результат:** Быстрый отклик, хорошее сглаживание возмущений
                    """,
                    "category": "автоматизация",
                    "tags": "ПИД, настройка, практика, руководство, примеры",
                    "difficulty_level": "высокий"
                },
                {
                    "file_name": "ПИД_регуляторы_в_промышленности.pdf",
                    "content": """
# ПИД-регуляторы в промышленности

## Области применения

### Химическая промышленность
- Регулирование температуры реакторов
- Контроль давления в системах
- Управление уровнем в емкостях
- Регулирование расхода веществ

### Пищевая промышленность
- Контроль температуры пастеризации
- Регулирование влажности
- Управление скоростью конвейеров
- Контроль качества продукции

### Металлургия
- Регулирование температуры печей
- Контроль состава сплавов
- Управление давлением в системах
- Регулирование скорости прокатки

### Энергетика
- Контроль температуры котлов
- Регулирование давления пара
- Управление частотой генераторов
- Контроль нагрузки турбин

## Типы промышленных ПИД-регуляторов

### Аналоговые регуляторы
- Простота настройки
- Высокая надежность
- Ограниченная функциональность
- Применение в простых системах

### Цифровые регуляторы
- Гибкость настройки
- Возможность программирования
- Связь с системами управления
- Применение в сложных системах

### Программируемые логические контроллеры (ПЛК)
- Интеграция с системами автоматизации
- Возможность сложной логики
- Связь с SCADA системами
- Применение в распределенных системах

## Особенности промышленного применения

### Условия эксплуатации
- Высокие температуры
- Вибрации и удары
- Электромагнитные помехи
- Агрессивные среды

### Требования к надежности
- Непрерывная работа 24/7
- Высокая точность регулирования
- Быстрое восстановление после сбоев
- Резервирование критических систем

### Интеграция с системами управления
- Связь с SCADA
- Интеграция с MES
- Обмен данными с ERP
- Удаленный мониторинг

## Стандарты и нормативы

### Международные стандарты
- IEC 61131-3 (программирование ПЛК)
- IEC 61508 (функциональная безопасность)
- ISA-88 (автоматизация производства)
- ISA-95 (интеграция систем)

### Российские стандарты
- ГОСТ Р 51841-2001 (ПИД-регуляторы)
- ГОСТ Р 51904-2002 (системы управления)
- ГОСТ Р 52070-2003 (безопасность систем)

## Современные тенденции

### Адаптивные системы
- Автоматическая настройка параметров
- Адаптация к изменяющимся условиям
- Самообучающиеся алгоритмы
- Предиктивное управление

### Цифровизация
- Интеграция с IoT
- Облачные вычисления
- Большие данные
- Искусственный интеллект

### Безопасность
- Кибербезопасность
- Функциональная безопасность
- Резервирование систем
- Аварийные отключения

## Экономические аспекты

### Эффективность внедрения
- Снижение энергопотребления
- Повышение качества продукции
- Уменьшение брака
- Оптимизация ресурсов

### Окупаемость инвестиций
- Срок окупаемости 1-3 года
- Экономия на энергоресурсах
- Повышение производительности
- Снижение эксплуатационных расходов

### Выбор поставщика
- Техническая поддержка
- Обучение персонала
- Гарантийное обслуживание
- Долгосрочное партнерство
                    """,
                    "category": "автоматизация",
                    "tags": "ПИД, промышленность, применение, стандарты, экономика",
                    "difficulty_level": "высокий"
                },
                {
                    "file_name": "Математические_основы_ПИД_регулирования.pdf",
                    "content": """
# Математические основы ПИД-регулирования

## Математическая модель

### Дифференциальное уравнение
ПИД-регулятор описывается дифференциальным уравнением:
d²u/dt² + (Kp/Kd)du/dt + (Ki/Kd)u = Kp/Kd * de/dt + Ki/Kd * e

### Передаточная функция
W(s) = Kp + Ki/s + Kd*s

Где s - оператор Лапласа

### Частотные характеристики
- Амплитудно-частотная характеристика (АЧХ)
- Фазо-частотная характеристика (ФЧХ)
- Логарифмические характеристики

## Анализ устойчивости

### Критерий Гурвица
Для устойчивости системы необходимо и достаточно, чтобы все главные миноры матрицы Гурвица были положительными.

### Критерий Найквиста
Система устойчива, если годограф разомкнутой системы не охватывает точку (-1, j0).

### Критерий Михайлова
Система устойчива, если годограф Михайлова проходит последовательно n квадрантов в положительном направлении.

## Синтез регулятора

### Метод модального управления
Выбор полюсов замкнутой системы для обеспечения желаемых динамических характеристик.

### Метод оптимального управления
Минимизация функционала качества:
J = ∫[x^T(t)Qx(t) + u^T(t)Ru(t)]dt

### Метод робастного управления
Обеспечение устойчивости при неопределенности параметров объекта.

## Дискретные ПИД-регуляторы

### Разностное уравнение
u(k) = u(k-1) + Kp[e(k) - e(k-1)] + Ki*T*e(k) + Kd/T[e(k) - 2e(k-1) + e(k-2)]

Где T - период дискретизации

### Z-преобразование
W(z) = Kp + Ki*T*z/(z-1) + Kd*(z-1)/(T*z)

### Критерии выбора периода дискретизации
- Теорема Котельникова: T ≤ π/ωmax
- Практическое правило: T ≤ Tmin/10
- Учет вычислительных ресурсов

## Нелинейные эффекты

### Насыщение исполнительного устройства
- Ограничение выходного сигнала
- Антивинд-ап алгоритмы
- Компенсация насыщения

### Зона нечувствительности
- Мертвая зона в системе
- Гистерезис
- Трение в механических системах

### Запаздывание
- Транспортное запаздывание
- Чистое запаздывание
- Методы компенсации

## Адаптивные алгоритмы

### Самонастраивающиеся регуляторы
- Идентификация параметров объекта
- Автоматическая настройка параметров регулятора
- Адаптация к изменяющимся условиям

### Нейросетевые регуляторы
- Обучение нейронной сети
- Аппроксимация нелинейных функций
- Адаптивное управление

### Нечеткие регуляторы
- Лингвистические правила
- Нечеткая логика
- Гибридные системы

## Численные методы

### Методы оптимизации
- Градиентные методы
- Методы случайного поиска
- Генетические алгоритмы

### Методы идентификации
- Метод наименьших квадратов
- Рекуррентные алгоритмы
- Методы максимального правдоподобия

### Симуляция систем
- Численное интегрирование
- Методы Рунге-Кутта
- Событийное моделирование
                    """,
                    "category": "автоматизация",
                    "tags": "ПИД, математика, теория, анализ, синтез",
                    "difficulty_level": "высокий"
                },
                {
                    "file_name": "Программирование_ПИД_регуляторов.pdf",
                    "content": """
# Программирование ПИД-регуляторов

## Языки программирования

### Ladder Logic (LD)
```
|--[PID]--|
|  SP: 100 |
|  PV: AI1 |
|  CV: AO1 |
|  Kp: 2.0 |
|  Ki: 0.1 |
|  Kd: 0.05|
```

### Function Block Diagram (FBD)
```
[AI1]--[PID_BLOCK]--[AO1]
        |    |
        |    +--[SP: 100]
        +--[PARAMS: Kp=2.0, Ki=0.1, Kd=0.05]
```

### Structured Text (ST)
```
PROGRAM PID_CONTROL
VAR
    pid_regulator : PID;
    setpoint : REAL := 100.0;
    process_value : REAL;
    control_value : REAL;
    kp : REAL := 2.0;
    ki : REAL := 0.1;
    kd : REAL := 0.05;
END_VAR

process_value := AI1;
pid_regulator(setpoint := setpoint,
              process_value := process_value,
              kp := kp,
              ki := ki,
              kd := kd,
              control_value => control_value);
AO1 := control_value;
END_PROGRAM
```

## Реализация на C/C++

### Базовый класс ПИД-регулятора
```cpp
class PIDController {
private:
    double kp, ki, kd;
    double integral, previous_error;
    double dt;
    double output_min, output_max;
    
public:
    PIDController(double kp, double ki, double kd, double dt) {
        this->kp = kp;
        this->ki = ki;
        this->kd = kd;
        this->dt = dt;
        this->integral = 0;
        this->previous_error = 0;
        this->output_min = 0;
        this->output_max = 100;
    }
    
    double calculate(double setpoint, double process_value) {
        double error = setpoint - process_value;
        
        // Пропорциональный член
        double proportional = kp * error;
        
        // Интегральный член
        integral += error * dt;
        double integral_term = ki * integral;
        
        // Дифференциальный член
        double derivative = (error - previous_error) / dt;
        double derivative_term = kd * derivative;
        
        // Выходной сигнал
        double output = proportional + integral_term + derivative_term;
        
        // Ограничение выходного сигнала
        if (output > output_max) output = output_max;
        if (output < output_min) output = output_min;
        
        previous_error = error;
        return output;
    }
    
    void setLimits(double min, double max) {
        output_min = min;
        output_max = max;
    }
    
    void reset() {
        integral = 0;
        previous_error = 0;
    }
};
```

## Реализация на Python

### Класс ПИД-регулятора
```python
import time

class PIDController:
    def __init__(self, kp, ki, kd, dt=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        
        self.integral = 0
        self.previous_error = 0
        self.previous_time = time.time()
        
        self.output_min = 0
        self.output_max = 100
    
    def calculate(self, setpoint, process_value):
        current_time = time.time()
        dt = current_time - self.previous_time
        
        if dt <= 0:
            dt = self.dt
        
        error = setpoint - process_value
        
        # Пропорциональный член
        proportional = self.kp * error
        
        # Интегральный член
        self.integral += error * dt
        integral_term = self.ki * self.integral
        
        # Дифференциальный член
        derivative = (error - self.previous_error) / dt
        derivative_term = self.kd * derivative
        
        # Выходной сигнал
        output = proportional + integral_term + derivative_term
        
        # Ограничение выходного сигнала
        output = max(self.output_min, min(self.output_max, output))
        
        self.previous_error = error
        self.previous_time = current_time
        
        return output
    
    def set_limits(self, min_val, max_val):
        self.output_min = min_val
        self.output_max = max_val
    
    def reset(self):
        self.integral = 0
        self.previous_error = 0
        self.previous_time = time.time()
```

## Антивинд-ап алгоритмы

### Условный интеграл
```cpp
double calculate_with_antiwindup(double setpoint, double process_value) {
    double error = setpoint - process_value;
    double output = kp * error + ki * integral + kd * derivative;
    
    // Антивинд-ап
    if (output > output_max) {
        output = output_max;
        if (error > 0) {
            integral += error * dt; // Интегрируем только если ошибка уменьшает выход
        }
    } else if (output < output_min) {
        output = output_min;
        if (error < 0) {
            integral += error * dt;
        }
    } else {
        integral += error * dt;
    }
    
    return output;
}
```

### Обратное вычисление
```cpp
double calculate_with_backcalculation(double setpoint, double process_value) {
    double error = setpoint - process_value;
    double output = kp * error + ki * integral + kd * derivative;
    
    // Ограничение
    double limited_output = max(output_min, min(output_max, output));
    
    // Обратное вычисление интеграла
    double back_calc = (limited_output - output) / ki;
    integral += error * dt + back_calc * dt;
    
    return limited_output;
}
```

## Фильтрация дифференциального члена

### Фильтр первого порядка
```cpp
class PIDWithFilter {
private:
    double derivative_filter;
    double alpha; // Коэффициент фильтра
    
public:
    PIDWithFilter(double kp, double ki, double kd, double dt, double filter_time) {
        this->kp = kp;
        this->ki = ki;
        this->kd = kd;
        this->dt = dt;
        this->alpha = dt / (filter_time + dt);
        this->derivative_filter = 0;
    }
    
    double calculate(double setpoint, double process_value) {
        double error = setpoint - process_value;
        double derivative = (error - previous_error) / dt;
        
        // Фильтрация дифференциального члена
        derivative_filter = alpha * derivative + (1 - alpha) * derivative_filter;
        
        double output = kp * error + ki * integral + kd * derivative_filter;
        
        integral += error * dt;
        previous_error = error;
        
        return output;
    }
};
```

## Тестирование и отладка

### Модульные тесты
```python
import unittest

class TestPIDController(unittest.TestCase):
    def setUp(self):
        self.pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
    
    def test_proportional_response(self):
        # Тест пропорционального отклика
        output = self.pid.calculate(setpoint=100, process_value=90)
        self.assertAlmostEqual(output, 10.0, places=2)
    
    def test_integral_elimination(self):
        # Тест устранения статической ошибки
        for _ in range(100):
            output = self.pid.calculate(setpoint=100, process_value=90)
        self.assertGreater(output, 10.0)  # Интегральный член должен увеличить выход
    
    def test_derivative_prediction(self):
        # Тест предсказания
        self.pid.calculate(setpoint=100, process_value=90)
        output = self.pid.calculate(setpoint=100, process_value=95)
        # Дифференциальный член должен уменьшить выход при уменьшении ошибки
        self.assertLess(output, 5.0)
```

### Симуляция системы
```python
def simulate_pid_system():
    pid = PIDController(kp=2.0, ki=0.1, kd=0.05)
    process_value = 0
    setpoint = 100
    
    for i in range(1000):
        control_value = pid.calculate(setpoint, process_value)
        # Простая модель процесса первого порядка
        process_value += (control_value - process_value) * 0.01
        
        if i % 100 == 0:
            print(f"Time: {i}, PV: {process_value:.2f}, CV: {control_value:.2f}")
```

## Оптимизация производительности

### Использование фиксированной точки
```cpp
// Использование целых чисел вместо float для ускорения
class FastPIDController {
private:
    int32_t kp, ki, kd;
    int32_t integral, previous_error;
    int32_t output_min, output_max;
    int32_t scale_factor = 1000; // Масштабирующий коэффициент
    
public:
    FastPIDController(int32_t kp, int32_t ki, int32_t kd) {
        this->kp = kp * scale_factor;
        this->ki = ki * scale_factor;
        this->kd = kd * scale_factor;
        this->integral = 0;
        this->previous_error = 0;
    }
    
    int32_t calculate(int32_t setpoint, int32_t process_value) {
        int32_t error = setpoint - process_value;
        
        int32_t proportional = (kp * error) / scale_factor;
        integral += error;
        int32_t integral_term = (ki * integral) / scale_factor;
        int32_t derivative = error - previous_error;
        int32_t derivative_term = (kd * derivative) / scale_factor;
        
        int32_t output = proportional + integral_term + derivative_term;
        
        output = max(output_min, min(output_max, output));
        
        previous_error = error;
        return output;
    }
};
```

## Интеграция с системами реального времени

### Использование таймеров
```cpp
void timer_interrupt_handler() {
    static PIDController pid(2.0, 0.1, 0.05, 0.01);
    
    double setpoint = read_setpoint();
    double process_value = read_sensor();
    
    double control_value = pid.calculate(setpoint, process_value);
    
    write_actuator(control_value);
}
```

### Приоритеты задач
```cpp
// Высокий приоритет для ПИД-регулятора
void pid_task() {
    while (true) {
        pid_controller.calculate();
        vTaskDelay(pdMS_TO_TICKS(10)); // 10 мс период
    }
}

// Низкий приоритет для интерфейса
void interface_task() {
    while (true) {
        update_display();
        vTaskDelay(pdMS_TO_TICKS(100)); // 100 мс период
    }
}
```
                    """,
                    "category": "автоматизация",
                    "tags": "ПИД, программирование, код, реализация, тестирование",
                    "difficulty_level": "высокий"
                }
            ]
            
            added_count = 0
            for doc in pid_documents:
                try:
                    # Проверяем, существует ли уже такой документ
                    cursor.execute("SELECT COUNT(*) FROM documents WHERE file_name = ?", (doc["file_name"],))
                    if cursor.fetchone()[0] == 0:
                        cursor.execute("""
                            INSERT INTO documents 
                            (file_path, file_name, content, category, tags, difficulty_level, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            f"documents/{doc['file_name']}",
                            doc["file_name"],
                            doc["content"],
                            doc["category"],
                            doc["tags"],
                            doc["difficulty_level"],
                            datetime.now().isoformat(),
                            datetime.now().isoformat()
                        ))
                        added_count += 1
                        print(f"✅ Добавлен документ: {doc['file_name']}")
                    else:
                        print(f"⚠️ Документ уже существует: {doc['file_name']}")
                        
                except Exception as e:
                    print(f"❌ Ошибка добавления документа {doc['file_name']}: {e}")
            
            self.connection.commit()
            print(f"\n🎉 Добавлено {added_count} новых документов по ПИД-регуляторам!")
            
            return added_count
            
        except Exception as e:
            print(f"❌ Ошибка добавления документов: {e}")
            return 0
    
    def update_synonyms_for_pid(self):
        """Обновление синонимов для ПИД-регуляторов"""
        try:
            cursor = self.connection.cursor()
            
            # Дополнительные синонимы для ПИД-регуляторов
            additional_synonyms = [
                ("симистр", "тиристор", "электротехника"),
                ("инвертор", "преобразователь частоты", "электротехника"),
                ("энкодер", "датчик положения", "автоматизация"),
                ("сервопривод", "серводвигатель", "автоматизация"),
                ("частотник", "частотный преобразователь", "электротехника"),
                ("регулятор температуры", "терморегулятор", "автоматизация"),
                ("контроллер давления", "регулятор давления", "автоматизация"),
                ("датчик уровня", "сенсор уровня", "автоматизация"),
                ("исполнительный механизм", "привод", "автоматизация"),
                ("обратная связь", "feedback", "автоматизация"),
                ("уставка", "заданное значение", "автоматизация"),
                ("перерегулирование", "overshoot", "автоматизация"),
                ("статическая ошибка", "steady-state error", "автоматизация"),
                ("время установления", "settling time", "автоматизация"),
                ("коэффициент усиления", "gain", "автоматизация")
            ]
            
            added_synonyms = 0
            for main_term, synonym, category in additional_synonyms:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO technical_synonyms 
                        (main_term, synonym, category, confidence)
                        VALUES (?, ?, ?, ?)
                    """, (main_term, synonym, category, 1.0))
                    added_synonyms += 1
                except:
                    pass
            
            self.connection.commit()
            print(f"✅ Добавлено {added_synonyms} новых синонимов для ПИД-регуляторов")
            
            return added_synonyms
            
        except Exception as e:
            print(f"❌ Ошибка обновления синонимов: {e}")
            return 0
    
    def close_connection(self):
        """Закрытие соединения с базой данных"""
        if self.connection:
            self.connection.close()
            print("✅ Соединение с БД закрыто")

def main():
    """Основная функция"""
    print("📚 ДОБАВЛЕНИЕ ДОКУМЕНТОВ ПО ПИД-РЕГУЛЯТОРАМ")
    print("=" * 60)
    
    adder = PIDDocumentAdder()
    
    try:
        # Добавляем документы
        docs_added = adder.add_pid_documents()
        
        # Обновляем синонимы
        synonyms_added = adder.update_synonyms_for_pid()
        
        print(f"\n📊 ИТОГИ:")
        print(f"  - Добавлено документов: {docs_added}")
        print(f"  - Добавлено синонимов: {synonyms_added}")
        print(f"  - Категория: автоматизация")
        print(f"  - Уровни сложности: средний, высокий")
        
        print(f"\n🎯 СЛЕДУЮЩИЕ ШАГИ:")
        print("  1. Перезапустите систему поиска")
        print("  2. Протестируйте поиск по ПИД-регуляторам")
        print("  3. Проверьте качество результатов")
        
        print(f"\n📅 Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
    finally:
        adder.close_connection()

if __name__ == "__main__":
    main()
