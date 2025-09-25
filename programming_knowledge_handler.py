#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обработчик знаний по программированию для Rubin AI
Специализированные ответы на вопросы о программировании, языках и алгоритмах
"""

import re
import logging
from typing import Dict, List, Optional
from datetime import datetime

class ProgrammingKnowledgeHandler:
    """Обработчик знаний по программированию"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # База знаний по программированию
        self.knowledge_base = {
            'cpp_vs_python': {
                'keywords': ['c++', 'python', 'сравни', 'compare', 'vs', 'против', 'автоматизация', 'automation'],
                'response_ru': """💻 **C++ vs Python для промышленной автоматизации:**

**C++ - преимущества:**
• **Высокая производительность** - компилируемый язык, быстрое выполнение
• **Низкоуровневый доступ** - прямая работа с памятью, регистрами, железом
• **Детерминированность** - предсказуемое время выполнения операций
• **Малое потребление памяти** - критично для встраиваемых систем
• **Реальное время** - поддержка жестких временных требований
• **Интеграция с PLC** - легкая работа с промышленными протоколами

**C++ - недостатки:**
• **Сложность разработки** - много кода для простых задач
• **Время разработки** - долгая отладка и тестирование
• **Управление памятью** - риск утечек памяти и ошибок
• **Портируемость** - зависимость от платформы и компилятора

---

**Python - преимущества:**
• **Быстрая разработка** - простой и понятный синтаксис
• **Богатые библиотеки** - NumPy, Pandas, Matplotlib, SciPy
• **Простота изучения** - низкий порог входа для новых разработчиков
• **Интерпретируемость** - быстрое тестирование и прототипирование
• **Кроссплатформенность** - работает на любой ОС
• **Анализ данных** - отличные инструменты для обработки данных

**Python - недостатки:**
• **Низкая производительность** - интерпретируемый язык медленнее
• **Потребление памяти** - больше RAM чем C++
• **GIL ограничения** - проблемы с многопоточностью
• **Не для реального времени** - непредсказуемые задержки сборщика мусора

---

**Рекомендации по выбору:**

**Выбирайте C++ для:**
• Управления двигателями и сервоприводами
• Обработки сигналов в реальном времени
• Программирования микроконтроллеров и PLC
• Критичных по безопасности систем
• Высокочастотных измерений и управления

**Выбирайте Python для:**
• Мониторинга и диспетчеризации (SCADA)
• Анализа производственных данных
• Создания веб-интерфейсов управления
• Прототипирования алгоритмов
• Машинного обучения для предиктивного обслуживания
• Интеграции различных систем через API

**Гибридный подход (лучшее решение):**
• **C++** - критичные компоненты реального времени
• **Python** - анализ данных, интерфейсы, отчеты
• **Связь через:** TCP/IP, REST API, shared memory, файлы

**Популярные комбинации:**
• C++ + Python через pybind11 или ctypes
• C++ для драйверов PLC, Python для HMI
• C++ микросервисы + Python веб-интерфейс
• C++ алгоритмы + Python аналитика

✅ **Вывод: Выбор зависит от конкретной задачи. Для максимальной эффективности используйте гибридный подход!**""",
                
                'response_en': """💻 **C++ vs Python for Industrial Automation:**

**C++ - Advantages:**
• **High Performance** - compiled language, fast execution
• **Low-level Access** - direct memory, register, hardware access
• **Deterministic** - predictable execution time
• **Low Memory Usage** - critical for embedded systems
• **Real-time Support** - hard real-time requirements
• **PLC Integration** - easy work with industrial protocols

**C++ - Disadvantages:**
• **Development Complexity** - lots of code for simple tasks
• **Development Time** - long debugging and testing
• **Memory Management** - risk of memory leaks and errors
• **Portability** - platform and compiler dependencies

---

**Python - Advantages:**
• **Rapid Development** - simple and clear syntax
• **Rich Libraries** - NumPy, Pandas, Matplotlib, SciPy
• **Easy to Learn** - low entry barrier for new developers
• **Interpreted** - fast testing and prototyping
• **Cross-platform** - works on any OS
• **Data Analysis** - excellent data processing tools

**Python - Disadvantages:**
• **Lower Performance** - interpreted language is slower
• **Memory Usage** - more RAM than C++
• **GIL Limitations** - multithreading issues
• **Not Real-time** - unpredictable garbage collector delays

---

**Selection Recommendations:**

**Choose C++ for:**
• Motor and servo control
• Real-time signal processing
• Microcontroller and PLC programming
• Safety-critical systems
• High-frequency measurements and control

**Choose Python for:**
• Monitoring and dispatching (SCADA)
• Production data analysis
• Web control interfaces
• Algorithm prototyping
• Machine learning for predictive maintenance
• System integration via APIs

**Hybrid Approach (best solution):**
• **C++** - real-time critical components
• **Python** - data analysis, interfaces, reports
• **Communication via:** TCP/IP, REST API, shared memory, files

**Popular Combinations:**
• C++ + Python via pybind11 or ctypes
• C++ for PLC drivers, Python for HMI
• C++ microservices + Python web interface
• C++ algorithms + Python analytics

✅ **Conclusion: Choice depends on specific task. For maximum efficiency use hybrid approach!**"""
            },
            
            'industrial_programming': {
                'keywords': ['промышленное программирование', 'industrial programming', 'plc', 'scada', 'автоматизация'],
                'response_ru': """🏭 **Промышленное программирование - полное руководство:**

## **Языки программирования PLC**

**Ladder Logic (LD):**
• **Принцип:** Визуальное программирование в виде релейных схем
• **Преимущества:** Интуитивно понятен электрикам
• **Применение:** Простые логические операции

**Structured Text (ST):**
• **Принцип:** Текстовый язык высокого уровня (Pascal/C)
• **Преимущества:** Мощные возможности, читаемость
• **Применение:** Сложные алгоритмы, математика

**Function Block Diagram (FBD):**
• **Принцип:** Графическое программирование блоками
• **Преимущества:** Модульность, переиспользование
• **Применение:** PID-регуляторы, системы управления

## **Среды разработки**

**Siemens TIA Portal:**
• Языки: LAD, STL, FBD, SCL, GRAPH
• Интеграция с HMI и диагностика

**Allen-Bradley Studio 5000:**
• Языки: Ladder, Structured Text, Function Block
• Интеграция с FactoryTalk

**Schneider Electric Unity Pro:**
• Модульная архитектура, открытость

## **Коммуникационные протоколы**

• **Modbus RTU/TCP** - простой, распространенный
• **Profibus** - высокоскоростной, надежный  
• **Ethernet/IP** - современный, масштабируемый
• **OPC UA** - универсальный, безопасный

## **Архитектура систем**

**Уровни автоматизации:**
• **Уровень 0:** Датчики, исполнительные механизмы
• **Уровень 1:** Контроллеры (PLC, DCS)
• **Уровень 2:** HMI, SCADA системы
• **Уровень 3:** MES системы
• **Уровень 4:** ERP системы

✅ **Промышленное программирование требует знания специфики производства и стандартов безопасности!**""",
                
                'response_en': """🏭 **Industrial Programming - Complete Guide:**

## **PLC Programming Languages**

**Ladder Logic (LD):**
• **Principle:** Visual programming like relay circuits
• **Advantages:** Intuitive for electricians
• **Application:** Simple logical operations

**Structured Text (ST):**
• **Principle:** High-level text language (Pascal/C-like)
• **Advantages:** Powerful capabilities, readability
• **Application:** Complex algorithms, mathematics

**Function Block Diagram (FBD):**
• **Principle:** Graphical programming with blocks
• **Advantages:** Modularity, code reuse
• **Application:** PID controllers, control systems

## **Development Environments**

**Siemens TIA Portal:**
• Languages: LAD, STL, FBD, SCL, GRAPH
• HMI integration and diagnostics

**Allen-Bradley Studio 5000:**
• Languages: Ladder, Structured Text, Function Block
• FactoryTalk integration

**Schneider Electric Unity Pro:**
• Modular architecture, openness

## **Communication Protocols**

• **Modbus RTU/TCP** - simple, widespread
• **Profibus** - high-speed, reliable
• **Ethernet/IP** - modern, scalable
• **OPC UA** - universal, secure

## **System Architecture**

**Automation Levels:**
• **Level 0:** Sensors, actuators
• **Level 1:** Controllers (PLC, DCS)
• **Level 2:** HMI, SCADA systems
• **Level 3:** MES systems
• **Level 4:** ERP systems

✅ **Industrial programming requires knowledge of production specifics and safety standards!**"""
            },
            
            'algorithms': {
                'keywords': ['алгоритм', 'algorithm', 'конвейер', 'conveyor', 'управление', 'control'],
                'response_ru': """🔧 **Алгоритмы управления в промышленной автоматизации:**

## **Алгоритм управления конвейером (Python):**

```python
import time
import threading
from enum import Enum

class ConveyorState(Enum):
    STOPPED = 0
    RUNNING = 1
    ERROR = 2
    MAINTENANCE = 3

class ConveyorController:
    def __init__(self):
        self.state = ConveyorState.STOPPED
        self.speed = 0  # м/мин
        self.items_count = 0
        self.emergency_stop = False
        self.sensors = {
            'start_button': False,
            'stop_button': False,
            'emergency': False,
            'overload': False,
            'item_detected': False
        }
    
    def start_conveyor(self):
        \"\"\"Запуск конвейера\"\"\"
        if self.emergency_stop:
            print("❌ Аварийная остановка активна!")
            return False
        
        if self.sensors['overload']:
            print("❌ Перегрузка конвейера!")
            return False
        
        self.state = ConveyorState.RUNNING
        self.speed = 10  # м/мин
        print("✅ Конвейер запущен")
        return True
    
    def stop_conveyor(self):
        \"\"\"Остановка конвейера\"\"\"
        self.state = ConveyorState.STOPPED
        self.speed = 0
        print("🛑 Конвейер остановлен")
    
    def emergency_stop_conveyor(self):
        \"\"\"Аварийная остановка\"\"\"
        self.emergency_stop = True
        self.state = ConveyorState.ERROR
        self.speed = 0
        print("🚨 АВАРИЙНАЯ ОСТАНОВКА!")
    
    def monitor_sensors(self):
        \"\"\"Мониторинг датчиков\"\"\"
        while True:
            if self.sensors['emergency']:
                self.emergency_stop_conveyor()
            
            if self.sensors['overload'] and self.state == ConveyorState.RUNNING:
                print("⚠️ Перегрузка! Снижаем скорость")
                self.speed = max(5, self.speed - 1)
            
            if self.sensors['item_detected']:
                self.items_count += 1
                print(f"📦 Предмет #{self.items_count} обнаружен")
            
            time.sleep(0.1)  # 100мс цикл
```

## **Алгоритм ПИД-регулятора (C++):**

```cpp
class PIDController {
private:
    double kp, ki, kd;  // Коэффициенты
    double prev_error, integral;
    
public:
    PIDController(double p, double i, double d) 
        : kp(p), ki(i), kd(d), prev_error(0), integral(0) {}
    
    double calculate(double setpoint, double measured_value, double dt) {
        double error = setpoint - measured_value;
        
        // Пропорциональная составляющая
        double proportional = kp * error;
        
        // Интегральная составляющая
        integral += error * dt;
        double integral_term = ki * integral;
        
        // Дифференциальная составляющая
        double derivative = (error - prev_error) / dt;
        double derivative_term = kd * derivative;
        
        prev_error = error;
        
        return proportional + integral_term + derivative_term;
    }
};
```

✅ **Ключевые принципы: безопасность, надежность, производительность!**""",
                
                'response_en': """🔧 **Control Algorithms in Industrial Automation:**

## **Conveyor Control Algorithm (Python):**

```python
import time
import threading
from enum import Enum

class ConveyorState(Enum):
    STOPPED = 0
    RUNNING = 1
    ERROR = 2
    MAINTENANCE = 3

class ConveyorController:
    def __init__(self):
        self.state = ConveyorState.STOPPED
        self.speed = 0  # m/min
        self.items_count = 0
        self.emergency_stop = False
        self.sensors = {
            'start_button': False,
            'stop_button': False,
            'emergency': False,
            'overload': False,
            'item_detected': False
        }
    
    def start_conveyor(self):
        \"\"\"Start conveyor\"\"\"
        if self.emergency_stop:
            print("❌ Emergency stop active!")
            return False
        
        if self.sensors['overload']:
            print("❌ Conveyor overload!")
            return False
        
        self.state = ConveyorState.RUNNING
        self.speed = 10  # m/min
        print("✅ Conveyor started")
        return True
    
    def stop_conveyor(self):
        \"\"\"Stop conveyor\"\"\"
        self.state = ConveyorState.STOPPED
        self.speed = 0
        print("🛑 Conveyor stopped")
    
    def emergency_stop_conveyor(self):
        \"\"\"Emergency stop\"\"\"
        self.emergency_stop = True
        self.state = ConveyorState.ERROR
        self.speed = 0
        print("🚨 EMERGENCY STOP!")
    
    def monitor_sensors(self):
        \"\"\"Sensor monitoring\"\"\"
        while True:
            if self.sensors['emergency']:
                self.emergency_stop_conveyor()
            
            if self.sensors['overload'] and self.state == ConveyorState.RUNNING:
                print("⚠️ Overload! Reducing speed")
                self.speed = max(5, self.speed - 1)
            
            if self.sensors['item_detected']:
                self.items_count += 1
                print(f"📦 Item #{self.items_count} detected")
            
            time.sleep(0.1)  # 100ms cycle
```

## **PID Controller Algorithm (C++):**

```cpp
class PIDController {
private:
    double kp, ki, kd;  // Coefficients
    double prev_error, integral;
    
public:
    PIDController(double p, double i, double d) 
        : kp(p), ki(i), kd(d), prev_error(0), integral(0) {}
    
    double calculate(double setpoint, double measured_value, double dt) {
        double error = setpoint - measured_value;
        
        // Proportional term
        double proportional = kp * error;
        
        // Integral term
        integral += error * dt;
        double integral_term = ki * integral;
        
        // Derivative term
        double derivative = (error - prev_error) / dt;
        double derivative_term = kd * derivative;
        
        prev_error = error;
        
        return proportional + integral_term + derivative_term;
    }
};
```

✅ **Key principles: safety, reliability, performance!**"""
            },
            
            'best_practices': {
                'keywords': ['лучшие практики', 'best practices', 'рекомендации', 'recommendations'],
                'response_ru': """📋 **Лучшие практики промышленного программирования:**

## **Разработка**
• **Version Control** - обязательное использование Git
• **Code Review** - проверка кода коллегами
• **Testing** - unit тесты, интеграционные тесты
• **Documentation** - подробная документация API
• **Continuous Integration** - автоматическая сборка и тесты

## **Безопасность**
• **Fail-Safe** - безопасное состояние при отказе
• **Redundancy** - резервирование критичных функций
• **Watchdog** - контроль работоспособности
• **Input Validation** - проверка всех входных данных
• **Error Handling** - обработка всех возможных ошибок

## **Производительность**
• **Real-time Constraints** - соблюдение временных ограничений
• **Memory Management** - эффективное использование памяти
• **Algorithm Optimization** - выбор оптимальных алгоритмов
• **Profiling** - измерение производительности
• **Caching** - кэширование часто используемых данных

## **Стандарты**
• **IEC 61131-3** - стандарт языков программирования PLC
• **IEC 61508** - функциональная безопасность
• **ISO 26262** - безопасность автомобильных систем
• **MISRA C** - стандарт безопасного программирования на C

✅ **Следование стандартам и лучшим практикам критично для промышленных систем!**""",
                
                'response_en': """📋 **Industrial Programming Best Practices:**

## **Development**
• **Version Control** - mandatory Git usage
• **Code Review** - peer code review
• **Testing** - unit tests, integration tests
• **Documentation** - detailed API documentation
• **Continuous Integration** - automated build and tests

## **Safety**
• **Fail-Safe** - safe state on failure
• **Redundancy** - backup of critical functions
• **Watchdog** - system health monitoring
• **Input Validation** - validate all inputs
• **Error Handling** - handle all possible errors

## **Performance**
• **Real-time Constraints** - meet timing requirements
• **Memory Management** - efficient memory usage
• **Algorithm Optimization** - choose optimal algorithms
• **Profiling** - measure performance
• **Caching** - cache frequently used data

## **Standards**
• **IEC 61131-3** - PLC programming languages standard
• **IEC 61508** - functional safety
• **ISO 26262** - automotive safety
• **MISRA C** - safe C programming standard

✅ **Following standards and best practices is critical for industrial systems!**"""
            }
        }
        
        self.logger.info("Обработчик знаний по программированию инициализирован")
    
    def detect_topic(self, message: str) -> Optional[str]:
        """Определение темы программного вопроса"""
        message_lower = message.lower()
        
        # Проверяем каждую тему в порядке приоритета
        topic_priorities = [
            'cpp_vs_python',
            'algorithms', 
            'industrial_programming',
            'best_practices'
        ]
        
        for topic in topic_priorities:
            keywords = self.knowledge_base[topic]['keywords']
            matches = sum(1 for keyword in keywords if keyword in message_lower)
            
            # Если найдено достаточно совпадений, возвращаем тему
            if matches >= 2:  # Минимум 2 ключевых слова
                return topic
            elif matches >= 1 and topic == 'cpp_vs_python':
                # Для сравнения языков достаточно одного совпадения
                return topic
        
        return None
    
    def detect_language(self, message: str) -> str:
        """Определение языка сообщения"""
        cyrillic_chars = len(re.findall(r'[а-яё]', message.lower()))
        latin_chars = len(re.findall(r'[a-z]', message.lower()))
        
        return 'ru' if cyrillic_chars > latin_chars else 'en'
    
    def handle_request(self, message: str) -> Dict:
        """Обработка программного запроса"""
        try:
            # Определяем тему и язык
            topic = self.detect_topic(message)
            language = self.detect_language(message)
            
            if not topic:
                # Общий ответ по программированию
                return self._get_general_programming_response(language)
            
            # Получаем специализированный ответ
            response_key = f'response_{language}'
            knowledge = self.knowledge_base[topic]
            
            if response_key in knowledge:
                response_text = knowledge[response_key]
            else:
                # Fallback на русский если нет перевода
                response_text = knowledge.get('response_ru', 'Информация недоступна')
            
            self.logger.info(f"Обработан программный запрос: тема='{topic}', язык='{language}'")
            
            return {
                'response': response_text,
                'provider': 'Programming Knowledge Handler',
                'category': 'programming',
                'topic': topic,
                'language': language,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки программного запроса: {e}")
            return {
                'response': f'Произошла ошибка при обработке программного вопроса: {str(e)}',
                'provider': 'Programming Knowledge Handler',
                'category': 'programming',
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_general_programming_response(self, language: str) -> Dict:
        """Общий ответ по программированию"""
        if language == 'ru':
            response = """💻 **Промышленное программирование - моя специализация!**

**Основные области:**
• **Языки программирования** - C++, Python, C#, Java
• **PLC программирование** - Ladder Logic, Structured Text
• **SCADA системы** - WinCC, FactoryTalk, Citect
• **Промышленные протоколы** - Modbus, Profibus, Ethernet/IP
• **Алгоритмы управления** - ПИД-регуляторы, машины состояний
• **Встраиваемые системы** - микроконтроллеры, реальное время

**Могу помочь с:**
• Сравнением языков программирования
• Выбором технологий для проектов
• Алгоритмами управления и автоматизации
• Лучшими практиками разработки
• Архитектурой промышленных систем

**Задайте конкретный вопрос** - например:
• "Сравни C++ и Python для автоматизации"
• "Как написать алгоритм управления конвейером?"
• "Какие лучшие практики промышленного программирования?"

✅ **Готов помочь с любыми вопросами по программированию!**"""
        else:
            response = """💻 **Industrial Programming - My Specialization!**

**Main Areas:**
• **Programming Languages** - C++, Python, C#, Java
• **PLC Programming** - Ladder Logic, Structured Text
• **SCADA Systems** - WinCC, FactoryTalk, Citect
• **Industrial Protocols** - Modbus, Profibus, Ethernet/IP
• **Control Algorithms** - PID controllers, state machines
• **Embedded Systems** - microcontrollers, real-time

**I can help with:**
• Programming language comparisons
• Technology selection for projects
• Control and automation algorithms
• Development best practices
• Industrial system architecture

**Ask a specific question** - for example:
• "Compare C++ and Python for automation"
• "How to write conveyor control algorithm?"
• "What are industrial programming best practices?"

✅ **Ready to help with any programming questions!**"""
        
        return {
            'response': response,
            'provider': 'Programming Knowledge Handler',
            'category': 'programming',
            'topic': 'general',
            'language': language,
            'success': True,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_supported_topics(self) -> List[str]:
        """Получение списка поддерживаемых тем"""
        return list(self.knowledge_base.keys())
    
    def get_statistics(self) -> Dict:
        """Получение статистики обработчика"""
        return {
            'supported_topics': len(self.knowledge_base),
            'topics': list(self.knowledge_base.keys()),
            'supported_languages': ['ru', 'en']
        }

# Глобальный экземпляр
_programming_handler_instance = None

def get_programming_handler() -> ProgrammingKnowledgeHandler:
    """Получение глобального экземпляра обработчика"""
    global _programming_handler_instance
    if _programming_handler_instance is None:
        _programming_handler_instance = ProgrammingKnowledgeHandler()
    return _programming_handler_instance

# Тестирование
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    handler = ProgrammingKnowledgeHandler()
    
    test_questions = [
        "Сравни C++ и Python для задач промышленной автоматизации",
        "Compare C++ and Python for automation",
        "Как написать алгоритм управления конвейером?",
        "What are the best practices for industrial programming?",
        "Промышленное программирование PLC",
        "Какой язык программирования выбрать?"
    ]
    
    print("=== ТЕСТИРОВАНИЕ ОБРАБОТЧИКА ПРОГРАММИРОВАНИЯ ===")
    for question in test_questions:
        print(f"\nВопрос: {question}")
        result = handler.handle_request(question)
        print(f"Тема: {result.get('topic', 'N/A')}")
        print(f"Язык: {result.get('language', 'N/A')}")
        print(f"Успех: {result.get('success', False)}")
        print("Ответ:", result['response'][:200] + "..." if len(result['response']) > 200 else result['response'])
    
    print(f"\nСтатистика: {handler.get_statistics()}")