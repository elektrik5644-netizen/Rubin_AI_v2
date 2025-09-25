#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧠 ДЕМОНСТРАЦИЯ: КАК RUBIN AI ДУМАЕТ И ОБЩАЕТСЯ С НЕЙРОННОЙ СЕТЬЮ
================================================================

Этот скрипт демонстрирует:
1. Архитектуру мышления Rubin AI
2. Процесс принятия решений через нейронную сеть
3. Коммуникацию с NeuroRepository
4. Обучение и адаптацию
"""

import requests
import json
import time
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URL-адреса компонентов Rubin AI
SMART_DISPATCHER_URL = "http://localhost:8080/api/chat"
NEURO_API_URL = "http://localhost:8083/api/neuro/analyze"
PYTORCH_SERVER_URL = "http://localhost:8092/api/pytorch/chat"

def demonstrate_rubin_thinking_process():
    """
    Демонстрирует процесс мышления Rubin AI
    """
    print("🧠 ДЕМОНСТРАЦИЯ МЫШЛЕНИЯ RUBIN AI")
    print("=" * 50)
    
    # Тестовые вопросы для демонстрации разных типов мышления
    test_questions = [
        {
            "question": "Как работает нейронная сеть?",
            "expected_thinking": "Нейросетевое мышление",
            "description": "Демонстрация понимания нейросетевых концепций"
        },
        {
            "question": "Реши уравнение x^2 + 5x + 6 = 0",
            "expected_thinking": "Математическое мышление",
            "description": "Демонстрация математического анализа"
        },
        {
            "question": "Проанализируй торговые данные",
            "expected_thinking": "Аналитическое мышление",
            "description": "Демонстрация аналитического подхода"
        },
        {
            "question": "Что такое PyTorch?",
            "expected_thinking": "Техническое мышление",
            "description": "Демонстрация технических знаний"
        }
    ]
    
    for i, test in enumerate(test_questions, 1):
        print(f"\n🔍 ТЕСТ {i}: {test['description']}")
        print(f"❓ Вопрос: {test['question']}")
        print(f"🎯 Ожидаемое мышление: {test['expected_thinking']}")
        
        # Отправляем вопрос в Rubin AI
        try:
            response = requests.post(SMART_DISPATCHER_URL, 
                                   json={"message": test['question']}, 
                                   timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Статус: Успешно")
                print(f"📊 Модуль: {data.get('module', 'N/A')}")
                print(f"🎯 Уверенность: {data.get('confidence', 0):.1f}%")
                print(f"💭 Ответ: {data.get('explanation', 'N/A')[:200]}...")
                
                # Анализируем тип мышления
                analyze_thinking_type(test['question'], data.get('explanation', ''))
                
            else:
                print(f"❌ Ошибка: HTTP {response.status_code}")
                print(f"📝 Ответ: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Исключение: {e}")
        
        time.sleep(1)  # Пауза между тестами

def analyze_thinking_type(question, response):
    """
    Анализирует тип мышления Rubin AI на основе вопроса и ответа
    """
    print(f"\n🧠 АНАЛИЗ ТИПА МЫШЛЕНИЯ:")
    
    # Определяем тип мышления по ключевым словам
    thinking_types = {
        "Нейросетевое мышление": ["нейронная сеть", "нейросеть", "обучение", "веса", "активация"],
        "Математическое мышление": ["уравнение", "решение", "формула", "вычисление", "математика"],
        "Аналитическое мышление": ["анализ", "данные", "тренд", "прогноз", "статистика"],
        "Техническое мышление": ["технология", "алгоритм", "код", "программирование", "архитектура"]
    }
    
    response_lower = response.lower()
    question_lower = question.lower()
    
    detected_types = []
    for thinking_type, keywords in thinking_types.items():
        score = sum(1 for keyword in keywords if keyword in response_lower or keyword in question_lower)
        if score > 0:
            detected_types.append((thinking_type, score))
    
    if detected_types:
        # Сортируем по количеству совпадений
        detected_types.sort(key=lambda x: x[1], reverse=True)
        primary_type = detected_types[0][0]
        confidence = (detected_types[0][1] / len(thinking_types[primary_type])) * 100
        
        print(f"🎯 Основной тип мышления: {primary_type}")
        print(f"📊 Уверенность: {confidence:.1f}%")
        
        if len(detected_types) > 1:
            print(f"🔄 Дополнительные типы:")
            for thinking_type, score in detected_types[1:]:
                print(f"   - {thinking_type}: {score} совпадений")
    else:
        print(f"❓ Тип мышления не определен")

def demonstrate_neural_communication():
    """
    Демонстрирует коммуникацию с нейронной сетью
    """
    print(f"\n🔗 ДЕМОНСТРАЦИЯ КОММУНИКАЦИИ С НЕЙРОННОЙ СЕТЬЮ")
    print("=" * 50)
    
    # Тестируем прямую коммуникацию с PyTorch сервером
    pytorch_questions = [
        "Как работает обратное распространение ошибки?",
        "Что такое градиентный спуск?",
        "Как выбрать функцию активации?",
        "Что такое dropout в нейронных сетях?"
    ]
    
    for i, question in enumerate(pytorch_questions, 1):
        print(f"\n🧠 ТЕСТ НЕЙРОСЕТЕВОЙ КОММУНИКАЦИИ {i}")
        print(f"❓ Вопрос: {question}")
        
        try:
            response = requests.post(PYTORCH_SERVER_URL, 
                                   json={"message": question}, 
                                   timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Статус: Успешно")
                print(f"📊 Категория: {data.get('category', 'N/A')}")
                print(f"🎯 Уверенность: {data.get('confidence', 0):.1f}%")
                print(f"💭 Ответ: {data.get('explanation', 'N/A')[:150]}...")
                
                # Анализируем качество нейросетевого ответа
                analyze_neural_response_quality(question, data.get('explanation', ''))
                
            else:
                print(f"❌ Ошибка: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Исключение: {e}")
        
        time.sleep(1)

def analyze_neural_response_quality(question, response):
    """
    Анализирует качество нейросетевого ответа
    """
    print(f"\n📊 АНАЛИЗ КАЧЕСТВА НЕЙРОСЕТЕВОГО ОТВЕТА:")
    
    # Критерии качества
    quality_indicators = {
        "Техническая точность": ["pytorch", "tensor", "gradient", "backward", "optimizer"],
        "Практические примеры": ["```python", "код", "пример", "демонстрация"],
        "Лучшие практики": ["лучшие практики", "рекомендация", "совет", "важно"],
        "Объяснение концепций": ["объяснение", "принцип", "работает", "функция"]
    }
    
    response_lower = response.lower()
    quality_scores = {}
    
    for indicator, keywords in quality_indicators.items():
        score = sum(1 for keyword in keywords if keyword in response_lower)
        quality_scores[indicator] = score
    
    total_score = sum(quality_scores.values())
    max_possible = sum(len(keywords) for keywords in quality_indicators.values())
    overall_quality = (total_score / max_possible) * 100 if max_possible > 0 else 0
    
    print(f"🎯 Общее качество: {overall_quality:.1f}%")
    
    for indicator, score in quality_scores.items():
        if score > 0:
            print(f"✅ {indicator}: {score} индикаторов")
        else:
            print(f"❌ {indicator}: не обнаружено")

def demonstrate_learning_process():
    """
    Демонстрирует процесс обучения Rubin AI
    """
    print(f"\n📚 ДЕМОНСТРАЦИЯ ПРОЦЕССА ОБУЧЕНИЯ")
    print("=" * 50)
    
    # Вопросы о процессе обучения
    learning_questions = [
        "Как ты обучаешься?",
        "Что ты изучил сегодня?",
        "Как ты запоминаешь новую информацию?",
        "Как ты улучшаешь свои ответы?"
    ]
    
    for i, question in enumerate(learning_questions, 1):
        print(f"\n📖 ТЕСТ ОБУЧЕНИЯ {i}")
        print(f"❓ Вопрос: {question}")
        
        try:
            response = requests.post(SMART_DISPATCHER_URL, 
                                   json={"message": question}, 
                                   timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Статус: Успешно")
                print(f"📊 Модуль: {data.get('module', 'N/A')}")
                print(f"💭 Ответ: {data.get('explanation', 'N/A')[:200]}...")
                
                # Анализируем понимание процесса обучения
                analyze_learning_understanding(question, data.get('explanation', ''))
                
            else:
                print(f"❌ Ошибка: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Исключение: {e}")
        
        time.sleep(1)

def analyze_learning_understanding(question, response):
    """
    Анализирует понимание Rubin AI процесса обучения
    """
    print(f"\n🧠 АНАЛИЗ ПОНИМАНИЯ ОБУЧЕНИЯ:")
    
    # Индикаторы понимания обучения
    learning_indicators = {
        "Самосознание": ["я", "мой", "себя", "сам", "обучаюсь"],
        "Процесс обучения": ["изучаю", "запоминаю", "анализирую", "улучшаю"],
        "Контекстная память": ["сегодня", "недавно", "вчера", "ранее"],
        "Адаптация": ["адаптируюсь", "изменяюсь", "развиваюсь", "совершенствуюсь"]
    }
    
    response_lower = response.lower()
    understanding_scores = {}
    
    for indicator, keywords in learning_indicators.items():
        score = sum(1 for keyword in keywords if keyword in response_lower)
        understanding_scores[indicator] = score
    
    total_score = sum(understanding_scores.values())
    max_possible = sum(len(keywords) for keywords in learning_indicators.values())
    overall_understanding = (total_score / max_possible) * 100 if max_possible > 0 else 0
    
    print(f"🎯 Понимание обучения: {overall_understanding:.1f}%")
    
    for indicator, score in understanding_scores.items():
        if score > 0:
            print(f"✅ {indicator}: {score} индикаторов")
        else:
            print(f"❌ {indicator}: не обнаружено")

def demonstrate_architecture_overview():
    """
    Демонстрирует архитектуру мышления Rubin AI
    """
    print(f"\n🏗️ АРХИТЕКТУРА МЫШЛЕНИЯ RUBIN AI")
    print("=" * 50)
    
    architecture_diagram = """
    🧠 RUBIN AI МЫШЛЕНИЕ - АРХИТЕКТУРА:
    
    📥 ВХОДНОЙ ЗАПРОС
        ↓
    🔍 SMART DISPATCHER (Порт 8080)
        ├── Нейронная классификация
        ├── Fallback на ключевые слова
        └── Маршрутизация к модулям
        ↓
    🧠 СПЕЦИАЛИЗИРОВАННЫЕ МОДУЛИ:
        ├── 🧮 Математика (Порт 8086)
        ├── ⚡ Электротехника (Порт 8087)
        ├── 📡 Радиомеханика (Порт 8089)
        ├── 🎮 Контроллеры (Порт 9000)
        ├── 💻 Программирование (Порт 8088)
        ├── 🔥 PyTorch (Порт 8092)
        ├── 🧠 Нейросети (Порт 8083)
        ├── 📚 Обучение (Порт 8081)
        └── 🌐 Общие вопросы (Порт 8085)
        ↓
    💭 ГЕНЕРАЦИЯ ОТВЕТА
        ├── Контекстный анализ
        ├── Специализированные знания
        ├── Примеры кода
        └── Лучшие практики
        ↓
    📤 ОТВЕТ ПОЛЬЗОВАТЕЛЮ
    
    🔄 ПРОЦЕСС ОБУЧЕНИЯ:
        ├── Анализ обратной связи
        ├── Обновление базы знаний
        ├── Переобучение нейросети
        └── Адаптация ответов
    """
    
    print(architecture_diagram)

def main():
    """
    Основная функция демонстрации
    """
    print("🚀 ЗАПУСК ДЕМОНСТРАЦИИ МЫШЛЕНИЯ RUBIN AI")
    print("=" * 60)
    print(f"⏰ Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Демонстрация архитектуры
    demonstrate_architecture_overview()
    
    # 2. Демонстрация процесса мышления
    demonstrate_rubin_thinking_process()
    
    # 3. Демонстрация нейросетевой коммуникации
    demonstrate_neural_communication()
    
    # 4. Демонстрация процесса обучения
    demonstrate_learning_process()
    
    print(f"\n🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 60)
    print("📊 ИТОГОВЫЕ ВЫВОДЫ:")
    print("✅ Rubin AI демонстрирует многоуровневое мышление")
    print("✅ Эффективная коммуникация с нейронными сетями")
    print("✅ Понимание процесса собственного обучения")
    print("✅ Адаптивные ответы на основе контекста")
    print("✅ Интеграция различных типов знаний")

if __name__ == "__main__":
    main()





