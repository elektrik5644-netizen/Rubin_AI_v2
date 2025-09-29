#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор вопросов для Rubin AI на основе базы данных
"""

import sqlite3
import random
from datetime import datetime

def analyze_existing_questions():
    """Анализ существующих вопросов в базе данных"""
    print("🔍 Анализ существующих вопросов в базе данных Rubin AI")
    print("=" * 60)
    
    try:
        conn = sqlite3.connect('rubin_learning.db')
        cursor = conn.cursor()
        
        # Получаем существующие вопросы
        cursor.execute("SELECT message, intent FROM interactions WHERE message IS NOT NULL")
        existing_questions = cursor.fetchall()
        
        print(f"📊 Найдено {len(existing_questions)} существующих вопросов:")
        for i, (question, intent) in enumerate(existing_questions, 1):
            print(f"  {i}. [{intent}] {question}")
        
        conn.close()
        return existing_questions
        
    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")
        return []

def generate_new_questions():
    """Генерация новых вопросов на основе анализа"""
    print(f"\n🎯 Генерация новых вопросов для Rubin AI")
    print("=" * 50)
    
    # База знаний по темам Rubin AI
    question_templates = {
        'controllers': [
            "Как настроить ПИД-регулятор для точного позиционирования?",
            "Что такое энкодер и какие типы энкодеров бывают?",
            "Как работает серводвигатель с обратной связью?",
            "Объясни принцип работы PMAC контроллера",
            "Как программировать PLC для автоматизации производства?",
            "Что такое HMI интерфейс и как его настроить?",
            "Как работает система SCADA в промышленности?",
            "Объясни протокол Modbus RTU и Modbus TCP",
            "Как настроить асинхронный двигатель с частотным преобразователем?",
            "Что такое энкодер абсолютный и инкрементальный?"
        ],
        'programming': [
            "Сравни C++ и Python для задач промышленной автоматизации",
            "Как написать алгоритм для управления конвейером на Python?",
            "Объясни принципы объектно-ориентированного программирования",
            "Как создать REST API для промышленного оборудования?",
            "Что такое многопоточность в программировании?",
            "Как работать с базами данных в промышленных системах?",
            "Объясни паттерны проектирования в программировании",
            "Как отлаживать код в промышленных приложениях?",
            "Что такое асинхронное программирование?",
            "Как оптимизировать производительность кода?"
        ],
        'electrical': [
            "Расскажи про закон Ома для полной цепи",
            "Как рассчитать мощность в трехфазной системе?",
            "Что такое коэффициент мощности и как его улучшить?",
            "Объясни принцип работы трансформатора",
            "Как защитить электрические цепи от короткого замыкания?",
            "Что такое активная и реактивная мощность?",
            "Как рассчитать ток короткого замыкания?",
            "Объясни принцип работы электродвигателя",
            "Что такое гармоники в электрических сетях?",
            "Как выбрать кабель для промышленного оборудования?"
        ],
        'radiomechanics': [
            "Какие бывают типы антенн и в чем их разница?",
            "Как рассчитать длину волны для радиосигнала?",
            "Что такое модуляция и демодуляция сигнала?",
            "Объясни принцип работы радиолокационной станции",
            "Как настроить радиопередатчик для связи?",
            "Что такое полоса пропускания антенны?",
            "Как рассчитать дальность радиосвязи?",
            "Объясни принцип работы спутниковой связи",
            "Что такое шум в радиосистемах?",
            "Как выбрать частоту для радиосвязи?"
        ],
        'general': [
            "Что такое промышленная автоматизация?",
            "Как работает система управления технологическими процессами?",
            "Объясни принципы работы робототехники",
            "Что такое кибербезопасность в промышленности?",
            "Как работает система мониторинга оборудования?",
            "Что такое цифровая трансформация производства?",
            "Объясни принципы работы IoT в промышленности",
            "Что такое машинное обучение в автоматизации?",
            "Как работает система предиктивного обслуживания?",
            "Что такое цифровой двойник оборудования?"
        ]
    }
    
    # Генерируем 10 вопросов
    generated_questions = []
    categories = list(question_templates.keys())
    
    for i in range(10):
        category = random.choice(categories)
        question = random.choice(question_templates[category])
        generated_questions.append({
            'question': question,
            'category': category,
            'number': i + 1
        })
    
    return generated_questions

def save_questions_to_database(questions):
    """Сохранение вопросов в базу данных"""
    print(f"\n💾 Сохранение вопросов в базу данных")
    print("=" * 40)
    
    try:
        conn = sqlite3.connect('rubin_learning.db')
        cursor = conn.cursor()
        
        # Создаем таблицу для сгенерированных вопросов, если её нет
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generated_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                category TEXT NOT NULL,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                used BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Сохраняем вопросы
        for q in questions:
            cursor.execute('''
                INSERT INTO generated_questions (question, category)
                VALUES (?, ?)
            ''', (q['question'], q['category']))
        
        conn.commit()
        conn.close()
        
        print("✅ Вопросы успешно сохранены в базу данных")
        
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")

def display_questions(questions):
    """Отображение сгенерированных вопросов"""
    print(f"\n📝 10 новых вопросов для Rubin AI:")
    print("=" * 50)
    
    for q in questions:
        print(f"\n{q['number']}. [{q['category'].upper()}]")
        print(f"   {q['question']}")
    
    print(f"\n🎯 Категории вопросов:")
    categories = {}
    for q in questions:
        categories[q['category']] = categories.get(q['category'], 0) + 1
    
    for category, count in categories.items():
        print(f"   • {category}: {count} вопросов")

def create_question_file(questions):
    """Создание файла с вопросами"""
    print(f"\n📄 Создание файла с вопросами")
    print("=" * 35)
    
    try:
        with open('RUBIN_QUESTIONS_DATABASE.md', 'w', encoding='utf-8') as f:
            f.write("# 🎯 База вопросов Rubin AI\n\n")
            f.write(f"**Дата создания:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Всего вопросов:** {len(questions)}\n\n")
            
            f.write("## 📋 Список вопросов\n\n")
            
            for q in questions:
                f.write(f"### {q['number']}. [{q['category'].upper()}]\n")
                f.write(f"{q['question']}\n\n")
            
            f.write("## 📊 Статистика по категориям\n\n")
            categories = {}
            for q in questions:
                categories[q['category']] = categories.get(q['category'], 0) + 1
            
            for category, count in categories.items():
                f.write(f"- **{category}**: {count} вопросов\n")
            
            f.write(f"\n## 🎯 Использование\n\n")
            f.write("Эти вопросы можно использовать для:\n")
            f.write("- Тестирования системы Rubin AI\n")
            f.write("- Обучения авто-тестировщика\n")
            f.write("- Проверки качества ответов\n")
            f.write("- Анализа производительности модулей\n")
        
        print("✅ Файл RUBIN_QUESTIONS_DATABASE.md создан")
        
    except Exception as e:
        print(f"❌ Ошибка создания файла: {e}")

def main():
    """Основная функция"""
    print("🚀 Генератор вопросов для Rubin AI")
    print("=" * 40)
    
    # Анализируем существующие вопросы
    existing_questions = analyze_existing_questions()
    
    # Генерируем новые вопросы
    new_questions = generate_new_questions()
    
    # Отображаем вопросы
    display_questions(new_questions)
    
    # Сохраняем в базу данных
    save_questions_to_database(new_questions)
    
    # Создаем файл с вопросами
    create_question_file(new_questions)
    
    print(f"\n🎉 Генерация завершена!")
    print(f"📊 Создано {len(new_questions)} новых вопросов")
    print(f"💾 Вопросы сохранены в базу данных и файл")

if __name__ == '__main__':
    main()























