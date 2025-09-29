#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Извлечение реальных вопросов из базы данных Rubin AI
"""

import sqlite3
import json
from datetime import datetime

def extract_real_questions():
    """Извлечение реальных вопросов из всех баз данных Rubin AI"""
    print("🔍 Извлечение реальных вопросов из базы данных Rubin AI")
    print("=" * 60)
    
    all_questions = []
    
    # Список баз данных для проверки
    databases = [
        'rubin_learning.db',
        'rubin_chatbot_unified.db', 
        'rubin_knowledge_base.db',
        'rubin_api_unified.db',
        'rubin_context.db',
        'rubin_simple_learning.db',
        'rubin_understanding.db'
    ]
    
    for db_name in databases:
        try:
            print(f"\n📊 Проверяем базу: {db_name}")
            conn = sqlite3.connect(db_name)
            cursor = conn.cursor()
            
            # Получаем список таблиц
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            for table_name, in tables:
                try:
                    # Пытаемся найти столбцы с вопросами/сообщениями
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    column_names = [col[1] for col in columns]
                    
                    # Ищем столбцы, которые могут содержать вопросы
                    question_columns = []
                    for col in column_names:
                        if any(keyword in col.lower() for keyword in ['message', 'question', 'query', 'text', 'content', 'input']):
                            question_columns.append(col)
                    
                    if question_columns:
                        print(f"  📋 Таблица {table_name}: найдены столбцы {question_columns}")
                        
                        # Извлекаем данные
                        for col in question_columns:
                            try:
                                cursor.execute(f"SELECT DISTINCT {col} FROM {table_name} WHERE {col} IS NOT NULL AND {col} != '' LIMIT 20")
                                questions = cursor.fetchall()
                                
                                for question, in questions:
                                    if isinstance(question, str) and len(question.strip()) > 10:
                                        all_questions.append({
                                            'question': question.strip(),
                                            'source_db': db_name,
                                            'source_table': table_name,
                                            'source_column': col
                                        })
                            except Exception as e:
                                print(f"    ❌ Ошибка чтения столбца {col}: {e}")
                
                except Exception as e:
                    print(f"  ❌ Ошибка обработки таблицы {table_name}: {e}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Ошибка подключения к {db_name}: {e}")
    
    return all_questions

def analyze_questions(questions):
    """Анализ извлеченных вопросов"""
    print(f"\n📊 Анализ извлеченных вопросов")
    print("=" * 40)
    
    print(f"📈 Всего найдено вопросов: {len(questions)}")
    
    # Группировка по источникам
    sources = {}
    for q in questions:
        source = q['source_db']
        if source not in sources:
            sources[source] = 0
        sources[source] += 1
    
    print(f"\n📋 Распределение по базам данных:")
    for source, count in sources.items():
        print(f"  • {source}: {count} вопросов")
    
    # Анализ длины вопросов
    lengths = [len(q['question']) for q in questions]
    if lengths:
        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)
        print(f"\n📏 Длина вопросов:")
        print(f"  • Средняя: {avg_length:.1f} символов")
        print(f"  • Минимальная: {min_length} символов")
        print(f"  • Максимальная: {max_length} символов")
    
    return questions

def select_best_questions(questions, count=10):
    """Выбор лучших вопросов для тестирования"""
    print(f"\n🎯 Выбор {count} лучших вопросов")
    print("=" * 35)
    
    # Фильтруем и сортируем вопросы
    filtered_questions = []
    
    for q in questions:
        question = q['question']
        
        # Фильтруем по длине и содержанию
        if (20 <= len(question) <= 200 and 
            not question.startswith('{') and  # не JSON
            not question.startswith('[') and  # не массив
            not question.isdigit() and        # не число
            '?' in question or               # содержит вопрос
            any(word in question.lower() for word in ['что', 'как', 'расскажи', 'объясни', 'помоги', 'сравни'])):  # вопросительные слова
            
            filtered_questions.append(q)
    
    # Сортируем по длине (предпочитаем средние по длине)
    filtered_questions.sort(key=lambda x: abs(len(x['question']) - 80))
    
    # Выбираем уникальные вопросы
    selected = []
    seen_questions = set()
    
    for q in filtered_questions:
        question_lower = q['question'].lower()
        if question_lower not in seen_questions:
            selected.append(q)
            seen_questions.add(question_lower)
            if len(selected) >= count:
                break
    
    print(f"✅ Выбрано {len(selected)} уникальных вопросов")
    
    return selected

def create_questions_file(questions):
    """Создание файла с реальными вопросами"""
    print(f"\n📄 Создание файла с реальными вопросами")
    print("=" * 45)
    
    try:
        with open('RUBIN_REAL_QUESTIONS.md', 'w', encoding='utf-8') as f:
            f.write("# 🎯 Реальные вопросы из базы данных Rubin AI\n\n")
            f.write(f"**Дата извлечения:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Всего вопросов:** {len(questions)}\n\n")
            
            f.write("## 📋 Список реальных вопросов\n\n")
            
            for i, q in enumerate(questions, 1):
                f.write(f"### {i}. {q['question']}\n")
                f.write(f"**Источник:** {q['source_db']} → {q['source_table']} → {q['source_column']}\n\n")
            
            f.write("## 📊 Статистика по источникам\n\n")
            sources = {}
            for q in questions:
                source = q['source_db']
                sources[source] = sources.get(source, 0) + 1
            
            for source, count in sources.items():
                f.write(f"- **{source}**: {count} вопросов\n")
            
            f.write(f"\n## 🎯 Использование\n\n")
            f.write("Эти вопросы извлечены из реальной базы данных Rubin AI и могут использоваться для:\n")
            f.write("- Тестирования системы\n")
            f.write("- Анализа качества ответов\n")
            f.write("- Обучения авто-тестировщика\n")
            f.write("- Проверки работы модулей\n")
        
        print("✅ Файл RUBIN_REAL_QUESTIONS.md создан")
        
    except Exception as e:
        print(f"❌ Ошибка создания файла: {e}")

def main():
    """Основная функция"""
    print("🚀 Извлечение реальных вопросов из базы данных Rubin AI")
    print("=" * 60)
    
    # Извлекаем вопросы
    all_questions = extract_real_questions()
    
    if not all_questions:
        print("❌ Не найдено вопросов в базе данных")
        return
    
    # Анализируем вопросы
    analyze_questions(all_questions)
    
    # Выбираем лучшие вопросы
    best_questions = select_best_questions(all_questions, 10)
    
    if not best_questions:
        print("❌ Не удалось выбрать подходящие вопросы")
        return
    
    # Создаем файл
    create_questions_file(best_questions)
    
    # Выводим результат
    print(f"\n🎉 Извлечение завершено!")
    print(f"📊 Найдено {len(all_questions)} вопросов")
    print(f"🎯 Выбрано {len(best_questions)} лучших вопросов")
    print(f"📄 Создан файл RUBIN_REAL_QUESTIONS.md")
    
    print(f"\n📋 Выбранные вопросы:")
    for i, q in enumerate(best_questions, 1):
        print(f"  {i}. {q['question'][:60]}...")

if __name__ == '__main__':
    main()























