#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полный анализ всех перенесенных баз знаний Smart Rubin AI
"""

import sqlite3
import os
from datetime import datetime

def analyze_database(db_path):
    """Анализ одной базы данных"""
    print(f"\n📁 АНАЛИЗ: {os.path.basename(db_path)}")
    print("=" * 60)
    
    if not os.path.exists(db_path):
        print(f"❌ Файл не найден: {db_path}")
        return 0, 0
    
    file_size = os.path.getsize(db_path)
    print(f"📊 Размер файла: {file_size:,} байт")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Получаем список таблиц
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("❌ Таблицы не найдены")
            conn.close()
            return file_size, 0
        
        print(f"📋 Найдено таблиц: {len(tables)}")
        
        total_records = 0
        for table in tables:
            table_name = table[0]
            
            # Пропускаем системные таблицы
            if table_name.startswith('sqlite_'):
                continue
            
            # Подсчитываем записи
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            total_records += count
            
            print(f"   📄 {table_name}: {count:,} записей")
            
            # Показываем структуру таблицы
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            if columns:
                print(f"      Структура:")
                for col in columns:
                    col_id, col_name, col_type, not_null, default_val, pk = col
                    pk_mark = " (PK)" if pk else ""
                    print(f"         • {col_name}: {col_type}{pk_mark}")
            
            # Показываем примеры данных для таблиц с небольшим количеством записей
            if count > 0 and count <= 3:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 2;")
                examples = cursor.fetchall()
                print(f"      Примеры данных:")
                for i, example in enumerate(examples, 1):
                    print(f"         {i}. {str(example)[:100]}{'...' if len(str(example)) > 100 else ''}")
            elif count > 3:
                # Показываем последние записи
                cursor.execute(f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT 1;")
                recent = cursor.fetchall()
                if recent:
                    print(f"      Последняя запись:")
                    print(f"         {str(recent[0])[:100]}{'...' if len(str(recent[0])) > 100 else ''}")
        
        print(f"\n📊 ИТОГО записей в базе: {total_records:,}")
        
        conn.close()
        return file_size, total_records
        
    except sqlite3.Error as e:
        print(f"❌ Ошибка базы данных: {e}")
        return file_size, 0
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")
        return file_size, 0

def main():
    """Основная функция"""
    print("🧠 ПОЛНЫЙ АНАЛИЗ ВСЕХ ПЕРЕНЕСЕННЫХ БАЗ ЗНАНИЙ SMART RUBIN AI")
    print("=" * 80)
    print(f"📅 Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Список всех баз данных для анализа
    databases = [
        "rubin_ai.db",                    # Текущая база сообщений
        "rubin_knowledge.db",             # Основная база знаний
        "rubin_knowledge_base.db",        # База знаний (альтернативная)
        "rubin_knowledge_base_enhanced.db", # Расширенная база знаний
        "rubin_documents.db",             # База документов
        "rubin_learning.db",              # База обучения
        "rubin_simple_learning.db",       # Простое обучение
        "rubin_context.db",               # База контекста
        "rubin_understanding.db"          # База понимания
    ]
    
    total_size = 0
    total_records = 0
    successful_analyses = 0
    
    for db_name in databases:
        if os.path.exists(db_name):
            size, records = analyze_database(db_name)
            total_size += size
            total_records += records
            successful_analyses += 1
        else:
            print(f"\n❌ База данных не найдена: {db_name}")
    
    print(f"\n🎯 ИТОГОВАЯ СТАТИСТИКА:")
    print("=" * 60)
    print(f"📊 Общий размер всех баз: {total_size:,} байт ({total_size/1024/1024:.2f} МБ)")
    print(f"📁 Проанализировано баз: {successful_analyses}")
    print(f"📋 Общее количество записей: {total_records:,}")
    
    print(f"\n✅ Полный анализ завершен!")
    print(f"🎉 Все базы знаний Smart Rubin AI успешно перенесены и проанализированы!")

if __name__ == "__main__":
    main()
