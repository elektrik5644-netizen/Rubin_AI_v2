#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ всех баз знаний Smart Rubin AI
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
        return
    
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
            return
        
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
            if count > 0 and count <= 5:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                examples = cursor.fetchall()
                print(f"      Примеры данных:")
                for i, example in enumerate(examples, 1):
                    print(f"         {i}. {example}")
            elif count > 5:
                # Показываем последние записи
                cursor.execute(f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT 2;")
                recent = cursor.fetchall()
                print(f"      Последние записи:")
                for i, record in enumerate(recent, 1):
                    print(f"         {i}. {record}")
        
        print(f"\n📊 ИТОГО записей в базе: {total_records:,}")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"❌ Ошибка базы данных: {e}")
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")

def main():
    """Основная функция"""
    print("🧠 ПОЛНЫЙ АНАЛИЗ БАЗ ЗНАНИЙ SMART RUBIN AI")
    print("=" * 80)
    print(f"📅 Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Список баз данных для анализа
    databases = [
        "rubin_ai.db",           # Текущая база сообщений
        "rubin_knowledge.db",    # Основная база знаний
        "rubin_knowledge_base.db", # База знаний (альтернативная)
        "rubin_documents.db",    # База документов
        "rubin_learning.db"      # База обучения
    ]
    
    total_size = 0
    total_records = 0
    
    for db_name in databases:
        if os.path.exists(db_name):
            analyze_database(db_name)
            total_size += os.path.getsize(db_name)
        else:
            print(f"\n❌ База данных не найдена: {db_name}")
    
    print(f"\n🎯 ИТОГОВАЯ СТАТИСТИКА:")
    print("=" * 60)
    print(f"📊 Общий размер всех баз: {total_size:,} байт ({total_size/1024/1024:.2f} МБ)")
    print(f"📁 Проанализировано баз: {len([db for db in databases if os.path.exists(db)])}")
    
    print(f"\n✅ Анализ завершен!")

if __name__ == "__main__":
    main()
