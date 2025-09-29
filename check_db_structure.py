#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка структуры базы данных
"""

import sqlite3

def check_database_structure():
    """Проверка структуры базы данных"""
    print("🔍 Проверка структуры базы данных...")
    
    try:
        conn = sqlite3.connect("rubin_ai_v2.db")
        cursor = conn.cursor()
        
        # Получаем список таблиц
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📋 Найденные таблицы: {[table[0] for table in tables]}")
        
        # Проверяем структуру таблицы documents
        if any('documents' in table for table in tables):
            cursor.execute("PRAGMA table_info(documents);")
            columns = cursor.fetchall()
            print("\n📄 Структура таблицы documents:")
            for col in columns:
                print(f"  - {col[1]} ({col[2]})")
        
        # Проверяем структуру таблицы synonyms
        if any('synonyms' in table for table in tables):
            cursor.execute("PRAGMA table_info(synonyms);")
            columns = cursor.fetchall()
            print("\n🔍 Структура таблицы synonyms:")
            for col in columns:
                print(f"  - {col[1]} ({col[2]})")
        
        # Проверяем количество записей
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        print(f"\n📊 Количество документов: {doc_count}")
        
        cursor.execute("SELECT COUNT(*) FROM synonyms")
        syn_count = cursor.fetchone()[0]
        print(f"📊 Количество синонимов: {syn_count}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    check_database_structure()






















