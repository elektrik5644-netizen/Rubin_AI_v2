#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка содержимого базы данных
"""

import sqlite3
import json

def check_database():
    """Проверка содержимого базы данных"""
    
    print("=== ПРОВЕРКА БАЗЫ ДАННЫХ RUBIN AI ===\n")
    
    try:
        # Подключение к базе данных
        conn = sqlite3.connect("rubin_ai_v2.db")
        cursor = conn.cursor()
        
        # Проверка таблиц
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📋 Найденные таблицы: {[table[0] for table in tables]}")
        
        # Проверка документов
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        print(f"📄 Общее количество документов: {doc_count}")
        
        if doc_count > 0:
            # Статистика по категориям
            cursor.execute("""
                SELECT category, COUNT(*) as count 
                FROM documents 
                GROUP BY category 
                ORDER BY count DESC
            """)
            categories = cursor.fetchall()
            
            print(f"\n📊 Статистика по категориям:")
            for category, count in categories:
                print(f"   {category}: {count} документов")
            
            # Последние загруженные документы
            cursor.execute("""
                SELECT file_name, category, created_at, LENGTH(content) as content_length
                FROM documents 
                ORDER BY created_at DESC 
                LIMIT 10
            """)
            recent_docs = cursor.fetchall()
            
            print(f"\n📝 Последние загруженные документы:")
            for doc in recent_docs:
                file_name, category, created_at, content_length = doc
                print(f"   📄 {file_name} ({category}) - {content_length} символов - {created_at}")
            
            # Поиск документов по ключевым словам
            keywords = ["ПИД", "электротехника", "Python", "антенна", "контроллер"]
            
            print(f"\n🔍 Поиск по ключевым словам:")
            for keyword in keywords:
                cursor.execute("""
                    SELECT file_name, category 
                    FROM documents 
                    WHERE content LIKE ? OR file_name LIKE ?
                    LIMIT 5
                """, (f'%{keyword}%', f'%{keyword}%'))
                
                results = cursor.fetchall()
                print(f"   '{keyword}': {len(results)} документов")
                for result in results:
                    print(f"      - {result[0]} ({result[1]})")
        
        # Проверка векторного поиска
        try:
            cursor.execute("SELECT COUNT(*) FROM document_vectors")
            vector_count = cursor.fetchone()[0]
            print(f"\n🧠 Векторных записей: {vector_count}")
        except:
            print(f"\n🧠 Таблица векторов не найдена")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"❌ Ошибка базы данных: {e}")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")

def search_specific_content():
    """Поиск конкретного содержимого"""
    
    print(f"\n=== ПОИСК КОНКРЕТНОГО СОДЕРЖИМОГО ===")
    
    try:
        conn = sqlite3.connect("rubin_ai_v2.db")
        cursor = conn.cursor()
        
        # Поиск документов с ПИД-регуляторами
        cursor.execute("""
            SELECT file_name, category, SUBSTR(content, 1, 200) as preview
            FROM documents 
            WHERE content LIKE '%ПИД%' OR content LIKE '%PID%'
            LIMIT 3
        """)
        
        pid_docs = cursor.fetchall()
        print(f"\n🎯 Документы с ПИД-регуляторами:")
        for doc in pid_docs:
            file_name, category, preview = doc
            print(f"   📄 {file_name} ({category})")
            print(f"      Превью: {preview}...")
            print()
        
        # Поиск документов с Python
        cursor.execute("""
            SELECT file_name, category, SUBSTR(content, 1, 200) as preview
            FROM documents 
            WHERE content LIKE '%Python%' OR content LIKE '%python%'
            LIMIT 3
        """)
        
        python_docs = cursor.fetchall()
        print(f"🐍 Документы с Python:")
        for doc in python_docs:
            file_name, category, preview = doc
            print(f"   📄 {file_name} ({category})")
            print(f"      Превью: {preview}...")
            print()
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Ошибка поиска: {e}")

if __name__ == "__main__":
    check_database()
    search_specific_content()