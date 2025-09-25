#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка базы знаний Smart Rubin AI
"""

import sqlite3
import os
from datetime import datetime

def check_knowledge_base():
    """Проверка состояния базы знаний"""
    print("🧠 ПРОВЕРКА БАЗЫ ЗНАНИЙ SMART RUBIN AI")
    print("=" * 60)
    
    db_path = "rubin_ai.db"
    
    # Проверяем существование файла БД
    if os.path.exists(db_path):
        print(f"✅ База данных найдена: {db_path}")
        file_size = os.path.getsize(db_path)
        print(f"📊 Размер файла: {file_size} байт")
    else:
        print(f"❌ База данных не найдена: {db_path}")
        return
    
    try:
        # Подключаемся к БД
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Проверяем структуру таблиц
        print(f"\n📋 СТРУКТУРА БАЗЫ ДАННЫХ:")
        print("-" * 40)
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if tables:
            for table in tables:
                table_name = table[0]
                print(f"📁 Таблица: {table_name}")
                
                # Получаем структуру таблицы
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                for col in columns:
                    col_id, col_name, col_type, not_null, default_val, pk = col
                    pk_mark = " (PRIMARY KEY)" if pk else ""
                    not_null_mark = " NOT NULL" if not_null else ""
                    default_mark = f" DEFAULT {default_val}" if default_val else ""
                    print(f"   📄 {col_name}: {col_type}{not_null_mark}{default_mark}{pk_mark}")
                
                # Подсчитываем количество записей
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                count = cursor.fetchone()[0]
                print(f"   📊 Записей: {count}")
                
                if count > 0:
                    # Показываем последние записи
                    cursor.execute(f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT 3;")
                    recent_records = cursor.fetchall()
                    
                    print(f"   🔍 Последние записи:")
                    for i, record in enumerate(recent_records, 1):
                        print(f"      {i}. {record}")
                
                print()
        else:
            print("❌ Таблицы не найдены")
        
        # Проверяем конкретно таблицу messages
        print(f"💬 АНАЛИЗ СООБЩЕНИЙ:")
        print("-" * 40)
        
        try:
            cursor.execute("SELECT COUNT(*) FROM messages;")
            total_messages = cursor.fetchone()[0]
            print(f"📊 Всего сообщений: {total_messages}")
            
            if total_messages > 0:
                # Статистика по времени
                cursor.execute("""
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as count
                    FROM messages 
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                    LIMIT 7;
                """)
                daily_stats = cursor.fetchall()
                
                print(f"📅 Статистика по дням (последние 7 дней):")
                for date, count in daily_stats:
                    print(f"   {date}: {count} сообщений")
                
                # Последние сообщения
                cursor.execute("""
                    SELECT message, response, timestamp 
                    FROM messages 
                    ORDER BY timestamp DESC 
                    LIMIT 5;
                """)
                recent_messages = cursor.fetchall()
                
                print(f"\n🔍 Последние 5 сообщений:")
                for i, (message, response, timestamp) in enumerate(recent_messages, 1):
                    print(f"   {i}. [{timestamp}]")
                    print(f"      Вопрос: {message[:100]}{'...' if len(message) > 100 else ''}")
                    print(f"      Ответ: {response[:100]}{'...' if len(response) > 100 else ''}")
                    print()
            
        except sqlite3.OperationalError as e:
            print(f"❌ Ошибка при анализе сообщений: {e}")
        
        conn.close()
        print("✅ Подключение к базе данных закрыто")
        
    except sqlite3.Error as e:
        print(f"❌ Ошибка базы данных: {e}")
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")

def check_server_status():
    """Проверка статуса сервера"""
    print(f"\n🌐 СТАТУС СЕРВЕРА:")
    print("-" * 40)
    
    import subprocess
    try:
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
        if ':8083' in result.stdout:
            print("✅ Smart Rubin AI сервер запущен на порту 8083")
            # Извлекаем PID
            lines = result.stdout.split('\n')
            for line in lines:
                if ':8083' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        print(f"   PID: {pid}")
        else:
            print("❌ Smart Rubin AI сервер не запущен")
    except Exception as e:
        print(f"❌ Ошибка проверки сервера: {e}")

if __name__ == "__main__":
    check_knowledge_base()
    check_server_status()
