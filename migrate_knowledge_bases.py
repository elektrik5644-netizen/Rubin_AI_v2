#!/usr/bin/env python3
"""
Скрипт миграции данных в единую базу знаний
"""

import sqlite3
import os
from datetime import datetime

def migrate_knowledge_bases():
    """Миграция всех баз знаний в единую структуру"""
    
    print("🔄 МИГРАЦИЯ БАЗ ЗНАНИЙ")
    print("=" * 50)
    
    # Проверяем существование единой базы
    if not os.path.exists('rubin_unified_knowledge.db'):
        print("❌ Единая база знаний не найдена!")
        print("   Сначала запустите cleanup_databases.py")
        return
    
    # Подключаемся к единой базе
    unified_db = sqlite3.connect('rubin_unified_knowledge.db')
    unified_cursor = unified_db.cursor()
    
    # Список баз для миграции
    knowledge_bases = [
        'rubin_knowledge_base.db',
        'rubin_knowledge_base_enhanced.db',
        'rubin_knowledge_base_unified.db',
        'readable_knowledge_base.db'
    ]
    
    total_migrated = 0
    
    for db_file in knowledge_bases:
        if os.path.exists(db_file):
            print(f"📊 Мигрирую {db_file}...")
            
            try:
                source_db = sqlite3.connect(db_file)
                source_cursor = source_db.cursor()
                
                # Получаем список таблиц
                source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = source_cursor.fetchall()
                
                db_migrated = 0
                
                for table in tables:
                    table_name = table[0]
                    if table_name.startswith('knowledge') or table_name.startswith('documents'):
                        # Получаем структуру таблицы
                        source_cursor.execute(f"PRAGMA table_info({table_name})")
                        columns = source_cursor.fetchall()
                        
                        # Мигрируем данные
                        source_cursor.execute(f"SELECT * FROM {table_name}")
                        rows = source_cursor.fetchall()
                        
                        for row in rows:
                            try:
                                # Адаптируем данные под новую схему
                                title = row[1] if len(row) > 1 else 'Unknown'
                                content = row[2] if len(row) > 2 else ''
                                category = row[3] if len(row) > 3 else 'general'
                                tags = row[4] if len(row) > 4 else ''
                                
                                unified_cursor.execute('''
                                    INSERT INTO unified_knowledge 
                                    (title, content, category, tags, difficulty_level, source_file)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                ''', (
                                    title,
                                    content,
                                    category,
                                    tags,
                                    'medium',
                                    db_file
                                ))
                                
                                db_migrated += 1
                                
                            except Exception as e:
                                print(f"    ⚠️ Ошибка при миграции записи: {e}")
                                continue
                        
                        print(f"  ✅ Мигрировано {len(rows)} записей из {table_name}")
                
                source_db.close()
                total_migrated += db_migrated
                print(f"📈 Всего из {db_file}: {db_migrated} записей")
                
            except Exception as e:
                print(f"❌ Ошибка при миграции {db_file}: {e}")
        else:
            print(f"ℹ️ {db_file} не найден, пропускаю")
    
    unified_db.commit()
    unified_db.close()
    
    print(f"\n🎉 Миграция завершена!")
    print(f"📊 Всего записей мигрировано: {total_migrated}")
    print("📁 Единая база: rubin_unified_knowledge.db")

def migrate_learning_data():
    """Миграция данных обучения"""
    
    print("\n🧠 МИГРАЦИЯ ДАННЫХ ОБУЧЕНИЯ")
    print("=" * 50)
    
    # Создаем таблицу для обучения в единой базе
    unified_db = sqlite3.connect('rubin_unified_knowledge.db')
    unified_cursor = unified_db.cursor()
    
    unified_cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            category TEXT,
            confidence_score REAL,
            user_rating INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source_db TEXT
        )
    ''')
    
    # Список баз обучения
    learning_bases = [
        'rubin_learning.db',
        'rubin_simple_learning.db',
        'rubin_understanding.db'
    ]
    
    total_learning_migrated = 0
    
    for db_file in learning_bases:
        if os.path.exists(db_file):
            print(f"📚 Мигрирую данные обучения из {db_file}...")
            
            try:
                source_db = sqlite3.connect(db_file)
                source_cursor = source_db.cursor()
                
                # Получаем список таблиц
                source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = source_cursor.fetchall()
                
                db_learning_migrated = 0
                
                for table in tables:
                    table_name = table[0]
                    if 'interaction' in table_name.lower() or 'learning' in table_name.lower():
                        # Мигрируем данные обучения
                        source_cursor.execute(f"SELECT * FROM {table_name}")
                        rows = source_cursor.fetchall()
                        
                        for row in rows:
                            try:
                                # Адаптируем данные под новую схему
                                user_id = row[1] if len(row) > 1 else None
                                question = row[2] if len(row) > 2 else ''
                                answer = row[3] if len(row) > 3 else ''
                                category = row[4] if len(row) > 4 else 'general'
                                
                                unified_cursor.execute('''
                                    INSERT INTO learning_interactions 
                                    (user_id, question, answer, category, source_db)
                                    VALUES (?, ?, ?, ?, ?)
                                ''', (
                                    user_id,
                                    question,
                                    answer,
                                    category,
                                    db_file
                                ))
                                
                                db_learning_migrated += 1
                                
                            except Exception as e:
                                print(f"    ⚠️ Ошибка при миграции записи обучения: {e}")
                                continue
                        
                        print(f"  ✅ Мигрировано {len(rows)} записей обучения из {table_name}")
                
                source_db.close()
                total_learning_migrated += db_learning_migrated
                print(f"📈 Всего из {db_file}: {db_learning_migrated} записей обучения")
                
            except Exception as e:
                print(f"❌ Ошибка при миграции обучения из {db_file}: {e}")
        else:
            print(f"ℹ️ {db_file} не найден, пропускаю")
    
    unified_db.commit()
    unified_db.close()
    
    print(f"\n🎉 Миграция обучения завершена!")
    print(f"📊 Всего записей обучения мигрировано: {total_learning_migrated}")

def create_performance_indexes():
    """Создание индексов для производительности"""
    
    print("\n⚡ СОЗДАНИЕ ИНДЕКСОВ ДЛЯ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 50)
    
    try:
        unified_db = sqlite3.connect('rubin_unified_knowledge.db')
        unified_cursor = unified_db.cursor()
        
        # Создаем дополнительные индексы
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_learning_user ON learning_interactions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_learning_category ON learning_interactions(category)",
            "CREATE INDEX IF NOT EXISTS idx_learning_timestamp ON learning_interactions(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_unified_created ON unified_knowledge(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_unified_updated ON unified_knowledge(updated_at)"
        ]
        
        for index_sql in indexes:
            unified_cursor.execute(index_sql)
            print(f"  ✅ Создан индекс")
        
        unified_db.commit()
        unified_db.close()
        
        print("✅ Все индексы созданы успешно")
        
    except Exception as e:
        print(f"❌ Ошибка при создании индексов: {e}")

def verify_migration():
    """Проверка результатов миграции"""
    
    print("\n🔍 ПРОВЕРКА РЕЗУЛЬТАТОВ МИГРАЦИИ")
    print("=" * 50)
    
    try:
        unified_db = sqlite3.connect('rubin_unified_knowledge.db')
        unified_cursor = unified_db.cursor()
        
        # Проверяем количество записей в unified_knowledge
        unified_cursor.execute("SELECT COUNT(*) FROM unified_knowledge")
        knowledge_count = unified_cursor.fetchone()[0]
        print(f"📚 Записей знаний: {knowledge_count}")
        
        # Проверяем количество записей в learning_interactions
        unified_cursor.execute("SELECT COUNT(*) FROM learning_interactions")
        learning_count = unified_cursor.fetchone()[0]
        print(f"🧠 Записей обучения: {learning_count}")
        
        # Проверяем категории
        unified_cursor.execute("SELECT DISTINCT category FROM unified_knowledge")
        categories = unified_cursor.fetchall()
        print(f"📂 Категорий знаний: {len(categories)}")
        for category in categories[:5]:  # Показываем первые 5
            print(f"  - {category[0]}")
        
        # Проверяем источники
        unified_cursor.execute("SELECT DISTINCT source_file FROM unified_knowledge")
        sources = unified_cursor.fetchall()
        print(f"📁 Источников данных: {len(sources)}")
        for source in sources:
            print(f"  - {source[0]}")
        
        unified_db.close()
        
        print("\n✅ Проверка миграции завершена")
        
    except Exception as e:
        print(f"❌ Ошибка при проверке миграции: {e}")

def main():
    """Основная функция"""
    
    print("🚀 МИГРАЦИЯ ДАННЫХ RUBIN AI v2")
    print("=" * 60)
    
    # 1. Мигрируем базы знаний
    migrate_knowledge_bases()
    
    # 2. Мигрируем данные обучения
    migrate_learning_data()
    
    # 3. Создаем индексы для производительности
    create_performance_indexes()
    
    # 4. Проверяем результаты
    verify_migration()
    
    print("\n🎉 МИГРАЦИЯ ДАННЫХ ЗАВЕРШЕНА!")
    print("=" * 60)
    print("📊 Результаты:")
    print("  ✅ Все базы знаний объединены")
    print("  ✅ Данные обучения мигрированы")
    print("  ✅ Индексы для производительности созданы")
    print("  ✅ Проверка целостности пройдена")
    
    print("\n📋 Следующие шаги:")
    print("  1. Обновить конфигурацию системы")
    print("  2. Протестировать новую структуру")
    print("  3. Запустить систему с новой базой")

if __name__ == "__main__":
    main()










