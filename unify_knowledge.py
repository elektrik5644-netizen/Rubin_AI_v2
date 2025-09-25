#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Унификация баз знаний Rubin AI v2
Объединяет фрагментированные базы знаний в единую структуру
"""

import sqlite3
import os
import shutil
from datetime import datetime
import hashlib

def backup_db(db_path):
    """Создает резервную копию базы данных"""
    if os.path.exists(db_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{db_path}.backup_{timestamp}"
        shutil.copy2(db_path, backup_path)
        print(f"✅ Создана резервная копия: {backup_path}")
        return backup_path
    return None

def get_table_schema(cursor, table_name):
    """Получает схему таблицы"""
    cursor.execute(f"PRAGMA table_info({table_name});")
    return [col[1] for col in cursor.fetchall()]

def migrate_table(src_cursor, dest_cursor, src_table, dest_table, column_mapping=None):
    """Мигрирует данные из одной таблицы в другую"""
    src_cursor.execute(f"SELECT * FROM {src_table};")
    rows = src_cursor.fetchall()
    
    if not rows:
        print(f"⚠️ Таблица {src_table} пуста, пропускаем миграцию")
        return 0
    
    src_cols = [description[0] for description in src_cursor.description]
    dest_cols = get_table_schema(dest_cursor, dest_table)
    
    # Определяем какие колонки мигрировать
    insert_cols = []
    select_cols = []
    
    for dest_col in dest_cols:
        if column_mapping and dest_col in column_mapping:
            mapped_col = column_mapping[dest_col]
            if mapped_col in src_cols:
                insert_cols.append(dest_col)
                select_cols.append(mapped_col)
        elif dest_col in src_cols:
            insert_cols.append(dest_col)
            select_cols.append(dest_col)
    
    if not insert_cols:
        print(f"❌ Нет общих колонок для миграции из {src_table} в {dest_table}")
        return 0
    
    # Подготавливаем данные для вставки
    placeholders = ", ".join(["?" for _ in insert_cols])
    insert_statement = f"INSERT OR IGNORE INTO {dest_table} ({', '.join(insert_cols)}) VALUES ({placeholders});"
    
    # Получаем данные с правильными колонками
    select_statement = ", ".join(select_cols)
    src_cursor.execute(f"SELECT {select_statement} FROM {src_table};")
    data_rows = src_cursor.fetchall()
    
    try:
        dest_cursor.executemany(insert_statement, data_rows)
        print(f"✅ Мигрировано {len(data_rows)} записей из {src_table} в {dest_table}")
        return len(data_rows)
    except sqlite3.Error as e:
        print(f"❌ Ошибка при миграции из {src_table} в {dest_table}: {e}")
        return 0

def unify_knowledge_bases():
    """Унифицирует базы знаний"""
    print("🔄 УНИФИКАЦИЯ БАЗ ЗНАНИЙ RUBIN AI V2")
    print("=" * 60)
    
    # Список баз знаний для унификации
    knowledge_dbs = [
        'rubin_knowledge_base.db',
        'rubin_knowledge_base_enhanced.db', 
        'rubin_knowledge_base_unified.db'
    ]
    
    # Целевая унифицированная база
    unified_db = 'rubin_knowledge_unified.db'
    
    # Создаем резервные копии
    print("📦 Создание резервных копий...")
    backups = []
    for db in knowledge_dbs:
        if os.path.exists(db):
            backup = backup_db(db)
            if backup:
                backups.append(backup)
    
    # Создаем унифицированную базу знаний
    print(f"\n🏗️ Создание унифицированной базы: {unified_db}")
    conn_unified = sqlite3.connect(unified_db)
    cursor_unified = conn_unified.cursor()
    
    # Создаем унифицированную схему
    cursor_unified.execute("""
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT,
            tags TEXT,
            difficulty_level TEXT DEFAULT 'medium',
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    cursor_unified.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            parent_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    cursor_unified.execute("""
        CREATE TABLE IF NOT EXISTS search_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            knowledge_id INTEGER,
            term TEXT NOT NULL,
            frequency INTEGER DEFAULT 1,
            position INTEGER,
            FOREIGN KEY (knowledge_id) REFERENCES knowledge (id)
        );
    """)
    
    cursor_unified.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            usage_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    cursor_unified.execute("""
        CREATE TABLE IF NOT EXISTS statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            metric_value TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Создаем индексы
    cursor_unified.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_title ON knowledge (title);")
    cursor_unified.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge (category);")
    cursor_unified.execute("CREATE INDEX IF NOT EXISTS idx_search_index_term ON search_index (term);")
    cursor_unified.execute("CREATE INDEX IF NOT EXISTS idx_tags_name ON tags (name);")
    
    conn_unified.commit()
    
    total_migrated = 0
    
    # Мигрируем данные из каждой базы знаний
    for db_name in knowledge_dbs:
        if not os.path.exists(db_name):
            print(f"⚠️ База знаний не найдена: {db_name}")
            continue
            
        print(f"\n📊 Миграция из: {db_name}")
        conn_source = sqlite3.connect(db_name)
        cursor_source = conn_source.cursor()
        
        # Получаем список таблиц
        cursor_source.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor_source.fetchall()
        
        print(f"   Найдено таблиц: {len(tables)}")
        
        # Мигрируем каждую таблицу
        for table in tables:
            table_name = table[0]
            
            if table_name == 'knowledge' or table_name == 'knowledge_entries' or table_name == 'knowledge_base':
                # Мигрируем основные записи знаний
                migrated = migrate_table(
                    cursor_source, cursor_unified, 
                    table_name, 'knowledge',
                    {
                        'title': 'title' if 'title' in get_table_schema(cursor_source, table_name) else 'name',
                        'content': 'content' if 'content' in get_table_schema(cursor_source, table_name) else 'description',
                        'category': 'category',
                        'tags': 'tags',
                        'difficulty_level': 'difficulty_level',
                        'source': 'source'
                    }
                )
                total_migrated += migrated
                
            elif table_name == 'categories':
                # Мигрируем категории
                migrated = migrate_table(
                    cursor_source, cursor_unified,
                    table_name, 'categories',
                    {
                        'name': 'name',
                        'description': 'description',
                        'parent_id': 'parent_id'
                    }
                )
                total_migrated += migrated
                
            elif table_name == 'search_index' or table_name == 'search_queries':
                # Мигрируем поисковые индексы
                migrated = migrate_table(
                    cursor_source, cursor_unified,
                    table_name, 'search_index',
                    {
                        'knowledge_id': 'knowledge_id' if 'knowledge_id' in get_table_schema(cursor_source, table_name) else 'id',
                        'term': 'term' if 'term' in get_table_schema(cursor_source, table_name) else 'query',
                        'frequency': 'frequency',
                        'position': 'position'
                    }
                )
                total_migrated += migrated
                
            elif table_name == 'tags':
                # Мигрируем теги
                migrated = migrate_table(
                    cursor_source, cursor_unified,
                    table_name, 'tags',
                    {
                        'name': 'name',
                        'description': 'description',
                        'usage_count': 'usage_count'
                    }
                )
                total_migrated += migrated
                
            elif table_name == 'statistics':
                # Мигрируем статистику
                migrated = migrate_table(
                    cursor_source, cursor_unified,
                    table_name, 'statistics',
                    {
                        'metric_name': 'metric_name',
                        'metric_value': 'metric_value'
                    }
                )
                total_migrated += migrated
        
        conn_source.close()
    
    # Оптимизируем унифицированную базу
    print(f"\n⚡ Оптимизация унифицированной базы...")
    cursor_unified.execute("VACUUM;")
    cursor_unified.execute("ANALYZE;")
    
    # Создаем полнотекстовый поиск
    print(f"🔍 Создание полнотекстового поиска...")
    cursor_unified.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
            title, content, tags, category
        );
    """)
    
    # Заполняем FTS таблицу
    cursor_unified.execute("""
        INSERT INTO knowledge_fts (rowid, title, content, tags, category)
        SELECT id, title, content, tags, category FROM knowledge;
    """)
    
    conn_unified.commit()
    conn_unified.close()
    
    # Получаем размеры
    original_size = sum(os.path.getsize(db) for db in knowledge_dbs if os.path.exists(db))
    new_size = os.path.getsize(unified_db)
    
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ УНИФИКАЦИИ")
    print("=" * 60)
    print(f"✅ Обработано баз знаний: {len(knowledge_dbs)}")
    print(f"✅ Мигрировано записей: {total_migrated}")
    print(f"✅ Исходный размер: {original_size / 1024:.1f} KB")
    print(f"✅ Новый размер: {new_size / 1024:.1f} KB")
    print(f"✅ Экономия места: {(original_size - new_size) / 1024:.1f} KB")
    
    print(f"\n🎯 Унифицированная база создана: {unified_db}")
    print("📦 Резервные копии сохранены в текущей директории")
    
    return unified_db

if __name__ == "__main__":
    try:
        unified_db = unify_knowledge_bases()
        print("\n🎉 Унификация завершена успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка при унификации: {e}")
        print("📦 Проверьте резервные копии для восстановления")





