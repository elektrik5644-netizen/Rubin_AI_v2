#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Консолидация баз документов Rubin AI v2
Объединяет дублирующие базы документов для экономии места и ускорения поиска
"""

import sqlite3
import os
import shutil
from datetime import datetime
import hashlib

def backup_database(db_path):
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

def calculate_content_hash(content):
    """Вычисляет хэш содержимого для обнаружения дубликатов"""
    if content is None:
        content = ""
    return hashlib.md5(str(content).encode('utf-8')).hexdigest()

def consolidate_documents():
    """Консолидирует базы документов"""
    print("🔄 КОНСОЛИДАЦИЯ БАЗ ДОКУМЕНТОВ RUBIN AI V2")
    print("=" * 60)
    
    # Основные базы документов для консолидации
    source_databases = [
        'rubin_ai_documents.db',
        'rubin_ai_v2.db'
    ]
    
    # Целевая консолидированная база
    target_db = 'rubin_documents_consolidated.db'
    
    # Создаем резервные копии
    print("📦 Создание резервных копий...")
    backups = []
    for db in source_databases:
        if os.path.exists(db):
            backup = backup_database(db)
            if backup:
                backups.append(backup)
    
    # Создаем целевую базу данных
    print(f"\n🏗️ Создание консолидированной базы: {target_db}")
    conn_target = sqlite3.connect(target_db)
    cursor_target = conn_target.cursor()
    
    # Создаем основные таблицы
    cursor_target.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            content TEXT NOT NULL,
            file_path TEXT,
            file_size INTEGER,
            file_type TEXT,
            category TEXT,
            tags TEXT,
            difficulty_level TEXT DEFAULT 'medium',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            content_hash TEXT,
            source_db TEXT
        );
    """)
    
    cursor_target.execute("""
        CREATE TABLE IF NOT EXISTS document_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            term TEXT NOT NULL,
            frequency INTEGER DEFAULT 1,
            position INTEGER,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        );
    """)
    
    cursor_target.execute("""
        CREATE TABLE IF NOT EXISTS document_vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            vector_data TEXT,
            vector_dimension INTEGER,
            model_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        );
    """)
    
    cursor_target.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            parent_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Создаем индексы для ускорения поиска
    cursor_target.execute("CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents (content_hash);")
    cursor_target.execute("CREATE INDEX IF NOT EXISTS idx_documents_category ON documents (category);")
    cursor_target.execute("CREATE INDEX IF NOT EXISTS idx_documents_file_name ON documents (file_name);")
    cursor_target.execute("CREATE INDEX IF NOT EXISTS idx_document_index_term ON document_index (term);")
    cursor_target.execute("CREATE INDEX IF NOT EXISTS idx_document_index_doc_id ON document_index (document_id);")
    
    conn_target.commit()
    
    # Статистика консолидации
    total_documents = 0
    duplicates_removed = 0
    content_hashes = set()
    
    # Обрабатываем каждую исходную базу
    for db_name in source_databases:
        if not os.path.exists(db_name):
            print(f"⚠️ База данных не найдена: {db_name}")
            continue
            
        print(f"\n📊 Обработка базы: {db_name}")
        conn_source = sqlite3.connect(db_name)
        cursor_source = conn_source.cursor()
        
        # Получаем все документы
        cursor_source.execute("SELECT * FROM documents;")
        documents = cursor_source.fetchall()
        
        print(f"   Найдено документов: {len(documents)}")
        
        # Получаем схему таблицы documents
        doc_columns = get_table_schema(cursor_source, 'documents')
        print(f"   Колонки: {doc_columns}")
        
        for doc in documents:
            # Создаем словарь из кортежа документа
            doc_dict = dict(zip(doc_columns, doc))
            
            # Вычисляем хэш содержимого
            content_hash = calculate_content_hash(doc_dict.get('content', ''))
            
            # Проверяем на дубликат
            if content_hash in content_hashes:
                duplicates_removed += 1
                print(f"   🔄 Пропущен дубликат: {doc_dict.get('file_name', 'Unknown')}")
                continue
            
            content_hashes.add(content_hash)
            
            # Вставляем документ в целевую базу
            cursor_target.execute("""
                INSERT INTO documents (
                    file_name, content, file_path, file_size, file_type,
                    category, tags, difficulty_level, content_hash, source_db
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_dict.get('file_name', ''),
                doc_dict.get('content', ''),
                doc_dict.get('file_path', ''),
                doc_dict.get('file_size', 0),
                doc_dict.get('file_type', ''),
                doc_dict.get('category', ''),
                doc_dict.get('tags', ''),
                doc_dict.get('difficulty_level', 'medium'),
                content_hash,
                db_name
            ))
            
            total_documents += 1
        
        # Мигрируем индексы документов
        print(f"   📇 Миграция индексов...")
        cursor_source.execute("SELECT * FROM document_index;")
        indexes = cursor_source.fetchall()
        
        index_columns = get_table_schema(cursor_source, 'document_index')
        print(f"   Найдено индексов: {len(indexes)}")
        
        for idx in indexes:
            idx_dict = dict(zip(index_columns, idx))
            cursor_target.execute("""
                INSERT INTO document_index (document_id, term, frequency, position)
                VALUES (?, ?, ?, ?)
            """, (
                idx_dict.get('document_id', 0),
                idx_dict.get('term', ''),
                idx_dict.get('frequency', 1),
                idx_dict.get('position', 0)
            ))
        
        # Мигрируем векторы документов
        print(f"   🧮 Миграция векторов...")
        cursor_source.execute("SELECT * FROM document_vectors;")
        vectors = cursor_source.fetchall()
        
        vector_columns = get_table_schema(cursor_source, 'document_vectors')
        print(f"   Найдено векторов: {len(vectors)}")
        
        for vec in vectors:
            vec_dict = dict(zip(vector_columns, vec))
            cursor_target.execute("""
                INSERT INTO document_vectors (document_id, vector_data, vector_dimension, model_name)
                VALUES (?, ?, ?, ?)
            """, (
                vec_dict.get('document_id', 0),
                vec_dict.get('vector_data', ''),
                vec_dict.get('vector_dimension', 0),
                vec_dict.get('model_name', '')
            ))
        
        conn_source.close()
    
    # Оптимизируем целевую базу
    print(f"\n⚡ Оптимизация консолидированной базы...")
    cursor_target.execute("VACUUM;")
    cursor_target.execute("ANALYZE;")
    
    conn_target.commit()
    conn_target.close()
    
    # Получаем размеры
    original_size = sum(os.path.getsize(db) for db in source_databases if os.path.exists(db))
    new_size = os.path.getsize(target_db)
    
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ КОНСОЛИДАЦИИ")
    print("=" * 60)
    print(f"✅ Обработано баз данных: {len(source_databases)}")
    print(f"✅ Всего документов: {total_documents}")
    print(f"✅ Удалено дубликатов: {duplicates_removed}")
    print(f"✅ Исходный размер: {original_size / 1024 / 1024:.1f} MB")
    print(f"✅ Новый размер: {new_size / 1024 / 1024:.1f} MB")
    print(f"✅ Экономия места: {(original_size - new_size) / 1024 / 1024:.1f} MB")
    print(f"✅ Процент экономии: {((original_size - new_size) / original_size * 100):.1f}%")
    
    print(f"\n🎯 Консолидированная база создана: {target_db}")
    print("📦 Резервные копии сохранены в текущей директории")
    
    return target_db

def cleanup_old_databases():
    """Удаляет старые базы данных после успешной консолидации"""
    print("\n🗑️ ОЧИСТКА СТАРЫХ БАЗ ДАННЫХ")
    print("=" * 40)
    
    old_databases = [
        'rubin_ai_documents.db',
        'rubin_ai_v2.db'
    ]
    
    for db in old_databases:
        if os.path.exists(db):
            # Проверяем, что резервная копия существует
            backup_exists = any(f.startswith(db) and 'backup' in f for f in os.listdir('.'))
            if backup_exists:
                os.remove(db)
                print(f"✅ Удалена старая база: {db}")
            else:
                print(f"⚠️ Пропущена база без резервной копии: {db}")

if __name__ == "__main__":
    try:
        # Консолидируем документы
        consolidated_db = consolidate_documents()
        
        # Спрашиваем пользователя о удалении старых баз
        print("\n❓ Удалить старые базы данных? (y/n): ", end="")
        response = input().lower().strip()
        
        if response in ['y', 'yes', 'да', 'д']:
            cleanup_old_databases()
        else:
            print("ℹ️ Старые базы данных сохранены")
        
        print("\n🎉 Консолидация завершена успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка при консолидации: {e}")
        print("📦 Проверьте резервные копии для восстановления")





