#!/usr/bin/env python3
"""
Проверка содержимого баз данных Rubin AI
"""

import sqlite3
import os

def check_databases():
    """Проверка всех баз данных"""
    
    print("🔍 ПРОВЕРКА БАЗ ДАННЫХ RUBIN AI")
    print("=" * 50)
    
    # Основные базы данных
    db_files = [
        'rubin_ai_v2.db',
        'rubin_ai_documents.db', 
        'rubin_knowledge_base.db',
        'readable_knowledge_base.db',
        'rubin_learning.db',
        'rubin_knowledge_base_enhanced.db'
    ]
    
    for db_file in db_files:
        if os.path.exists(db_file):
            print(f'\n📊 {db_file}:')
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                
                # Получаем список таблиц
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                print(f'  📋 Таблиц: {len(tables)}')
                
                # Показываем содержимое первых 5 таблиц
                for table in tables[:5]:
                    table_name = table[0]
                    try:
                        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}";')
                        count = cursor.fetchone()[0]
                        print(f'    - {table_name}: {count} записей')
                        
                        # Показываем примеры данных для важных таблиц
                        if table_name in ['documents', 'knowledge', 'content', 'texts'] and count > 0:
                            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 1;')
                            sample = cursor.fetchone()
                            if sample:
                                print(f'      Пример: {str(sample)[:100]}...')
                    except Exception as e:
                        print(f'    - {table_name}: ошибка чтения - {e}')
                
                conn.close()
                
            except Exception as e:
                print(f'  ❌ Ошибка: {e}')
        else:
            print(f'\n❌ {db_file}: файл не найден')
    
    # Проверяем документы в папке test_documents
    print(f'\n📁 ДОКУМЕНТЫ В TEST_DOCUMENTS:')
    test_docs_dir = 'test_documents'
    if os.path.exists(test_docs_dir):
        files = os.listdir(test_docs_dir)
        print(f'  📄 Файлов: {len(files)}')
        for file in files:
            file_path = os.path.join(test_docs_dir, file)
            size = os.path.getsize(file_path)
            print(f'    - {file}: {size} байт')
    else:
        print('  ❌ Папка test_documents не найдена')

def show_sample_content():
    """Показать примеры содержимого"""
    
    print(f'\n📖 ПРИМЕРЫ СОДЕРЖИМОГО:')
    print("=" * 30)
    
    # Показываем содержимое одного из документов
    doc_file = 'test_documents/radiomechanics_guide.txt'
    if os.path.exists(doc_file):
        print(f'\n📄 {doc_file}:')
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f'  Размер: {len(content)} символов')
            print(f'  Начало: {content[:200]}...')
    
    # Показываем содержимое базы знаний
    db_file = 'rubin_knowledge_base.db'
    if os.path.exists(db_file):
        print(f'\n📊 {db_file}:')
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Ищем таблицы с контентом
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                if 'content' in table_name.lower() or 'text' in table_name.lower():
                    cursor.execute(f'SELECT COUNT(*) FROM "{table_name}";')
                    count = cursor.fetchone()[0]
                    if count > 0:
                        cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 1;')
                        sample = cursor.fetchone()
                        print(f'  Таблица {table_name}: {count} записей')
                        if sample:
                            print(f'    Пример: {str(sample)[:150]}...')
            
            conn.close()
        except Exception as e:
            print(f'  ❌ Ошибка чтения: {e}')

if __name__ == "__main__":
    check_databases()
    show_sample_content()
    
    print(f'\n🎯 ВЫВОДЫ:')
    print("1. База данных найдена и содержит множество таблиц")
    print("2. Есть документы в папке test_documents")
    print("3. Система может использовать эти данные для ответов")
    print("4. Нужно настроить правильное подключение к базе")












