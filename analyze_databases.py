#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ баз данных Rubin AI v2 для выявления дублирования и оптимизации
"""

import sqlite3
import os
from datetime import datetime

def analyze_database(db_path):
    """Анализирует структуру базы данных"""
    if not os.path.exists(db_path):
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Получаем список таблиц
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        # Получаем размер базы данных
        file_size = os.path.getsize(db_path)
        
        # Анализируем каждую таблицу
        table_info = {}
        total_records = 0
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            total_records += count
            
            # Получаем схему таблицы
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            table_info[table_name] = {
                'records': count,
                'columns': [col[1] for col in columns]
            }
        
        conn.close()
        
        return {
            'file_size': file_size,
            'tables': table_info,
            'total_records': total_records,
            'last_modified': datetime.fromtimestamp(os.path.getmtime(db_path))
        }
        
    except Exception as e:
        print(f"Ошибка при анализе {db_path}: {e}")
        return None

def main():
    print("🔍 АНАЛИЗ БАЗ ДАННЫХ RUBIN AI V2")
    print("=" * 60)
    
    # Список всех баз данных
    db_files = [f for f in os.listdir('.') if f.endswith('.db')]
    
    analysis_results = {}
    
    for db_file in sorted(db_files):
        print(f"\n📊 Анализирую: {db_file}")
        result = analyze_database(db_file)
        
        if result:
            analysis_results[db_file] = result
            print(f"   Размер: {result['file_size']:,} байт ({result['file_size']/1024/1024:.1f} MB)")
            print(f"   Таблиц: {len(result['tables'])}")
            print(f"   Записей: {result['total_records']:,}")
            print(f"   Изменен: {result['last_modified']}")
            
            for table_name, info in result['tables'].items():
                print(f"     - {table_name}: {info['records']} записей, {len(info['columns'])} колонок")
        else:
            print(f"   ❌ Не удалось проанализировать")
    
    # Анализ дублирования
    print("\n" + "=" * 60)
    print("🔍 АНАЛИЗ ДУБЛИРОВАНИЯ")
    print("=" * 60)
    
    # Группируем базы по схожей структуре
    knowledge_bases = []
    document_bases = []
    learning_bases = []
    other_bases = []
    
    for db_file, result in analysis_results.items():
        if not result:
            continue
            
        tables = list(result['tables'].keys())
        
        if any('knowledge' in table.lower() for table in tables):
            knowledge_bases.append((db_file, result))
        elif any('document' in table.lower() for table in tables):
            document_bases.append((db_file, result))
        elif any('learning' in table.lower() for table in tables):
            learning_bases.append((db_file, result))
        else:
            other_bases.append((db_file, result))
    
    print(f"\n📚 Базы знаний ({len(knowledge_bases)}):")
    for db_file, result in knowledge_bases:
        print(f"   - {db_file}: {result['file_size']/1024/1024:.1f} MB, {result['total_records']} записей")
    
    print(f"\n📄 Базы документов ({len(document_bases)}):")
    for db_file, result in document_bases:
        print(f"   - {db_file}: {result['file_size']/1024/1024:.1f} MB, {result['total_records']} записей")
    
    print(f"\n🧠 Базы обучения ({len(learning_bases)}):")
    for db_file, result in learning_bases:
        print(f"   - {db_file}: {result['file_size']/1024/1024:.1f} MB, {result['total_records']} записей")
    
    print(f"\n🔧 Прочие базы ({len(other_bases)}):")
    for db_file, result in other_bases:
        print(f"   - {db_file}: {result['file_size']/1024/1024:.1f} MB, {result['total_records']} записей")
    
    # Рекомендации по оптимизации
    print("\n" + "=" * 60)
    print("💡 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ")
    print("=" * 60)
    
    total_size = sum(result['file_size'] for result in analysis_results.values())
    print(f"Общий размер всех баз: {total_size/1024/1024:.1f} MB")
    
    # Выявляем потенциальные дубликаты
    if len(document_bases) > 1:
        print(f"\n⚠️  Обнаружено {len(document_bases)} баз документов - возможное дублирование!")
        print("   Рекомендуется объединить в одну базу")
    
    if len(knowledge_bases) > 3:
        print(f"\n⚠️  Обнаружено {len(knowledge_bases)} баз знаний - избыточность!")
        print("   Рекомендуется консолидация")
    
    # Пустые базы
    empty_bases = [db_file for db_file, result in analysis_results.items() 
                   if result and result['total_records'] == 0]
    if empty_bases:
        print(f"\n🗑️  Пустые базы данных ({len(empty_bases)}):")
        for db_file in empty_bases:
            print(f"   - {db_file}")
        print("   Рекомендуется удалить пустые базы")
    
    print(f"\n✅ Анализ завершен. Найдено {len(analysis_results)} баз данных")

if __name__ == "__main__":
    main()





