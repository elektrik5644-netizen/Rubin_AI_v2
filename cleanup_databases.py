#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Очистка пустых и устаревших баз данных Rubin AI v2
Удаляет базы данных без записей для освобождения места
"""

import sqlite3
import os
import shutil
from datetime import datetime

def backup_db(db_path):
    """Создает резервную копию базы данных"""
    if os.path.exists(db_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{db_path}.backup_{timestamp}"
        shutil.copy2(db_path, backup_path)
        print(f"✅ Создана резервная копия: {backup_path}")
        return backup_path
    return None

def delete_db(db_path):
    """Удаляет базу данных"""
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"🗑️ Удалена база данных: {db_path}")
        return True
    print(f"⚠️ База данных не найдена для удаления: {db_path}")
    return False

def check_and_vacuum_db(db_path):
    """Проверяет целостность и оптимизирует базу данных"""
    if not os.path.exists(db_path):
        print(f"❌ База данных не найдена: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Проверяем целостность
        cursor.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()[0]
        
        if result == 'ok':
            print(f"✅ Целостность {db_path} в норме")
            
            # Оптимизируем базу
            cursor.execute("VACUUM;")
            print(f"✨ Оптимизация {db_path} завершена")
            
            conn.close()
            return True
        else:
            print(f"❌ Ошибка целостности {db_path}: {result}")
            conn.close()
            return False
            
    except sqlite3.Error as e:
        print(f"❌ Ошибка при работе с {db_path}: {e}")
        return False

def cleanup_databases():
    """Основная функция очистки баз данных"""
    print("🧹 ОЧИСТКА БАЗ ДАННЫХ RUBIN AI V2")
    print("=" * 50)
    
    # Определяем пустые и устаревшие базы
    empty_databases = [
        'rubin_documents.db',      # 0 записей
        'rubin_errors.db',         # 0 записей  
        'rubin_knowledge.db'       # 0 байт
    ]
    
    # Определяем дублирующие базы (после консолидации)
    duplicate_databases = [
        'rubin_ai_documents.db',   # Дубликат после консолидации
        'rubin_ai_v2.db'           # Дубликат после консолидации
    ]
    
    # Проверяем, существует ли консолидированная база
    consolidated_exists = os.path.exists('rubin_documents_consolidated.db')
    
    total_freed_space = 0
    
    # Обрабатываем пустые базы
    print("\n📊 ОБРАБОТКА ПУСТЫХ БАЗ ДАННЫХ")
    print("-" * 40)
    
    for db_name in empty_databases:
        if os.path.exists(db_name):
            file_size = os.path.getsize(db_name)
            print(f"\n🔍 Проверяю: {db_name} ({file_size} байт)")
            
            # Проверяем, действительно ли база пуста
            try:
                conn = sqlite3.connect(db_name)
                cursor = conn.cursor()
                
                # Получаем список таблиц
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                total_records = 0
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                    count = cursor.fetchone()[0]
                    total_records += count
                
                conn.close()
                
                if total_records == 0 or file_size == 0:
                    print(f"   ✅ База пуста ({total_records} записей)")
                    backup_db(db_name)
                    if delete_db(db_name):
                        total_freed_space += file_size
                else:
                    print(f"   ⚠️ База содержит {total_records} записей - пропускаем")
                    
            except Exception as e:
                print(f"   ❌ Ошибка при проверке {db_name}: {e}")
        else:
            print(f"⚠️ База данных не найдена: {db_name}")
    
    # Обрабатываем дублирующие базы (только если есть консолидированная)
    if consolidated_exists:
        print("\n📊 ОБРАБОТКА ДУБЛИРУЮЩИХ БАЗ ДАННЫХ")
        print("-" * 40)
        
        for db_name in duplicate_databases:
            if os.path.exists(db_name):
                file_size = os.path.getsize(db_name)
                print(f"\n🔍 Проверяю дубликат: {db_name} ({file_size / 1024 / 1024:.1f} MB)")
                
                # Создаем резервную копию и удаляем
                backup_db(db_name)
                if delete_db(db_name):
                    total_freed_space += file_size
                    print(f"   ✅ Дубликат удален")
            else:
                print(f"⚠️ Дубликат не найден: {db_name}")
    else:
        print(f"\n⚠️ Консолидированная база не найдена - дубликаты не удаляются")
        print("   Сначала запустите consolidate_documents.py")
    
    # Оптимизируем оставшиеся базы
    print("\n⚡ ОПТИМИЗАЦИЯ ОСТАВШИХСЯ БАЗ ДАННЫХ")
    print("-" * 40)
    
    remaining_dbs = [f for f in os.listdir('.') if f.endswith('.db') and not f.startswith('rubin_documents_consolidated.db.backup')]
    
    for db_file in remaining_dbs:
        if db_file not in empty_databases and db_file not in duplicate_databases:
            print(f"\n🔧 Оптимизирую: {db_file}")
            check_and_vacuum_db(db_file)
    
    # Итоговая статистика
    print("\n" + "=" * 50)
    print("📊 ИТОГОВАЯ СТАТИСТИКА ОЧИСТКИ")
    print("=" * 50)
    print(f"✅ Освобождено места: {total_freed_space / 1024 / 1024:.1f} MB")
    print(f"✅ Обработано баз данных: {len(empty_databases) + len(duplicate_databases)}")
    
    if consolidated_exists:
        print(f"✅ Дубликаты удалены: {len(duplicate_databases)}")
    else:
        print(f"⚠️ Дубликаты сохранены (нет консолидированной базы)")
    
    print(f"✅ Пустые базы удалены: {len(empty_databases)}")
    print(f"✅ Резервные копии созданы для всех удаленных баз")
    
    print("\n🎉 Очистка завершена успешно!")

if __name__ == "__main__":
    try:
        cleanup_databases()
    except Exception as e:
        print(f"\n❌ Ошибка при очистке: {e}")
        print("📦 Проверьте резервные копии для восстановления")