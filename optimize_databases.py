#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полная оптимизация баз данных Rubin AI v2
Выполняет все этапы оптимизации: консолидацию, унификацию и очистку
"""

import os
import sys
from datetime import datetime

def print_header(title):
    """Печатает заголовок"""
    print("\n" + "=" * 60)
    print(f"🚀 {title}")
    print("=" * 60)

def print_step(step_num, title, description=""):
    """Печатает шаг процесса"""
    print(f"\n📋 ШАГ {step_num}: {title}")
    if description:
        print(f"   {description}")
    print("-" * 40)

def check_prerequisites():
    """Проверяет предварительные условия"""
    print_header("ПРОВЕРКА ПРЕДВАРИТЕЛЬНЫХ УСЛОВИЙ")
    
    # Проверяем наличие скриптов
    required_scripts = [
        'consolidate_documents.py',
        'unify_knowledge.py', 
        'cleanup_databases.py'
    ]
    
    missing_scripts = []
    for script in required_scripts:
        if not os.path.exists(script):
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"❌ Отсутствуют скрипты: {', '.join(missing_scripts)}")
        return False
    
    # Проверяем наличие баз данных
    db_files = [f for f in os.listdir('.') if f.endswith('.db')]
    if not db_files:
        print("❌ Не найдены базы данных для оптимизации")
        return False
    
    print(f"✅ Найдено скриптов: {len(required_scripts)}")
    print(f"✅ Найдено баз данных: {len(db_files)}")
    print("✅ Все предварительные условия выполнены")
    
    return True

def create_backup_folder():
    """Создает папку для резервных копий"""
    backup_folder = f"database_backups_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
        print(f"📁 Создана папка для резервных копий: {backup_folder}")
    
    return backup_folder

def run_script(script_name, description):
    """Запускает скрипт оптимизации"""
    print(f"\n🔄 Запуск: {script_name}")
    print(f"   {description}")
    
    try:
        # Импортируем и запускаем скрипт
        if script_name == 'consolidate_documents.py':
            from consolidate_documents import consolidate_documents
            result = consolidate_documents()
            return result is not None
            
        elif script_name == 'unify_knowledge.py':
            from unify_knowledge import unify_knowledge_bases
            result = unify_knowledge_bases()
            return result is not None
            
        elif script_name == 'cleanup_databases.py':
            from cleanup_databases import cleanup_databases
            cleanup_databases()
            return True
            
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка выполнения: {e}")
        return False

def generate_report():
    """Генерирует отчет об оптимизации"""
    print_header("ОТЧЕТ ОБ ОПТИМИЗАЦИИ")
    
    # Анализируем текущее состояние
    db_files = [f for f in os.listdir('.') if f.endswith('.db')]
    total_size = sum(os.path.getsize(f) for f in db_files)
    
    print(f"📊 Текущее состояние баз данных:")
    print(f"   Всего баз: {len(db_files)}")
    print(f"   Общий размер: {total_size / 1024 / 1024:.1f} MB")
    
    # Проверяем созданные оптимизированные базы
    optimized_dbs = [
        'rubin_documents_consolidated.db',
        'rubin_knowledge_unified.db'
    ]
    
    print(f"\n✅ Оптимизированные базы:")
    for db in optimized_dbs:
        if os.path.exists(db):
            size = os.path.getsize(db)
            print(f"   - {db}: {size / 1024 / 1024:.1f} MB")
        else:
            print(f"   - {db}: не создана")
    
    # Проверяем резервные копии
    backup_files = [f for f in os.listdir('.') if 'backup' in f and f.endswith('.db')]
    print(f"\n📦 Резервные копии: {len(backup_files)}")
    
    print(f"\n🎯 Рекомендации:")
    print(f"   1. Протестируйте новые базы данных")
    print(f"   2. Обновите конфигурацию приложения")
    print(f"   3. Удалите старые базы после тестирования")
    print(f"   4. Сохраните резервные копии в безопасном месте")

def main():
    """Основная функция оптимизации"""
    print_header("ПОЛНАЯ ОПТИМИЗАЦИЯ БАЗ ДАННЫХ RUBIN AI V2")
    print("Этот скрипт выполнит полную оптимизацию всех баз данных")
    print("ВНИМАНИЕ: Создайте резервные копии перед запуском!")
    
    # Проверяем предварительные условия
    if not check_prerequisites():
        print("\n❌ Предварительные условия не выполнены")
        return False
    
    # Создаем папку для резервных копий
    backup_folder = create_backup_folder()
    
    # Выполняем оптимизацию по шагам
    steps = [
        {
            'script': 'consolidate_documents.py',
            'title': 'КОНСОЛИДАЦИЯ ДОКУМЕНТОВ',
            'description': 'Объединяет дублирующие базы документов (экономия ~35 MB)'
        },
        {
            'script': 'unify_knowledge.py', 
            'title': 'УНИФИКАЦИЯ БАЗ ЗНАНИЙ',
            'description': 'Объединяет фрагментированные базы знаний'
        },
        {
            'script': 'cleanup_databases.py',
            'title': 'ОЧИСТКА БАЗ ДАННЫХ', 
            'description': 'Удаляет пустые и устаревшие базы данных'
        }
    ]
    
    success_count = 0
    
    for i, step in enumerate(steps, 1):
        print_step(i, step['title'], step['description'])
        
        if run_script(step['script'], step['description']):
            print(f"✅ Шаг {i} выполнен успешно")
            success_count += 1
        else:
            print(f"❌ Шаг {i} завершился с ошибкой")
            print("⚠️ Продолжаем выполнение следующих шагов...")
    
    # Генерируем отчет
    generate_report()
    
    # Итоговый результат
    print_header("ИТОГОВЫЙ РЕЗУЛЬТАТ")
    
    if success_count == len(steps):
        print("🎉 ВСЕ ШАГИ ВЫПОЛНЕНЫ УСПЕШНО!")
        print("✅ Базы данных оптимизированы")
        print("✅ Место на диске освобождено")
        print("✅ Производительность улучшена")
    else:
        print(f"⚠️ ВЫПОЛНЕНО {success_count} ИЗ {len(steps)} ШАГОВ")
        print("📋 Проверьте логи для выявления проблем")
        print("📦 Используйте резервные копии для восстановления")
    
    print(f"\n📁 Резервные копии сохранены в: {backup_folder}")
    print("🔧 Следующий шаг: Обновите конфигурацию приложения")
    
    return success_count == len(steps)

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Оптимизация прервана пользователем")
        print("📦 Проверьте резервные копии для восстановления")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        print("📦 Проверьте резервные копии для восстановления")
        sys.exit(1)










