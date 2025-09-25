#!/usr/bin/env python3
"""
Проверка загруженных материалов
"""

import sqlite3

def check_uploaded_materials():
    """Проверка загруженных материалов"""
    
    print("🔍 ПРОВЕРКА ЗАГРУЖЕННЫХ МАТЕРИАЛОВ")
    print("=" * 50)
    
    conn = sqlite3.connect('rubin_knowledge_base.db')
    cursor = conn.cursor()
    
    # Проверяем все записи
    cursor.execute('SELECT title, category, subject, content FROM knowledge_entries')
    results = cursor.fetchall()
    
    print(f"📊 Всего записей в базе: {len(results)}")
    print("\n📋 Все записи:")
    
    for title, category, subject, content in results:
        print(f"  • {title}")
        print(f"    Категория: {category}")
        print(f"    Предмет: {subject}")
        print(f"    Содержание: {content[:100]}...")
        print()
    
    # Проверяем записи по новым предметам
    print("\n🧪 Записи по новым предметам:")
    cursor.execute('SELECT title, category, subject FROM knowledge_entries WHERE category IN ("chemistry", "physics", "mathematics")')
    new_results = cursor.fetchall()
    
    if new_results:
        for title, category, subject in new_results:
            print(f"  • {title} ({category}/{subject})")
    else:
        print("  ❌ Записи по новым предметам не найдены")
    
    conn.close()

if __name__ == "__main__":
    check_uploaded_materials()












