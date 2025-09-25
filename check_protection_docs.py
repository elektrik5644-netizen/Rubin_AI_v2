#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3

def check_protection_documents():
    """Проверяем наличие документов по защите от короткого замыкания"""
    
    conn = sqlite3.connect('rubin_ai_documents.db')
    cursor = conn.cursor()
    
    # Ищем документы по ключевым словам
    keywords = ['защит', 'предохранитель', 'коротк', 'автомат', 'выключатель', 'защита']
    
    print("🔍 **ПОИСК ДОКУМЕНТОВ ПО ЗАЩИТЕ ОТ КОРОТКОГО ЗАМЫКАНИЯ**")
    print("=" * 60)
    
    for keyword in keywords:
        cursor.execute('''
            SELECT file_name, category, LENGTH(content) as content_length
            FROM documents 
            WHERE content LIKE ? 
            LIMIT 5
        ''', (f'%{keyword}%',))
        
        results = cursor.fetchall()
        
        if results:
            print(f"\n📄 **Документы со словом '{keyword}':**")
            for row in results:
                print(f"  - {row[0]} ({row[1]}) - {row[2]} символов")
        else:
            print(f"\n❌ Документы со словом '{keyword}' не найдены")
    
    # Общая статистика
    cursor.execute('SELECT COUNT(*) FROM documents')
    total_docs = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(DISTINCT category) FROM documents')
    total_categories = cursor.fetchone()[0]
    
    print(f"\n📊 **ОБЩАЯ СТАТИСТИКА:**")
    print(f"  Всего документов: {total_docs}")
    print(f"  Категорий: {total_categories}")
    
    # Категории
    cursor.execute('SELECT category, COUNT(*) FROM documents GROUP BY category')
    categories = cursor.fetchall()
    
    print(f"\n📁 **КАТЕГОРИИ ДОКУМЕНТОВ:**")
    for category, count in categories:
        print(f"  - {category}: {count} документов")
    
    conn.close()

if __name__ == "__main__":
    check_protection_documents()

















