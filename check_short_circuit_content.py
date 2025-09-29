#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import re

def check_short_circuit_content():
    """Проверяем содержимое документов о коротком замыкании"""
    
    conn = sqlite3.connect('rubin_ai_documents.db')
    cursor = conn.cursor()
    
    # Ищем документы с информацией о коротком замыкании
    cursor.execute('''
        SELECT file_name, content 
        FROM documents 
        WHERE content LIKE "%коротк%" 
        AND category = "Электротехника"
        LIMIT 3
    ''')
    
    results = cursor.fetchall()
    
    print("🔍 **СОДЕРЖИМОЕ ДОКУМЕНТОВ О КОРОТКОМ ЗАМЫКАНИИ**")
    print("=" * 60)
    
    for i, (file_name, content) in enumerate(results, 1):
        print(f"\n📄 **Документ {i}: {file_name}**")
        print("-" * 50)
        
        # Ищем фрагменты с упоминанием короткого замыкания
        sentences = content.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            if 'коротк' in sentence.lower() and len(sentence.strip()) > 20:
                relevant_sentences.append(sentence.strip())
                if len(relevant_sentences) >= 3:  # Ограничиваем количество
                    break
        
        if relevant_sentences:
            print("📝 **Релевантные фрагменты:**")
            for j, sentence in enumerate(relevant_sentences, 1):
                print(f"  {j}. {sentence[:200]}...")
        else:
            print("❌ Релевантные фрагменты не найдены")
    
    conn.close()

if __name__ == "__main__":
    check_short_circuit_content()






















