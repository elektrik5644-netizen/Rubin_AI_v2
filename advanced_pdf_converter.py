#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Продвинутый конвертер PDF в текст с улучшенным извлечением и форматированием
"""

import sqlite3
import os
import hashlib
from datetime import datetime
import PyPDF2
import re

def extract_text_with_structure(pdf_path):
    """Извлекает текст из PDF с сохранением структуры"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            print(f"📊 Количество страниц: {len(pdf_reader.pages)}")
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text:
                    # Добавляем номер страницы
                    text += f"\n--- СТРАНИЦА {page_num + 1} ---\n"
                    text += page_text + "\n"
                
                print(f"   Страница {page_num + 1}: {len(page_text) if page_text else 0} символов")
        
        return text.strip()
        
    except Exception as e:
        print(f"❌ Ошибка извлечения текста: {e}")
        return None

def improve_text_structure(text):
    """Улучшает структуру текста"""
    if not text:
        return ""
    
    # Разделяем на строки
    lines = text.split('\n')
    improved_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Пропускаем пустые строки
        if not line:
            continue
        
        # Улучшаем заголовки
        if re.match(r'^\d+\.', line):  # Нумерованные пункты
            improved_lines.append(f"\n{line}")
        elif re.match(r'^\d+\.\d+', line):  # Подпункты
            improved_lines.append(f"  {line}")
        elif line.isupper() and len(line) > 10:  # Заголовки заглавными буквами
            improved_lines.append(f"\n{line}")
        else:
            improved_lines.append(line)
    
    return '\n'.join(improved_lines)

def create_structured_text(text):
    """Создает структурированный текст из PDF"""
    if not text:
        return ""
    
    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text)
    
    # Разделяем на абзацы
    paragraphs = text.split('. ')
    
    structured_text = ""
    current_section = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # Определяем тип контента
        if re.match(r'^\d+\.', paragraph):  # Нумерованный пункт
            if current_section:
                structured_text += f"\n{current_section}\n"
            current_section = paragraph
        elif paragraph.isupper() and len(paragraph) > 10:  # Заголовок
            if current_section:
                structured_text += f"\n{current_section}\n"
            structured_text += f"\n{paragraph}\n"
            current_section = ""
        else:
            if current_section:
                current_section += f". {paragraph}"
            else:
                structured_text += f"{paragraph}. "
    
    # Добавляем последний раздел
    if current_section:
        structured_text += f"\n{current_section}\n"
    
    return structured_text.strip()

def update_pdf_with_improved_text(db_path="rubin_ai_documents.db"):
    """Обновляет PDF файл в базе данных с улучшенным текстом"""
    
    if not os.path.exists(db_path):
        print(f"❌ База данных не найдена: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Находим PDF файл
        cursor.execute("""
            SELECT id, file_name, content, LENGTH(content) as content_length
            FROM documents 
            WHERE file_name LIKE '%05.Руководство по наладке и диагностике%'
        """)
        
        result = cursor.fetchone()
        if not result:
            print("❌ PDF файл не найден в базе данных")
            conn.close()
            return False
        
        doc_id, file_name, old_content, old_length = result
        print(f"📋 Найден документ:")
        print(f"   ID: {doc_id}")
        print(f"   Файл: {file_name}")
        print(f"   Текущий размер: {old_length} символов")
        
        # Улучшаем текст
        print("🔄 Улучшаем структуру текста...")
        improved_content = improve_text_structure(old_content)
        
        # Создаем структурированный текст
        print("🔄 Создаем структурированный текст...")
        structured_content = create_structured_text(improved_content)
        
        new_length = len(structured_content)
        print(f"   Новый размер: {new_length} символов")
        print(f"   Изменение: {old_length - new_length} символов")
        
        # Показываем превью
        preview = structured_content[:800].replace('\n', ' ')
        print(f"   Превью: {preview}...")
        
        # Вычисляем новый хеш
        new_hash = hashlib.md5(structured_content.encode('utf-8')).hexdigest()
        
        # Обновляем запись
        cursor.execute("""
            UPDATE documents 
            SET content = ?, 
                file_hash = ?,
                updated_at = ?
            WHERE id = ?
        """, (structured_content, new_hash, datetime.now().isoformat(), doc_id))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Файл {file_name} успешно обновлен с улучшенной структурой")
        print(f"   Размер изменен с {old_length} на {new_length} символов")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка обновления базы данных: {e}")
        return False

def main():
    """Основная функция"""
    print("🔄 Продвинутый конвертер PDF в текст")
    print("=" * 50)
    
    success = update_pdf_with_improved_text()
    
    if success:
        print("\n🎉 Конвертация завершена успешно!")
        print("💡 Теперь перезапустите AI Чат сервис для применения изменений")
    else:
        print("\n❌ Конвертация не удалась")

if __name__ == "__main__":
    main()






















