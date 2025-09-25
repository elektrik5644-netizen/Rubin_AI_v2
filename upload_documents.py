#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для загрузки документов в базу данных Rubin AI
"""

import os
import sys
import sqlite3
import json
from datetime import datetime
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentUploader:
    def __init__(self, db_path="rubin_ai_v2.db"):
        self.db_path = db_path
        self.connection = None
    
    def connect(self):
        """Подключение к базе данных"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.execute("PRAGMA foreign_keys = ON")
            logger.info(f"Подключение к базе данных {self.db_path} установлено")
            return True
        except sqlite3.Error as e:
            logger.error(f"Ошибка подключения к базе данных: {e}")
            return False
    
    def create_tables(self):
        """Создание таблиц если они не существуют"""
        try:
            cursor = self.connection.cursor()
            
            # Таблица документов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица для векторного поиска
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS document_vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    vector BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                )
            ''')
            
            # Индексы для ускорения поиска
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_documents_category 
                ON documents (category)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_documents_filename 
                ON documents (file_name)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_vectors_document_id 
                ON document_vectors (document_id)
            ''')
            
            self.connection.commit()
            logger.info("Таблицы созданы успешно")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Ошибка создания таблиц: {e}")
            return False
    
    def read_file_content(self, file_path):
        """Чтение содержимого файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except UnicodeDecodeError:
            # Попробуем другие кодировки
            try:
                with open(file_path, 'r', encoding='cp1251') as f:
                    content = f.read()
                return content
            except:
                logger.error(f"Не удалось прочитать файл {file_path}")
                return None
        except Exception as e:
            logger.error(f"Ошибка чтения файла {file_path}: {e}")
            return None
    
    def upload_document(self, file_path, category, title=None, metadata=None):
        """Загрузка документа в базу данных"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"Файл {file_path} не найден")
                return False
            
            # Чтение содержимого файла
            content = self.read_file_content(file_path)
            if content is None:
                return False
            
            # Определение имени файла
            file_name = os.path.basename(file_path)
            if title:
                file_name = title
            
            # Подготовка метаданных
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'file_size': os.path.getsize(file_path),
                'file_extension': os.path.splitext(file_path)[1],
                'upload_time': datetime.now().isoformat()
            })
            
            # Вставка в базу данных
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO documents (file_name, file_path, category, content, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (file_name, file_path, category, content, json.dumps(metadata)))
            
            document_id = cursor.lastrowid
            self.connection.commit()
            
            logger.info(f"Документ {file_name} загружен с ID {document_id}")
            return document_id
            
        except sqlite3.Error as e:
            logger.error(f"Ошибка загрузки документа: {e}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            return False
    
    def upload_directory(self, directory_path, category):
        """Загрузка всех документов из директории"""
        if not os.path.exists(directory_path):
            logger.error(f"Директория {directory_path} не найдена")
            return False
        
        uploaded_count = 0
        failed_count = 0
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            if os.path.isfile(file_path):
                # Определение подкатегории по расширению файла
                ext = os.path.splitext(filename)[1].lower()
                subcategory = self.get_subcategory(ext)
                
                # Загрузка документа
                if self.upload_document(file_path, category, metadata={'subcategory': subcategory}):
                    uploaded_count += 1
                else:
                    failed_count += 1
        
        logger.info(f"Загружено: {uploaded_count}, Ошибок: {failed_count}")
        return uploaded_count > 0
    
    def get_subcategory(self, file_extension):
        """Определение подкатегории по расширению файла"""
        subcategories = {
            '.txt': 'text',
            '.py': 'python',
            '.js': 'javascript',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.xml': 'xml',
            '.md': 'markdown',
            '.pdf': 'pdf',
            '.doc': 'word',
            '.docx': 'word',
            '.xls': 'excel',
            '.xlsx': 'excel'
        }
        return subcategories.get(file_extension, 'unknown')
    
    def get_documents_count(self):
        """Получение количества документов в базе"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            count = cursor.fetchone()[0]
            return count
        except sqlite3.Error as e:
            logger.error(f"Ошибка получения количества документов: {e}")
            return 0
    
    def get_documents_by_category(self):
        """Получение статистики по категориям"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT category, COUNT(*) as count 
                FROM documents 
                GROUP BY category 
                ORDER BY count DESC
            ''')
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return []
    
    def close(self):
        """Закрытие подключения"""
        if self.connection:
            self.connection.close()
            logger.info("Подключение к базе данных закрыто")

def main():
    """Основная функция"""
    logger.info("=== ЗАГРУЗКА ДОКУМЕНТОВ В RUBIN AI ===")
    
    # Создание загрузчика
    uploader = DocumentUploader()
    
    if not uploader.connect():
        logger.error("Не удалось подключиться к базе данных")
        return
    
    # Создание таблиц
    if not uploader.create_tables():
        logger.error("Не удалось создать таблицы")
        return
    
    # Загрузка документов
    documents_to_upload = [
        {
            'file_path': 'test_documents/electrical_guide.txt',
            'category': 'electrical',
            'title': 'Руководство по электротехнике',
            'metadata': {
                'description': 'Полное руководство по основам электротехники',
                'author': 'Rubin AI',
                'version': '1.0'
            }
        },
        {
            'file_path': 'test_documents/programming_guide.txt',
            'category': 'programming',
            'title': 'Руководство по программированию',
            'metadata': {
                'description': 'Руководство по программированию на различных языках',
                'author': 'Rubin AI',
                'version': '1.0'
            }
        },
        {
            'file_path': 'test_documents/controllers_guide.txt',
            'category': 'controllers',
            'title': 'Руководство по контроллерам и автоматизации',
            'metadata': {
                'description': 'Руководство по ПЛК, ПИД-регуляторам и автоматизации',
                'author': 'Rubin AI',
                'version': '1.0'
            }
        },
        {
            'file_path': 'test_documents/radiomechanics_guide.txt',
            'category': 'radiomechanics',
            'title': 'Руководство по радиомеханике',
            'metadata': {
                'description': 'Руководство по радиотехнике и антеннам',
                'author': 'Rubin AI',
                'version': '1.0'
            }
        },
        {
            'file_path': 'test_documents/python_examples.py',
            'category': 'programming',
            'title': 'Примеры кода на Python',
            'metadata': {
                'description': 'Практические примеры кода на Python',
                'author': 'Rubin AI',
                'version': '1.0',
                'language': 'python'
            }
        }
    ]
    
    # Загрузка каждого документа
    uploaded_count = 0
    for doc_info in documents_to_upload:
        logger.info(f"Загрузка: {doc_info['title']}")
        
        if uploader.upload_document(
            doc_info['file_path'],
            doc_info['category'],
            doc_info['title'],
            doc_info['metadata']
        ):
            uploaded_count += 1
        else:
            logger.error(f"Не удалось загрузить: {doc_info['title']}")
    
    # Статистика
    total_documents = uploader.get_documents_count()
    category_stats = uploader.get_documents_by_category()
    
    logger.info(f"\n=== СТАТИСТИКА ===")
    logger.info(f"Всего документов в базе: {total_documents}")
    logger.info(f"Загружено в этой сессии: {uploaded_count}")
    
    if category_stats:
        logger.info("По категориям:")
        for category, count in category_stats:
            logger.info(f"  {category}: {count} документов")
    
    # Закрытие подключения
    uploader.close()
    
    logger.info("=== ЗАГРУЗКА ЗАВЕРШЕНА ===")

if __name__ == "__main__":
    main()

















