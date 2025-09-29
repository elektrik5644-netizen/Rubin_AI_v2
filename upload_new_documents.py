#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для загрузки новых документов в базу данных Rubin AI
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
    
    def read_file_content(self, file_path):
        """Чтение содержимого файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except UnicodeDecodeError:
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
    logger.info("=== ЗАГРУЗКА НОВЫХ ДОКУМЕНТОВ В RUBIN AI ===")
    
    # Создание загрузчика
    uploader = DocumentUploader()
    
    if not uploader.connect():
        logger.error("Не удалось подключиться к базе данных")
        return
    
    # Загрузка новых документов
    new_documents = [
        {
            'file_path': 'test_documents/automation_examples.txt',
            'category': 'automation',
            'title': 'Примеры автоматизации',
            'metadata': {
                'description': 'Практические примеры систем автоматизации',
                'author': 'Rubin AI',
                'version': '1.0'
            }
        },
        {
            'file_path': 'test_documents/electrical_circuits.txt',
            'category': 'electrical',
            'title': 'Электрические схемы и расчеты',
            'metadata': {
                'description': 'Расчеты электрических цепей и схем',
                'author': 'Rubin AI',
                'version': '1.0'
            }
        }
    ]
    
    # Загрузка каждого документа
    uploaded_count = 0
    for doc_info in new_documents:
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






















