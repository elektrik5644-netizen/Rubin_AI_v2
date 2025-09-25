#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для индексации существующих документов в векторном пространстве
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path

# Добавляем текущую папку в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector_search import VectorSearchEngine
from document_loader import DocumentLoader

def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('index_documents.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def index_all_documents():
    """Индексация всех документов в векторном пространстве"""
    logger = setup_logging()
    
    try:
        logger.info("🚀 Начало индексации документов для векторного поиска")
        
        # Инициализация движков
        doc_loader = DocumentLoader()
        vector_engine = VectorSearchEngine()
        
        # Проверка доступности модели
        if not vector_engine.model:
            logger.error("❌ Модель для генерации embeddings не загружена")
            logger.info("💡 Установите sentence-transformers: pip install sentence-transformers")
            return False
            
        # Получение всех документов из базы
        conn = sqlite3.connect(doc_loader.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, file_name, content, category, metadata
            FROM documents
            WHERE content IS NOT NULL AND content != ''
            ORDER BY id
        ''')
        
        documents = cursor.fetchall()
        conn.close()
        
        if not documents:
            logger.warning("⚠️ Документы не найдены в базе данных")
            return False
            
        logger.info(f"📄 Найдено {len(documents)} документов для индексации")
        
        # Индексация документов
        indexed_count = 0
        failed_count = 0
        
        for doc_id, file_name, content, category, metadata in documents:
            try:
                logger.info(f"📄 Индексация документа {doc_id}: {file_name}")
                
                # Подготовка метаданных
                doc_metadata = {
                    'file_name': file_name,
                    'category': category or 'unknown',
                    'metadata': metadata or ''
                }
                
                # Индексация в векторном пространстве
                success = vector_engine.index_document(doc_id, content, doc_metadata)
                
                if success:
                    indexed_count += 1
                    logger.info(f"✅ Документ {doc_id} проиндексирован")
                else:
                    failed_count += 1
                    logger.warning(f"⚠️ Не удалось проиндексировать документ {doc_id}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ Ошибка индексации документа {doc_id}: {e}")
                
        # Построение FAISS индекса
        logger.info("🔨 Построение FAISS индекса для быстрого поиска...")
        faiss_success = vector_engine.build_faiss_index()
        
        # Статистика
        logger.info("📊 Статистика индексации:")
        logger.info(f"   Всего документов: {len(documents)}")
        logger.info(f"   Успешно проиндексировано: {indexed_count}")
        logger.info(f"   Ошибок: {failed_count}")
        logger.info(f"   FAISS индекс: {'✅ Построен' if faiss_success else '❌ Не построен'}")
        
        # Получение финальной статистики
        stats = vector_engine.get_stats()
        logger.info("📈 Финальная статистика векторного поиска:")
        for key, value in stats.items():
            logger.info(f"   {key}: {value}")
            
        return indexed_count > 0
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка индексации: {e}")
        return False

def test_vector_search():
    """Тестирование векторного поиска"""
    logger = setup_logging()
    
    try:
        logger.info("🧪 Тестирование векторного поиска")
        
        vector_engine = VectorSearchEngine()
        
        if not vector_engine.model:
            logger.error("❌ Модель не загружена, тестирование невозможно")
            return False
            
        # Тестовые запросы
        test_queries = [
            "ПИД регулятор",
            "электротехника",
            "программирование",
            "контроллер",
            "радиомеханика"
        ]
        
        for query in test_queries:
            logger.info(f"🔍 Тестовый запрос: '{query}'")
            
            results = vector_engine.search_similar(query, top_k=3, threshold=0.5)
            
            if results:
                logger.info(f"   Найдено {len(results)} результатов:")
                for i, result in enumerate(results, 1):
                    logger.info(f"   {i}. Документ {result['document_id']} (сходство: {result['similarity']:.3f})")
            else:
                logger.info("   Результаты не найдены")
                
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования: {e}")
        return False

if __name__ == "__main__":
    print("🎯 ИНДЕКСАЦИЯ ДОКУМЕНТОВ ДЛЯ ВЕКТОРНОГО ПОИСКА")
    print("=" * 60)
    
    # Индексация документов
    success = index_all_documents()
    
    if success:
        print("\n✅ Индексация завершена успешно!")
        
        # Тестирование
        print("\n🧪 Тестирование векторного поиска...")
        test_success = test_vector_search()
        
        if test_success:
            print("✅ Тестирование прошло успешно!")
            print("\n🚀 Векторный поиск готов к использованию!")
            print("📊 API доступен по адресу: http://localhost:8091")
        else:
            print("⚠️ Тестирование завершилось с ошибками")
    else:
        print("❌ Индексация завершилась с ошибками")
        print("💡 Проверьте логи для получения подробной информации")

















