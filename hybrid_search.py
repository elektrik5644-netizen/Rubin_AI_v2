#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Гибридный поисковик для Rubin AI v2.0
Объединяет текстовый и векторный поиск для максимальной эффективности
"""

import logging
import time
from typing import List, Dict, Optional, Tuple
from vector_search import VectorSearchEngine
from document_loader import DocumentLoader

class HybridSearchEngine:
    """Гибридный поисковик, объединяющий текстовый и векторный поиск"""
    
    def __init__(self, db_path="rubin_ai_documents.db"):
        self.db_path = db_path
        self.setup_logging()
        
        # Инициализация компонентов
        self.text_search = DocumentLoader(db_path)
        self.vector_search = VectorSearchEngine(db_path)
        
        # Настройки поиска
        self.text_weight = 0.4  # Вес текстового поиска
        self.vector_weight = 0.6  # Вес векторного поиска
        self.min_similarity = 0.6  # Минимальное сходство для векторного поиска
        
    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('hybrid_search.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def search(self, query: str, limit: int = 10, search_type: str = "hybrid") -> List[Dict]:
        """
        Гибридный поиск документов
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            search_type: Тип поиска ("text", "vector", "hybrid")
            
        Returns:
            Список найденных документов с метаданными
        """
        start_time = time.time()
        
        try:
            if search_type == "text":
                results = self._text_only_search(query, limit)
            elif search_type == "vector":
                results = self._vector_only_search(query, limit)
            else:  # hybrid
                results = self._hybrid_search(query, limit)
                
            search_time = time.time() - start_time
            self.logger.info(f"🔍 Поиск завершен за {search_time:.3f}с, найдено {len(results)} результатов")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка гибридного поиска: {e}")
            return []
            
    def _text_only_search(self, query: str, limit: int) -> List[Dict]:
        """Только текстовый поиск"""
        self.logger.info("📝 Выполняется текстовый поиск")
        
        try:
            # Получение результатов текстового поиска
            text_results = self.text_search.search_documents(query, limit * 2)
            
            # Форматирование результатов
            results = []
            for row in text_results:
                doc_id, file_name, category, metadata, keywords = row
                
                # Получение полного содержимого
                content = self._get_document_content(doc_id)
                
                results.append({
                    'document_id': doc_id,
                    'file_name': file_name,
                    'category': category,
                    'content': content,  # Полный контент
                    'content_preview': content[:300] + "..." if content and len(content) > 300 else content,
                    'keywords': keywords.split(',') if keywords else [],
                    'search_type': 'text',
                    'relevance_score': 1.0,  # Базовый score для текстового поиска
                    'metadata': metadata
                })
                
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка текстового поиска: {e}")
            return []
            
    def _vector_only_search(self, query: str, limit: int) -> List[Dict]:
        """Только векторный поиск"""
        self.logger.info("🧠 Выполняется векторный поиск")
        
        try:
            # Получение результатов векторного поиска
            vector_results = self.vector_search.search_similar(
                query, 
                top_k=limit * 2, 
                threshold=self.min_similarity
            )
            
            # Форматирование результатов
            results = []
            for result in vector_results:
                doc_id = result['document_id']
                similarity = result['similarity']
                metadata = result['metadata']
                
                # Получение полного содержимого
                content = self.vector_search.get_document_content(doc_id)
                
                # Получение дополнительной информации из базы
                doc_info = self._get_document_info(doc_id)
                
                results.append({
                    'document_id': doc_id,
                    'file_name': doc_info.get('file_name', 'Unknown'),
                    'category': doc_info.get('category', metadata.get('category', 'Unknown')),
                    'content': content,  # Полный контент вместо preview
                    'content_preview': metadata.get('preview', content[:300] + "..." if content and len(content) > 300 else content),
                    'keywords': [],
                    'search_type': 'vector',
                    'relevance_score': similarity,
                    'similarity': similarity,
                    'metadata': doc_info.get('metadata', '')
                })
                
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка векторного поиска: {e}")
            return []
            
    def _hybrid_search(self, query: str, limit: int) -> List[Dict]:
        """Гибридный поиск (текстовый + векторный)"""
        self.logger.info("🔄 Выполняется гибридный поиск")
        
        try:
            # Параллельное выполнение поисков
            text_results = self._text_only_search(query, limit * 2)
            vector_results = self._vector_only_search(query, limit * 2)
            
            # Объединение и ранжирование результатов
            combined_results = self._merge_and_rank_results(
                text_results, 
                vector_results, 
                query
            )
            
            return combined_results[:limit]
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка гибридного поиска: {e}")
            return []
            
    def _merge_and_rank_results(self, text_results: List[Dict], vector_results: List[Dict], query: str) -> List[Dict]:
        """Объединение и ранжирование результатов"""
        try:
            # Создание словаря для объединения результатов
            combined = {}
            
            # Добавление текстовых результатов
            for result in text_results:
                doc_id = result['document_id']
                combined[doc_id] = {
                    **result,
                    'text_score': 1.0,
                    'vector_score': 0.0,
                    'combined_score': self.text_weight
                }
                
            # Добавление векторных результатов
            for result in vector_results:
                doc_id = result['document_id']
                if doc_id in combined:
                    # Обновление существующего результата
                    combined[doc_id]['vector_score'] = result['similarity']
                    combined[doc_id]['similarity'] = result['similarity']
                    combined[doc_id]['combined_score'] = (
                        self.text_weight * combined[doc_id]['text_score'] +
                        self.vector_weight * result['similarity']
                    )
                else:
                    # Добавление нового результата
                    combined[doc_id] = {
                        **result,
                        'text_score': 0.0,
                        'vector_score': result['similarity'],
                        'combined_score': self.vector_weight * result['similarity']
                    }
                    
            # Сортировка по комбинированному score
            sorted_results = sorted(
                combined.values(), 
                key=lambda x: x['combined_score'], 
                reverse=True
            )
            
            # Добавление информации о типе поиска
            for result in sorted_results:
                if result['text_score'] > 0 and result['vector_score'] > 0:
                    result['search_type'] = 'hybrid'
                elif result['text_score'] > 0:
                    result['search_type'] = 'text'
                else:
                    result['search_type'] = 'vector'
                    
            return sorted_results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка объединения результатов: {e}")
            return []
            
    def _get_document_content(self, doc_id: int) -> Optional[str]:
        """Получение содержимого документа"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT content FROM documents WHERE id = ?', (doc_id,))
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения содержимого документа {doc_id}: {e}")
            return None
            
    def _get_document_info(self, doc_id: int) -> Dict:
        """Получение информации о документе"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT file_name, category, metadata 
                FROM documents 
                WHERE id = ?
            ''', (doc_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'file_name': result[0],
                    'category': result[1],
                    'metadata': result[2]
                }
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения информации о документе {doc_id}: {e}")
            return {}
            
    def index_document(self, document_id: int, content: str, metadata: Dict = None) -> bool:
        """Индексация документа в векторном пространстве"""
        return self.vector_search.index_document(document_id, content, metadata)
        
    def build_vector_index(self) -> bool:
        """Построение векторного индекса"""
        return self.vector_search.build_faiss_index()
        
    def get_search_stats(self) -> Dict:
        """Получение статистики поиска"""
        text_stats = self.text_search.get_document_stats()
        vector_stats = self.vector_search.get_stats()
        
        return {
            'text_search': text_stats,
            'vector_search': vector_stats,
            'hybrid_enabled': True,
            'weights': {
                'text_weight': self.text_weight,
                'vector_weight': self.vector_weight
            }
        }
        
    def set_search_weights(self, text_weight: float, vector_weight: float):
        """Установка весов для гибридного поиска"""
        if text_weight + vector_weight != 1.0:
            self.logger.warning("⚠️ Сумма весов не равна 1.0, нормализация...")
            total = text_weight + vector_weight
            text_weight /= total
            vector_weight /= total
            
        self.text_weight = text_weight
        self.vector_weight = vector_weight
        
        self.logger.info(f"✅ Веса поиска обновлены: text={text_weight:.2f}, vector={vector_weight:.2f}")

if __name__ == "__main__":
    # Тестирование гибридного поиска
    engine = HybridSearchEngine()
    
    # Получение статистики
    stats = engine.get_search_stats()
    print("📊 Статистика гибридного поиска:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
        
    # Тестовый поиск
    test_query = "ПИД регулятор"
    print(f"\n🔍 Тестовый поиск: '{test_query}'")
    
    results = engine.search(test_query, limit=3)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['file_name']}")
        print(f"   Категория: {result['category']}")
        print(f"   Тип поиска: {result['search_type']}")
        print(f"   Релевантность: {result.get('relevance_score', 0):.3f}")
        if 'similarity' in result:
            print(f"   Сходство: {result['similarity']:.3f}")
        print(f"   Превью: {result['content_preview'][:100]}...")
