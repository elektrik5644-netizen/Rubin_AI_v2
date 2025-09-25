#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Последовательный поисковик для Rubin AI v2.0
Цепочка: Векторный поиск → Текстовый поиск → Диспетчер → LLM
"""

import logging
import time
from typing import List, Dict, Optional, Tuple
from vector_search import VectorSearchEngine
from document_loader import DocumentLoader

class SequentialSearchEngine:
    """Последовательный поисковик с цепочкой обработки"""
    
    def __init__(self, db_path="rubin_ai_v2.db"):
        self.db_path = db_path
        self.setup_logging()
        
        # Инициализация компонентов
        self.text_search = DocumentLoader(db_path)
        self.vector_search = VectorSearchEngine(db_path)
        
        # Настройки поиска
        self.vector_candidates_limit = 20  # Количество кандидатов от векторного поиска
        self.final_results_limit = 5       # Финальное количество результатов
        self.min_similarity = 0.6          # Минимальное сходство для векторного поиска
        
    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('sequential_search.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Последовательный поиск с цепочкой обработки
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            
        Returns:
            Список найденных документов с метаданными
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🔄 Начинаем последовательный поиск для запроса: '{query}'")
            
            # Этап 1: Векторный поиск находит релевантные файлы
            vector_candidates = self._stage1_vector_search(query)
            self.logger.info(f"📁 Этап 1: Векторный поиск нашел {len(vector_candidates)} кандидатов")
            
            if not vector_candidates:
                self.logger.warning("❌ Векторный поиск не нашел кандидатов")
                # Используем fallback поиск
                fallback_results = self._fallback_text_search(query)
                if fallback_results:
                    # Применяем этапы 2 и 3 к fallback результатам
                    text_results = self._stage2_text_search(query, fallback_results)
                    final_results = self._stage3_ranking(query, text_results, limit)
                    
                    elapsed = time.time() - start_time
                    self.logger.info(f"✅ Последовательный поиск завершен за {elapsed:.3f}с")
                    return final_results
                else:
                    return []
            
            # Этап 2: Текстовый поиск извлекает конкретный контент
            text_results = self._stage2_text_search(query, vector_candidates)
            self.logger.info(f"📝 Этап 2: Текстовый поиск обработал {len(text_results)} документов")
            
            if not text_results:
                self.logger.warning("❌ Текстовый поиск не нашел релевантного контента")
                return []
            
            # Этап 3: Ранжирование и фильтрация результатов
            final_results = self._stage3_ranking(query, text_results, limit)
            self.logger.info(f"🎯 Этап 3: Финальный отбор {len(final_results)} результатов")
            
            search_time = time.time() - start_time
            self.logger.info(f"✅ Последовательный поиск завершен за {search_time:.3f}с")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка последовательного поиска: {e}")
            return []
    
    def _stage1_vector_search(self, query: str) -> List[Dict]:
        """
        Этап 1: Векторный поиск находит релевантные файлы
        
        Returns:
            Список кандидатов с метаданными файлов
        """
        self.logger.info("🧠 Этап 1: Векторный поиск файлов...")
        
        try:
            # Получение результатов векторного поиска
            vector_results = self.vector_search.search_similar(
                query, 
                top_k=self.vector_candidates_limit, 
                threshold=self.min_similarity
            )
            
            # Форматирование кандидатов
            candidates = []
            for result in vector_results:
                doc_id = result['document_id']
                similarity = result['similarity']
                metadata = result['metadata']
                
                # Получение информации о файле
                doc_info = self._get_document_info(doc_id)
                
                candidates.append({
                    'document_id': doc_id,
                    'file_name': doc_info.get('file_name', 'Unknown'),
                    'category': doc_info.get('category', metadata.get('category', 'Unknown')),
                    'similarity': similarity,
                    'metadata': doc_info.get('metadata', ''),
                    'stage': 'vector_candidate'
                })
            
            self.logger.info(f"📁 Найдено {len(candidates)} файлов-кандидатов")
            
            # Если векторный поиск не нашел кандидатов, используем fallback
            if len(candidates) == 0:
                self.logger.warning("❌ Векторный поиск не нашел кандидатов, используем fallback")
                return self._fallback_text_search(query)
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка векторного поиска: {e}")
            # Fallback: используем текстовый поиск для всех документов
            return self._fallback_text_search(query)
    
    def _stage2_text_search(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Этап 2: Текстовый поиск извлекает конкретный контент из найденных файлов
        
        Args:
            query: Поисковый запрос
            candidates: Список кандидатов от векторного поиска
            
        Returns:
            Список документов с извлеченным контентом
        """
        self.logger.info("📝 Этап 2: Текстовый поиск контента...")
        
        try:
            results = []
            
            for candidate in candidates:
                doc_id = candidate['document_id']
                file_name = candidate['file_name']
                
                # Получение полного содержимого файла
                content = self._get_document_content(doc_id)
                
                if not content:
                    self.logger.warning(f"⚠️ Не удалось получить содержимое файла {file_name}")
                    continue
                
                # Поиск релевантных фрагментов в содержимом
                relevant_fragments = self._extract_relevant_fragments(query, content)
                
                if relevant_fragments:
                    # Вычисление релевантности текстового поиска
                    text_relevance = self._calculate_text_relevance(query, relevant_fragments)
                    
                    results.append({
                        'document_id': doc_id,
                        'file_name': file_name,
                        'category': candidate['category'],
                        'content_preview': ' '.join(relevant_fragments[:3]),  # Первые 3 фрагмента
                        'full_content': content,
                        'relevant_fragments': relevant_fragments,
                        'vector_similarity': candidate['similarity'],
                        'text_relevance': text_relevance,
                        'combined_score': self._calculate_combined_score(
                            candidate['similarity'], 
                            text_relevance
                        ),
                        'metadata': candidate['metadata'],
                        'stage': 'text_processed'
                    })
                    
                    self.logger.debug(f"✅ Обработан файл {file_name}: {len(relevant_fragments)} фрагментов")
                else:
                    self.logger.debug(f"⚠️ В файле {file_name} не найдено релевантных фрагментов")
            
            self.logger.info(f"📝 Обработано {len(results)} файлов с релевантным контентом")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка текстового поиска: {e}")
            return []
    
    def _stage3_ranking(self, query: str, text_results: List[Dict], limit: int) -> List[Dict]:
        """
        Этап 3: Ранжирование и финальный отбор результатов
        
        Args:
            query: Поисковый запрос
            text_results: Результаты текстового поиска
            limit: Максимальное количество результатов
            
        Returns:
            Финальный список ранжированных результатов
        """
        self.logger.info("🎯 Этап 3: Ранжирование результатов...")
        
        try:
            # Сортировка по комбинированному score
            sorted_results = sorted(
                text_results, 
                key=lambda x: x['combined_score'], 
                reverse=True
            )
            
            # Ограничение количества результатов
            final_results = sorted_results[:limit]
            
            # Добавление финальной информации
            for i, result in enumerate(final_results):
                result['rank'] = i + 1
                result['stage'] = 'final'
                result['search_type'] = 'sequential'
                
                # Удаление полного контента для экономии памяти
                if 'full_content' in result:
                    del result['full_content']
            
            self.logger.info(f"🎯 Отобрано {len(final_results)} финальных результатов")
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка ранжирования: {e}")
            return []
    
    def _extract_relevant_fragments(self, query: str, content: str) -> List[str]:
        """Извлечение релевантных фрагментов из контента"""
        try:
            import re
            
            # Разбиваем контент на предложения
            sentences = re.split(r'[.!?]+', content)
            
            # Ключевые слова из запроса
            query_words = [word.lower() for word in query.split() if len(word) > 2]
            
            relevant_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:  # Пропускаем слишком короткие предложения
                    continue
                
                sentence_lower = sentence.lower()
                # Проверяем наличие ключевых слов
                if any(word in sentence_lower for word in query_words):
                    relevant_sentences.append(sentence)
            
            # Если не найдено релевантных предложений, берем первые
            if not relevant_sentences:
                relevant_sentences = [s.strip() for s in sentences[:5] if len(s.strip()) > 10]
            
            return relevant_sentences[:10]  # Максимум 10 предложений
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка извлечения фрагментов: {e}")
            return []
    
    def _calculate_text_relevance(self, query: str, fragments: List[str]) -> float:
        """Вычисление релевантности текстового поиска"""
        try:
            if not fragments:
                return 0.0
            
            query_words = set(query.lower().split())
            total_score = 0.0
            
            for fragment in fragments:
                fragment_words = set(fragment.lower().split())
                common_words = query_words.intersection(fragment_words)
                
                if query_words:
                    fragment_score = len(common_words) / len(query_words)
                    total_score += fragment_score
            
            return min(total_score / len(fragments), 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка вычисления релевантности: {e}")
            return 0.0
    
    def _calculate_combined_score(self, vector_similarity: float, text_relevance: float) -> float:
        """Вычисление комбинированного score"""
        # Веса: 60% векторный поиск, 40% текстовый поиск
        vector_weight = 0.6
        text_weight = 0.4
        
        return (vector_weight * vector_similarity) + (text_weight * text_relevance)
    
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
                    'metadata': result[2] or ''
                }
            else:
                return {
                    'file_name': 'Unknown',
                    'category': 'Unknown',
                    'metadata': ''
                }
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения информации о документе {doc_id}: {e}")
            return {
                'file_name': 'Unknown',
                'category': 'Unknown',
                'metadata': ''
            }
    
    def _fallback_text_search(self, query: str) -> List[Dict]:
        """
        Fallback текстовый поиск когда векторный поиск недоступен
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Список кандидатов на основе текстового поиска
        """
        self.logger.info("🔄 Fallback: Текстовый поиск по всем документам...")
        
        try:
            # Получаем все документы из базы данных
            all_docs = self.text_search.get_all_documents()
            
            if not all_docs:
                self.logger.warning("⚠️ Нет документов в базе данных")
                return []
            
            candidates = []
            query_words = query.lower().split()
            
            for doc in all_docs:
                doc_id = doc['id']
                content = doc.get('content', '').lower()
                file_name = doc.get('file_name', 'Unknown')
                category = doc.get('category', 'Unknown')
                
                # Простой подсчет совпадений слов
                matches = sum(1 for word in query_words if word in content)
                
                if matches > 0:
                    # Вычисляем простую релевантность
                    relevance = matches / len(query_words)
                    
                    candidates.append({
                        'document_id': doc_id,
                        'file_name': file_name,
                        'category': category,
                        'similarity': relevance,
                        'metadata': doc.get('metadata', ''),
                        'stage': 'text_fallback'
                    })
            
            # Сортируем по релевантности
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Ограничиваем количество кандидатов
            candidates = candidates[:self.vector_candidates_limit]
            
            self.logger.info(f"📁 Fallback поиск нашел {len(candidates)} кандидатов")
            return candidates
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка fallback поиска: {e}")
            return []

    def get_search_statistics(self) -> Dict:
        """Получение статистики поиска"""
        return {
            'vector_candidates_limit': self.vector_candidates_limit,
            'final_results_limit': self.final_results_limit,
            'min_similarity': self.min_similarity,
            'search_type': 'sequential'
        }
