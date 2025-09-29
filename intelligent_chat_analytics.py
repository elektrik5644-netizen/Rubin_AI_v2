#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интегрированный чат с аналитикой ответов
Автоматически анализирует и улучшает ответы Rubin AI
"""

import logging
import requests
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from response_analytics import get_quality_controller
from enhanced_smart_dispatcher import get_enhanced_dispatcher
from intelligent_knowledge_chat import get_rubin_chat

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentChatWithAnalytics:
    """Интеллектуальный чат с аналитикой ответов"""
    
    def __init__(self):
        self.quality_controller = get_quality_controller()
        self.enhanced_dispatcher = get_enhanced_dispatcher()
        self.knowledge_chat = get_rubin_chat()
        self.chat_history = []
        self.analytics_history = []
        
        # Настройки
        self.auto_improvement_enabled = True
        self.quality_threshold = 0.6
        self.max_improvement_attempts = 2
    
    def process_question(self, question: str, user_id: str = "default") -> Dict[str, Any]:
        """Обрабатывает вопрос с полной аналитикой"""
        try:
            logger.info(f"💬 Обрабатываю вопрос: {question[:50]}...")
            
            # Шаг 1: Получаем базовый ответ через улучшенный диспетчер
            dispatcher_result = self.enhanced_dispatcher.route_question(question)
            
            if not dispatcher_result['success']:
                return self._create_error_response(question, dispatcher_result['response'])
            
            # Шаг 2: Анализируем качество ответа
            quality_result = self.quality_controller.process_response(
                question, 
                dispatcher_result['response'], 
                dispatcher_result['server_type']
            )
            
            # Шаг 3: Интегрируем с базой знаний
            knowledge_enhancement = self._get_knowledge_enhancement(question, user_id)
            
            # Шаг 4: Формируем финальный ответ
            final_response = self._create_final_response(
                quality_result, 
                knowledge_enhancement, 
                dispatcher_result
            )
            
            # Шаг 5: Сохраняем в историю
            self._save_to_history(question, final_response, quality_result, user_id)
            
            # Шаг 6: Возвращаем результат
            return {
                'question': question,
                'response': final_response,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'analytics': {
                    'quality_score': quality_result['analysis']['quality_score'],
                    'quality_status': quality_result['quality_status'],
                    'correction_applied': quality_result['correction_applied'],
                    'issues_found': len(quality_result['analysis']['issues']),
                    'suggestions_count': len(quality_result['analysis']['suggestions']),
                    'knowledge_enhanced': bool(knowledge_enhancement)
                },
                'metadata': {
                    'server_type': dispatcher_result['server_type'],
                    'complexity': dispatcher_result['complexity'],
                    'response_length': len(final_response),
                    'processing_time': time.time()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки вопроса: {e}")
            return self._create_error_response(question, f"Ошибка обработки: {str(e)}")
    
    def _create_error_response(self, question: str, error_message: str) -> Dict[str, Any]:
        """Создает ответ об ошибке"""
        return {
            'question': question,
            'response': f"❌ {error_message}\n\nПопробуйте переформулировать вопрос или обратитесь к администратору.",
            'user_id': 'system',
            'timestamp': datetime.now().isoformat(),
            'analytics': {
                'quality_score': 0.0,
                'quality_status': 'error',
                'correction_applied': False,
                'issues_found': 1,
                'suggestions_count': 0,
                'knowledge_enhanced': False
            },
            'metadata': {
                'server_type': 'error',
                'complexity': {'level': 'unknown'},
                'response_length': len(error_message),
                'processing_time': time.time()
            }
        }
    
    def _get_knowledge_enhancement(self, question: str, user_id: str) -> Optional[str]:
        """Получает улучшения от базы знаний"""
        try:
            knowledge_result = self.knowledge_chat.process_message(question, user_id)
            
            # Проверяем, содержит ли результат предложения новых знаний
            if 'предлагает добавить' in knowledge_result:
                return knowledge_result
            
            # Если это обычный ответ из базы знаний
            if len(knowledge_result) > 50 and 'Не нашел' not in knowledge_result:
                return knowledge_result
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка интеграции с базой знаний: {e}")
            return None
    
    def _create_final_response(self, quality_result: Dict[str, Any], 
                              knowledge_enhancement: Optional[str],
                              dispatcher_result: Dict[str, Any]) -> str:
        """Создает финальный ответ"""
        response_parts = []
        
        # Основной ответ (исправленный или оригинальный)
        main_response = quality_result['final_response']
        response_parts.append(main_response)
        
        # Добавляем информацию о качестве
        if quality_result['correction_applied']:
            response_parts.append(f"\n**🔧 АВТОМАТИЧЕСКОЕ УЛУЧШЕНИЕ:**")
            response_parts.append(f"Ответ был автоматически улучшен для повышения качества.")
            response_parts.append(f"Балл качества: {quality_result['analysis']['quality_score']}/1.0")
        
        # Добавляем улучшения от базы знаний
        if knowledge_enhancement:
            response_parts.append(f"\n**🧠 ДОПОЛНЕНИЯ ИЗ БАЗЫ ЗНАНИЙ:**")
            response_parts.append(knowledge_enhancement)
        
        # Добавляем метаинформацию о качестве
        if quality_result['analysis']['issues']:
            response_parts.append(f"\n**📊 АНАЛИЗ КАЧЕСТВА:**")
            response_parts.append(f"• Статус качества: {quality_result['quality_status']}")
            response_parts.append(f"• Найдено проблем: {len(quality_result['analysis']['issues'])}")
            
            if quality_result['analysis']['suggestions']:
                response_parts.append(f"• Предложения по улучшению:")
                for suggestion in quality_result['analysis']['suggestions'][:3]:  # Показываем первые 3
                    response_parts.append(f"  - {suggestion}")
        
        return '\n'.join(response_parts)
    
    def _save_to_history(self, question: str, response: str, 
                        quality_result: Dict[str, Any], user_id: str):
        """Сохраняет в историю чата и аналитики"""
        # История чата
        chat_entry = {
            'question': question,
            'response': response,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'quality_score': quality_result['analysis']['quality_score']
        }
        self.chat_history.append(chat_entry)
        
        # История аналитики
        analytics_entry = {
            'question': question,
            'quality_analysis': quality_result['analysis'],
            'correction_applied': quality_result['correction_applied'],
            'quality_status': quality_result['quality_status'],
            'timestamp': datetime.now().isoformat()
        }
        self.analytics_history.append(analytics_entry)
    
    def get_chat_history(self, user_id: str = "default", limit: int = 10) -> List[Dict]:
        """Получает историю чата"""
        user_history = [entry for entry in self.chat_history if entry['user_id'] == user_id]
        return user_history[-limit:]
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Получает сводку аналитики"""
        if not self.analytics_history:
            return {'message': 'Нет данных для анализа'}
        
        total_responses = len(self.analytics_history)
        avg_quality = sum(entry['quality_analysis']['quality_score'] 
                         for entry in self.analytics_history) / total_responses
        
        corrections_applied = sum(1 for entry in self.analytics_history 
                                if entry['correction_applied'])
        
        quality_distribution = {}
        for entry in self.analytics_history:
            status = entry['quality_status']
            quality_distribution[status] = quality_distribution.get(status, 0) + 1
        
        return {
            'total_responses': total_responses,
            'average_quality_score': round(avg_quality, 2),
            'corrections_applied': corrections_applied,
            'correction_rate': round(corrections_applied / total_responses * 100, 1),
            'quality_distribution': quality_distribution,
            'recent_trend': self._calculate_recent_trend()
        }
    
    def _calculate_recent_trend(self) -> str:
        """Рассчитывает тренд качества за последние ответы"""
        if len(self.analytics_history) < 5:
            return 'insufficient_data'
        
        recent_scores = [entry['quality_analysis']['quality_score'] 
                        for entry in self.analytics_history[-5:]]
        
        if recent_scores[-1] > recent_scores[0]:
            return 'improving'
        elif recent_scores[-1] < recent_scores[0]:
            return 'declining'
        else:
            return 'stable'
    
    def configure_settings(self, auto_improvement: bool = None, 
                          quality_threshold: float = None):
        """Настраивает параметры чата"""
        if auto_improvement is not None:
            self.auto_improvement_enabled = auto_improvement
        if quality_threshold is not None:
            self.quality_threshold = quality_threshold
            self.quality_controller.configure_quality_settings(threshold=quality_threshold)
        
        logger.info(f"⚙️ Настройки чата обновлены: auto_improvement={self.auto_improvement_enabled}, threshold={self.quality_threshold}")

# Глобальный экземпляр
intelligent_chat = None

def get_intelligent_chat():
    """Получает глобальный экземпляр интеллектуального чата"""
    global intelligent_chat
    if intelligent_chat is None:
        intelligent_chat = IntelligentChatWithAnalytics()
    return intelligent_chat

if __name__ == "__main__":
    print("🚀 Тестирование интеллектуального чата с аналитикой")
    
    chat = get_intelligent_chat()
    
    # Тестовые вопросы
    test_questions = [
        "Что такое закон Ома?",
        "Реши уравнение x^2 + 5x + 6 = 0",
        "Как работает транзистор?",
        "Напиши программу на Python для сортировки",
        "Объясни принцип работы ПЛК"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"Вопрос {i}: {question}")
        print(f"{'='*80}")
        
        result = chat.process_question(question)
        
        print(f"📊 Качество: {result['analytics']['quality_score']}")
        print(f"🎯 Статус: {result['analytics']['quality_status']}")
        print(f"🔧 Исправлен: {result['analytics']['correction_applied']}")
        print(f"⚠️ Проблем: {result['analytics']['issues_found']}")
        print(f"💡 Предложений: {result['analytics']['suggestions_count']}")
        print(f"🧠 Улучшен знаниями: {result['analytics']['knowledge_enhanced']}")
        print(f"📏 Длина: {result['metadata']['response_length']} символов")
        
        print(f"\n📝 Ответ:")
        print(result['response'][:300] + "..." if len(result['response']) > 300 else result['response'])
        
        time.sleep(1)
    
    # Сводка аналитики
    print(f"\n{'='*80}")
    print("📊 СВОДКА АНАЛИТИКИ")
    print(f"{'='*80}")
    
    summary = chat.get_analytics_summary()
    print(f"• Всего ответов: {summary['total_responses']}")
    print(f"• Среднее качество: {summary['average_quality_score']}")
    print(f"• Исправлений применено: {summary['corrections_applied']}")
    print(f"• Процент исправлений: {summary['correction_rate']}%")
    print(f"• Тренд качества: {summary['recent_trend']}")
    print(f"• Распределение качества: {summary['quality_distribution']}")
    
    print(f"\n✅ Тестирование завершено!")










