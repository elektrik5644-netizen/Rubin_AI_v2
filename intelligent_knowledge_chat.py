#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интегрированный модуль предложения знаний для Rubin AI
Автоматически предлагает новые знания в чате и запрашивает подтверждение
"""

import logging
import re
from typing import Dict, List, Optional, Any
from central_knowledge_base import get_knowledge_base, get_suggestion_engine

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentKnowledgeManager:
    """Интеллектуальный менеджер знаний с автоматическими предложениями"""
    
    def __init__(self):
        self.kb = get_knowledge_base()
        self.engine = get_suggestion_engine()
        self.active_suggestions = {}  # Активные предложения в сессии
        self.user_preferences = {
            'auto_suggest': True,
            'suggestion_threshold': 0.6,
            'max_suggestions_per_session': 3
        }
    
    def process_question(self, question: str, user_id: str = "default") -> Dict[str, Any]:
        """Обрабатывает вопрос и предлагает новые знания"""
        try:
            logger.info(f"🧠 Обработка вопроса: {question[:50]}...")
            
            # Сначала ищем в существующих знаниях
            existing_knowledge = self.kb.search_knowledge(question)
            
            # Анализируем вопрос на предмет новых знаний
            suggestion_data = self.engine.analyze_question(question)
            
            response = {
                'question': question,
                'existing_knowledge': existing_knowledge,
                'suggestion': None,
                'suggestion_message': None,
                'needs_user_input': False
            }
            
            # Если есть предложение и пользователь разрешил авто-предложения
            if suggestion_data and self.user_preferences['auto_suggest']:
                suggestion_id = suggestion_data['suggestion_id']
                
                # Проверяем лимит предложений в сессии
                if len(self.active_suggestions) < self.user_preferences['max_suggestions_per_session']:
                    self.active_suggestions[suggestion_id] = {
                        'user_id': user_id,
                        'question': question,
                        'suggestion': suggestion_data['suggestion'],
                        'timestamp': suggestion_data.get('timestamp')
                    }
                    
                    response['suggestion'] = suggestion_data
                    response['suggestion_message'] = self.engine.generate_suggestion_message(suggestion_data)
                    response['needs_user_input'] = True
                    
                    logger.info(f"💡 Предложено новое знание: {suggestion_data['suggestion']['title']}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки вопроса: {e}")
            return {
                'question': question,
                'existing_knowledge': [],
                'suggestion': None,
                'suggestion_message': None,
                'needs_user_input': False,
                'error': str(e)
            }
    
    def handle_user_feedback(self, feedback: str, user_id: str = "default") -> Dict[str, Any]:
        """Обрабатывает обратную связь пользователя"""
        try:
            feedback_lower = feedback.lower().strip()
            
            # Парсим команды
            if feedback_lower.startswith('approve'):
                suggestion_id = self._extract_id(feedback_lower)
                return self._approve_suggestion(suggestion_id, user_id)
            
            elif feedback_lower.startswith('reject'):
                suggestion_id = self._extract_id(feedback_lower)
                return self._reject_suggestion(suggestion_id, user_id)
            
            elif feedback_lower.startswith('edit'):
                suggestion_id = self._extract_id(feedback_lower)
                return self._edit_suggestion(suggestion_id, user_id, feedback)
            
            elif feedback_lower in ['да', 'yes', 'да, добавить', 'добавить']:
                return self._approve_latest_suggestion(user_id)
            
            elif feedback_lower in ['нет', 'no', 'не нужно', 'отклонить']:
                return self._reject_latest_suggestion(user_id)
            
            else:
                return {
                    'status': 'unknown_command',
                    'message': 'Не понял команду. Используйте: approve, reject, edit или просто "да"/"нет"'
                }
                
        except Exception as e:
            logger.error(f"❌ Ошибка обработки обратной связи: {e}")
            return {
                'status': 'error',
                'message': f'Ошибка: {str(e)}'
            }
    
    def _extract_id(self, feedback: str) -> Optional[int]:
        """Извлекает ID из команды"""
        try:
            # Ищем числа в строке
            numbers = re.findall(r'\d+', feedback)
            if numbers:
                return int(numbers[0])
            return None
        except:
            return None
    
    def _approve_suggestion(self, suggestion_id: int, user_id: str) -> Dict[str, Any]:
        """Подтверждает предложение"""
        try:
            if suggestion_id not in self.active_suggestions:
                return {
                    'status': 'error',
                    'message': f'Предложение {suggestion_id} не найдено в активных'
                }
            
            success = self.kb.approve_suggestion(suggestion_id, f"Подтверждено пользователем {user_id}")
            
            if success:
                suggestion_info = self.active_suggestions.pop(suggestion_id, {})
                return {
                    'status': 'approved',
                    'message': f'✅ Знание "{suggestion_info.get("suggestion", {}).get("title", "Неизвестно")}" добавлено в базу!',
                    'suggestion_id': suggestion_id
                }
            else:
                return {
                    'status': 'error',
                    'message': f'❌ Не удалось добавить предложение {suggestion_id}'
                }
                
        except Exception as e:
            logger.error(f"❌ Ошибка подтверждения предложения: {e}")
            return {
                'status': 'error',
                'message': f'Ошибка: {str(e)}'
            }
    
    def _reject_suggestion(self, suggestion_id: int, user_id: str) -> Dict[str, Any]:
        """Отклоняет предложение"""
        try:
            success = self.kb.reject_suggestion(suggestion_id, f"Отклонено пользователем {user_id}")
            
            if success:
                suggestion_info = self.active_suggestions.pop(suggestion_id, {})
                return {
                    'status': 'rejected',
                    'message': f'❌ Предложение "{suggestion_info.get("suggestion", {}).get("title", "Неизвестно")}" отклонено',
                    'suggestion_id': suggestion_id
                }
            else:
                return {
                    'status': 'error',
                    'message': f'❌ Не удалось отклонить предложение {suggestion_id}'
                }
                
        except Exception as e:
            logger.error(f"❌ Ошибка отклонения предложения: {e}")
            return {
                'status': 'error',
                'message': f'Ошибка: {str(e)}'
            }
    
    def _edit_suggestion(self, suggestion_id: int, user_id: str, feedback: str) -> Dict[str, Any]:
        """Редактирует предложение"""
        try:
            if suggestion_id not in self.active_suggestions:
                return {
                    'status': 'error',
                    'message': f'Предложение {suggestion_id} не найдено в активных'
                }
            
            # Извлекаем изменения из обратной связи
            # Это упрощенная версия - в реальности нужен более сложный парсинг
            suggestion_info = self.active_suggestions[suggestion_id]
            
            return {
                'status': 'edit_requested',
                'message': f'✏️ Редактирование предложения {suggestion_id}. Отправьте исправленную версию.',
                'current_suggestion': suggestion_info['suggestion'],
                'suggestion_id': suggestion_id
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка редактирования предложения: {e}")
            return {
                'status': 'error',
                'message': f'Ошибка: {str(e)}'
            }
    
    def _approve_latest_suggestion(self, user_id: str) -> Dict[str, Any]:
        """Подтверждает последнее предложение"""
        if not self.active_suggestions:
            return {
                'status': 'no_suggestions',
                'message': 'Нет активных предложений для подтверждения'
            }
        
        latest_id = max(self.active_suggestions.keys())
        return self._approve_suggestion(latest_id, user_id)
    
    def _reject_latest_suggestion(self, user_id: str) -> Dict[str, Any]:
        """Отклоняет последнее предложение"""
        if not self.active_suggestions:
            return {
                'status': 'no_suggestions',
                'message': 'Нет активных предложений для отклонения'
            }
        
        latest_id = max(self.active_suggestions.keys())
        return self._reject_suggestion(latest_id, user_id)
    
    def get_active_suggestions(self, user_id: str = "default") -> List[Dict]:
        """Получает активные предложения для пользователя"""
        user_suggestions = []
        for suggestion_id, suggestion_data in self.active_suggestions.items():
            if suggestion_data['user_id'] == user_id:
                user_suggestions.append({
                    'id': suggestion_id,
                    'question': suggestion_data['question'],
                    'suggestion': suggestion_data['suggestion']
                })
        
        return user_suggestions
    
    def get_knowledge_stats(self) -> Dict:
        """Получает статистику знаний"""
        return self.kb.get_knowledge_stats()
    
    def configure_preferences(self, preferences: Dict[str, Any]):
        """Настраивает предпочтения пользователя"""
        self.user_preferences.update(preferences)
        logger.info(f"✅ Настройки обновлены: {preferences}")

class RubinChatWithKnowledge:
    """Интегрированный чат Rubin AI с автоматическими предложениями знаний"""
    
    def __init__(self):
        self.knowledge_manager = IntelligentKnowledgeManager()
        self.chat_history = []
    
    def process_message(self, message: str, user_id: str = "default") -> str:
        """Обрабатывает сообщение пользователя"""
        try:
            # Проверяем, не является ли это обратной связью
            if self._is_feedback_message(message):
                feedback_result = self.knowledge_manager.handle_user_feedback(message, user_id)
                return self._format_feedback_response(feedback_result)
            
            # Обрабатываем как обычный вопрос
            result = self.knowledge_manager.process_question(message, user_id)
            
            # Формируем ответ
            response_parts = []
            
            # Добавляем найденные знания
            if result['existing_knowledge']:
                response_parts.append("📚 **Найденные знания:**")
                for knowledge in result['existing_knowledge'][:3]:  # Показываем первые 3
                    response_parts.append(f"• **{knowledge['title']}**: {knowledge['content']}")
                    if knowledge['formulas']:
                        response_parts.append(f"  *Формулы: {knowledge['formulas']}*")
            
            # Добавляем предложение нового знания
            if result['suggestion_message']:
                response_parts.append("\n" + result['suggestion_message'])
            
            # Если нет знаний и предложений
            if not result['existing_knowledge'] and not result['suggestion']:
                response_parts.append("🤔 Не нашел подходящих знаний в базе. Попробуйте переформулировать вопрос.")
            
            # Сохраняем в историю
            self.chat_history.append({
                'user_id': user_id,
                'message': message,
                'response': '\n'.join(response_parts),
                'timestamp': result.get('timestamp')
            })
            
            return '\n'.join(response_parts)
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки сообщения: {e}")
            return f"❌ Произошла ошибка: {str(e)}"
    
    def _is_feedback_message(self, message: str) -> bool:
        """Проверяет, является ли сообщение обратной связью"""
        feedback_keywords = [
            'approve', 'reject', 'edit', 'да', 'нет', 'yes', 'no',
            'добавить', 'отклонить', 'редактировать'
        ]
        
        message_lower = message.lower().strip()
        return any(keyword in message_lower for keyword in feedback_keywords)
    
    def _format_feedback_response(self, feedback_result: Dict) -> str:
        """Форматирует ответ на обратную связь"""
        status = feedback_result.get('status', 'unknown')
        message = feedback_result.get('message', 'Неизвестная ошибка')
        
        if status == 'approved':
            return f"✅ {message}\n\nТеперь это знание доступно для поиска!"
        elif status == 'rejected':
            return f"❌ {message}"
        elif status == 'edit_requested':
            return f"✏️ {message}"
        elif status == 'no_suggestions':
            return f"ℹ️ {message}"
        else:
            return f"⚠️ {message}"
    
    def get_chat_history(self, user_id: str = "default") -> List[Dict]:
        """Получает историю чата для пользователя"""
        return [msg for msg in self.chat_history if msg['user_id'] == user_id]
    
    def get_knowledge_stats(self) -> Dict:
        """Получает статистику знаний"""
        return self.knowledge_manager.get_knowledge_stats()

# Глобальный экземпляр чата
rubin_chat = None

def get_rubin_chat():
    """Получает глобальный экземпляр чата"""
    global rubin_chat
    if rubin_chat is None:
        rubin_chat = RubinChatWithKnowledge()
    return rubin_chat

if __name__ == "__main__":
    print("🚀 Тестирование интегрированного чата Rubin AI с предложениями знаний")
    
    # Инициализация
    chat = get_rubin_chat()
    
    # Тестирование
    test_messages = [
        "Что такое закон Ома?",
        "Как работает транзистор?",
        "да",  # Подтверждение предложения
        "Реши уравнение x^2 + 5x + 6 = 0",
        "Что такое ШИМ?",
        "approve 1"  # Подтверждение конкретного предложения
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{'='*60}")
        print(f"Сообщение {i}: {message}")
        print(f"{'='*60}")
        
        response = chat.process_message(message)
        print(f"Ответ: {response}")
    
    # Статистика
    stats = chat.get_knowledge_stats()
    print(f"\n📊 Финальная статистика:")
    print(f"• Всего знаний: {stats['total_facts']}")
    print(f"• Ожидающих подтверждения: {stats['pending_suggestions']}")
    print(f"• Подтвержденных: {stats['approved_suggestions']}")
    
    print(f"\n✅ Тестирование завершено!")










