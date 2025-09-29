#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интеграция Центральной Базы Знаний с Smart Dispatcher
"""

import requests
import logging
from typing import Dict, Any, Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeIntegration:
    """Интеграция с Центральной Базой Знаний"""
    
    def __init__(self, knowledge_api_url: str = "http://localhost:8093"):
        self.api_url = knowledge_api_url
        self.session = requests.Session()
        self.session.timeout = 5
    
    def enhance_response_with_knowledge(self, question: str, original_response: str, 
                                      user_id: str = "default") -> Dict[str, Any]:
        """Улучшает ответ знаниями из базы"""
        try:
            logger.info(f"🧠 Улучшение ответа знаниями для: {question[:50]}...")
            
            # Отправляем вопрос в систему знаний
            response = self.session.post(f"{self.api_url}/api/knowledge/chat", 
                                       json={
                                           'message': question,
                                           'user_id': user_id
                                       })
            
            if response.status_code == 200:
                knowledge_data = response.json()
                
                # Формируем улучшенный ответ
                enhanced_response = {
                    'original_response': original_response,
                    'knowledge_response': knowledge_data['response'],
                    'has_suggestions': knowledge_data['active_suggestions'] > 0,
                    'suggestions': knowledge_data['suggestions'],
                    'enhanced': True
                }
                
                # Если есть предложения знаний, добавляем их к ответу
                if knowledge_data['active_suggestions'] > 0:
                    enhanced_response['needs_user_confirmation'] = True
                    enhanced_response['suggestion_message'] = knowledge_data['response']
                
                logger.info(f"✅ Ответ улучшен знаниями")
                return enhanced_response
            
            else:
                logger.warning(f"⚠️ Ошибка API знаний: {response.status_code}")
                return {
                    'original_response': original_response,
                    'knowledge_response': None,
                    'has_suggestions': False,
                    'enhanced': False,
                    'error': f"API error: {response.status_code}"
                }
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Не удалось подключиться к API знаний: {e}")
            return {
                'original_response': original_response,
                'knowledge_response': None,
                'has_suggestions': False,
                'enhanced': False,
                'error': f"Connection error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"❌ Ошибка улучшения ответа: {e}")
            return {
                'original_response': original_response,
                'knowledge_response': None,
                'has_suggestions': False,
                'enhanced': False,
                'error': str(e)
            }
    
    def search_knowledge(self, query: str, category: str = None) -> List[Dict]:
        """Поиск в базе знаний"""
        try:
            params = {'q': query}
            if category:
                params['category'] = category
            
            response = self.session.get(f"{self.api_url}/api/knowledge/search", 
                                      params=params)
            
            if response.status_code == 200:
                data = response.json()
                return data['results']
            else:
                logger.warning(f"⚠️ Ошибка поиска знаний: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Ошибка поиска знаний: {e}")
            return []
    
    def handle_knowledge_feedback(self, feedback: str, user_id: str = "default") -> Dict[str, Any]:
        """Обрабатывает обратную связь по знаниям"""
        try:
            response = self.session.post(f"{self.api_url}/api/knowledge/chat", 
                                       json={
                                           'message': feedback,
                                           'user_id': user_id
                                       })
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'error': f"API error: {response.status_code}",
                    'response': 'Не удалось обработать обратную связь'
                }
                
        except Exception as e:
            logger.error(f"❌ Ошибка обработки обратной связи: {e}")
            return {
                'error': str(e),
                'response': 'Ошибка обработки обратной связи'
            }
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Получает статистику базы знаний"""
        try:
            response = self.session.get(f"{self.api_url}/api/knowledge/stats")
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f"API error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики: {e}")
            return {'error': str(e)}
    
    def is_knowledge_api_available(self) -> bool:
        """Проверяет доступность API знаний"""
        try:
            response = self.session.get(f"{self.api_url}/api/knowledge/health", timeout=2)
            return response.status_code == 200
        except:
            return False

def enhance_smart_dispatcher_response(question: str, original_response: str, 
                                    user_id: str = "default") -> str:
    """Улучшает ответ Smart Dispatcher знаниями"""
    try:
        knowledge_integration = KnowledgeIntegration()
        
        if not knowledge_integration.is_knowledge_api_available():
            logger.warning("⚠️ API знаний недоступен, возвращаем оригинальный ответ")
            return original_response
        
        # Улучшаем ответ знаниями
        enhanced = knowledge_integration.enhance_response_with_knowledge(
            question, original_response, user_id
        )
        
        if enhanced['enhanced'] and enhanced['knowledge_response']:
            # Объединяем оригинальный ответ с знаниями
            if enhanced['has_suggestions']:
                # Если есть предложения, показываем их
                return f"{original_response}\n\n{enhanced['suggestion_message']}"
            else:
                # Если есть найденные знания, добавляем их
                return f"{original_response}\n\n📚 **Дополнительные знания:**\n{enhanced['knowledge_response']}"
        
        return original_response
        
    except Exception as e:
        logger.error(f"❌ Ошибка улучшения ответа: {e}")
        return original_response

# Пример интеграции с существующим Smart Dispatcher
def enhanced_smart_dispatcher_handler(question: str, user_id: str = "default") -> Dict[str, Any]:
    """Улучшенный обработчик Smart Dispatcher с интеграцией знаний"""
    try:
        # Здесь был бы вызов оригинального Smart Dispatcher
        # Для демонстрации используем заглушку
        original_response = f"Оригинальный ответ на вопрос: {question}"
        
        # Улучшаем ответ знаниями
        knowledge_integration = KnowledgeIntegration()
        
        if knowledge_integration.is_knowledge_api_available():
            enhanced = knowledge_integration.enhance_response_with_knowledge(
                question, original_response, user_id
            )
            
            return {
                'response': enhanced.get('knowledge_response', original_response),
                'original_response': original_response,
                'enhanced_with_knowledge': enhanced['enhanced'],
                'has_suggestions': enhanced.get('has_suggestions', False),
                'suggestions': enhanced.get('suggestions', []),
                'needs_user_confirmation': enhanced.get('needs_user_confirmation', False)
            }
        else:
            return {
                'response': original_response,
                'original_response': original_response,
                'enhanced_with_knowledge': False,
                'has_suggestions': False,
                'suggestions': [],
                'needs_user_confirmation': False
            }
            
    except Exception as e:
        logger.error(f"❌ Ошибка обработки: {e}")
        return {
            'response': f"Ошибка обработки: {str(e)}",
            'original_response': "",
            'enhanced_with_knowledge': False,
            'has_suggestions': False,
            'suggestions': [],
            'needs_user_confirmation': False
        }

if __name__ == "__main__":
    print("🚀 Тестирование интеграции Центральной Базы Знаний")
    
    # Тестирование
    knowledge_integration = KnowledgeIntegration()
    
    # Проверяем доступность API
    if knowledge_integration.is_knowledge_api_available():
        print("✅ API знаний доступен")
        
        # Тестируем улучшение ответа
        test_questions = [
            "Что такое закон Ома?",
            "Как работает транзистор?",
            "Реши уравнение x^2 + 5x + 6 = 0"
        ]
        
        for question in test_questions:
            print(f"\n{'='*60}")
            print(f"Вопрос: {question}")
            print(f"{'='*60}")
            
            enhanced = knowledge_integration.enhance_response_with_knowledge(
                question, f"Оригинальный ответ на: {question}"
            )
            
            print(f"Улучшен: {enhanced['enhanced']}")
            print(f"Есть предложения: {enhanced['has_suggestions']}")
            if enhanced['knowledge_response']:
                print(f"Ответ с знаниями: {enhanced['knowledge_response'][:100]}...")
        
        # Статистика
        stats = knowledge_integration.get_knowledge_stats()
        print(f"\n📊 Статистика знаний: {stats}")
        
    else:
        print("❌ API знаний недоступен")
    
    print(f"\n✅ Тестирование завершено!")










