#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced GAI Server for Rubin AI v2
Расширенный сервер генеративного ИИ
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
from datetime import datetime
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import requests
import base64
from PIL import Image
import io

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class GAIGenerator:
    """Генератор контента для Rubin AI v2"""
    
    def __init__(self):
        self.text_generators = {}
        self.image_generators = {}
        self.code_generators = {}
        self.load_models()
    
    def load_models(self):
        """Загрузка моделей генерации"""
        try:
            # Текстовые генераторы
            self.text_generators = {
                'technical': pipeline('text-generation', 
                                    model='microsoft/DialoGPT-medium',
                                    device=0 if torch.cuda.is_available() else -1),
                'mathematical': pipeline('text-generation',
                                        model='microsoft/CodeBERT',
                                        device=0 if torch.cuda.is_available() else -1)
            }
            
            # Генераторы кода
            self.code_generators = {
                'python': pipeline('text-generation',
                                  model='microsoft/CodeGPT-py',
                                  device=0 if torch.cuda.is_available() else -1),
                'javascript': pipeline('text-generation',
                                      model='microsoft/CodeGPT-js',
                                      device=0 if torch.cuda.is_available() else -1)
            }
            
            logger.info("✅ GAI модели загружены успешно")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки моделей: {e}")
            self.text_generators = {}
            self.code_generators = {}
    
    def generate_technical_explanation(self, topic, context=""):
        """Генерация технического объяснения"""
        try:
            if 'technical' not in self.text_generators:
                return self._fallback_explanation(topic)
            
            prompt = f"Объясни тему '{topic}' в контексте: {context}"
            
            result = self.text_generators['technical'](
                prompt,
                max_length=5000,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            return {
                'success': True,
                'explanation': result[0]['generated_text'],
                'topic': topic,
                'method': 'GAI_generation'
            }
            
        except Exception as e:
            logger.error(f"Ошибка генерации объяснения: {e}")
            return self._fallback_explanation(topic)
    
    def generate_code(self, requirements, language='python'):
        """Генерация кода по требованиям"""
        try:
            if language not in self.code_generators:
                return self._fallback_code(requirements, language)
            
            prompt = f"Напиши код на {language} для: {requirements}"
            
            result = self.code_generators[language](
                prompt,
                max_length=5000,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True
            )
            
            return {
                'success': True,
                'code': result[0]['generated_text'],
                'language': language,
                'requirements': requirements,
                'method': 'GAI_generation'
            }
            
        except Exception as e:
            logger.error(f"Ошибка генерации кода: {e}")
            return self._fallback_code(requirements, language)
    
    def generate_mathematical_solution(self, problem):
        """Генерация математического решения"""
        try:
            if 'mathematical' not in self.text_generators:
                return self._fallback_math_solution(problem)
            
            prompt = f"Реши математическую задачу: {problem}"
            
            result = self.text_generators['mathematical'](
                prompt,
                max_length=5000,
                num_return_sequences=1,
                temperature=0.5,
                do_sample=True
            )
            
            return {
                'success': True,
                'solution': result[0]['generated_text'],
                'problem': problem,
                'method': 'GAI_generation'
            }
            
        except Exception as e:
            logger.error(f"Ошибка генерации решения: {e}")
            return self._fallback_math_solution(problem)
    
    def generate_diagram_description(self, concept):
        """Генерация описания диаграммы"""
        try:
            # Простая генерация описания для создания диаграмм
            descriptions = {
                'electrical_circuit': f"Электрическая схема для {concept}",
                'flowchart': f"Блок-схема процесса {concept}",
                'architecture': f"Архитектурная диаграмма {concept}",
                'mathematical_graph': f"График функции {concept}"
            }
            
            return {
                'success': True,
                'description': descriptions.get(concept, f"Диаграмма для {concept}"),
                'concept': concept,
                'method': 'template_generation'
            }
            
        except Exception as e:
            logger.error(f"Ошибка генерации описания: {e}")
            return {'success': False, 'error': str(e)}
    
    def _fallback_explanation(self, topic):
        """Fallback объяснение"""
        return {
            'success': True,
            'explanation': f"Тема '{topic}' требует детального изучения. Рекомендую обратиться к специализированной литературе.",
            'topic': topic,
            'method': 'fallback'
        }
    
    def _fallback_code(self, requirements, language):
        """Fallback код"""
        return {
            'success': True,
            'code': f"# {requirements}\n# Требуется дополнительная разработка\nprint('Hello, {language}!')",
            'language': language,
            'requirements': requirements,
            'method': 'fallback'
        }
    
    def _fallback_math_solution(self, problem):
        """Fallback математическое решение"""
        return {
            'success': True,
            'solution': f"Задача: {problem}\nРешение требует дополнительного анализа.",
            'problem': problem,
            'method': 'fallback'
        }

# Инициализация генератора
gai_generator = GAIGenerator()

@app.route('/api/gai/generate_text', methods=['POST'])
def generate_text():
    """Генерация текста"""
    try:
        data = request.get_json()
        topic = data.get('topic', '')
        context = data.get('context', '')
        generation_type = data.get('type', 'technical')
        
        if not topic:
            return jsonify({'error': 'Тема не может быть пустой'}), 400
        
        if generation_type == 'technical':
            result = gai_generator.generate_technical_explanation(topic, context)
        elif generation_type == 'mathematical':
            result = gai_generator.generate_mathematical_solution(topic)
        else:
            return jsonify({'error': f'Неизвестный тип генерации: {generation_type}'}), 400
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка генерации текста: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/gai/generate_code', methods=['POST'])
def generate_code():
    """Генерация кода"""
    try:
        data = request.get_json()
        requirements = data.get('requirements', '')
        language = data.get('language', 'python')
        
        if not requirements:
            return jsonify({'error': 'Требования не могут быть пустыми'}), 400
        
        result = gai_generator.generate_code(requirements, language)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка генерации кода: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/gai/generate_diagram', methods=['POST'])
def generate_diagram():
    """Генерация описания диаграммы"""
    try:
        data = request.get_json()
        concept = data.get('concept', '')
        
        if not concept:
            return jsonify({'error': 'Концепция не может быть пустой'}), 400
        
        result = gai_generator.generate_diagram_description(concept)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка генерации диаграммы: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Проверка здоровья GAI сервера"""
    return jsonify({
        'service': 'Enhanced GAI Server',
        'status': 'online',
        'version': '1.0',
        'models_loaded': {
            'text_generators': len(gai_generator.text_generators),
            'code_generators': len(gai_generator.code_generators),
            'image_generators': len(gai_generator.image_generators)
        },
        'capabilities': ['text_generation', 'code_generation', 'diagram_generation']
    })

@app.route('/api/gai/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        'service': 'Enhanced GAI Server',
        'status': 'online',
        'version': '1.0',
        'models_loaded': {
            'text_generators': len(gai_generator.text_generators),
            'code_generators': len(gai_generator.code_generators),
            'image_generators': len(gai_generator.image_generators)
        },
        'gpu_available': torch.cuda.is_available(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/gai/status', methods=['GET'])
def get_status():
    """Статус сервера"""
    return jsonify({
        'service': 'Enhanced GAI Server',
        'status': 'running',
        'port': 8104,
        'endpoints': [
            '/api/gai/generate_text',
            '/api/gai/generate_code',
            '/api/gai/generate_diagram',
            '/api/gai/health',
            '/api/gai/status'
        ],
        'capabilities': [
            'Генерация технических объяснений',
            'Генерация программного кода',
            'Генерация математических решений',
            'Генерация описаний диаграмм',
            'Мультимодальная генерация'
        ],
        'models': {
            'text': list(gai_generator.text_generators.keys()),
            'code': list(gai_generator.code_generators.keys()),
            'image': list(gai_generator.image_generators.keys())
        }
    })

if __name__ == '__main__':
    print("🎨 Enhanced GAI Server запущен")
    print("URL: http://localhost:8104")
    print("Endpoints:")
    print("  - POST /api/gai/generate_text - Генерация текста")
    print("  - POST /api/gai/generate_code - Генерация кода")
    print("  - POST /api/gai/generate_diagram - Генерация диаграмм")
    print("  - GET /api/gai/health - Проверка здоровья")
    print("  - GET /api/gai/status - Статус сервера")
    app.run(host='0.0.0.0', port=8104, debug=False)
