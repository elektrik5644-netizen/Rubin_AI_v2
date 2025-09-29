#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль анализа графиков для Rubin AI
Интеграция с MCSetup для анализа данных приводов
"""

from flask import Flask, request, jsonify
import logging
import json
import requests
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Конфигурация
MCSETUP_BRIDGE_URL = "http://localhost:8096"
RUBIN_SMART_DISPATCHER_URL = "http://localhost:8080"

class GraphAnalyzer:
    """Анализатор графиков для приводов"""
    
    def __init__(self):
        self.analysis_cache = {}
        self.graph_patterns = {
            'position': ['pos', 'position', 'позиц', 'координат'],
            'velocity': ['vel', 'velocity', 'скорост', 'speed'],
            'current': ['cur', 'current', 'ток', 'ampere'],
            'torque': ['torque', 'момент', 'force'],
            'temperature': ['temp', 'температур', 'heat']
        }
    
    def analyze_motor_trends(self, motor_data):
        """Анализирует тренды моторов"""
        try:
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'trend_analysis': {},
                'anomalies': [],
                'recommendations': []
            }
            
            for motor_name, data in motor_data.items():
                motor_analysis = {
                    'data_points': len(data.get('sources', [])),
                    'gather_period': int(data.get('gather_period', 10)),
                    'performance_score': 0
                }
                
                # Анализ частоты сбора данных
                gather_period = motor_analysis['gather_period']
                if gather_period <= 5:
                    motor_analysis['performance_score'] += 30
                    analysis['recommendations'].append(f"Мотор {motor_name}: Отличная частота сбора данных")
                elif gather_period <= 20:
                    motor_analysis['performance_score'] += 20
                    analysis['recommendations'].append(f"Мотор {motor_name}: Хорошая частота сбора данных")
                else:
                    motor_analysis['performance_score'] += 10
                    analysis['recommendations'].append(f"Мотор {motor_name}: Низкая частота сбора данных")
                
                # Анализ количества источников
                sources_count = motor_analysis['data_points']
                if sources_count >= 6:
                    motor_analysis['performance_score'] += 25
                    analysis['recommendations'].append(f"Мотор {motor_name}: Полный мониторинг параметров")
                elif sources_count >= 3:
                    motor_analysis['performance_score'] += 15
                    analysis['recommendations'].append(f"Мотор {motor_name}: Базовый мониторинг")
                else:
                    motor_analysis['performance_score'] += 5
                    analysis['anomalies'].append(f"Мотор {motor_name}: Недостаточный мониторинг")
                
                # Анализ типов данных
                data_types = self._analyze_data_types(data.get('sources', []))
                motor_analysis['data_types'] = data_types
                
                if 'position' in data_types and 'velocity' in data_types and 'current' in data_types:
                    motor_analysis['performance_score'] += 25
                    analysis['recommendations'].append(f"Мотор {motor_name}: Полный набор данных для анализа")
                elif 'position' in data_types and 'velocity' in data_types:
                    motor_analysis['performance_score'] += 15
                    analysis['recommendations'].append(f"Мотор {motor_name}: Хороший набор данных")
                else:
                    motor_analysis['performance_score'] += 5
                    analysis['anomalies'].append(f"Мотор {motor_name}: Неполный набор данных")
                
                # Определение общего рейтинга
                if motor_analysis['performance_score'] >= 80:
                    motor_analysis['rating'] = 'excellent'
                elif motor_analysis['performance_score'] >= 60:
                    motor_analysis['rating'] = 'good'
                elif motor_analysis['performance_score'] >= 40:
                    motor_analysis['rating'] = 'fair'
                else:
                    motor_analysis['rating'] = 'poor'
                
                analysis['trend_analysis'][motor_name] = motor_analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа трендов: {e}")
            return {'error': str(e)}
    
    def _analyze_data_types(self, sources):
        """Анализирует типы данных в источниках"""
        data_types = set()
        
        for source in sources:
            name = source.get('name', '').lower()
            address = source.get('address', '').lower()
            
            for data_type, keywords in self.graph_patterns.items():
                if any(keyword in name or keyword in address for keyword in keywords):
                    data_types.add(data_type)
                    break
        
        return list(data_types)
    
    def generate_performance_report(self, analysis_data):
        """Генерирует отчет о производительности"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_motors': len(analysis_data.get('trend_analysis', {})),
                    'excellent_motors': 0,
                    'good_motors': 0,
                    'fair_motors': 0,
                    'poor_motors': 0
                },
                'detailed_analysis': {},
                'recommendations': [],
                'warnings': []
            }
            
            # Подсчет рейтингов
            for motor_name, motor_data in analysis_data.get('trend_analysis', {}).items():
                rating = motor_data.get('rating', 'unknown')
                if rating == 'excellent':
                    report['summary']['excellent_motors'] += 1
                elif rating == 'good':
                    report['summary']['good_motors'] += 1
                elif rating == 'fair':
                    report['summary']['fair_motors'] += 1
                elif rating == 'poor':
                    report['summary']['poor_motors'] += 1
                
                report['detailed_analysis'][motor_name] = {
                    'rating': rating,
                    'performance_score': motor_data.get('performance_score', 0),
                    'data_types': motor_data.get('data_types', []),
                    'gather_period': motor_data.get('gather_period', 0)
                }
            
            # Генерация рекомендаций
            if report['summary']['poor_motors'] > 0:
                report['warnings'].append(f"⚠️ {report['summary']['poor_motors']} моторов требуют внимания")
            
            if report['summary']['excellent_motors'] > report['summary']['total_motors'] / 2:
                report['recommendations'].append("✅ Большинство моторов работают отлично")
            else:
                report['recommendations'].append("💡 Рекомендуется оптимизировать настройки моторов")
            
            # Добавляем общие рекомендации
            report['recommendations'].extend([
                "📊 Регулярно анализируйте графики производительности",
                "⚙️ Оптимизируйте частоту сбора данных",
                "🔧 Настройте аварийные пороги",
                "📈 Используйте прогнозную аналитику"
            ])
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации отчета: {e}")
            return {'error': str(e)}
    
    def create_visualization(self, motor_data):
        """Создает визуализацию данных моторов"""
        try:
            # Создаем график производительности моторов
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            motor_names = list(motor_data.keys())
            performance_scores = [motor_data[name].get('performance_score', 0) for name in motor_names]
            gather_periods = [motor_data[name].get('gather_period', 10) for name in motor_names]
            
            # График производительности
            bars = ax1.bar(motor_names, performance_scores, color=['green' if score >= 80 else 'orange' if score >= 60 else 'red' for score in performance_scores])
            ax1.set_title('Производительность моторов', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Оценка производительности')
            ax1.set_ylim(0, 100)
            
            # Добавляем значения на столбцы
            for bar, score in zip(bars, performance_scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{score}', ha='center', va='bottom', fontweight='bold')
            
            # График частоты сбора данных
            ax2.bar(motor_names, gather_periods, color='skyblue', alpha=0.7)
            ax2.set_title('Частота сбора данных моторов', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Период сбора (мс)')
            ax2.set_xlabel('Моторы')
            
            # Добавляем значения на столбцы
            for i, period in enumerate(gather_periods):
                ax2.text(i, period + 0.5, f'{period}мс', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Конвертируем в base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_base64
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания визуализации: {e}")
            return None

# Инициализация анализатора
graph_analyzer = GraphAnalyzer()

@app.route('/api/graph/analyze/motors', methods=['POST'])
def analyze_motor_graphs():
    """Анализ графиков моторов"""
    try:
        data = request.get_json() or {}
        motor_id = data.get('motor_id')
        
        logger.info(f"📊 Анализ графиков моторов: {motor_id or 'все'}")
        
        # Получаем данные от MCSetup Bridge
        try:
            mcsetup_response = requests.post(
                f"{MCSETUP_BRIDGE_URL}/api/mcsetup/analyze/motors",
                json={'motor_id': motor_id},
                timeout=10
            )
            
            if mcsetup_response.status_code != 200:
                return jsonify({
                    'status': 'error',
                    'error': 'MCSetup Bridge недоступен'
                }), 503
            
            mcsetup_data = mcsetup_response.json()
            
        except requests.exceptions.RequestException:
            return jsonify({
                'status': 'error',
                'error': 'Ошибка подключения к MCSetup Bridge'
            }), 503
        
        # Анализируем данные
        analysis = graph_analyzer.analyze_motor_trends(mcsetup_data.get('analysis', {}).get('performance_metrics', {}))
        
        return jsonify({
            'status': 'success',
            'analysis': analysis,
            'mcsetup_data': mcsetup_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка анализа графиков моторов: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/graph/report', methods=['POST'])
def generate_performance_report():
    """Генерация отчета о производительности"""
    try:
        data = request.get_json() or {}
        motor_id = data.get('motor_id')
        
        logger.info(f"📋 Генерация отчета: {motor_id or 'все'}")
        
        # Получаем анализ
        analysis_response = requests.post(
            f"{MCSETUP_BRIDGE_URL}/api/mcsetup/analyze/motors",
            json={'motor_id': motor_id},
            timeout=10
        )
        
        if analysis_response.status_code != 200:
            return jsonify({
                'status': 'error',
                'error': 'Ошибка получения данных анализа'
            }), 503
        
        analysis_data = analysis_response.json()
        
        # Генерируем отчет
        report = graph_analyzer.generate_performance_report(analysis_data.get('analysis', {}))
        
        return jsonify({
            'status': 'success',
            'report': report,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка генерации отчета: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/graph/visualize', methods=['POST'])
def create_graph_visualization():
    """Создание визуализации графиков"""
    try:
        data = request.get_json() or {}
        motor_id = data.get('motor_id')
        
        logger.info(f"📈 Создание визуализации: {motor_id or 'все'}")
        
        # Получаем данные анализа
        analysis_response = requests.post(
            f"{MCSETUP_BRIDGE_URL}/api/mcsetup/analyze/motors",
            json={'motor_id': motor_id},
            timeout=10
        )
        
        if analysis_response.status_code != 200:
            return jsonify({
                'status': 'error',
                'error': 'Ошибка получения данных для визуализации'
            }), 503
        
        analysis_data = analysis_response.json()
        motor_data = analysis_data.get('analysis', {}).get('performance_metrics', {})
        
        # Создаем визуализацию
        visualization = graph_analyzer.create_visualization(motor_data)
        
        if visualization:
            return jsonify({
                'status': 'success',
                'visualization': {
                    'image_base64': visualization,
                    'format': 'png',
                    'description': 'График производительности моторов'
                },
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Ошибка создания визуализации'
            }), 500
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания визуализации: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/graph/integrate/rubin', methods=['POST'])
def integrate_with_rubin():
    """Интеграция с Rubin AI"""
    try:
        data = request.get_json() or {}
        query = data.get('query', '')
        analysis_type = data.get('analysis_type', 'graph_analysis')
        
        logger.info(f"🤖 Интеграция с Rubin AI: {query[:50]}...")
        
        # Отправляем запрос в Rubin AI
        rubin_payload = {
            'message': f"Анализ графиков приводов: {query}",
            'context': {
                'source': 'graph_analyzer',
                'analysis_type': analysis_type,
                'capabilities': [
                    'Анализ производительности моторов',
                    'Генерация отчетов',
                    'Создание визуализаций',
                    'Интеграция с MCSetup'
                ]
            }
        }
        
        try:
            response = requests.post(
                f"{RUBIN_SMART_DISPATCHER_URL}/api/chat",
                json=rubin_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                rubin_response = response.json()
                return jsonify({
                    'status': 'success',
                    'rubin_response': rubin_response,
                    'integration_status': 'successful',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'status': 'error',
                    'error': f'Rubin AI недоступен: {response.status_code}',
                    'integration_status': 'failed'
                }), 503
                
        except requests.exceptions.RequestException as e:
            return jsonify({
                'status': 'error',
                'error': f'Ошибка подключения к Rubin AI: {e}',
                'integration_status': 'failed'
            }), 503
        
    except Exception as e:
        logger.error(f"❌ Ошибка интеграции с Rubin AI: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/graph/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервиса"""
    return jsonify({
        'service': 'graph_analyzer',
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        'mcsetup_bridge_connected': True,  # Будет проверяться при интеграции
        'rubin_ai_connected': True,  # Будет проверяться при интеграции
        'capabilities': [
            'Анализ трендов моторов',
            'Генерация отчетов производительности',
            'Создание визуализаций',
            'Интеграция с Rubin AI'
        ]
    })

if __name__ == '__main__':
    print("📊 Graph Analyzer Server запущен")
    print("URL: http://localhost:8097")
    print("Доступные эндпоинты:")
    print("  - POST /api/graph/analyze/motors - анализ графиков моторов")
    print("  - POST /api/graph/report - генерация отчета")
    print("  - POST /api/graph/visualize - создание визуализации")
    print("  - POST /api/graph/integrate/rubin - интеграция с Rubin AI")
    print("  - GET /api/graph/health - проверка здоровья")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8097, debug=False)



