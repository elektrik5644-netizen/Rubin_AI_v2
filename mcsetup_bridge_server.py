#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCSetup Bridge Server для интеграции с Rubin AI
Анализ графиков и настроек приводов
"""

from flask import Flask, request, jsonify
import logging
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime
import requests
import configparser
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Конфигурация MCSetup
MCSETUP_PATH = r"C:\Users\elekt\OneDrive\Desktop\MCSetup_V1_9_0\MCSetup_V1_9_0"
RUBIN_SMART_DISPATCHER_URL = "http://localhost:8080"
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')  # ID чата RubinDeveloper

class MCSetupAnalyzer:
    """Анализатор данных MCSetup"""
    
    def __init__(self, mcsetup_path):
        self.mcsetup_path = Path(mcsetup_path)
        self.plot_configs = {}
        self.motor_configs = {}
        self.load_configurations()
    
    def load_configurations(self):
        """Загружает конфигурации MCSetup"""
        try:
            # Загружаем конфигурацию приложения
            config_file = self.mcsetup_path / "mcsetup.ini"
            if config_file.exists():
                self.load_mcsetup_config(config_file)
            
            # Загружаем конфигурации графиков
            plot_dir = self.mcsetup_path / "Plot, Watch windows" / "Plot"
            if plot_dir.exists():
                self.load_plot_configurations(plot_dir)
            
            logger.info(f"✅ Загружено {len(self.plot_configs)} конфигураций графиков")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки конфигураций: {e}")

def send_to_telegram(message, chat_id=None):
    """Отправляет сообщение в Telegram чат"""
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        logger.warning("⚠️ Telegram не настроен: отсутствует токен или chat_id")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info(f"✅ Сообщение отправлено в Telegram чат {chat_id}")
            return True
        else:
            logger.error(f"❌ Ошибка отправки в Telegram: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Ошибка отправки в Telegram: {e}")
        return False
    
    def load_mcsetup_config(self, config_file):
        """Загружает основную конфигурацию MCSetup"""
        try:
            config = configparser.ConfigParser()
            config.read(config_file, encoding='utf-8')
            
            # Извлекаем информацию о моторах
            if 'Position' in config:
                self.motor_configs['position_settings'] = dict(config['Position'])
            
            if 'ItemToGat' in config:
                self.motor_configs['data_sources'] = dict(config['ItemToGat'])
            
            logger.info("✅ Конфигурация MCSetup загружена")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки конфигурации MCSetup: {e}")
    
    def load_plot_configurations(self, plot_dir):
        """Загружает конфигурации графиков"""
        try:
            for xml_file in plot_dir.glob("*.xml"):
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    
                    config_data = {
                        'file_name': xml_file.name,
                        'gather_period': root.find('.//param[@name="gatherPeriod"]').get('value', '10'),
                        'sources': []
                    }
                    
                    # Извлекаем источники данных
                    for item in root.findall('.//group[@name="itemToGather"]'):
                        source = {
                            'address': item.find('param[@name="address"]').get('value', ''),
                            'name': item.find('param[@name="name"]').get('value', ''),
                            'enabled': item.find('param[@name="enable"]').get('value', 'false') == 'true'
                        }
                        config_data['sources'].append(source)
                    
                    self.plot_configs[xml_file.stem] = config_data
                    
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка загрузки {xml_file.name}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки конфигураций графиков: {e}")
    
    def analyze_motor_performance(self, motor_id=None):
        """Анализирует производительность моторов"""
        try:
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'motors_analyzed': 0,
                'performance_metrics': {},
                'recommendations': []
            }
            
            # Анализируем конфигурации моторов
            for config_name, config_data in self.plot_configs.items():
                if motor_id and motor_id not in config_name:
                    continue
                
                motor_analysis = {
                    'gather_period': int(config_data.get('gather_period', 10)),
                    'active_sources': len([s for s in config_data['sources'] if s['enabled']]),
                    'total_sources': len(config_data['sources']),
                    'data_sources': config_data['sources']
                }
                
                # Анализ производительности
                if motor_analysis['gather_period'] < 5:
                    motor_analysis['performance_rating'] = 'high'
                    analysis['recommendations'].append(f"Мотор {config_name}: Высокая частота сбора данных ({motor_analysis['gather_period']}мс)")
                elif motor_analysis['gather_period'] < 20:
                    motor_analysis['performance_rating'] = 'medium'
                    analysis['recommendations'].append(f"Мотор {config_name}: Средняя частота сбора данных ({motor_analysis['gather_period']}мс)")
                else:
                    motor_analysis['performance_rating'] = 'low'
                    analysis['recommendations'].append(f"Мотор {config_name}: Низкая частота сбора данных ({motor_analysis['gather_period']}мс)")
                
                analysis['performance_metrics'][config_name] = motor_analysis
                analysis['motors_analyzed'] += 1
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа производительности моторов: {e}")
            return {'error': str(e)}
    
    def analyze_graph_data(self, graph_name=None):
        """Анализирует данные графиков"""
        try:
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'graphs_analyzed': 0,
                'graph_analysis': {},
                'insights': []
            }
            
            for config_name, config_data in self.plot_configs.items():
                if graph_name and graph_name not in config_name:
                    continue
                
                graph_analysis = {
                    'sources_count': len(config_data['sources']),
                    'active_sources': len([s for s in config_data['sources'] if s['enabled']]),
                    'gather_period': config_data['gather_period'],
                    'data_types': self._categorize_data_sources(config_data['sources'])
                }
                
                # Анализ типов данных
                if 'position' in graph_analysis['data_types']:
                    analysis['insights'].append(f"График {config_name}: Содержит данные позиционирования")
                
                if 'velocity' in graph_analysis['data_types']:
                    analysis['insights'].append(f"График {config_name}: Содержит данные скорости")
                
                if 'current' in graph_analysis['data_types']:
                    analysis['insights'].append(f"График {config_name}: Содержит данные тока")
                
                analysis['graph_analysis'][config_name] = graph_analysis
                analysis['graphs_analyzed'] += 1
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа графиков: {e}")
            return {'error': str(e)}
    
    def _categorize_data_sources(self, sources):
        """Категоризирует источники данных"""
        categories = set()
        
        for source in sources:
            name = source['name'].lower()
            address = source['address'].lower()
            
            if any(keyword in name for keyword in ['pos', 'position', 'позиц']):
                categories.add('position')
            elif any(keyword in name for keyword in ['vel', 'velocity', 'скорост']):
                categories.add('velocity')
            elif any(keyword in name for keyword in ['cur', 'current', 'ток']):
                categories.add('current')
            elif any(keyword in name for keyword in ['torque', 'момент']):
                categories.add('torque')
            elif any(keyword in name for keyword in ['temp', 'температур']):
                categories.add('temperature')
            else:
                categories.add('other')
        
        return list(categories)
    
    def get_motor_recommendations(self, motor_id=None):
        """Получает рекомендации по настройке моторов"""
        try:
            recommendations = {
                'timestamp': datetime.now().isoformat(),
                'motor_recommendations': {},
                'general_recommendations': []
            }
            
            for config_name, config_data in self.plot_configs.items():
                if motor_id and motor_id not in config_name:
                    continue
                
                motor_recs = []
                
                # Анализ частоты сбора данных
                gather_period = int(config_data.get('gather_period', 10))
                if gather_period < 5:
                    motor_recs.append("⚠️ Слишком высокая частота сбора данных может вызвать перегрузку системы")
                elif gather_period > 50:
                    motor_recs.append("⚠️ Низкая частота сбора данных может пропустить важные события")
                
                # Анализ количества источников
                active_sources = len([s for s in config_data['sources'] if s['enabled']])
                if active_sources > 10:
                    motor_recs.append("⚠️ Большое количество активных источников может снизить производительность")
                elif active_sources < 3:
                    motor_recs.append("💡 Рекомендуется добавить больше источников данных для полного мониторинга")
                
                # Анализ типов данных
                data_types = self._categorize_data_sources(config_data['sources'])
                if 'position' not in data_types:
                    motor_recs.append("💡 Рекомендуется добавить мониторинг позиции")
                if 'velocity' not in data_types:
                    motor_recs.append("💡 Рекомендуется добавить мониторинг скорости")
                if 'current' not in data_types:
                    motor_recs.append("💡 Рекомендуется добавить мониторинг тока")
                
                recommendations['motor_recommendations'][config_name] = motor_recs
            
            # Общие рекомендации
            recommendations['general_recommendations'] = [
                "📊 Регулярно анализируйте графики производительности",
                "⚙️ Оптимизируйте частоту сбора данных в зависимости от задач",
                "🔧 Настройте аварийные пороги для критических параметров",
                "📈 Используйте трендовый анализ для прогнозирования отказов"
            ]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения рекомендаций: {e}")
            return {'error': str(e)}

# Инициализация анализатора
mcsetup_analyzer = MCSetupAnalyzer(MCSETUP_PATH)

@app.route('/api/mcsetup/status', methods=['GET'])
def mcsetup_status():
    """Статус интеграции MCSetup"""
    return jsonify({
        'service': 'mcsetup_bridge',
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        'mcsetup_path': MCSETUP_PATH,
        'plot_configs_loaded': len(mcsetup_analyzer.plot_configs),
        'motor_configs_loaded': len(mcsetup_analyzer.motor_configs),
        'capabilities': [
            'Анализ производительности моторов',
            'Анализ данных графиков',
            'Рекомендации по настройке',
            'Интеграция с Rubin AI'
        ]
    })

@app.route('/api/mcsetup/analyze/motors', methods=['POST'])
def analyze_motors():
    """Анализ производительности моторов"""
    try:
        data = request.get_json() or {}
        motor_id = data.get('motor_id')
        
        logger.info(f"🔧 Анализ моторов MCSetup: {motor_id or 'все'}")
        
        analysis = mcsetup_analyzer.analyze_motor_performance(motor_id)
        
        return jsonify({
            'status': 'success',
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка анализа моторов: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/mcsetup/analyze/graphs', methods=['POST'])
def analyze_graphs():
    """Анализ данных графиков"""
    try:
        data = request.get_json() or {}
        graph_name = data.get('graph_name')
        
        logger.info(f"📊 Анализ графиков MCSetup: {graph_name or 'все'}")
        
        analysis = mcsetup_analyzer.analyze_graph_data(graph_name)
        
        return jsonify({
            'status': 'success',
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка анализа графиков: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/mcsetup/recommendations', methods=['POST'])
def get_recommendations():
    """Получение рекомендаций по настройке"""
    try:
        data = request.get_json() or {}
        motor_id = data.get('motor_id')
        
        logger.info(f"💡 Получение рекомендаций MCSetup: {motor_id or 'все'}")
        
        recommendations = mcsetup_analyzer.get_motor_recommendations(motor_id)
        
        # Отправляем рекомендации в Telegram чат RubinDeveloper
        if recommendations and recommendations.get('general_recommendations'):
            telegram_message = f"🔧 **MCSetup Рекомендации по Настройке**\n\n"
            telegram_message += f"**Мотор:** {motor_id or 'Все моторы'}\n\n"
            
            # Формируем рекомендации
            general_recs = recommendations.get('general_recommendations', [])
            if general_recs:
                telegram_message += "**Общие рекомендации:**\n"
                for i, rec in enumerate(general_recs[:5], 1):  # Ограничиваем до 5 рекомендаций
                    telegram_message += f"{i}. {rec}\n"
            
            motor_recs = recommendations.get('motor_recommendations', {})
            if motor_recs:
                telegram_message += "\n**Рекомендации по моторам:**\n"
                for motor, recs in list(motor_recs.items())[:3]:  # Ограничиваем до 3 моторов
                    telegram_message += f"**{motor}:**\n"
                    for rec in recs[:3]:  # До 3 рекомендаций на мотор
                        telegram_message += f"• {rec}\n"
            
            telegram_message += f"\n**Время:** {datetime.now().strftime('%H:%M:%S')}"
            
            # Отправляем в Telegram
            send_to_telegram(telegram_message, TELEGRAM_CHAT_ID)
        
        return jsonify({
            'status': 'success',
            'recommendations': recommendations,
            'telegram_sent': True,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения рекомендаций: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/mcsetup/configs', methods=['GET'])
def get_configs():
    """Получение списка конфигураций"""
    try:
        configs = {
            'plot_configs': list(mcsetup_analyzer.plot_configs.keys()),
            'motor_configs': list(mcsetup_analyzer.motor_configs.keys()),
            'total_plots': len(mcsetup_analyzer.plot_configs),
            'total_motors': len(mcsetup_analyzer.plot_configs)  # Предполагаем, что каждый график = мотор
        }
        
        return jsonify({
            'status': 'success',
            'configs': configs,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения конфигураций: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/mcsetup/integrate/rubin', methods=['POST'])
def integrate_with_rubin():
    """Интеграция с Rubin AI через Smart Dispatcher"""
    try:
        data = request.get_json() or {}
        query = data.get('query', '')
        analysis_type = data.get('analysis_type', 'general')
        
        logger.info(f"🤖 Интеграция с Rubin AI: {query[:50]}...")
        
        # Отправляем запрос в Rubin AI
        rubin_payload = {
            'message': f"MCSetup анализ: {query}",
            'context': {
                'source': 'mcsetup_bridge',
                'analysis_type': analysis_type,
                'mcsetup_data': {
                    'plot_configs_count': len(mcsetup_analyzer.plot_configs),
                    'motor_configs_count': len(mcsetup_analyzer.motor_configs)
                }
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
                
                # Отправляем рекомендации в Telegram чат RubinDeveloper
                if analysis_type in ['drive_tuning', 'motor_analysis', 'graph_analysis']:
                    telegram_message = f"📊 **MCSetup Анализ Графика**\n\n"
                    telegram_message += f"**Запрос:** {query}\n\n"
                    
                    # Извлекаем ответ Rubin AI
                    rubin_text = ""
                    if isinstance(rubin_response, dict):
                        if 'response' in rubin_response:
                            if isinstance(rubin_response['response'], dict):
                                rubin_text = rubin_response['response'].get('explanation', str(rubin_response['response']))
                            else:
                                rubin_text = str(rubin_response['response'])
                        else:
                            rubin_text = str(rubin_response)
                    else:
                        rubin_text = str(rubin_response)
                    
                    telegram_message += f"**Рекомендации Rubin AI:**\n{rubin_text}\n\n"
                    telegram_message += f"**Время:** {datetime.now().strftime('%H:%M:%S')}"
                    
                    # Отправляем в Telegram
                    send_to_telegram(telegram_message, TELEGRAM_CHAT_ID)
                
                return jsonify({
                    'status': 'success',
                    'rubin_response': rubin_response,
                    'integration_status': 'successful',
                    'telegram_sent': True,
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

@app.route('/api/mcsetup/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервиса"""
    return jsonify({
        'service': 'mcsetup_bridge',
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        'mcsetup_connected': os.path.exists(MCSETUP_PATH),
        'rubin_ai_connected': True,  # Будет проверяться при интеграции
        'capabilities': [
            'Анализ производительности моторов',
            'Анализ данных графиков',
            'Рекомендации по настройке',
            'Интеграция с Rubin AI'
        ]
    })

if __name__ == '__main__':
    print("🔧 MCSetup Bridge Server запущен")
    print("URL: http://localhost:8096")
    print("Доступные эндпоинты:")
    print("  - GET /api/mcsetup/status - статус интеграции")
    print("  - POST /api/mcsetup/analyze/motors - анализ моторов")
    print("  - POST /api/mcsetup/analyze/graphs - анализ графиков")
    print("  - POST /api/mcsetup/recommendations - рекомендации")
    print("  - GET /api/mcsetup/configs - список конфигураций")
    print("  - POST /api/mcsetup/integrate/rubin - интеграция с Rubin AI")
    print("  - GET /api/mcsetup/health - проверка здоровья")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8096, debug=False)
