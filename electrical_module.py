#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль электротехники для Rubin AI v2
Основной модуль для обработки электротехнических запросов
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

# Импортируем утилиты
from electrical_utils import get_electrical_utils
from electrical_knowledge_handler import get_electrical_handler

logger = logging.getLogger(__name__)

class ElectricalModule:
    """Основной модуль электротехники"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.utils = get_electrical_utils()
        self.knowledge_handler = get_electrical_handler()
        self.logger.info("Модуль электротехники инициализирован")
    
    def process_request(self, request_data: Dict) -> Dict:
        """
        Обработка запроса по электротехнике
        
        Args:
            request_data: Данные запроса
        
        Returns:
            Dict с ответом
        """
        try:
            request_type = request_data.get('type', 'explain')
            concept = request_data.get('concept', '')
            parameters = request_data.get('parameters', {})
            
            if request_type == 'explain':
                # Объяснение концепций
                return self._handle_explanation(concept)
            elif request_type == 'calculate':
                # Выполнение расчетов
                return self._handle_calculation(parameters)
            elif request_type == 'knowledge':
                # Работа с базой знаний
                return self._handle_knowledge_request(concept)
            else:
                return {
                    'success': False,
                    'error': f'Неизвестный тип запроса: {request_type}',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Ошибка обработки запроса: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _handle_explanation(self, concept: str) -> Dict:
        """Обработка запросов на объяснение"""
        try:
            if not concept:
                return {
                    'success': False,
                    'error': 'Не указана концепция для объяснения',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Используем обработчик знаний
            result = self.knowledge_handler.handle_request(concept)
            
            return {
                'success': result.get('success', True),
                'concept': concept,
                'response': result.get('response', ''),
                'provider': result.get('provider', 'Electrical Module'),
                'category': result.get('category', 'electrical'),
                'topic': result.get('topic', 'general'),
                'language': result.get('language', 'ru'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки объяснения: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _handle_calculation(self, parameters: Dict) -> Dict:
        """Обработка запросов на расчеты"""
        try:
            calculation_type = parameters.get('type', '')
            
            if calculation_type == 'ohm_law':
                # Расчет по закону Ома
                voltage = parameters.get('voltage')
                current = parameters.get('current')
                resistance = parameters.get('resistance')
                power = parameters.get('power')
                
                result = self.utils.ohm_law_calculate(
                    voltage=voltage,
                    current=current,
                    resistance=resistance,
                    power=power
                )
                
                return {
                    'success': result.get('success', True),
                    'calculation_type': 'ohm_law',
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
            
            elif calculation_type == 'power':
                # Расчет мощности
                voltage = parameters.get('voltage')
                current = parameters.get('current')
                power_factor = parameters.get('power_factor', 1.0)
                
                result = self.utils.power_calculation(
                    voltage=voltage,
                    current=current,
                    power_factor=power_factor
                )
                
                return {
                    'success': result.get('success', True),
                    'calculation_type': 'power',
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
            
            elif calculation_type == 'three_phase_power':
                # Расчет трехфазной мощности
                line_voltage = parameters.get('line_voltage')
                line_current = parameters.get('line_current')
                power_factor = parameters.get('power_factor', 1.0)
                
                result = self.utils.three_phase_power(
                    line_voltage=line_voltage,
                    line_current=line_current,
                    power_factor=power_factor
                )
                
                return {
                    'success': result.get('success', True),
                    'calculation_type': 'three_phase_power',
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
            
            elif calculation_type == 'resistor_combination':
                # Расчет сопротивлений
                resistors = parameters.get('resistors', [])
                connection_type = parameters.get('connection_type', 'series')
                
                result = self.utils.resistor_combination(
                    resistors=resistors,
                    connection_type=connection_type
                )
                
                return {
                    'success': result.get('success', True),
                    'calculation_type': 'resistor_combination',
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
            
            elif calculation_type == 'capacitor_combination':
                # Расчет емкостей
                capacitors = parameters.get('capacitors', [])
                connection_type = parameters.get('connection_type', 'parallel')
                
                result = self.utils.capacitor_combination(
                    capacitors=capacitors,
                    connection_type=connection_type
                )
                
                return {
                    'success': result.get('success', True),
                    'calculation_type': 'capacitor_combination',
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
            
            elif calculation_type == 'voltage_divider':
                # Расчет делителя напряжения
                input_voltage = parameters.get('input_voltage')
                r1 = parameters.get('r1')
                r2 = parameters.get('r2')
                
                result = self.utils.voltage_divider(
                    input_voltage=input_voltage,
                    r1=r1,
                    r2=r2
                )
                
                return {
                    'success': result.get('success', True),
                    'calculation_type': 'voltage_divider',
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
            
            else:
                return {
                    'success': False,
                    'error': f'Неизвестный тип расчета: {calculation_type}',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Ошибка обработки расчета: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _handle_knowledge_request(self, concept: str) -> Dict:
        """Обработка запросов к базе знаний"""
        try:
            if not concept:
                return {
                    'success': False,
                    'error': 'Не указана концепция для поиска в базе знаний',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Используем обработчик знаний
            result = self.knowledge_handler.handle_request(concept)
            
            return {
                'success': result.get('success', True),
                'concept': concept,
                'response': result.get('response', ''),
                'provider': result.get('provider', 'Electrical Module'),
                'category': result.get('category', 'electrical'),
                'topic': result.get('topic', 'general'),
                'language': result.get('language', 'ru'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки запроса к базе знаний: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_supported_calculations(self) -> List[str]:
        """Получение списка поддерживаемых расчетов"""
        return [
            'ohm_law',
            'power',
            'three_phase_power',
            'resistor_combination',
            'capacitor_combination',
            'voltage_divider'
        ]
    
    def get_supported_topics(self) -> List[str]:
        """Получение списка поддерживаемых тем"""
        return self.knowledge_handler.get_supported_topics()
    
    def get_module_info(self) -> Dict:
        """Получение информации о модуле"""
        return {
            'name': 'Electrical Module',
            'version': '1.0.0',
            'description': 'Модуль электротехники для Rubin AI v2',
            'supported_calculations': self.get_supported_calculations(),
            'supported_topics': self.get_supported_topics(),
            'timestamp': datetime.now().isoformat()
        }

# Глобальный экземпляр
_electrical_module_instance = None

def get_electrical_module() -> ElectricalModule:
    """Получение глобального экземпляра модуля"""
    global _electrical_module_instance
    if _electrical_module_instance is None:
        _electrical_module_instance = ElectricalModule()
    return _electrical_module_instance

# Тестирование
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    module = ElectricalModule()
    
    print("=== ТЕСТИРОВАНИЕ МОДУЛЯ ЭЛЕКТРОТЕХНИКИ ===")
    
    # Тест объяснения
    print("\n1. Тест объяснения:")
    result = module.process_request({
        'type': 'explain',
        'concept': 'закон ома'
    })
    print(f"Объяснение закона Ома: {result.get('success', False)}")
    
    # Тест расчета
    print("\n2. Тест расчета:")
    result = module.process_request({
        'type': 'calculate',
        'parameters': {
            'type': 'ohm_law',
            'voltage': 12,
            'resistance': 4
        }
    })
    print(f"Расчет по закону Ома: {result.get('success', False)}")
    
    # Тест базы знаний
    print("\n3. Тест базы знаний:")
    result = module.process_request({
        'type': 'knowledge',
        'concept': 'защита от короткого замыкания'
    })
    print(f"База знаний: {result.get('success', False)}")
    
    # Информация о модуле
    print("\n4. Информация о модуле:")
    info = module.get_module_info()
    print(f"Название: {info['name']}")
    print(f"Версия: {info['version']}")
    print(f"Поддерживаемые расчеты: {len(info['supported_calculations'])}")
    print(f"Поддерживаемые темы: {len(info['supported_topics'])}")
    
    print("\n✅ Тестирование завершено!")











