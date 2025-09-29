#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Утилиты для электротехники Rubin AI v2
Вспомогательные функции для расчетов и обработки электротехнических данных
"""

import math
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class ElectricalUtils:
    """Класс утилит для электротехнических расчетов"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Инициализация утилит электротехники")
    
    def ohm_law_calculate(self, voltage: float = None, current: float = None, 
                          resistance: float = None, power: float = None) -> Dict:
        """
        Расчет по закону Ома
        
        Args:
            voltage: Напряжение (В)
            current: Ток (А)
            resistance: Сопротивление (Ом)
            power: Мощность (Вт)
        
        Returns:
            Dict с результатами расчетов
        """
        try:
            result = {}
            
            if voltage is not None and resistance is not None:
                # U и R известны
                current = voltage / resistance
                power = voltage * current
                result = {
                    'voltage': voltage,
                    'current': round(current, 3),
                    'resistance': resistance,
                    'power': round(power, 3),
                    'formula': 'I = U / R, P = U × I'
                }
            elif current is not None and resistance is not None:
                # I и R известны
                voltage = current * resistance
                power = voltage * current
                result = {
                    'voltage': round(voltage, 3),
                    'current': current,
                    'resistance': resistance,
                    'power': round(power, 3),
                    'formula': 'U = I × R, P = U × I'
                }
            elif voltage is not None and current is not None:
                # U и I известны
                resistance = voltage / current
                power = voltage * current
                result = {
                    'voltage': voltage,
                    'current': current,
                    'resistance': round(resistance, 3),
                    'power': round(power, 3),
                    'formula': 'R = U / I, P = U × I'
                }
            elif power is not None and voltage is not None:
                # P и U известны
                current = power / voltage
                resistance = voltage / current
                result = {
                    'voltage': voltage,
                    'current': round(current, 3),
                    'resistance': round(resistance, 3),
                    'power': power,
                    'formula': 'I = P / U, R = U / I'
                }
            elif power is not None and current is not None:
                # P и I известны
                voltage = power / current
                resistance = voltage / current
                result = {
                    'voltage': round(voltage, 3),
                    'current': current,
                    'resistance': round(resistance, 3),
                    'power': power,
                    'formula': 'U = P / I, R = U / I'
                }
            elif power is not None and resistance is not None:
                # P и R известны
                current = math.sqrt(power / resistance)
                voltage = current * resistance
                result = {
                    'voltage': round(voltage, 3),
                    'current': round(current, 3),
                    'resistance': resistance,
                    'power': power,
                    'formula': 'I = √(P / R), U = I × R'
                }
            else:
                raise ValueError("Недостаточно параметров для расчета")
            
            result['success'] = True
            result['timestamp'] = datetime.now().isoformat()
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета по закону Ома: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def power_calculation(self, voltage: float, current: float, 
                         power_factor: float = 1.0) -> Dict:
        """
        Расчет мощности в однофазной цепи
        
        Args:
            voltage: Напряжение (В)
            current: Ток (А)
            power_factor: Коэффициент мощности (cos φ)
        
        Returns:
            Dict с результатами расчетов
        """
        try:
            apparent_power = voltage * current  # Полная мощность (ВА)
            active_power = apparent_power * power_factor  # Активная мощность (Вт)
            reactive_power = apparent_power * math.sqrt(1 - power_factor**2)  # Реактивная мощность (ВАр)
            
            result = {
                'voltage': voltage,
                'current': current,
                'power_factor': power_factor,
                'apparent_power': round(apparent_power, 3),
                'active_power': round(active_power, 3),
                'reactive_power': round(reactive_power, 3),
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета мощности: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def three_phase_power(self, line_voltage: float, line_current: float, 
                         power_factor: float = 1.0) -> Dict:
        """
        Расчет мощности в трехфазной системе
        
        Args:
            line_voltage: Линейное напряжение (В)
            line_current: Линейный ток (А)
            power_factor: Коэффициент мощности (cos φ)
        
        Returns:
            Dict с результатами расчетов
        """
        try:
            sqrt3 = math.sqrt(3)
            
            # Полная мощность
            apparent_power = sqrt3 * line_voltage * line_current
            
            # Активная мощность
            active_power = apparent_power * power_factor
            
            # Реактивная мощность
            reactive_power = apparent_power * math.sqrt(1 - power_factor**2)
            
            result = {
                'line_voltage': line_voltage,
                'line_current': line_current,
                'power_factor': power_factor,
                'apparent_power': round(apparent_power, 3),
                'active_power': round(active_power, 3),
                'reactive_power': round(reactive_power, 3),
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета трехфазной мощности: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def resistor_combination(self, resistors: List[float], 
                           connection_type: str = 'series') -> Dict:
        """
        Расчет эквивалентного сопротивления
        
        Args:
            resistors: Список сопротивлений (Ом)
            connection_type: Тип соединения ('series' или 'parallel')
        
        Returns:
            Dict с результатами расчетов
        """
        try:
            if not resistors:
                raise ValueError("Список сопротивлений не может быть пустым")
            
            if connection_type == 'series':
                # Последовательное соединение
                equivalent_resistance = sum(resistors)
                formula = "R = R1 + R2 + R3 + ..."
            elif connection_type == 'parallel':
                # Параллельное соединение
                reciprocal_sum = sum(1/r for r in resistors)
                equivalent_resistance = 1 / reciprocal_sum
                formula = "1/R = 1/R1 + 1/R2 + 1/R3 + ..."
            else:
                raise ValueError("Неверный тип соединения. Используйте 'series' или 'parallel'")
            
            result = {
                'resistors': resistors,
                'connection_type': connection_type,
                'equivalent_resistance': round(equivalent_resistance, 3),
                'formula': formula,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета сопротивлений: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def capacitor_combination(self, capacitors: List[float], 
                            connection_type: str = 'parallel') -> Dict:
        """
        Расчет эквивалентной емкости
        
        Args:
            capacitors: Список емкостей (мкФ)
            connection_type: Тип соединения ('series' или 'parallel')
        
        Returns:
            Dict с результатами расчетов
        """
        try:
            if not capacitors:
                raise ValueError("Список емкостей не может быть пустым")
            
            if connection_type == 'parallel':
                # Параллельное соединение
                equivalent_capacitance = sum(capacitors)
                formula = "C = C1 + C2 + C3 + ..."
            elif connection_type == 'series':
                # Последовательное соединение
                reciprocal_sum = sum(1/c for c in capacitors)
                equivalent_capacitance = 1 / reciprocal_sum
                formula = "1/C = 1/C1 + 1/C2 + 1/C3 + ..."
            else:
                raise ValueError("Неверный тип соединения. Используйте 'series' или 'parallel'")
            
            result = {
                'capacitors': capacitors,
                'connection_type': connection_type,
                'equivalent_capacitance': round(equivalent_capacitance, 3),
                'formula': formula,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета емкостей: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def voltage_divider(self, input_voltage: float, r1: float, r2: float) -> Dict:
        """
        Расчет делителя напряжения
        
        Args:
            input_voltage: Входное напряжение (В)
            r1: Первое сопротивление (Ом)
            r2: Второе сопротивление (Ом)
        
        Returns:
            Dict с результатами расчетов
        """
        try:
            total_resistance = r1 + r2
            current = input_voltage / total_resistance
            voltage_r1 = current * r1
            voltage_r2 = current * r2
            
            result = {
                'input_voltage': input_voltage,
                'r1': r1,
                'r2': r2,
                'total_resistance': round(total_resistance, 3),
                'current': round(current, 3),
                'voltage_r1': round(voltage_r1, 3),
                'voltage_r2': round(voltage_r2, 3),
                'formula': 'U1 = U × R1 / (R1 + R2), U2 = U × R2 / (R1 + R2)',
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета делителя напряжения: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_unit_prefixes(self) -> Dict[str, float]:
        """Получение префиксов единиц измерения"""
        return {
            'pico': 1e-12,    # п
            'nano': 1e-9,     # н
            'micro': 1e-6,    # мк
            'milli': 1e-3,    # м
            'kilo': 1e3,      # к
            'mega': 1e6,       # М
            'giga': 1e9,       # Г
            'tera': 1e12      # Т
        }
    
    def convert_units(self, value: float, from_prefix: str, to_prefix: str) -> Dict:
        """
        Конвертация единиц измерения
        
        Args:
            value: Значение для конвертации
            from_prefix: Исходный префикс
            to_prefix: Целевой префикс
        
        Returns:
            Dict с результатами конвертации
        """
        try:
            prefixes = self.get_unit_prefixes()
            
            if from_prefix not in prefixes or to_prefix not in prefixes:
                raise ValueError("Неверный префикс единицы измерения")
            
            # Конвертируем в базовую единицу
            base_value = value * prefixes[from_prefix]
            
            # Конвертируем в целевую единицу
            converted_value = base_value / prefixes[to_prefix]
            
            result = {
                'original_value': value,
                'original_prefix': from_prefix,
                'converted_value': round(converted_value, 6),
                'converted_prefix': to_prefix,
                'conversion_factor': round(prefixes[from_prefix] / prefixes[to_prefix], 6),
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка конвертации единиц: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Глобальный экземпляр
_electrical_utils_instance = None

def get_electrical_utils() -> ElectricalUtils:
    """Получение глобального экземпляра утилит"""
    global _electrical_utils_instance
    if _electrical_utils_instance is None:
        _electrical_utils_instance = ElectricalUtils()
    return _electrical_utils_instance

# Тестирование
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    utils = ElectricalUtils()
    
    print("=== ТЕСТИРОВАНИЕ УТИЛИТ ЭЛЕКТРОТЕХНИКИ ===")
    
    # Тест закона Ома
    print("\n1. Тест закона Ома:")
    result = utils.ohm_law_calculate(voltage=12, resistance=4)
    print(f"U=12В, R=4Ом: {result}")
    
    # Тест мощности
    print("\n2. Тест мощности:")
    result = utils.power_calculation(voltage=220, current=5, power_factor=0.8)
    print(f"U=220В, I=5А, cosφ=0.8: {result}")
    
    # Тест трехфазной мощности
    print("\n3. Тест трехфазной мощности:")
    result = utils.three_phase_power(line_voltage=380, line_current=10, power_factor=0.9)
    print(f"Uл=380В, Iл=10А, cosφ=0.9: {result}")
    
    # Тест сопротивлений
    print("\n4. Тест сопротивлений:")
    result = utils.resistor_combination([10, 20, 30], 'series')
    print(f"Последовательное соединение [10, 20, 30]Ом: {result}")
    
    result = utils.resistor_combination([10, 20, 30], 'parallel')
    print(f"Параллельное соединение [10, 20, 30]Ом: {result}")
    
    # Тест делителя напряжения
    print("\n5. Тест делителя напряжения:")
    result = utils.voltage_divider(input_voltage=12, r1=1000, r2=2000)
    print(f"U=12В, R1=1кОм, R2=2кОм: {result}")
    
    print("\n✅ Тестирование завершено!")











