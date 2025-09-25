#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интеграция ConfigurationManager и EventSystem с VMB630
Демонстрация улучшенной архитектуры с применением паттернов проектирования
"""

import os
import time
import threading
from typing import Dict, List, Any
from vmb630_configuration_manager import ConfigurationManager, EventSystem, MotorStatusObserver, ErrorObserver, PositionObserver

class VMB630Controller:
    """
    Контроллер VMB630 с интеграцией ConfigurationManager и EventSystem
    Демонстрирует улучшенную архитектуру с применением паттернов
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Инициализация паттернов проектирования
        self.config_manager = ConfigurationManager()
        self.event_system = EventSystem()
        
        # Состояние системы
        self.motors = {}
        self.axes = {}
        self.spindles = {}
        self.system_state = "INITIALIZING"
        
        # Наблюдатели
        self.observers = {}
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Инициализация системы VMB630"""
        self.logger.info("🚀 Инициализация VMB630 с улучшенной архитектурой...")
        
        # Загрузка конфигураций
        self.logger.info("📋 Загрузка конфигураций...")
        success = self.config_manager.load_all_configurations()
        
        if not success:
            self.logger.error("❌ Не удалось загрузить конфигурации")
            return
        
        # Инициализация наблюдателей
        self._setup_observers()
        
        # Инициализация компонентов
        self._initialize_motors()
        self._initialize_axes()
        self._initialize_spindles()
        
        self.system_state = "READY"
        self.event_system.notify("system_status", "READY", "VMB630Controller")
        
        self.logger.info("✅ VMB630 инициализирован с улучшенной архитектурой!")
    
    def _setup_observers(self):
        """Настройка наблюдателей"""
        self.logger.info("👁️ Настройка наблюдателей...")
        
        # Создание наблюдателей для каждой оси
        axes = ['X', 'Y1', 'Y2', 'Z', 'B', 'C']
        for axis in axes:
            motor_observer = MotorStatusObserver(f"Motor_{axis}")
            position_observer = PositionObserver(f"Axis_{axis}")
            
            self.observers[f"motor_{axis}"] = motor_observer
            self.observers[f"position_{axis}"] = position_observer
            
            # Подписка на события
            self.event_system.subscribe("motor_status", motor_observer.update, f"Motor{axis}Observer")
            self.event_system.subscribe("position_update", position_observer.update, f"Position{axis}Observer")
        
        # Общий наблюдатель ошибок
        error_observer = ErrorObserver()
        self.observers["error"] = error_observer
        self.event_system.subscribe("error", error_observer.update, "ErrorObserver")
        
        self.logger.info(f"✅ Настроено {len(self.observers)} наблюдателей")
    
    def _initialize_motors(self):
        """Инициализация моторов на основе конфигурации"""
        self.logger.info("🔧 Инициализация моторов...")
        
        # Получаем информацию о VMB630
        vmb_info = self.config_manager.get_vmb630_info()
        
        # Парсим информацию о моторах из VMB630_info.txt
        if vmb_info:
            lines = vmb_info.split('\n')
            current_axis = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('Axis '):
                    current_axis = line.split(':')[0].replace('Axis ', '')
                    self.motors[current_axis] = {
                        'type': 'Linear' if current_axis in ['X', 'Y1', 'Y2', 'Z'] else 'Rotary',
                        'pwm_channel': self._extract_pwm_channel(current_axis),
                        'status': 'STOPPED',
                        'position': 0.0,
                        'encoder': 'BISS'
                    }
                elif line.startswith('Spindle '):
                    spindle_name = line.split(':')[0].replace('Spindle ', '')
                    self.motors[spindle_name] = {
                        'type': 'Spindle',
                        'pwm_channel': self._extract_pwm_channel(spindle_name),
                        'status': 'STOPPED',
                        'speed': 0,
                        'encoder': 'BISS'
                    }
        
        self.logger.info(f"✅ Инициализировано {len(self.motors)} моторов")
    
    def _extract_pwm_channel(self, component_name: str) -> int:
        """Извлечение PWM канала из имени компонента"""
        # Простое сопоставление на основе анализа VMB630_info.txt
        channel_map = {
            'X': 1, 'Y1': 2, 'Y2': 7, 'Z': 3,
            'B': 5, 'C': 6, 'S': 4, 'S1': 8
        }
        return channel_map.get(component_name, 0)
    
    def _initialize_axes(self):
        """Инициализация осей"""
        self.logger.info("📍 Инициализация осей...")
        
        axes_info = {
            'X': {'type': 'Linear', 'resolution': 0.001, 'max_speed': 50},
            'Y1': {'type': 'Linear', 'resolution': 0.001, 'max_speed': 50},
            'Y2': {'type': 'Linear', 'resolution': 0.001, 'max_speed': 50},
            'Z': {'type': 'Linear', 'resolution': 0.001, 'max_speed': 30},
            'B': {'type': 'Rotary', 'resolution': 0.001, 'max_speed': 1000},
            'C': {'type': 'Rotary', 'resolution': 0.001, 'max_speed': 1000}
        }
        
        for axis_name, info in axes_info.items():
            self.axes[axis_name] = {
                'name': axis_name,
                'type': info['type'],
                'resolution': info['resolution'],
                'max_speed': info['max_speed'],
                'position': 0.0,
                'status': 'IDLE'
            }
        
        self.logger.info(f"✅ Инициализировано {len(self.axes)} осей")
    
    def _initialize_spindles(self):
        """Инициализация шпинделей"""
        self.logger.info("⚡ Инициализация шпинделей...")
        
        spindles_info = {
            'S': {'max_speed': 8000, 'power': 15},
            'S1': {'max_speed': 6000, 'power': 10}
        }
        
        for spindle_name, info in spindles_info.items():
            self.spindles[spindle_name] = {
                'name': spindle_name,
                'max_speed': info['max_speed'],
                'power': info['power'],
                'current_speed': 0,
                'status': 'STOPPED'
            }
        
        self.logger.info(f"✅ Инициализировано {len(self.spindles)} шпинделей")
    
    def start_motor(self, motor_name: str) -> bool:
        """Запуск мотора"""
        try:
            if motor_name in self.motors:
                self.motors[motor_name]['status'] = 'RUNNING'
                self.event_system.notify("motor_status", "STARTED", f"Motor_{motor_name}")
                self.logger.info(f"🔧 Мотор {motor_name} запущен")
                return True
            else:
                self.event_system.notify("error", f"Motor {motor_name} not found", "VMB630Controller")
                return False
        except Exception as e:
            self.event_system.notify("error", f"Failed to start motor {motor_name}: {e}", "VMB630Controller")
            return False
    
    def stop_motor(self, motor_name: str) -> bool:
        """Остановка мотора"""
        try:
            if motor_name in self.motors:
                self.motors[motor_name]['status'] = 'STOPPED'
                self.event_system.notify("motor_status", "STOPPED", f"Motor_{motor_name}")
                self.logger.info(f"🔧 Мотор {motor_name} остановлен")
                return True
            else:
                self.event_system.notify("error", f"Motor {motor_name} not found", "VMB630Controller")
                return False
        except Exception as e:
            self.event_system.notify("error", f"Failed to stop motor {motor_name}: {e}", "VMB630Controller")
            return False
    
    def move_axis(self, axis_name: str, target_position: float) -> bool:
        """Перемещение оси"""
        try:
            if axis_name in self.axes:
                current_pos = self.axes[axis_name]['position']
                self.axes[axis_name]['position'] = target_position
                self.axes[axis_name]['status'] = 'MOVING'
                
                # Обновляем позицию мотора
                if axis_name in self.motors:
                    self.motors[axis_name]['position'] = target_position
                
                self.event_system.notify("position_update", target_position, f"Axis_{axis_name}")
                self.logger.info(f"📍 Ось {axis_name}: {current_pos} → {target_position}")
                
                # Симуляция завершения движения
                threading.Timer(0.1, lambda: self._finish_movement(axis_name)).start()
                return True
            else:
                self.event_system.notify("error", f"Axis {axis_name} not found", "VMB630Controller")
                return False
        except Exception as e:
            self.event_system.notify("error", f"Failed to move axis {axis_name}: {e}", "VMB630Controller")
            return False
    
    def _finish_movement(self, axis_name: str):
        """Завершение движения оси"""
        self.axes[axis_name]['status'] = 'IDLE'
        self.event_system.notify("movement_complete", axis_name, f"Axis_{axis_name}")
    
    def set_spindle_speed(self, spindle_name: str, speed: int) -> bool:
        """Установка скорости шпинделя"""
        try:
            if spindle_name in self.spindles:
                max_speed = self.spindles[spindle_name]['max_speed']
                if speed <= max_speed:
                    self.spindles[spindle_name]['current_speed'] = speed
                    self.spindles[spindle_name]['status'] = 'RUNNING' if speed > 0 else 'STOPPED'
                    
                    self.event_system.notify("spindle_speed", speed, f"Spindle_{spindle_name}")
                    self.logger.info(f"⚡ Шпиндель {spindle_name}: {speed} об/мин")
                    return True
                else:
                    self.event_system.notify("error", f"Speed {speed} exceeds max speed {max_speed} for spindle {spindle_name}", "VMB630Controller")
                    return False
            else:
                self.event_system.notify("error", f"Spindle {spindle_name} not found", "VMB630Controller")
                return False
        except Exception as e:
            self.event_system.notify("error", f"Failed to set spindle speed {spindle_name}: {e}", "VMB630Controller")
            return False
    
    def get_system_status(self) -> Dict:
        """Получение статуса системы"""
        return {
            'system_state': self.system_state,
            'motors': {name: motor['status'] for name, motor in self.motors.items()},
            'axes': {name: {'position': axis['position'], 'status': axis['status']} for name, axis in self.axes.items()},
            'spindles': {name: {'speed': spindle['current_speed'], 'status': spindle['status']} for name, spindle in self.spindles.items()},
            'config_status': self.config_manager.get_config_status(),
            'event_status': self.event_system.get_observers_status()
        }
    
    def get_event_history(self, event_type: str = None, limit: int = 10) -> List[Dict]:
        """Получение истории событий"""
        return self.event_system.get_event_history(event_type, limit)
    
    def reload_configuration(self, config_name: str) -> bool:
        """Перезагрузка конфигурации"""
        success = self.config_manager.reload_config(config_name)
        if success:
            self.event_system.notify("config_reloaded", config_name, "VMB630Controller")
        return success


def demonstrate_vmb630_integration():
    """Демонстрация интеграции VMB630 с паттернами проектирования"""
    
    print("🚀 ДЕМОНСТРАЦИЯ ИНТЕГРАЦИИ VMB630 С ПАТТЕРНАМИ ПРОЕКТИРОВАНИЯ")
    print("=" * 80)
    
    # Инициализация контроллера
    print("\n🔧 ИНИЦИАЛИЗАЦИЯ VMB630 КОНТРОЛЛЕРА:")
    controller = VMB630Controller()
    
    # Получение статуса системы
    print("\n📊 СТАТУС СИСТЕМЫ:")
    status = controller.get_system_status()
    print(f"  Состояние системы: {status['system_state']}")
    print(f"  Моторы: {len(status['motors'])}")
    print(f"  Оси: {len(status['axes'])}")
    print(f"  Шпиндели: {len(status['spindles'])}")
    print(f"  Конфигураций: {status['config_status']['total_configs']}")
    print(f"  Наблюдателей: {status['event_status']['total_observers']}")
    
    # Демонстрация управления моторами
    print("\n🔧 УПРАВЛЕНИЕ МОТОРАМИ:")
    controller.start_motor("X")
    controller.start_motor("Y1")
    controller.start_motor("Z")
    
    time.sleep(0.5)
    
    # Демонстрация перемещения осей
    print("\n📍 ПЕРЕМЕЩЕНИЕ ОСЕЙ:")
    controller.move_axis("X", 100.0)
    controller.move_axis("Y1", 50.0)
    controller.move_axis("Z", 25.0)
    
    time.sleep(0.5)
    
    # Демонстрация управления шпинделями
    print("\n⚡ УПРАВЛЕНИЕ ШПИНДЕЛЯМИ:")
    controller.set_spindle_speed("S", 5000)
    controller.set_spindle_speed("S1", 3000)
    
    time.sleep(0.5)
    
    # Демонстрация ошибок
    print("\n❌ ДЕМОНСТРАЦИЯ ОБРАБОТКИ ОШИБОК:")
    controller.move_axis("INVALID_AXIS", 100.0)  # Ошибка
    controller.set_spindle_speed("S", 10000)  # Превышение максимальной скорости
    
    time.sleep(0.5)
    
    # Получение истории событий
    print("\n📜 ИСТОРИЯ СОБЫТИЙ:")
    history = controller.get_event_history(limit=10)
    for event in history:
        print(f"  {event['timestamp'].strftime('%H:%M:%S.%f')[:-3]} - {event['type']}: {event['data']} (от {event['source']})")
    
    # Финальный статус
    print("\n📊 ФИНАЛЬНЫЙ СТАТУС СИСТЕМЫ:")
    final_status = controller.get_system_status()
    
    print("  🔧 Моторы:")
    for name, motor_status in final_status['motors'].items():
        print(f"    {name}: {motor_status}")
    
    print("  📍 Оси:")
    for name, axis_info in final_status['axes'].items():
        print(f"    {name}: позиция {axis_info['position']}, статус {axis_info['status']}")
    
    print("  ⚡ Шпиндели:")
    for name, spindle_info in final_status['spindles'].items():
        print(f"    {name}: скорость {spindle_info['speed']} об/мин, статус {spindle_info['status']}")
    
    print("\n🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
    print("VMB630 успешно интегрирован с паттернами проектирования!")
    print("\n✅ Преимущества новой архитектуры:")
    print("  - Единая точка доступа к конфигурациям (Singleton)")
    print("  - Слабая связанность через систему событий (Observer)")
    print("  - Централизованное логирование и мониторинг")
    print("  - Легкость расширения и тестирования")
    print("  - Улучшенная отказоустойчивость")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    demonstrate_vmb630_integration()





