#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Расширенная архитектура VMB630 с полным набором паттернов проектирования
Factory, Strategy, Command + существующие Singleton и Observer
"""

import os
import time
import threading
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from datetime import datetime
from vmb630_configuration_manager import ConfigurationManager, EventSystem

# ============================================================================
# FACTORY PATTERN - Создание моторов и осей
# ============================================================================

class MotorType(Enum):
    """Типы моторов"""
    LINEAR = "linear"
    ROTARY = "rotary"
    SPINDLE = "spindle"

class AxisType(Enum):
    """Типы осей"""
    LINEAR = "linear"
    ROTARY = "rotary"
    VIRTUAL = "virtual"

class IMotor(ABC):
    """Интерфейс мотора"""
    
    @abstractmethod
    def start(self) -> bool:
        """Запуск мотора"""
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """Остановка мотора"""
        pass
    
    @abstractmethod
    def get_status(self) -> str:
        """Получение статуса мотора"""
        pass
    
    @abstractmethod
    def get_position(self) -> float:
        """Получение позиции мотора"""
        pass

class LinearMotor(IMotor):
    """Линейный мотор"""
    
    def __init__(self, motor_id: str, pwm_channel: int, max_speed: float = 50.0):
        self.motor_id = motor_id
        self.pwm_channel = pwm_channel
        self.max_speed = max_speed
        self.status = "STOPPED"
        self.position = 0.0
        self.logger = logging.getLogger(__name__)
    
    def start(self) -> bool:
        """Запуск линейного мотора"""
        try:
            self.status = "RUNNING"
            self.logger.info(f"🔧 Линейный мотор {self.motor_id} запущен (PWM: {self.pwm_channel})")
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка запуска мотора {self.motor_id}: {e}")
            return False
    
    def stop(self) -> bool:
        """Остановка линейного мотора"""
        try:
            self.status = "STOPPED"
            self.logger.info(f"🔧 Линейный мотор {self.motor_id} остановлен")
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка остановки мотора {self.motor_id}: {e}")
            return False
    
    def get_status(self) -> str:
        """Получение статуса мотора"""
        return self.status
    
    def get_position(self) -> float:
        """Получение позиции мотора"""
        return self.position

class RotaryMotor(IMotor):
    """Вращательный мотор"""
    
    def __init__(self, motor_id: str, pwm_channel: int, max_speed: float = 1000.0):
        self.motor_id = motor_id
        self.pwm_channel = pwm_channel
        self.max_speed = max_speed
        self.status = "STOPPED"
        self.position = 0.0
        self.logger = logging.getLogger(__name__)
    
    def start(self) -> bool:
        """Запуск вращательного мотора"""
        try:
            self.status = "RUNNING"
            self.logger.info(f"🔄 Вращательный мотор {self.motor_id} запущен (PWM: {self.pwm_channel})")
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка запуска мотора {self.motor_id}: {e}")
            return False
    
    def stop(self) -> bool:
        """Остановка вращательного мотора"""
        try:
            self.status = "STOPPED"
            self.logger.info(f"🔄 Вращательный мотор {self.motor_id} остановлен")
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка остановки мотора {self.motor_id}: {e}")
            return False
    
    def get_status(self) -> str:
        """Получение статуса мотора"""
        return self.status
    
    def get_position(self) -> float:
        """Получение позиции мотора"""
        return self.position

class SpindleMotor(IMotor):
    """Шпиндель"""
    
    def __init__(self, motor_id: str, pwm_channel: int, max_speed: float = 8000.0, power: float = 15.0):
        self.motor_id = motor_id
        self.pwm_channel = pwm_channel
        self.max_speed = max_speed
        self.power = power
        self.status = "STOPPED"
        self.current_speed = 0
        self.logger = logging.getLogger(__name__)
    
    def start(self) -> bool:
        """Запуск шпинделя"""
        try:
            self.status = "RUNNING"
            self.logger.info(f"⚡ Шпиндель {self.motor_id} запущен (PWM: {self.pwm_channel}, {self.power}кВт)")
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка запуска шпинделя {self.motor_id}: {e}")
            return False
    
    def stop(self) -> bool:
        """Остановка шпинделя"""
        try:
            self.status = "STOPPED"
            self.current_speed = 0
            self.logger.info(f"⚡ Шпиндель {self.motor_id} остановлен")
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка остановки шпинделя {self.motor_id}: {e}")
            return False
    
    def get_status(self) -> str:
        """Получение статуса шпинделя"""
        return self.status
    
    def get_position(self) -> float:
        """Получение скорости шпинделя"""
        return self.current_speed

class IAxis(ABC):
    """Интерфейс оси"""
    
    @abstractmethod
    def move_to(self, position: float) -> bool:
        """Перемещение к позиции"""
        pass
    
    @abstractmethod
    def get_position(self) -> float:
        """Получение текущей позиции"""
        pass
    
    @abstractmethod
    def get_status(self) -> str:
        """Получение статуса оси"""
        pass

class LinearAxis(IAxis):
    """Линейная ось"""
    
    def __init__(self, axis_name: str, resolution: float = 0.001, max_speed: float = 50.0):
        self.axis_name = axis_name
        self.resolution = resolution
        self.max_speed = max_speed
        self.position = 0.0
        self.status = "IDLE"
        self.logger = logging.getLogger(__name__)
    
    def move_to(self, position: float) -> bool:
        """Перемещение линейной оси"""
        try:
            self.status = "MOVING"
            old_position = self.position
            self.position = position
            self.logger.info(f"📍 Линейная ось {self.axis_name}: {old_position} → {position} мм")
            
            # Симуляция завершения движения
            threading.Timer(0.1, lambda: self._finish_movement()).start()
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка перемещения оси {self.axis_name}: {e}")
            return False
    
    def _finish_movement(self):
        """Завершение движения"""
        self.status = "IDLE"
    
    def get_position(self) -> float:
        """Получение позиции оси"""
        return self.position
    
    def get_status(self) -> str:
        """Получение статуса оси"""
        return self.status

class RotaryAxis(IAxis):
    """Вращательная ось"""
    
    def __init__(self, axis_name: str, resolution: float = 0.001, max_speed: float = 1000.0):
        self.axis_name = axis_name
        self.resolution = resolution
        self.max_speed = max_speed
        self.position = 0.0
        self.status = "IDLE"
        self.logger = logging.getLogger(__name__)
    
    def move_to(self, position: float) -> bool:
        """Перемещение вращательной оси"""
        try:
            self.status = "MOVING"
            old_position = self.position
            self.position = position
            self.logger.info(f"🔄 Вращательная ось {self.axis_name}: {old_position} → {position}°")
            
            # Симуляция завершения движения
            threading.Timer(0.1, lambda: self._finish_movement()).start()
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка перемещения оси {self.axis_name}: {e}")
            return False
    
    def _finish_movement(self):
        """Завершение движения"""
        self.status = "IDLE"
    
    def get_position(self) -> float:
        """Получение позиции оси"""
        return self.position
    
    def get_status(self) -> str:
        """Получение статуса оси"""
        return self.status

class MotorFactory:
    """Фабрика моторов"""
    
    @staticmethod
    def create_motor(motor_type: MotorType, motor_id: str, **kwargs) -> IMotor:
        """Создание мотора по типу"""
        if motor_type == MotorType.LINEAR:
            return LinearMotor(motor_id, **kwargs)
        elif motor_type == MotorType.ROTARY:
            return RotaryMotor(motor_id, **kwargs)
        elif motor_type == MotorType.SPINDLE:
            return SpindleMotor(motor_id, **kwargs)
        else:
            raise ValueError(f"Неизвестный тип мотора: {motor_type}")

class AxisFactory:
    """Фабрика осей"""
    
    @staticmethod
    def create_axis(axis_type: AxisType, axis_name: str, **kwargs) -> IAxis:
        """Создание оси по типу"""
        if axis_type == AxisType.LINEAR:
            return LinearAxis(axis_name, **kwargs)
        elif axis_type == AxisType.ROTARY:
            return RotaryAxis(axis_name, **kwargs)
        else:
            raise ValueError(f"Неизвестный тип оси: {axis_type}")

# ============================================================================
# STRATEGY PATTERN - Алгоритмы управления
# ============================================================================

class IControlStrategy(ABC):
    """Интерфейс стратегии управления"""
    
    @abstractmethod
    def execute(self, target_position: float, current_position: float) -> Dict[str, Any]:
        """Выполнение стратегии управления"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Получение имени стратегии"""
        pass

class LinearControlStrategy(IControlStrategy):
    """Стратегия управления линейными осями"""
    
    def __init__(self, max_speed: float = 50.0, acceleration: float = 10.0):
        self.max_speed = max_speed
        self.acceleration = acceleration
    
    def execute(self, target_position: float, current_position: float) -> Dict[str, Any]:
        """Выполнение линейного управления"""
        distance = abs(target_position - current_position)
        direction = 1 if target_position > current_position else -1
        
        # Расчет времени движения
        time_to_max_speed = self.max_speed / self.acceleration
        distance_to_max_speed = 0.5 * self.acceleration * time_to_max_speed ** 2
        
        if distance <= 2 * distance_to_max_speed:
            # Треугольный профиль скорости
            time_total = 2 * (distance / self.acceleration) ** 0.5
        else:
            # Трапецеидальный профиль скорости
            time_constant = (distance - 2 * distance_to_max_speed) / self.max_speed
            time_total = 2 * time_to_max_speed + time_constant
        
        return {
            'strategy': 'linear',
            'distance': distance,
            'direction': direction,
            'time_total': time_total,
            'max_speed': self.max_speed,
            'acceleration': self.acceleration
        }
    
    def get_name(self) -> str:
        return "Linear Control Strategy"

class RotaryControlStrategy(IControlStrategy):
    """Стратегия управления вращательными осями"""
    
    def __init__(self, max_speed: float = 1000.0, acceleration: float = 500.0):
        self.max_speed = max_speed
        self.acceleration = acceleration
    
    def execute(self, target_position: float, current_position: float) -> Dict[str, Any]:
        """Выполнение вращательного управления"""
        # Нормализация углов
        angle_diff = target_position - current_position
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        
        distance = abs(angle_diff)
        direction = 1 if angle_diff > 0 else -1
        
        # Расчет времени движения
        time_to_max_speed = self.max_speed / self.acceleration
        distance_to_max_speed = 0.5 * self.acceleration * time_to_max_speed ** 2
        
        if distance <= 2 * distance_to_max_speed:
            time_total = 2 * (distance / self.acceleration) ** 0.5
        else:
            time_constant = (distance - 2 * distance_to_max_speed) / self.max_speed
            time_total = 2 * time_to_max_speed + time_constant
        
        return {
            'strategy': 'rotary',
            'distance': distance,
            'direction': direction,
            'time_total': time_total,
            'max_speed': self.max_speed,
            'acceleration': self.acceleration,
            'angle_diff': angle_diff
        }
    
    def get_name(self) -> str:
        return "Rotary Control Strategy"

class GantryControlStrategy(IControlStrategy):
    """Стратегия управления Gantry (синхронизация двух осей)"""
    
    def __init__(self, max_speed: float = 50.0, sync_tolerance: float = 0.001):
        self.max_speed = max_speed
        self.sync_tolerance = sync_tolerance
    
    def execute(self, target_position: float, current_position: float) -> Dict[str, Any]:
        """Выполнение Gantry управления"""
        distance = abs(target_position - current_position)
        direction = 1 if target_position > current_position else -1
        
        # Синхронизированное движение двух осей
        sync_factor = 0.95  # Коэффициент синхронизации
        
        return {
            'strategy': 'gantry',
            'distance': distance,
            'direction': direction,
            'time_total': distance / (self.max_speed * sync_factor),
            'max_speed': self.max_speed * sync_factor,
            'sync_tolerance': self.sync_tolerance,
            'sync_factor': sync_factor
        }
    
    def get_name(self) -> str:
        return "Gantry Control Strategy"

# ============================================================================
# COMMAND PATTERN - Операции с отменой
# ============================================================================

class ICommand(ABC):
    """Интерфейс команды"""
    
    @abstractmethod
    def execute(self) -> bool:
        """Выполнение команды"""
        pass
    
    @abstractmethod
    def undo(self) -> bool:
        """Отмена команды"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Получение описания команды"""
        pass

class MoveCommand(ICommand):
    """Команда перемещения"""
    
    def __init__(self, axis: IAxis, target_position: float, event_system: EventSystem):
        self.axis = axis
        self.target_position = target_position
        self.previous_position = axis.get_position()
        self.event_system = event_system
        self.executed = False
    
    def execute(self) -> bool:
        """Выполнение команды перемещения"""
        try:
            success = self.axis.move_to(self.target_position)
            if success:
                self.executed = True
                self.event_system.notify("command_executed", {
                    'command': 'move',
                    'axis': self.axis.axis_name,
                    'target': self.target_position
                }, "MoveCommand")
            return success
        except Exception as e:
            self.event_system.notify("command_error", {
                'command': 'move',
                'error': str(e)
            }, "MoveCommand")
            return False
    
    def undo(self) -> bool:
        """Отмена команды перемещения"""
        try:
            if self.executed:
                success = self.axis.move_to(self.previous_position)
                if success:
                    self.event_system.notify("command_undone", {
                        'command': 'move',
                        'axis': self.axis.axis_name,
                        'restored': self.previous_position
                    }, "MoveCommand")
                return success
            return True
        except Exception as e:
            self.event_system.notify("command_error", {
                'command': 'undo_move',
                'error': str(e)
            }, "MoveCommand")
            return False
    
    def get_description(self) -> str:
        return f"Переместить {self.axis.axis_name} в позицию {self.target_position}"

class MotorControlCommand(ICommand):
    """Команда управления мотором"""
    
    def __init__(self, motor: IMotor, action: str, event_system: EventSystem):
        self.motor = motor
        self.action = action  # "start" или "stop"
        self.event_system = event_system
        self.executed = False
        self.previous_status = motor.get_status()
    
    def execute(self) -> bool:
        """Выполнение команды управления мотором"""
        try:
            if self.action == "start":
                success = self.motor.start()
            elif self.action == "stop":
                success = self.motor.stop()
            else:
                return False
            
            if success:
                self.executed = True
                self.event_system.notify("command_executed", {
                    'command': 'motor_control',
                    'motor': self.motor.motor_id,
                    'action': self.action
                }, "MotorControlCommand")
            return success
        except Exception as e:
            self.event_system.notify("command_error", {
                'command': 'motor_control',
                'error': str(e)
            }, "MotorControlCommand")
            return False
    
    def undo(self) -> bool:
        """Отмена команды управления мотором"""
        try:
            if self.executed:
                # Восстанавливаем предыдущее состояние
                if self.previous_status == "RUNNING":
                    success = self.motor.start()
                else:
                    success = self.motor.stop()
                
                if success:
                    self.event_system.notify("command_undone", {
                        'command': 'motor_control',
                        'motor': self.motor.motor_id,
                        'restored': self.previous_status
                    }, "MotorControlCommand")
                return success
            return True
        except Exception as e:
            self.event_system.notify("command_error", {
                'command': 'undo_motor_control',
                'error': str(e)
            }, "MotorControlCommand")
            return False
    
    def get_description(self) -> str:
        return f"{self.action.capitalize()} мотор {self.motor.motor_id}"

class CommandInvoker:
    """Вызыватель команд с поддержкой отмены"""
    
    def __init__(self, event_system: EventSystem):
        self.event_system = event_system
        self.command_history = []
        self.max_history = 100
        self.logger = logging.getLogger(__name__)
    
    def execute_command(self, command: ICommand) -> bool:
        """Выполнение команды"""
        try:
            success = command.execute()
            if success:
                self.command_history.append(command)
                if len(self.command_history) > self.max_history:
                    self.command_history.pop(0)
                
                self.logger.info(f"✅ Выполнена команда: {command.get_description()}")
            else:
                self.logger.error(f"❌ Ошибка выполнения команды: {command.get_description()}")
            
            return success
        except Exception as e:
            self.logger.error(f"❌ Исключение при выполнении команды: {e}")
            return False
    
    def undo_last_command(self) -> bool:
        """Отмена последней команды"""
        if not self.command_history:
            self.logger.warning("⚠️ Нет команд для отмены")
            return False
        
        try:
            last_command = self.command_history.pop()
            success = last_command.undo()
            
            if success:
                self.logger.info(f"↩️ Отменена команда: {last_command.get_description()}")
            else:
                self.logger.error(f"❌ Ошибка отмены команды: {last_command.get_description()}")
                # Возвращаем команду в историю при ошибке отмены
                self.command_history.append(last_command)
            
            return success
        except Exception as e:
            self.logger.error(f"❌ Исключение при отмене команды: {e}")
            return False
    
    def get_command_history(self) -> List[str]:
        """Получение истории команд"""
        return [cmd.get_description() for cmd in self.command_history]

# ============================================================================
# ИНТЕГРИРОВАННАЯ СИСТЕМА VMB630
# ============================================================================

class VMB630AdvancedController:
    """Продвинутый контроллер VMB630 с полным набором паттернов"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Инициализация существующих паттернов
        self.config_manager = ConfigurationManager()
        self.event_system = EventSystem()
        
        # Инициализация новых паттернов
        self.motor_factory = MotorFactory()
        self.axis_factory = AxisFactory()
        self.command_invoker = CommandInvoker(self.event_system)
        
        # Стратегии управления
        self.control_strategies = {
            'linear': LinearControlStrategy(),
            'rotary': RotaryControlStrategy(),
            'gantry': GantryControlStrategy()
        }
        
        # Компоненты системы
        self.motors = {}
        self.axes = {}
        self.system_state = "INITIALIZING"
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Инициализация расширенной системы VMB630"""
        self.logger.info("🚀 Инициализация расширенной системы VMB630...")
        
        # Загрузка конфигураций
        success = self.config_manager.load_all_configurations()
        if not success:
            self.logger.error("❌ Не удалось загрузить конфигурации")
            return
        
        # Создание моторов через Factory
        self._create_motors()
        
        # Создание осей через Factory
        self._create_axes()
        
        # Настройка наблюдателей
        self._setup_observers()
        
        self.system_state = "READY"
        self.event_system.notify("system_ready", "VMB630 Advanced System", "VMB630AdvancedController")
        
        self.logger.info("✅ Расширенная система VMB630 инициализирована!")
    
    def _create_motors(self):
        """Создание моторов через Factory Pattern"""
        self.logger.info("🏭 Создание моторов через Factory...")
        
        # Линейные моторы
        linear_motors = [
            ('X', MotorType.LINEAR, {'pwm_channel': 1, 'max_speed': 50.0}),
            ('Y1', MotorType.LINEAR, {'pwm_channel': 2, 'max_speed': 50.0}),
            ('Y2', MotorType.LINEAR, {'pwm_channel': 7, 'max_speed': 50.0}),
            ('Z', MotorType.LINEAR, {'pwm_channel': 3, 'max_speed': 30.0})
        ]
        
        # Вращательные моторы
        rotary_motors = [
            ('B', MotorType.ROTARY, {'pwm_channel': 5, 'max_speed': 1000.0}),
            ('C', MotorType.ROTARY, {'pwm_channel': 6, 'max_speed': 1000.0})
        ]
        
        # Шпиндели
        spindle_motors = [
            ('S', MotorType.SPINDLE, {'pwm_channel': 4, 'max_speed': 8000.0, 'power': 15.0}),
            ('S1', MotorType.SPINDLE, {'pwm_channel': 8, 'max_speed': 6000.0, 'power': 10.0})
        ]
        
        # Создание всех моторов
        all_motors = linear_motors + rotary_motors + spindle_motors
        
        for motor_id, motor_type, params in all_motors:
            try:
                motor = self.motor_factory.create_motor(motor_type, motor_id, **params)
                self.motors[motor_id] = motor
                self.logger.info(f"✅ Создан {motor_type.value} мотор {motor_id}")
            except Exception as e:
                self.logger.error(f"❌ Ошибка создания мотора {motor_id}: {e}")
        
        self.logger.info(f"🏭 Создано {len(self.motors)} моторов")
    
    def _create_axes(self):
        """Создание осей через Factory Pattern"""
        self.logger.info("🏭 Создание осей через Factory...")
        
        # Линейные оси
        linear_axes = [
            ('X', AxisType.LINEAR, {'resolution': 0.001, 'max_speed': 50.0}),
            ('Y1', AxisType.LINEAR, {'resolution': 0.001, 'max_speed': 50.0}),
            ('Y2', AxisType.LINEAR, {'resolution': 0.001, 'max_speed': 50.0}),
            ('Z', AxisType.LINEAR, {'resolution': 0.001, 'max_speed': 30.0})
        ]
        
        # Вращательные оси
        rotary_axes = [
            ('B', AxisType.ROTARY, {'resolution': 0.001, 'max_speed': 1000.0}),
            ('C', AxisType.ROTARY, {'resolution': 0.001, 'max_speed': 1000.0})
        ]
        
        # Создание всех осей
        all_axes = linear_axes + rotary_axes
        
        for axis_name, axis_type, params in all_axes:
            try:
                axis = self.axis_factory.create_axis(axis_type, axis_name, **params)
                self.axes[axis_name] = axis
                self.logger.info(f"✅ Создана {axis_type.value} ось {axis_name}")
            except Exception as e:
                self.logger.error(f"❌ Ошибка создания оси {axis_name}: {e}")
        
        self.logger.info(f"🏭 Создано {len(self.axes)} осей")
    
    def _setup_observers(self):
        """Настройка наблюдателей"""
        from vmb630_configuration_manager import MotorStatusObserver, ErrorObserver, PositionObserver
        
        # Создание наблюдателей для каждой оси
        for axis_name in self.axes.keys():
            motor_observer = MotorStatusObserver(f"Motor_{axis_name}")
            position_observer = PositionObserver(f"Axis_{axis_name}")
            
            self.event_system.subscribe("motor_status", motor_observer.update, f"Motor{axis_name}Observer")
            self.event_system.subscribe("position_update", position_observer.update, f"Position{axis_name}Observer")
        
        # Общий наблюдатель ошибок
        error_observer = ErrorObserver()
        self.event_system.subscribe("error", error_observer.update, "ErrorObserver")
    
    def move_axis_with_strategy(self, axis_name: str, target_position: float, strategy_name: str = None) -> bool:
        """Перемещение оси с использованием стратегии управления"""
        try:
            if axis_name not in self.axes:
                self.event_system.notify("error", f"Axis {axis_name} not found", "VMB630AdvancedController")
                return False
            
            axis = self.axes[axis_name]
            current_position = axis.get_position()
            
            # Определение стратегии
            if strategy_name is None:
                if axis_name in ['X', 'Y1', 'Y2', 'Z']:
                    strategy_name = 'linear'
                elif axis_name in ['B', 'C']:
                    strategy_name = 'rotary'
                else:
                    strategy_name = 'linear'
            
            if strategy_name not in self.control_strategies:
                self.event_system.notify("error", f"Strategy {strategy_name} not found", "VMB630AdvancedController")
                return False
            
            # Выполнение стратегии
            strategy = self.control_strategies[strategy_name]
            control_params = strategy.execute(target_position, current_position)
            
            self.logger.info(f"🎯 Стратегия {strategy.get_name()}: {control_params}")
            
            # Создание и выполнение команды перемещения
            move_command = MoveCommand(axis, target_position, self.event_system)
            success = self.command_invoker.execute_command(move_command)
            
            if success:
                self.event_system.notify("strategy_executed", {
                    'axis': axis_name,
                    'strategy': strategy_name,
                    'params': control_params
                }, "VMB630AdvancedController")
            
            return success
            
        except Exception as e:
            self.event_system.notify("error", f"Move axis error: {e}", "VMB630AdvancedController")
            return False
    
    def control_motor_with_command(self, motor_id: str, action: str) -> bool:
        """Управление мотором через команды"""
        try:
            if motor_id not in self.motors:
                self.event_system.notify("error", f"Motor {motor_id} not found", "VMB630AdvancedController")
                return False
            
            motor = self.motors[motor_id]
            
            # Создание и выполнение команды управления мотором
            motor_command = MotorControlCommand(motor, action, self.event_system)
            success = self.command_invoker.execute_command(motor_command)
            
            return success
            
        except Exception as e:
            self.event_system.notify("error", f"Motor control error: {e}", "VMB630AdvancedController")
            return False
    
    def undo_last_operation(self) -> bool:
        """Отмена последней операции"""
        return self.command_invoker.undo_last_command()
    
    def get_system_status(self) -> Dict:
        """Получение статуса расширенной системы"""
        return {
            'system_state': self.system_state,
            'motors': {name: {'status': motor.get_status(), 'position': motor.get_position()} 
                      for name, motor in self.motors.items()},
            'axes': {name: {'position': axis.get_position(), 'status': axis.get_status()} 
                    for name, axis in self.axes.items()},
            'strategies': {name: strategy.get_name() for name, strategy in self.control_strategies.items()},
            'command_history': self.command_invoker.get_command_history(),
            'config_status': self.config_manager.get_config_status(),
            'event_status': self.event_system.get_observers_status()
        }
    
    def get_event_history(self, event_type: str = None, limit: int = 10) -> List[Dict]:
        """Получение истории событий"""
        return self.event_system.get_event_history(event_type, limit)
    
    def get_command_history(self) -> List[str]:
        """Получение истории команд"""
        return self.command_invoker.get_command_history()


# ============================================================================
# ДЕМОНСТРАЦИЯ РАСШИРЕННОЙ АРХИТЕКТУРЫ
# ============================================================================

def demonstrate_advanced_vmb630_architecture():
    """Демонстрация расширенной архитектуры VMB630"""
    
    print("🚀 ДЕМОНСТРАЦИЯ РАСШИРЕННОЙ АРХИТЕКТУРЫ VMB630")
    print("=" * 80)
    print("Паттерны: Singleton + Observer + Factory + Strategy + Command")
    print("=" * 80)
    
    # Инициализация расширенного контроллера
    print("\n🔧 ИНИЦИАЛИЗАЦИЯ РАСШИРЕННОГО КОНТРОЛЛЕРА:")
    controller = VMB630AdvancedController()
    
    # Получение статуса системы
    print("\n📊 СТАТУС РАСШИРЕННОЙ СИСТЕМЫ:")
    status = controller.get_system_status()
    print(f"  Состояние системы: {status['system_state']}")
    print(f"  Моторы: {len(status['motors'])}")
    print(f"  Оси: {len(status['axes'])}")
    print(f"  Стратегии: {len(status['strategies'])}")
    print(f"  Команд в истории: {len(status['command_history'])}")
    
    # Демонстрация Factory Pattern
    print("\n🏭 ДЕМОНСТРАЦИЯ FACTORY PATTERN:")
    print("  Созданные моторы:")
    for motor_id, motor_info in status['motors'].items():
        print(f"    {motor_id}: {motor_info['status']}")
    
    print("  Созданные оси:")
    for axis_name, axis_info in status['axes'].items():
        print(f"    {axis_name}: позиция {axis_info['position']}, статус {axis_info['status']}")
    
    # Демонстрация Strategy Pattern
    print("\n🎯 ДЕМОНСТРАЦИЯ STRATEGY PATTERN:")
    print("  Доступные стратегии:")
    for strategy_name, strategy_description in status['strategies'].items():
        print(f"    {strategy_name}: {strategy_description}")
    
    # Демонстрация Command Pattern
    print("\n⚡ ДЕМОНСТРАЦИЯ COMMAND PATTERN:")
    
    # Управление моторами через команды
    print("  Управление моторами через команды:")
    controller.control_motor_with_command("X", "start")
    controller.control_motor_with_command("Y1", "start")
    controller.control_motor_with_command("Z", "start")
    
    time.sleep(0.5)
    
    # Перемещение осей с различными стратегиями
    print("  Перемещение осей с различными стратегиями:")
    controller.move_axis_with_strategy("X", 100.0, "linear")
    controller.move_axis_with_strategy("Y1", 50.0, "linear")
    controller.move_axis_with_strategy("B", 45.0, "rotary")
    controller.move_axis_with_strategy("C", 90.0, "rotary")
    
    time.sleep(0.5)
    
    # Демонстрация отмены операций
    print("  Демонстрация отмены операций:")
    print("    Отмена последней операции...")
    controller.undo_last_operation()
    
    print("    Отмена еще одной операции...")
    controller.undo_last_operation()
    
    time.sleep(0.5)
    
    # Получение истории команд
    print("\n📜 ИСТОРИЯ КОМАНД:")
    command_history = controller.get_command_history()
    for i, command_desc in enumerate(command_history[-5:], 1):  # Последние 5 команд
        print(f"  {i}. {command_desc}")
    
    # Получение истории событий
    print("\n📡 ИСТОРИЯ СОБЫТИЙ:")
    event_history = controller.get_event_history(limit=10)
    for event in event_history:
        print(f"  {event['timestamp'].strftime('%H:%M:%S.%f')[:-3]} - {event['type']}: {event['data']}")
    
    # Финальный статус
    print("\n📊 ФИНАЛЬНЫЙ СТАТУС РАСШИРЕННОЙ СИСТЕМЫ:")
    final_status = controller.get_system_status()
    
    print("  🔧 Моторы:")
    for name, motor_info in final_status['motors'].items():
        print(f"    {name}: {motor_info['status']}, позиция {motor_info['position']}")
    
    print("  📍 Оси:")
    for name, axis_info in final_status['axes'].items():
        print(f"    {name}: позиция {axis_info['position']}, статус {axis_info['status']}")
    
    print("\n🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
    print("Расширенная архитектура VMB630 с полным набором паттернов готова!")
    
    print("\n✅ РЕАЛИЗОВАННЫЕ ПАТТЕРНЫ:")
    print("  🏗️ Singleton - ConfigurationManager (единая точка доступа к конфигурациям)")
    print("  👁️ Observer - EventSystem (слабая связанность через события)")
    print("  🏭 Factory - MotorFactory, AxisFactory (создание компонентов)")
    print("  🎯 Strategy - ControlStrategy (алгоритмы управления)")
    print("  ⚡ Command - CommandInvoker (операции с отменой)")
    
    print("\n🚀 ПРЕИМУЩЕСТВА РАСШИРЕННОЙ АРХИТЕКТУРЫ:")
    print("  - Модульность и расширяемость")
    print("  - Слабая связанность компонентов")
    print("  - Возможность отмены операций")
    print("  - Гибкие алгоритмы управления")
    print("  - Централизованное управление конфигурациями")
    print("  - Полная система мониторинга и логирования")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    demonstrate_advanced_vmb630_architecture()
