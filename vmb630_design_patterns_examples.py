#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Примеры применения паттернов проектирования для улучшения архитектуры VMB630
"""

# ============================================================================
# 1. ПАТТЕРН SINGLETON - ConfigurationManager
# ============================================================================

class ConfigurationManager:
    """Singleton для управления всеми конфигурациями VMB630"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._configs = {}
            self._load_configurations()
            ConfigurationManager._initialized = True
    
    def _load_configurations(self):
        """Загрузка всех конфигураций"""
        self._configs = {
            'definitions': self._load_xml_config('define.xml'),
            'start_config': self._load_cfg_config('start.cfg'),
            'errors': self._load_xml_config('errors.xml'),
            'motors': self._load_cfg_config('motors.cfg'),
            'plc': self._load_ini_config('plc.ini')
        }
    
    def get_definition(self, key):
        """Получение определения по ключу"""
        return self._configs['definitions'].get(key)
    
    def get_start_config(self):
        """Получение конфигурации запуска"""
        return self._configs['start_config']
    
    def get_error_codes(self):
        """Получение кодов ошибок"""
        return self._configs['errors']
    
    def get_motor_config(self, motor_id):
        """Получение конфигурации мотора"""
        return self._configs['motors'].get(f'motor_{motor_id}')
    
    def _load_xml_config(self, filename):
        """Загрузка XML конфигурации"""
        # Здесь будет логика загрузки XML
        return {}
    
    def _load_cfg_config(self, filename):
        """Загрузка CFG конфигурации"""
        # Здесь будет логика загрузки CFG
        return {}
    
    def _load_ini_config(self, filename):
        """Загрузка INI конфигурации"""
        # Здесь будет логика загрузки INI
        return {}

# ============================================================================
# 2. ПАТТЕРН FACTORY - MotorFactory и AxisFactory
# ============================================================================

from abc import ABC, abstractmethod

class IMotor(ABC):
    """Интерфейс мотора"""
    
    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def stop(self):
        pass
    
    @abstractmethod
    def get_position(self):
        pass
    
    @abstractmethod
    def move_to(self, position):
        pass

class LinearMotor(IMotor):
    """Линейный мотор"""
    
    def __init__(self, motor_id, pwm_channel, encoder_config):
        self.motor_id = motor_id
        self.pwm_channel = pwm_channel
        self.encoder_config = encoder_config
        self.position = 0.0
        self.is_running = False
    
    def start(self):
        self.is_running = True
        print(f"Linear Motor {self.motor_id} started on PWM channel {self.pwm_channel}")
    
    def stop(self):
        self.is_running = False
        print(f"Linear Motor {self.motor_id} stopped")
    
    def get_position(self):
        return self.position
    
    def move_to(self, position):
        self.position = position
        print(f"Linear Motor {self.motor_id} moved to position {position}")

class RotaryMotor(IMotor):
    """Вращательный мотор"""
    
    def __init__(self, motor_id, pwm_channel, encoder_config):
        self.motor_id = motor_id
        self.pwm_channel = pwm_channel
        self.encoder_config = encoder_config
        self.angle = 0.0
        self.is_running = False
    
    def start(self):
        self.is_running = True
        print(f"Rotary Motor {self.motor_id} started on PWM channel {self.pwm_channel}")
    
    def stop(self):
        self.is_running = False
        print(f"Rotary Motor {self.motor_id} stopped")
    
    def get_position(self):
        return self.angle
    
    def move_to(self, angle):
        self.angle = angle
        print(f"Rotary Motor {self.motor_id} rotated to angle {angle}")

class MotorFactory:
    """Фабрика для создания моторов"""
    
    @staticmethod
    def create_linear_motor(motor_id, pwm_channel, encoder_config):
        return LinearMotor(motor_id, pwm_channel, encoder_config)
    
    @staticmethod
    def create_rotary_motor(motor_id, pwm_channel, encoder_config):
        return RotaryMotor(motor_id, pwm_channel, encoder_config)
    
    @staticmethod
    def create_motor(motor_type, motor_id, pwm_channel, encoder_config):
        if motor_type == "linear":
            return MotorFactory.create_linear_motor(motor_id, pwm_channel, encoder_config)
        elif motor_type == "rotary":
            return MotorFactory.create_rotary_motor(motor_id, pwm_channel, encoder_config)
        else:
            raise ValueError(f"Unknown motor type: {motor_type}")

class IAxis(ABC):
    """Интерфейс оси"""
    
    @abstractmethod
    def move_to(self, position):
        pass
    
    @abstractmethod
    def get_position(self):
        pass
    
    @abstractmethod
    def calibrate(self):
        pass

class LinearAxis(IAxis):
    """Линейная ось"""
    
    def __init__(self, name, motor):
        self.name = name
        self.motor = motor
        self.position = 0.0
    
    def move_to(self, position):
        self.motor.move_to(position)
        self.position = position
        print(f"Linear Axis {self.name} moved to {position}")
    
    def get_position(self):
        return self.motor.get_position()
    
    def calibrate(self):
        print(f"Linear Axis {self.name} calibrated")

class RotaryAxis(IAxis):
    """Вращательная ось"""
    
    def __init__(self, name, motor):
        self.name = name
        self.motor = motor
        self.angle = 0.0
    
    def move_to(self, angle):
        self.motor.move_to(angle)
        self.angle = angle
        print(f"Rotary Axis {self.name} rotated to {angle}")
    
    def get_position(self):
        return self.motor.get_position()
    
    def calibrate(self):
        print(f"Rotary Axis {self.name} calibrated")

class AxisFactory:
    """Фабрика для создания осей"""
    
    @staticmethod
    def create_linear_axis(name, motor):
        return LinearAxis(name, motor)
    
    @staticmethod
    def create_rotary_axis(name, motor):
        return RotaryAxis(name, motor)
    
    @staticmethod
    def create_axis(axis_type, name, motor):
        if axis_type == "linear":
            return AxisFactory.create_linear_axis(name, motor)
        elif axis_type == "rotary":
            return AxisFactory.create_rotary_axis(name, motor)
        else:
            raise ValueError(f"Unknown axis type: {axis_type}")

# ============================================================================
# 3. ПАТТЕРН OBSERVER - EventSystem
# ============================================================================

class EventSystem:
    """Система событий для мониторинга состояния"""
    
    def __init__(self):
        self._observers = {}
    
    def subscribe(self, event_type, observer):
        """Подписка на событие"""
        if event_type not in self._observers:
            self._observers[event_type] = []
        self._observers[event_type].append(observer)
    
    def unsubscribe(self, event_type, observer):
        """Отписка от события"""
        if event_type in self._observers:
            self._observers[event_type].remove(observer)
    
    def notify(self, event_type, data):
        """Уведомление наблюдателей"""
        if event_type in self._observers:
            for observer in self._observers[event_type]:
                observer.update(event_type, data)

class MotorStatusObserver:
    """Наблюдатель за состоянием моторов"""
    
    def __init__(self, motor_id):
        self.motor_id = motor_id
    
    def update(self, event_type, data):
        if event_type == "motor_status":
            print(f"Motor {self.motor_id} status: {data}")

class ErrorObserver:
    """Наблюдатель за ошибками"""
    
    def update(self, event_type, data):
        if event_type == "error":
            print(f"Error occurred: {data}")

# ============================================================================
# 4. ПАТТЕРН STRATEGY - ControlStrategy
# ============================================================================

class ControlStrategy(ABC):
    """Абстрактная стратегия управления"""
    
    @abstractmethod
    def control(self, axis, target_position):
        pass

class LinearControlStrategy(ControlStrategy):
    """Стратегия управления линейными осями"""
    
    def control(self, axis, target_position):
        print(f"Linear control: moving {axis.name} to {target_position}")
        axis.move_to(target_position)

class RotaryControlStrategy(ControlStrategy):
    """Стратегия управления вращательными осями"""
    
    def control(self, axis, target_position):
        print(f"Rotary control: rotating {axis.name} to {target_position}")
        axis.move_to(target_position)

class GantryControlStrategy(ControlStrategy):
    """Стратегия управления GANTRY осями"""
    
    def control(self, axis, target_position):
        print(f"Gantry control: synchronizing {axis.name} to {target_position}")
        axis.move_to(target_position)

# ============================================================================
# 5. ПАТТЕРН COMMAND - Command Pattern
# ============================================================================

class Command(ABC):
    """Абстрактная команда"""
    
    @abstractmethod
    def execute(self):
        pass
    
    @abstractmethod
    def undo(self):
        pass

class MoveCommand(Command):
    """Команда перемещения оси"""
    
    def __init__(self, axis, target_position):
        self.axis = axis
        self.target_position = target_position
        self.previous_position = None
    
    def execute(self):
        self.previous_position = self.axis.get_position()
        self.axis.move_to(self.target_position)
    
    def undo(self):
        if self.previous_position is not None:
            self.axis.move_to(self.previous_position)

class SpindleCommand(Command):
    """Команда управления шпинделем"""
    
    def __init__(self, spindle, action, value=None):
        self.spindle = spindle
        self.action = action
        self.value = value
        self.previous_state = None
    
    def execute(self):
        self.previous_state = self.spindle.get_state()
        if self.action == "start":
            self.spindle.start()
        elif self.action == "stop":
            self.spindle.stop()
        elif self.action == "set_speed":
            self.spindle.set_speed(self.value)
    
    def undo(self):
        if self.previous_state is not None:
            self.spindle.restore_state(self.previous_state)

class CommandInvoker:
    """Выполнитель команд"""
    
    def __init__(self):
        self.history = []
    
    def execute_command(self, command):
        command.execute()
        self.history.append(command)
    
    def undo_last_command(self):
        if self.history:
            command = self.history.pop()
            command.undo()

# ============================================================================
# 6. ПАТТЕРН STATE - StateMachine
# ============================================================================

class State(ABC):
    """Абстрактное состояние"""
    
    @abstractmethod
    def enter(self):
        pass
    
    @abstractmethod
    def exit(self):
        pass
    
    @abstractmethod
    def handle_event(self, event):
        pass

class IdleState(State):
    """Состояние ожидания"""
    
    def enter(self):
        print("Machine entered Idle state")
    
    def exit(self):
        print("Machine exited Idle state")
    
    def handle_event(self, event):
        if event == "start":
            return RunningState()
        elif event == "calibrate":
            return CalibrationState()
        return self

class RunningState(State):
    """Состояние работы"""
    
    def enter(self):
        print("Machine entered Running state")
    
    def exit(self):
        print("Machine exited Running state")
    
    def handle_event(self, event):
        if event == "stop":
            return IdleState()
        elif event == "error":
            return ErrorState()
        return self

class ErrorState(State):
    """Состояние ошибки"""
    
    def enter(self):
        print("Machine entered Error state")
    
    def exit(self):
        print("Machine exited Error state")
    
    def handle_event(self, event):
        if event == "reset":
            return IdleState()
        return self

class CalibrationState(State):
    """Состояние калибровки"""
    
    def enter(self):
        print("Machine entered Calibration state")
    
    def exit(self):
        print("Machine exited Calibration state")
    
    def handle_event(self, event):
        if event == "complete":
            return IdleState()
        elif event == "error":
            return ErrorState()
        return self

class StateMachine:
    """Машина состояний"""
    
    def __init__(self):
        self.current_state = IdleState()
        self.current_state.enter()
    
    def handle_event(self, event):
        new_state = self.current_state.handle_event(event)
        if new_state != self.current_state:
            self.current_state.exit()
            self.current_state = new_state
            self.current_state.enter()

# ============================================================================
# 7. ПАТТЕРН FACADE - VMB630Facade
# ============================================================================

class VMB630Facade:
    """Фасад для упрощения взаимодействия с VMB630"""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.event_system = EventSystem()
        self.command_invoker = CommandInvoker()
        self.state_machine = StateMachine()
        self.axes = {}
        self.motors = {}
        self._initialize_system()
    
    def _initialize_system(self):
        """Инициализация системы"""
        print("Initializing VMB630 system...")
        
        # Создание моторов
        self.motors['X'] = MotorFactory.create_linear_motor(1, 1, "BISS")
        self.motors['Y1'] = MotorFactory.create_linear_motor(2, 2, "BISS")
        self.motors['Y2'] = MotorFactory.create_linear_motor(7, 7, "BISS")
        self.motors['Z'] = MotorFactory.create_linear_motor(3, 3, "BISS")
        self.motors['B'] = MotorFactory.create_rotary_motor(5, 5, "BISS")
        self.motors['C'] = MotorFactory.create_rotary_motor(6, 6, "BISS")
        
        # Создание осей
        self.axes['X'] = AxisFactory.create_linear_axis('X', self.motors['X'])
        self.axes['Y1'] = AxisFactory.create_linear_axis('Y1', self.motors['Y1'])
        self.axes['Y2'] = AxisFactory.create_linear_axis('Y2', self.motors['Y2'])
        self.axes['Z'] = AxisFactory.create_linear_axis('Z', self.motors['Z'])
        self.axes['B'] = AxisFactory.create_rotary_axis('B', self.motors['B'])
        self.axes['C'] = AxisFactory.create_rotary_axis('C', self.motors['C'])
        
        # Подписка на события
        self.event_system.subscribe("motor_status", MotorStatusObserver("X"))
        self.event_system.subscribe("error", ErrorObserver())
        
        print("VMB630 system initialized successfully!")
    
    def start_machine(self):
        """Запуск станка"""
        self.state_machine.handle_event("start")
        for motor in self.motors.values():
            motor.start()
    
    def stop_machine(self):
        """Остановка станка"""
        self.state_machine.handle_event("stop")
        for motor in self.motors.values():
            motor.stop()
    
    def move_axis(self, axis_name, position):
        """Перемещение оси"""
        if axis_name in self.axes:
            command = MoveCommand(self.axes[axis_name], position)
            self.command_invoker.execute_command(command)
        else:
            print(f"Axis {axis_name} not found")
    
    def get_status(self):
        """Получение статуса системы"""
        status = {
            "state": type(self.state_machine.current_state).__name__,
            "axes": {name: axis.get_position() for name, axis in self.axes.items()},
            "motors": {name: motor.is_running for name, motor in self.motors.items()}
        }
        return status
    
    def calibrate_axes(self):
        """Калибровка осей"""
        self.state_machine.handle_event("calibrate")
        for axis in self.axes.values():
            axis.calibrate()
        self.state_machine.handle_event("complete")

# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

def demonstrate_patterns():
    """Демонстрация применения паттернов"""
    
    print("🚀 ДЕМОНСТРАЦИЯ ПАТТЕРНОВ ПРОЕКТИРОВАНИЯ ДЛЯ VMB630")
    print("=" * 60)
    
    # Создание фасада
    vmb630 = VMB630Facade()
    
    print(f"\n📊 Статус системы:")
    status = vmb630.get_status()
    print(f"  Состояние: {status['state']}")
    print(f"  Оси: {status['axes']}")
    print(f"  Моторы: {status['motors']}")
    
    print(f"\n🎮 Управление станком:")
    vmb630.start_machine()
    vmb630.move_axis('X', 100.0)
    vmb630.move_axis('Y1', 50.0)
    vmb630.move_axis('Z', 25.0)
    
    print(f"\n📊 Обновленный статус:")
    status = vmb630.get_status()
    print(f"  Состояние: {status['state']}")
    print(f"  Оси: {status['axes']}")
    
    print(f"\n🔧 Калибровка:")
    vmb630.calibrate_axes()
    
    print(f"\n⏹️ Остановка:")
    vmb630.stop_machine()
    
    print(f"\n✅ Демонстрация завершена!")

if __name__ == "__main__":
    demonstrate_patterns()





