#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConfigurationManager и EventSystem для VMB630
Реализация паттернов Singleton и Observer для улучшения архитектуры
"""

import os
import json
import xml.etree.ElementTree as ET
import configparser
import logging
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import threading
from datetime import datetime

class ConfigurationManager:
    """
    Singleton для управления всеми конфигурациями VMB630
    Паттерн Singleton обеспечивает единую точку доступа к конфигурациям
    """
    
    _instance = None
    _initialized = False
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not ConfigurationManager._initialized:
            self.logger = logging.getLogger(__name__)
            self._configs = {}
            self._config_paths = {}
            self._last_modified = {}
            self._setup_logging()
            self._initialize_default_paths()
            ConfigurationManager._initialized = True
    
    def _setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger.info("ConfigurationManager инициализирован")
    
    def _initialize_default_paths(self):
        """Инициализация путей к конфигурационным файлам"""
        # Базовый путь к проекту VMB630
        base_path = r"C:\Users\elekt\OneDrive\Desktop\VMB630_v_005_019_000"
        
        self._config_paths = {
            'definitions': os.path.join(base_path, 'define.xml'),
            'start_config': os.path.join(base_path, 'start.cfg'),
            'start_stend': os.path.join(base_path, 'start_stend.cfg'),
            'pult_ctrl': os.path.join(base_path, 'pult_ctrl.cfg'),
            'errors': os.path.join(base_path, 'errors.xml'),
            'motors': os.path.join(base_path, 'motors.cfg'),
            'axes': os.path.join(base_path, 'axes.cfg'),
            'spindles': os.path.join(base_path, 'spindles.cfg'),
            'encoders': os.path.join(base_path, 'encoders.cfg'),
            'plc_config': os.path.join(base_path, 'plc_config.ini'),
            'vmb630_info': os.path.join(base_path, 'VMB630_info.txt')
        }
        
        self.logger.info(f"Инициализированы пути к конфигурациям: {len(self._config_paths)} файлов")
    
    def load_all_configurations(self) -> bool:
        """Загрузка всех конфигураций"""
        try:
            self.logger.info("Начинаем загрузку всех конфигураций...")
            
            # Загружаем XML файлы
            xml_files = ['definitions', 'errors']
            for config_name in xml_files:
                if config_name in self._config_paths:
                    self._load_xml_config(config_name)
            
            # Загружаем CFG файлы
            cfg_files = ['start_config', 'start_stend', 'pult_ctrl', 'motors', 'axes', 'spindles', 'encoders']
            for config_name in cfg_files:
                if config_name in self._config_paths:
                    self._load_cfg_config(config_name)
            
            # Загружаем INI файлы
            ini_files = ['plc_config']
            for config_name in ini_files:
                if config_name in self._config_paths:
                    self._load_ini_config(config_name)
            
            # Загружаем TXT файлы
            txt_files = ['vmb630_info']
            for config_name in txt_files:
                if config_name in self._config_paths:
                    self._load_txt_config(config_name)
            
            self.logger.info(f"✅ Загружено {len(self._configs)} конфигураций")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки конфигураций: {e}")
            return False
    
    def _load_xml_config(self, config_name: str):
        """Загрузка XML конфигурации"""
        try:
            file_path = self._config_paths[config_name]
            if os.path.exists(file_path):
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # Преобразуем XML в словарь
                config_data = self._xml_to_dict(root)
                self._configs[config_name] = config_data
                self._last_modified[config_name] = os.path.getmtime(file_path)
                
                self.logger.info(f"✅ Загружен XML: {config_name} ({len(config_data)} элементов)")
            else:
                self.logger.warning(f"⚠️ Файл не найден: {file_path}")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки XML {config_name}: {e}")
    
    def _load_cfg_config(self, config_name: str):
        """Загрузка CFG конфигурации"""
        try:
            file_path = self._config_paths[config_name]
            if os.path.exists(file_path):
                config_data = {}
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                current_section = 'default'
                config_data[current_section] = []
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith(';'):
                        continue
                    
                    if line.startswith('[') and line.endswith(']'):
                        current_section = line[1:-1]
                        config_data[current_section] = []
                    else:
                        config_data[current_section].append(line)
                
                self._configs[config_name] = config_data
                self._last_modified[config_name] = os.path.getmtime(file_path)
                
                self.logger.info(f"✅ Загружен CFG: {config_name} ({len(config_data)} секций)")
            else:
                self.logger.warning(f"⚠️ Файл не найден: {file_path}")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки CFG {config_name}: {e}")
    
    def _load_ini_config(self, config_name: str):
        """Загрузка INI конфигурации"""
        try:
            file_path = self._config_paths[config_name]
            if os.path.exists(file_path):
                config = configparser.ConfigParser()
                config.read(file_path, encoding='utf-8')
                
                config_data = {}
                for section in config.sections():
                    config_data[section] = dict(config[section])
                
                self._configs[config_name] = config_data
                self._last_modified[config_name] = os.path.getmtime(file_path)
                
                self.logger.info(f"✅ Загружен INI: {config_name} ({len(config_data)} секций)")
            else:
                self.logger.warning(f"⚠️ Файл не найден: {file_path}")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки INI {config_name}: {e}")
    
    def _load_txt_config(self, config_name: str):
        """Загрузка TXT конфигурации"""
        try:
            file_path = self._config_paths[config_name]
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                self._configs[config_name] = {'content': content}
                self._last_modified[config_name] = os.path.getmtime(file_path)
                
                self.logger.info(f"✅ Загружен TXT: {config_name}")
            else:
                self.logger.warning(f"⚠️ Файл не найден: {file_path}")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки TXT {config_name}: {e}")
    
    def _xml_to_dict(self, element) -> Dict:
        """Преобразование XML элемента в словарь"""
        result = {}
        
        # Добавляем атрибуты
        if element.attrib:
            result['@attributes'] = element.attrib
        
        # Добавляем текст
        if element.text and element.text.strip():
            result['text'] = element.text.strip()
        
        # Добавляем дочерние элементы
        for child in element:
            child_data = self._xml_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        
        return result
    
    # Методы доступа к конфигурациям
    def get_definition(self, key: str) -> Any:
        """Получение определения по ключу"""
        if 'definitions' in self._configs:
            return self._find_in_dict(self._configs['definitions'], key)
        return None
    
    def get_start_config(self) -> Dict:
        """Получение конфигурации запуска"""
        return self._configs.get('start_config', {})
    
    def get_error_codes(self) -> Dict:
        """Получение кодов ошибок"""
        return self._configs.get('errors', {})
    
    def get_motor_config(self, motor_id: str) -> Dict:
        """Получение конфигурации мотора"""
        motors_config = self._configs.get('motors', {})
        return self._find_motor_config(motors_config, motor_id)
    
    def get_axis_config(self, axis_name: str) -> Dict:
        """Получение конфигурации оси"""
        axes_config = self._configs.get('axes', {})
        return self._find_axis_config(axes_config, axis_name)
    
    def get_spindle_config(self, spindle_name: str) -> Dict:
        """Получение конфигурации шпинделя"""
        spindles_config = self._configs.get('spindles', {})
        return self._find_spindle_config(spindles_config, spindle_name)
    
    def get_plc_config(self, section: str = None) -> Dict:
        """Получение конфигурации PLC"""
        plc_config = self._configs.get('plc_config', {})
        if section:
            return plc_config.get(section, {})
        return plc_config
    
    def get_vmb630_info(self) -> str:
        """Получение информации о VMB630"""
        info_config = self._configs.get('vmb630_info', {})
        return info_config.get('content', '')
    
    def _find_in_dict(self, data: Dict, key: str) -> Any:
        """Поиск ключа в словаре"""
        if isinstance(data, dict):
            if key in data:
                return data[key]
            for value in data.values():
                result = self._find_in_dict(value, key)
                if result is not None:
                    return result
        elif isinstance(data, list):
            for item in data:
                result = self._find_in_dict(item, key)
                if result is not None:
                    return result
        return None
    
    def _find_motor_config(self, motors_config: Dict, motor_id: str) -> Dict:
        """Поиск конфигурации мотора"""
        for section_name, section_data in motors_config.items():
            if isinstance(section_data, list):
                for item in section_data:
                    if isinstance(item, str) and motor_id.lower() in item.lower():
                        return {'section': section_name, 'config': item}
        return {}
    
    def _find_axis_config(self, axes_config: Dict, axis_name: str) -> Dict:
        """Поиск конфигурации оси"""
        for section_name, section_data in axes_config.items():
            if isinstance(section_data, list):
                for item in section_data:
                    if isinstance(item, str) and axis_name.upper() in item.upper():
                        return {'section': section_name, 'config': item}
        return {}
    
    def _find_spindle_config(self, spindles_config: Dict, spindle_name: str) -> Dict:
        """Поиск конфигурации шпинделя"""
        for section_name, section_data in spindles_config.items():
            if isinstance(section_data, list):
                for item in section_data:
                    if isinstance(item, str) and spindle_name.upper() in item.upper():
                        return {'section': section_name, 'config': item}
        return {}
    
    def reload_config(self, config_name: str) -> bool:
        """Перезагрузка конкретной конфигурации"""
        try:
            if config_name in self._config_paths:
                if config_name in ['definitions', 'errors']:
                    self._load_xml_config(config_name)
                elif config_name in ['start_config', 'start_stend', 'pult_ctrl', 'motors', 'axes', 'spindles', 'encoders']:
                    self._load_cfg_config(config_name)
                elif config_name in ['plc_config']:
                    self._load_ini_config(config_name)
                elif config_name in ['vmb630_info']:
                    self._load_txt_config(config_name)
                
                self.logger.info(f"✅ Перезагружена конфигурация: {config_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"❌ Ошибка перезагрузки {config_name}: {e}")
            return False
    
    def get_config_status(self) -> Dict:
        """Получение статуса всех конфигураций"""
        status = {
            'total_configs': len(self._configs),
            'loaded_configs': list(self._configs.keys()),
            'config_details': {}
        }
        
        for config_name, config_data in self._configs.items():
            status['config_details'][config_name] = {
                'loaded': True,
                'size': len(str(config_data)),
                'last_modified': self._last_modified.get(config_name, 0)
            }
        
        return status


class EventSystem:
    """
    Система событий для мониторинга состояния VMB630
    Паттерн Observer обеспечивает слабую связанность между компонентами
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._observers = {}
        self._event_history = []
        self._max_history = 1000
        self._lock = threading.Lock()
        self.logger.info("EventSystem инициализирован")
    
    def subscribe(self, event_type: str, observer: Callable, observer_name: str = None):
        """Подписка на событие"""
        with self._lock:
            if event_type not in self._observers:
                self._observers[event_type] = []
            
            observer_info = {
                'callback': observer,
                'name': observer_name or f"Observer_{len(self._observers[event_type])}",
                'subscribed_at': datetime.now()
            }
            
            self._observers[event_type].append(observer_info)
            self.logger.info(f"✅ Подписка на {event_type}: {observer_info['name']}")
    
    def unsubscribe(self, event_type: str, observer_name: str):
        """Отписка от события"""
        with self._lock:
            if event_type in self._observers:
                self._observers[event_type] = [
                    obs for obs in self._observers[event_type] 
                    if obs['name'] != observer_name
                ]
                self.logger.info(f"✅ Отписка от {event_type}: {observer_name}")
    
    def notify(self, event_type: str, data: Any, source: str = "Unknown"):
        """Уведомление наблюдателей"""
        with self._lock:
            # Добавляем событие в историю
            event = {
                'type': event_type,
                'data': data,
                'source': source,
                'timestamp': datetime.now()
            }
            
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
            
            # Уведомляем наблюдателей
            if event_type in self._observers:
                for observer_info in self._observers[event_type]:
                    try:
                        observer_info['callback'](event)
                        self.logger.debug(f"📡 Уведомлен {observer_info['name']} о {event_type}")
                    except Exception as e:
                        self.logger.error(f"❌ Ошибка уведомления {observer_info['name']}: {e}")
            
            self.logger.info(f"📡 Событие {event_type} от {source}: {len(self._observers.get(event_type, []))} наблюдателей")
    
    def get_event_history(self, event_type: str = None, limit: int = 100) -> List[Dict]:
        """Получение истории событий"""
        with self._lock:
            if event_type:
                filtered_events = [e for e in self._event_history if e['type'] == event_type]
            else:
                filtered_events = self._event_history
            
            return filtered_events[-limit:]
    
    def get_observers_status(self) -> Dict:
        """Получение статуса наблюдателей"""
        with self._lock:
            status = {
                'total_event_types': len(self._observers),
                'total_observers': sum(len(obs) for obs in self._observers.values()),
                'event_types': {}
            }
            
            for event_type, observers in self._observers.items():
                status['event_types'][event_type] = {
                    'observer_count': len(observers),
                    'observers': [obs['name'] for obs in observers]
                }
            
            return status
    
    def clear_history(self):
        """Очистка истории событий"""
        with self._lock:
            self._event_history.clear()
            self.logger.info("🧹 История событий очищена")


# Наблюдатели для VMB630
class MotorStatusObserver:
    """Наблюдатель за состоянием моторов"""
    
    def __init__(self, motor_id: str):
        self.motor_id = motor_id
        self.logger = logging.getLogger(__name__)
        self.status_history = []
    
    def update(self, event: Dict):
        """Обновление состояния мотора"""
        if event['type'] == "motor_status":
            status_data = {
                'motor_id': self.motor_id,
                'status': event['data'],
                'timestamp': event['timestamp'],
                'source': event['source']
            }
            self.status_history.append(status_data)
            
            # Ограничиваем историю
            if len(self.status_history) > 100:
                self.status_history.pop(0)
            
            self.logger.info(f"🔧 Мотор {self.motor_id}: {event['data']}")


class ErrorObserver:
    """Наблюдатель за ошибками"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history = []
    
    def update(self, event: Dict):
        """Обработка ошибки"""
        if event['type'] == "error":
            error_data = {
                'error': event['data'],
                'timestamp': event['timestamp'],
                'source': event['source']
            }
            self.error_history.append(error_data)
            
            # Ограничиваем историю
            if len(self.error_history) > 50:
                self.error_history.pop(0)
            
            self.logger.error(f"❌ Ошибка: {event['data']} от {event['source']}")


class PositionObserver:
    """Наблюдатель за позициями осей"""
    
    def __init__(self, axis_name: str):
        self.axis_name = axis_name
        self.logger = logging.getLogger(__name__)
        self.position_history = []
    
    def update(self, event: Dict):
        """Обновление позиции оси"""
        if event['type'] == "position_update":
            position_data = {
                'axis': self.axis_name,
                'position': event['data'],
                'timestamp': event['timestamp'],
                'source': event['source']
            }
            self.position_history.append(position_data)
            
            # Ограничиваем историю
            if len(self.position_history) > 200:
                self.position_history.pop(0)
            
            self.logger.debug(f"📍 Ось {self.axis_name}: {event['data']}")


# Демонстрация использования
def demonstrate_configuration_and_events():
    """Демонстрация работы ConfigurationManager и EventSystem"""
    
    print("🚀 ДЕМОНСТРАЦИЯ CONFIGURATIONMANAGER И EVENTSYSTEM ДЛЯ VMB630")
    print("=" * 70)
    
    # Инициализация ConfigurationManager
    print("\n📋 ИНИЦИАЛИЗАЦИЯ CONFIGURATIONMANAGER:")
    config_manager = ConfigurationManager()
    
    # Загрузка конфигураций
    print("\n🔄 ЗАГРУЗКА КОНФИГУРАЦИЙ:")
    success = config_manager.load_all_configurations()
    
    if success:
        print("✅ Все конфигурации загружены успешно!")
        
        # Получение статуса
        status = config_manager.get_config_status()
        print(f"\n📊 СТАТУС КОНФИГУРАЦИЙ:")
        print(f"  Всего конфигураций: {status['total_configs']}")
        print(f"  Загруженные: {', '.join(status['loaded_configs'])}")
        
        # Примеры использования
        print(f"\n🔍 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:")
        
        # Получение информации о VMB630
        vmb_info = config_manager.get_vmb630_info()
        if vmb_info:
            print(f"  📄 VMB630 Info: {vmb_info[:100]}...")
        
        # Получение конфигурации мотора
        motor_config = config_manager.get_motor_config("X")
        if motor_config:
            print(f"  🔧 Motor X Config: {motor_config}")
        
        # Получение конфигурации оси
        axis_config = config_manager.get_axis_config("X")
        if axis_config:
            print(f"  📍 Axis X Config: {axis_config}")
    
    # Инициализация EventSystem
    print(f"\n📡 ИНИЦИАЛИЗАЦИЯ EVENTSYSTEM:")
    event_system = EventSystem()
    
    # Создание наблюдателей
    print(f"\n👁️ СОЗДАНИЕ НАБЛЮДАТЕЛЕЙ:")
    motor_observer = MotorStatusObserver("X")
    error_observer = ErrorObserver()
    position_observer = PositionObserver("X")
    
    # Подписка на события
    print(f"\n📝 ПОДПИСКА НА СОБЫТИЯ:")
    event_system.subscribe("motor_status", motor_observer.update, "MotorXObserver")
    event_system.subscribe("error", error_observer.update, "ErrorObserver")
    event_system.subscribe("position_update", position_observer.update, "PositionXObserver")
    
    # Генерация событий
    print(f"\n📡 ГЕНЕРАЦИЯ СОБЫТИЙ:")
    event_system.notify("motor_status", "STARTED", "MotorController")
    event_system.notify("position_update", 100.5, "AxisController")
    event_system.notify("motor_status", "RUNNING", "MotorController")
    event_system.notify("position_update", 150.2, "AxisController")
    event_system.notify("error", "Position limit exceeded", "SafetySystem")
    
    # Получение статуса наблюдателей
    observers_status = event_system.get_observers_status()
    print(f"\n📊 СТАТУС НАБЛЮДАТЕЛЕЙ:")
    print(f"  Типов событий: {observers_status['total_event_types']}")
    print(f"  Всего наблюдателей: {observers_status['total_observers']}")
    
    for event_type, info in observers_status['event_types'].items():
        print(f"  {event_type}: {info['observer_count']} наблюдателей")
    
    # Получение истории событий
    print(f"\n📜 ИСТОРИЯ СОБЫТИЙ:")
    history = event_system.get_event_history(limit=5)
    for event in history:
        print(f"  {event['timestamp'].strftime('%H:%M:%S')} - {event['type']}: {event['data']}")
    
    print(f"\n🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
    print("ConfigurationManager и EventSystem готовы к использованию в VMB630!")


if __name__ == "__main__":
    demonstrate_configuration_and_events()










