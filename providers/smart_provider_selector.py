"""
Rubin AI v2.0 - Умный селектор провайдеров
Автоматический выбор оптимального AI провайдера для задачи
"""

import re
from typing import Dict, List, Optional, Any, Tuple
import logging

from .base_provider import BaseProvider, TaskType, ResponseFormat

class SmartProviderSelector:
    """Умный селектор для выбора оптимального AI провайдера"""
    
    def __init__(self):
        self.providers: Dict[str, BaseProvider] = {}
        self.logger = logging.getLogger("rubin_ai.smart_selector")
        self.task_patterns = self._initialize_task_patterns()
        
    def _initialize_task_patterns(self) -> Dict[str, List[str]]:
        """Инициализация паттернов для определения типа задачи"""
        return {
            # Программирование и разработка
            TaskType.CODE_ANALYSIS: [
                r'анализ.*код', r'провер.*код', r'ошибк.*код', r'debug', r'отладк',
                r'code.*analysis', r'code.*review', r'code.*check', r'баг', r'bug',
                r'исправ.*ошибк', r'fix.*bug', r'логи.*ошибк', r'error.*log'
            ],
            TaskType.CODE_GENERATION: [
                r'созда.*код', r'напиш.*код', r'генерир.*код', r'generate.*code',
                r'write.*code', r'код.*для', r'функци.*для', r'алгоритм.*для',
                r'программ.*для', r'скрипт.*для', r'класс.*для', r'method.*for'
            ],
            
            # Программирование и ООП
            'programming_analysis': [
                r'программирован', r'programming', r'код', r'code', r'алгоритм', r'algorithm',
                r'объектно.*ориентированн', r'object.*oriented', r'ооп', r'oop',
                r'принцип.*программирован', r'programming.*principles', r'класс', r'class',
                r'объект', r'object', r'наследован', r'inheritance', r'полиморфизм', r'polymorphism',
                r'инкапсуляц', r'encapsulation', r'абстракц', r'abstraction', r'интерфейс', r'interface',
                r'метод', r'method', r'функц', r'function', r'переменн', r'variable'
            ],
            TaskType.SECURITY_CHECK: [
                r'безопасност', r'уязвимост', r'security', r'vulnerability', r'защит',
                r'провер.*безопасност', r'security.*check', r'хакер', r'hacker',
                r'взлом', r'breach', r'шифрован', r'encryption', r'аутентификац'
            ],
            
            # Промышленная автоматизация
            TaskType.PLC_ANALYSIS: [
                r'plc', r'логическ.*контроллер', r'промышленн.*автоматизац',
                r'iec.*61131', r'step.*7', r'tia.*portal', r'сименс', r'siemens',
                r'allen.*bradley', r'rockwell', r'omron', r'mitsubishi',
                r'ladder.*logic', r'лестничн.*логик', r'функциональн.*блок',
                r'structured.*text', r'структурированн.*текст', r'instruction.*list',
                r'modbus', r'rtu', r'tcp', r'протокол', r'protocol',
                r'программирован.*plc', r'programming.*plc', r'автоматизац.*производств',
                r'automation.*production', r'промышленн.*контроллер'
            ],
            TaskType.PMAC_ANALYSIS: [
                r'pmac', r'delta.*tau', r'чпу.*контроллер', r'cnc.*контроллер',
                r'серво.*двигател', r'servo.*motor', r'шагов.*двигател', r'stepper',
                r'энкодер', r'encoder', r'обратн.*связ', r'feedback'
            ],
            TaskType.CNC_ANALYSIS: [
                r'g.*code', r'gcode', r'чпу', r'cnc', r'фрезер', r'токарн',
                r'станок', r'machine.*tool', r'обработк.*металл', r'machining',
                r'координат.*систем', r'coordinate.*system', r'траектори', r'path'
            ],
            
            # Электроника и схемотехника
            TaskType.SCHEMATIC_ANALYSIS: [
                r'схем', r'чертеж', r'электрическ.*схем', r'принципиальн.*схем',
                r'schematic', r'circuit.*diagram', r'электронн.*схем', r'pcb',
                r'печатн.*плат', r'printed.*circuit', r'компонент', r'component',
                r'резистор', r'resistor', r'конденсатор', r'capacitor', r'транзистор'
            ],
            
            # Датчики и измерения
            'sensors_analysis': [
                r'датчик', r'sensor', r'измерен', r'measurement', r'температур',
                r'temperature', r'давлен', r'pressure', r'поток', r'flow',
                r'позиц', r'position', r'скорост', r'velocity', r'ускорен', r'acceleration',
                r'термопар', r'thermocouple', r'rtd', r'потенциометр', r'potentiometer'
            ],
            
            # Радиомеханика и связь
            'radiomechanics_analysis': [
                r'радио', r'radio', r'радиопередатчик', r'radio.*transmitter', r'радиосвяз', r'radio.*communication',
                r'частота', r'frequency', r'антенн', r'antenna', r'полос.*пропускан', r'bandwidth',
                r'радиоволн', r'radio.*wave', r'передатчик', r'transmitter', r'приемник', r'receiver',
                r'модуляц', r'modulation', r'демодуляц', r'demodulation', r'сигнал', r'signal',
                r'радиочастот', r'radio.*frequency', r'связ', r'communication', r'передач', r'transmission'
            ],
            
            # Электрика и силовая электроника
            'electrical_analysis': [
                r'электрическ', r'electrical', r'контактор', r'contactor', r'реле', r'relay',
                r'выключател', r'switch', r'автомат', r'circuit.*breaker', r'защит',
                r'protection', r'мощност', r'power', r'напряжен', r'voltage', r'ток', r'current',
                r'частотн.*преобразовател', r'frequency.*converter', r'инвертор', r'inverter',
                r'трансформатор', r'transformer', r'принцип.*работ', r'principle.*work',
                r'электрическ.*цеп', r'electrical.*circuit', r'коротк.*замыкан', r'short.*circuit',
                r'защит.*цеп', r'circuit.*protection', r'напряжен', r'voltage', r'мощност', r'power'
            ],
            
            # Мультимедиа и анализ
            TaskType.IMAGE_ANALYSIS: [
                r'анализ.*изображен', r'что.*на.*картинк', r'распозна.*изображен',
                r'image.*analysis', r'picture.*analysis', r'фото', r'photo',
                r'картинк', r'image', r'видео', r'video', r'камер', r'camera'
            ],
            TaskType.SPEECH_TO_TEXT: [
                r'распозна.*реч', r'преобраз.*реч.*текст', r'speech.*to.*text',
                r'что.*сказал', r'расшифров.*аудио', r'аудио', r'audio',
                r'звук', r'sound', r'микрофон', r'microphone'
            ],
            
            # Документация и обучение
            TaskType.TECHNICAL_DOCUMENTATION: [
                r'документац', r'техническ.*документ', r'руководств', r'инструкц',
                r'technical.*documentation', r'manual', r'guide', r'спецификац',
                r'specification', r'техническ.*задан', r'technical.*requirement'
            ],
            TaskType.DOCUMENTATION: [
                r'документац', r'описани', r'комментар', r'documentation',
                r'comment', r'опис.*код', r'readme', r'api.*doc', r'help'
            ],
            
            # Наука и математика
            'science_analysis': [
                r'физик', r'physics', r'математик', r'mathematics', r'формул', r'formula',
                r'расчет', r'calculation', r'уравнен', r'equation', r'интеграл', r'integral',
                r'дифференциал', r'differential', r'статистик', r'statistics', r'вероятност'
            ],
            
            # ИИ и машинное обучение
            'ai_analysis': [
                r'искусственн.*интеллект', r'artificial.*intelligence', r'машинн.*обучен',
                r'machine.*learning', r'нейронн.*сет', r'neural.*network', r'алгоритм',
                r'algorithm', r'модел', r'model', r'тренировк', r'training', r'данн', r'data'
            ],
            
            # Общие технические вопросы
            'general_technical': [
                r'как.*работает', r'how.*works', r'принцип.*работ', r'principle',
                r'устройств', r'device', r'механизм', r'mechanism', r'систем', r'system',
                r'процесс', r'process', r'технологи', r'technology', r'инженер', r'engineer'
            ]
        }
    
    def register_provider(self, provider: BaseProvider):
        """Зарегистрировать провайдера"""
        self.providers[provider.name] = provider
        self.logger.info(f"Зарегистрирован провайдер: {provider.name}")
    
    def detect_task_type(self, message: str, context: Optional[Dict] = None) -> str:
        """Определить тип задачи на основе сообщения и контекста"""
        message_lower = message.lower()
        
        # Проверяем контекст
        if context and 'task_type' in context:
            return context['task_type']
        
        # Проверяем наличие файлов в контексте
        if context:
            if 'image_path' in context:
                return TaskType.IMAGE_ANALYSIS
            elif 'audio_path' in context:
                return TaskType.SPEECH_TO_TEXT
            elif 'file_extension' in context:
                ext = context['file_extension'].lower()
                if ext in ['.plc', '.st', '.iec']:
                    return TaskType.PLC_ANALYSIS
                elif ext in ['.pmac', '.pma']:
                    return TaskType.PMAC_ANALYSIS
                elif ext in ['.gcode', '.nc', '.cnc']:
                    return TaskType.CNC_ANALYSIS
                elif ext in ['.sch', '.brd', '.kicad_pcb']:
                    return TaskType.SCHEMATIC_ANALYSIS
        
        # Анализируем текст сообщения с улучшенным алгоритмом
        task_scores = {}
        
        for task_type, patterns in self.task_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, message_lower)
                if matches:
                    # Учитываем количество совпадений и их длину
                    score += len(matches) * (1 + len(pattern) / 20)  # Более длинные паттерны важнее
            
            # Дополнительные бонусы за ключевые слова
            if task_type == TaskType.CODE_ANALYSIS:
                code_bonus_words = ['ошибка', 'error', 'баг', 'bug', 'не работает', 'doesn\'t work']
                score += sum(2 for word in code_bonus_words if word in message_lower)
            elif task_type == TaskType.CODE_GENERATION:
                gen_bonus_words = ['создать', 'create', 'написать', 'write', 'сделать', 'make']
                score += sum(2 for word in gen_bonus_words if word in message_lower)
            elif task_type == 'sensors_analysis':
                sensor_bonus_words = ['калибровка', 'calibration', 'точность', 'accuracy', 'погрешность']
                score += sum(1.5 for word in sensor_bonus_words if word in message_lower)
            elif task_type == 'electrical_analysis':
                elec_bonus_words = ['короткое замыкание', 'short circuit', 'перегрузка', 'overload']
                score += sum(1.5 for word in elec_bonus_words if word in message_lower)
            
            task_scores[task_type] = score
        
        # Находим задачу с максимальным счетом
        if task_scores:
            best_task = max(task_scores, key=task_scores.get)
            if task_scores[best_task] > 0:
                self.logger.info(f"Определена категория: {best_task} (счет: {task_scores[best_task]:.2f})")
                return best_task
        
        # Проверяем ключевые слова для кода (улучшенная версия)
        code_keywords = ['код', 'code', 'функция', 'function', 'класс', 'class', 'метод', 'method', 
                        'программ', 'program', 'алгоритм', 'algorithm', 'скрипт', 'script']
        if any(keyword in message_lower for keyword in code_keywords):
            return TaskType.CODE_ANALYSIS
        
        # Проверяем технические вопросы
        tech_keywords = ['как работает', 'how works', 'принцип', 'principle', 'устройство', 'device']
        if any(keyword in message_lower for keyword in tech_keywords):
            return 'general_technical'
        
        # По умолчанию - общий чат
        return TaskType.GENERAL_CHAT
    
    def select_best_provider(self, task_type: str, context: Optional[Dict] = None) -> Optional[BaseProvider]:
        """Выбрать лучший провайдер для задачи"""
        available_providers = []
        
        for provider in self.providers.values():
            if provider.is_available and provider.is_suitable_for_task(task_type):
                priority = provider.get_priority_for_task(task_type)
                available_providers.append((priority, provider))
        
        if not available_providers:
            # Если нет подходящих провайдеров, выбираем любой доступный
            for provider in self.providers.values():
                if provider.is_available:
                    available_providers.append((provider.priority, provider))
        
        if not available_providers:
            return None
        
        # Сортируем по приоритету (меньше число = выше приоритет)
        available_providers.sort(key=lambda x: x[0])
        
        best_provider = available_providers[0][1]
        self.logger.info(f"Выбран провайдер {best_provider.name} для задачи {task_type}")
        
        return best_provider
    
    def get_response(self, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Получить ответ от оптимального провайдера"""
        try:
            # Определяем тип задачи
            task_type = self.detect_task_type(message, context)
            
            # Выбираем лучший провайдер
            provider = self.select_best_provider(task_type, context)
            
            if not provider:
                return ResponseFormat.create_error_response(
                    "Нет доступных AI провайдеров",
                    "smart_selector",
                    task_type
                )
            
            # Обновляем контекст с типом задачи
            if context is None:
                context = {}
            context['task_type'] = task_type
            
            # Получаем ответ от провайдера
            response = provider.get_response(message, context)
            
            # Добавляем информацию о выборе провайдера
            response['provider_selection'] = {
                'selected_provider': provider.name,
                'task_type': task_type,
                'available_providers': [p.name for p in self.providers.values() if p.is_available],
                'selection_reason': f"Выбран {provider.name} для задачи {task_type}"
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Ошибка в SmartProviderSelector: {e}")
            return ResponseFormat.create_error_response(
                f"Ошибка выбора провайдера: {str(e)}",
                "smart_selector",
                context.get('task_type', TaskType.GENERAL_CHAT) if context else TaskType.GENERAL_CHAT
            )
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Получить статус всех провайдеров"""
        status = {
            'total_providers': len(self.providers),
            'available_providers': 0,
            'providers': {}
        }
        
        for name, provider in self.providers.items():
            provider_status = provider.get_status()
            status['providers'][name] = provider_status
            
            if provider.is_available:
                status['available_providers'] += 1
        
        return status
    
    def get_capabilities_summary(self) -> Dict[str, List[str]]:
        """Получить сводку возможностей всех провайдеров"""
        capabilities = {}
        
        for name, provider in self.providers.items():
            if provider.is_available:
                capabilities[name] = provider.get_capabilities()
        
        return capabilities
    
    def test_provider(self, provider_name: str, test_message: str = "Тестовое сообщение") -> Dict[str, Any]:
        """Протестировать конкретный провайдер"""
        if provider_name not in self.providers:
            return {
                'success': False,
                'error': f"Провайдер {provider_name} не найден"
            }
        
        provider = self.providers[provider_name]
        
        if not provider.is_available:
            return {
                'success': False,
                'error': f"Провайдер {provider_name} недоступен"
            }
        
        try:
            response = provider.get_response(test_message)
            return {
                'success': True,
                'provider': provider_name,
                'response': response
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Ошибка тестирования {provider_name}: {str(e)}"
            }
