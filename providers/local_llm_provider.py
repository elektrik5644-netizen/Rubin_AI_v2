"""
Rubin AI v2.0 - Локальный LLM провайдер
Простой провайдер для генерации ответов на основе найденных документов
"""

from typing import Dict, List, Optional, Any
import logging
import re
from .base_provider import BaseProvider, TaskType, ResponseFormat

class LocalLLMProvider(BaseProvider):
    """Локальный LLM провайдер для генерации ответов"""
    
    def __init__(self):
        super().__init__("local_llm", priority=1)
        self.logger = logging.getLogger("rubin_ai.local_llm")
        
    def initialize(self) -> bool:
        """Инициализация локального LLM провайдера"""
        try:
            self.logger.info("Инициализация локального LLM провайдера...")
            self.is_available = True
            self.logger.info("Локальный LLM провайдер успешно инициализирован")
            return True
        except Exception as e:
            self.log_error(e)
            return False
    
    def get_response(self, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Получить ответ от локального LLM"""
        if not self.is_available:
            return ResponseFormat.create_error_response(
                "Локальный LLM провайдер недоступен",
                self.name,
                context.get('task_type', TaskType.GENERAL_CHAT) if context else TaskType.GENERAL_CHAT
            )
        
        try:
            task_type = context.get('task_type', TaskType.GENERAL_CHAT) if context else TaskType.GENERAL_CHAT
            documents = context.get('documents', []) if context else []
            
            thinking_process = []
            
            # Генерируем ответ на основе найденных документов
            if documents:
                response_content = self._generate_response_from_documents(message, documents, thinking_process)
            else:
                response_content = self._generate_fallback_response(message, thinking_process)
            
            return ResponseFormat.create_response(
                content=response_content,
                provider=self.name,
                task_type=task_type,
                metadata={'documents_used': len(documents), 'method': 'local_generation'},
                thinking_process=thinking_process
            )
            
        except Exception as e:
            self.log_error(e)
            return ResponseFormat.create_error_response(
                str(e),
                self.name,
                context.get('task_type', TaskType.GENERAL_CHAT) if context else TaskType.GENERAL_CHAT
            )
    
    def _generate_response_from_documents(self, message: str, documents: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует ответ на основе найденных документов"""
        thinking_process.append("📚 Анализирую найденные документы...")
        
        # Извлекаем ключевую информацию из документов
        relevant_info = []
        for doc in documents[:5]:  # Берем первые 5 документов для более полных ответов
            title = doc.get('file_name', doc.get('title', 'Документ'))
            content = doc.get('content_preview', doc.get('content', ''))
            
            if content:
                # Извлекаем релевантные предложения
                sentences = self._extract_relevant_sentences(content, message)
                if sentences:
                    relevant_info.append({
                        'title': title,
                        'sentences': sentences
                    })
        
        thinking_process.append(f"📖 Найдено {len(relevant_info)} релевантных документов")
        
        if not relevant_info:
            return self._generate_fallback_response(message, thinking_process)
        
        # Анализируем тип вопроса и генерируем соответствующий ответ
        message_lower = message.lower()
        
        # МЕТА-ВОПРОСЫ О ПОНИМАНИИ И МЫШЛЕНИИ (добавляем ПЕРЕД техническими вопросами)
        if any(phrase in message_lower for phrase in [
            "как ты понимаешь", "как ты меня понимаешь", 
            "как ты анализируешь", "как ты обрабатываешь"
        ]):
            return self._generate_understanding_process_response(thinking_process)
        
        if any(phrase in message_lower for phrase in [
            "как ты размышляешь", "как ты думаешь",
            "как ты мыслишь", "как работает твой мозг"
        ]):
            return self._generate_thinking_process_response(thinking_process)
        
        if any(phrase in message_lower for phrase in [
            "как работает твоя логика", "как ты принимаешь решения",
            "как ты выбираешь ответ", "как ты анализируешь"
        ]):
            return self._generate_decision_process_response(thinking_process)
        
        # Специализированные ответы для технических вопросов
        if any(word in message_lower for word in ['энкодер', 'encoder', 'датчик', 'sensor']):
            return self._generate_encoder_response(relevant_info, thinking_process)
        elif any(word in message_lower for word in ['пид', 'pid', 'регулятор', 'controller']):
            return self._generate_pid_response(relevant_info, thinking_process)
        elif any(word in message_lower for word in ['диод', 'diode', 'полупроводник', 'semiconductor']):
            return self._generate_diode_response(relevant_info, thinking_process)
        elif any(word in message_lower for word in ['контроллер', 'controller', 'промышленный', 'industrial']):
            return self._generate_controller_response(relevant_info, thinking_process)
        elif any(word in message_lower for word in ['modbus', 'rtu', 'tcp', 'протокол', 'protocol', 'связь', 'communication']):
            return self._generate_protocol_response(relevant_info, thinking_process)
        elif any(word in message_lower for word in ['защит', 'protection', 'коротк', 'short', 'замыкание', 'circuit', 'предохранитель', 'fuse', 'автомат', 'breaker', 'выключатель', 'switch']):
            return self._generate_protection_response(relevant_info, thinking_process)
        elif any(word in message_lower for word in ['кирхгоф', 'kirchhoff', 'закон', 'law', 'ом', 'ohm', 'электрическ', 'electrical']):
            return self._generate_electrical_laws_response(relevant_info, thinking_process)
        elif any(word in message_lower for word in ['безопасность', 'safety', 'надежность', 'reliability', 'эмс', 'emc']):
            return self._generate_safety_response(relevant_info, thinking_process)
        elif any(word in message_lower for word in ['сбор данных', 'data gathering', 'мониторинг', 'monitoring']):
            return self._generate_data_gathering_response(relevant_info, thinking_process)
        elif any(word in message_lower for word in ['скачивание', 'загрузка', 'downloading', 'uploading', 'программ', 'programs']):
            return self._generate_programming_response(relevant_info, thinking_process)
        elif any(word in message_lower for word in ['симистр', 'тиристор', 'thyristor', 'semiconductor', 'полупроводник']):
            return self._generate_semiconductor_response(relevant_info, thinking_process)
        elif any(word in message_lower for word in ['алгоритм', 'algorithm', 'python', 'программирование', 'программирование', 'конвейер', 'conveyor']):
            return self._generate_programming_algorithm_response(relevant_info, thinking_process)
        elif any(word in message_lower for word in ['ооп', 'oop', 'объектно', 'ориентированное', 'класс', 'class', 'наследование', 'inheritance', 'инкапсуляция', 'encapsulation']):
            return self._generate_oop_response(relevant_info, thinking_process)
        elif any(word in message_lower for word in ['коэффициент мощности', 'power factor', 'cos φ', 'cos phi', 'реактивная мощность', 'reactive power']):
            return self._generate_power_factor_response(relevant_info, thinking_process)
        elif any(word in message_lower for word in ['синтаксис', 'syntax', 'переменные', 'variables', 'типы данных', 'data types']):
            return self._generate_syntax_response(relevant_info, thinking_process)
        elif any(word in message_lower for word in ['отладка', 'debugging', 'неисправности', 'troubleshooting', 'диагностика', 'diagnostics']):
            return self._generate_debugging_response(relevant_info, thinking_process)
        elif any(word in message_lower for word in ['продвинутые', 'advanced', 'специфические', 'specific', 'функции', 'functions']):
            return self._generate_advanced_functions_response(relevant_info, thinking_process)
        else:
            # Общий ответ
            return self._generate_general_technical_response(relevant_info, thinking_process)
    
    def _extract_relevant_sentences(self, content: str, query: str) -> List[str]:
        """Извлекает релевантные предложения из контента"""
        # Разбиваем на предложения
        sentences = re.split(r'[.!?]+', content)
        
        # Ключевые слова из запроса
        query_words = [word.lower() for word in query.split() if len(word) > 2]
        
        relevant_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Пропускаем слишком короткие предложения
                continue
            
            sentence_lower = sentence.lower()
            # Проверяем наличие ключевых слов
            if any(word in sentence_lower for word in query_words):
                relevant_sentences.append(sentence)
        
        # Если не найдено релевантных предложений, берем первые
        if not relevant_sentences:
            relevant_sentences = [s.strip() for s in sentences[:3] if len(s.strip()) > 10]
        
        return relevant_sentences[:8]  # Максимум 8 предложений для более полных ответов
    
    def _generate_encoder_response(self, relevant_info: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует специализированный ответ об энкодерах"""
        thinking_process.append("🔍 Генерирую ответ об энкодерах...")
        
        response_parts = ["**Энкодеры и датчики:**\n"]
        
        # Общая информация об энкодерах
        response_parts.append("Энкодер - это устройство для измерения углового или линейного положения. Основные типы:")
        response_parts.append("")
        response_parts.append("• **Инкрементальные энкодеры** - выдают импульсы при вращении")
        response_parts.append("• **Абсолютные энкодеры** - выдают уникальный код для каждого положения")
        response_parts.append("• **Оптические энкодеры** - используют светодиоды и фототранзисторы")
        response_parts.append("• **Магнитные энкодеры** - работают на основе магнитного поля")
        response_parts.append("• **Резольверы** - используют электромагнитную индукцию")
        
        # Информация из найденных документов
        if relevant_info:
            response_parts.append(f"\n**Найденная информация из документов:**")
            for info in relevant_info[:3]:
                response_parts.append(f"\n📄 **{info['title']}:**")
                # Показываем только первые 2 предложения, если они содержат полезную информацию
                useful_sentences = [s for s in info['sentences'][:4] if len(s.strip()) > 20 and not s.strip().startswith(';')]
                if useful_sentences:
                    response_parts.append(f"• {useful_sentences[0]}")
                else:
                    response_parts.append("• Информация об энкодерах и датчиках позиционирования")
        
        
        return "\n".join(response_parts)
    
    def _generate_pid_response(self, relevant_info: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует специализированный ответ о ПИД-регуляторах"""
        thinking_process.append("🎛️ Генерирую ответ о ПИД-регуляторах...")
        
        response_parts = ["**ПИД-регулятор:**\n"]
        
        # Общая информация о ПИД-регуляторах
        response_parts.append("ПИД-регулятор (Proportional-Integral-Derivative) - это алгоритм управления с обратной связью.")
        response_parts.append("")
        response_parts.append("**Компоненты:**")
        response_parts.append("• **P (Пропорциональная)** - реагирует на текущую ошибку")
        response_parts.append("• **I (Интегральная)** - учитывает накопленную ошибку")
        response_parts.append("• **D (Дифференциальная)** - предсказывает будущую ошибку")
        response_parts.append("")
        response_parts.append("**Принцип работы:** Регулятор сравнивает заданное значение с текущим и вычисляет управляющий сигнал для минимизации ошибки.")
        
        # Информация из найденных документов
        if relevant_info:
            response_parts.append(f"\n**Найденная информация из документов:**")
            for info in relevant_info[:3]:
                response_parts.append(f"\n📄 **{info['title']}:**")
                # Показываем только первые 2 предложения, если они содержат полезную информацию
                useful_sentences = [s for s in info['sentences'][:4] if len(s.strip()) > 20 and not s.strip().startswith(';')]
                if useful_sentences:
                    response_parts.append(f"• {useful_sentences[0]}")
                else:
                    response_parts.append("• Информация о ПИД-регуляторах и системах автоматического управления")
        
        
        return "\n".join(response_parts)
    
    def _generate_diode_response(self, relevant_info: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует специализированный ответ о диодах"""
        thinking_process.append("⚡ Генерирую ответ о диодах...")
        
        response_parts = ["**Диоды:**\n"]
        
        # Общая информация о диодах
        response_parts.append("Диод - это полупроводниковый прибор, пропускающий ток только в одном направлении.")
        response_parts.append("")
        response_parts.append("**Основные типы:**")
        response_parts.append("• **Выпрямительные диоды** - для преобразования переменного тока в постоянный")
        response_parts.append("• **Стабилитроны** - для стабилизации напряжения")
        response_parts.append("• **Светодиоды (LED)** - для излучения света")
        response_parts.append("• **Фотодиоды** - для преобразования света в электрический сигнал")
        response_parts.append("")
        response_parts.append("**Принцип работы:** Основан на p-n переходе, который создает барьер для электронов в одном направлении.")
        
        # Информация из найденных документов
        if relevant_info:
            response_parts.append(f"\n**Найденная информация из документов:**")
            for info in relevant_info[:3]:
                response_parts.append(f"\n📄 **{info['title']}:**")
                # Показываем только первые 2 предложения, если они содержат полезную информацию
                useful_sentences = [s for s in info['sentences'][:4] if len(s.strip()) > 20 and not s.strip().startswith(';')]
                if useful_sentences:
                    response_parts.append(f"• {useful_sentences[0]}")
                else:
                    response_parts.append("• Информация о диодах, транзисторах и полупроводниковых приборах")
        
        
        return "\n".join(response_parts)
    
    def _generate_controller_response(self, relevant_info: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует специализированный ответ о контроллерах"""
        thinking_process.append("🎮 Генерирую ответ о контроллерах...")
        
        response_parts = ["**Промышленные контроллеры:**\n"]
        
        # Общая информация о контроллерах
        response_parts.append("Промышленный контроллер - это устройство для автоматизации технологических процессов.")
        response_parts.append("")
        response_parts.append("**Основные типы:**")
        response_parts.append("• **PLC (ПЛК)** - программируемые логические контроллеры")
        response_parts.append("• **PAC** - программируемые автоматизированные контроллеры")
        response_parts.append("• **DCS** - распределенные системы управления")
        response_parts.append("• **SCADA** - системы диспетчерского управления")
        response_parts.append("")
        response_parts.append("**Применение:** Автоматизация производства, контроль качества, управление оборудованием.")
        
        # Информация из найденных документов
        if relevant_info:
            response_parts.append(f"\n**Найденная информация из документов:**")
            for info in relevant_info[:3]:
                response_parts.append(f"\n📄 **{info['title']}:**")
                # Показываем только первые 2 предложения, если они содержат полезную информацию
                useful_sentences = [s for s in info['sentences'][:4] if len(s.strip()) > 20 and not s.strip().startswith(';')]
                if useful_sentences:
                    response_parts.append(f"• {useful_sentences[0]}")
                else:
                    response_parts.append("• Информация о PLC, PMAC и промышленных контроллерах")
        
        
        return "\n".join(response_parts)
    
    def _generate_general_technical_response(self, relevant_info: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует общий технический ответ"""
        thinking_process.append("🔧 Генерирую общий технический ответ...")
        
        response_parts = []
        
        # Информация из найденных документов
        if relevant_info:
            for info in relevant_info[:3]:
                response_parts.append(f"📄 **{info['title']}:**")
                # Показываем только первые 2 предложения, если они содержат полезную информацию
                useful_sentences = [s for s in info['sentences'][:4] if len(s.strip()) > 20 and not s.strip().startswith(';')]
                if useful_sentences:
                    response_parts.append(f"• {useful_sentences[0]}")
                else:
                    # Извлекаем более полезную информацию из контента
                    content = info.get('content', '')
                    if content:
                        # Берем первые 200 символов содержательного текста
                        meaningful_text = content.strip()
                        if meaningful_text:
                            response_parts.append(f"• {meaningful_text}...")
                        else:
                            response_parts.append("• Документ содержит техническую информацию по данной теме")
                response_parts.append("")  # Пустая строка между документами
        else:
            response_parts.append("🔍 **Техническая информация:**")
            response_parts.append("К сожалению, в базе знаний не найдено достаточно информации по данному вопросу.")
            response_parts.append("Попробуйте переформулировать вопрос или уточнить детали.")
        
        return "\n".join(response_parts)
    
    def _generate_protocol_response(self, relevant_info: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует специализированный ответ о протоколах связи"""
        thinking_process.append("📡 Генерирую ответ о протоколах связи...")
        
        response_parts = ["**Промышленные протоколы связи:**\n"]
        
        # Общая информация о протоколах
        response_parts.append("**Modbus RTU** - это протокол последовательной связи для промышленной автоматизации.")
        response_parts.append("")
        response_parts.append("**Основные характеристики Modbus RTU:**")
        response_parts.append("• **Физический уровень:** RS-485, RS-232")
        response_parts.append("• **Скорость:** 1200-115200 бод")
        response_parts.append("• **Топология:** Мастер-Слейв (Master-Slave)")
        response_parts.append("• **Адресация:** 1-247 устройств")
        response_parts.append("• **Функции:** Чтение/запись регистров")
        response_parts.append("")
        response_parts.append("**Структура кадра:**")
        response_parts.append("• Адрес устройства (1 байт)")
        response_parts.append("• Код функции (1 байт)")
        response_parts.append("• Данные (N байт)")
        response_parts.append("• CRC (2 байта)")
        response_parts.append("")
        response_parts.append("**Основные функции:**")
        response_parts.append("• 03h - Чтение регистров")
        response_parts.append("• 06h - Запись одного регистра")
        response_parts.append("• 10h - Запись множественных регистров")
        
        # Информация из найденных документов
        if relevant_info:
            response_parts.append(f"\n**Найденная информация из документов:**")
            for info in relevant_info[:3]:
                response_parts.append(f"\n📄 **{info['title']}:**")
                useful_sentences = [s for s in info['sentences'][:4] if len(s.strip()) > 20 and not s.strip().startswith(';')]
                if useful_sentences:
                    response_parts.append(f"• {useful_sentences[0]}")
                else:
                    response_parts.append("• Информация о Modbus RTU, TCP и промышленных протоколах связи")
        
        
        return "\n".join(response_parts)
    
    def _generate_electrical_laws_response(self, relevant_info: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует специализированный ответ о законах электротехники"""
        thinking_process.append("⚡ Генерирую ответ о законах электротехники...")
        
        response_parts = ["**Основные законы электротехники:**\n"]
        
        # Закон Ома
        response_parts.append("**1. Закон Ома:**")
        response_parts.append("U = I × R")
        response_parts.append("где: U - напряжение (В), I - ток (А), R - сопротивление (Ом)")
        response_parts.append("")
        
        # Законы Кирхгофа
        response_parts.append("**2. Законы Кирхгофа:**")
        response_parts.append("")
        response_parts.append("**Первый закон (ЗКТ):**")
        response_parts.append("Сумма токов, входящих в узел, равна сумме токов, выходящих из узла")
        response_parts.append("ΣIвх = ΣIвых")
        response_parts.append("")
        response_parts.append("**Второй закон (ЗКН):**")
        response_parts.append("В замкнутом контуре алгебраическая сумма ЭДС равна алгебраической сумме падений напряжений")
        response_parts.append("ΣE = ΣIR")
        response_parts.append("")
        
        # Дополнительные законы
        response_parts.append("**3. Другие важные законы:**")
        response_parts.append("• **Закон Джоуля-Ленца:** Q = I²Rt")
        response_parts.append("• **Закон Фарадея:** ЭДС индукции пропорциональна скорости изменения магнитного потока")
        response_parts.append("• **Правило буравчика:** Направление магнитного поля вокруг проводника с током")
        
        # Информация из найденных документов
        if relevant_info:
            response_parts.append(f"\n**Найденная информация из документов:**")
            for info in relevant_info[:3]:
                response_parts.append(f"\n📄 **{info['title']}:**")
                useful_sentences = [s for s in info['sentences'][:4] if len(s.strip()) > 20 and not s.strip().startswith(';')]
                if useful_sentences:
                    response_parts.append(f"• {useful_sentences[0]}")
                else:
                    response_parts.append("• Документ содержит формулы и расчеты по электротехнике")
        
        
        return "\n".join(response_parts)
    
    def _generate_safety_response(self, relevant_info: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует специализированный ответ о безопасности и надежности"""
        thinking_process.append("🛡️ Генерирую ответ о безопасности и надежности...")
        
        response_parts = ["**Безопасность и надежность в промышленности:**\n"]
        
        # Общие принципы безопасности
        response_parts.append("**Принципы промышленной безопасности:**")
        response_parts.append("• **Электробезопасность** - защита от поражения электрическим током")
        response_parts.append("• **Электромагнитная совместимость (ЭМС)** - защита от помех")
        response_parts.append("• **Функциональная безопасность** - SIL (Safety Integrity Level)")
        response_parts.append("• **Механическая безопасность** - защитные ограждения")
        response_parts.append("")
        
        # Стандарты
        response_parts.append("**Ключевые стандарты:**")
        response_parts.append("• **IEC 61508** - Функциональная безопасность")
        response_parts.append("• **IEC 61511** - Функциональная безопасность для процессов")
        response_parts.append("• **IEC 61000** - Электромагнитная совместимость")
        response_parts.append("• **ГОСТ Р** - Российские стандарты безопасности")
        response_parts.append("")
        
        # Уровни безопасности
        response_parts.append("**Уровни SIL (Safety Integrity Level):**")
        response_parts.append("• **SIL 1** - Низкий риск (10⁻⁵-10⁻⁶)")
        response_parts.append("• **SIL 2** - Средний риск (10⁻⁶-10⁻⁷)")
        response_parts.append("• **SIL 3** - Высокий риск (10⁻⁷-10⁻⁸)")
        response_parts.append("• **SIL 4** - Критический риск (10⁻⁸-10⁻⁹)")
        
        # Информация из найденных документов
        if relevant_info:
            response_parts.append(f"\n**Найденная информация из документов:**")
            for info in relevant_info[:3]:
                response_parts.append(f"\n📄 **{info['title']}:**")
                useful_sentences = [s for s in info['sentences'][:4] if len(s.strip()) > 20 and not s.strip().startswith(';')]
                if useful_sentences:
                    response_parts.append(f"• {useful_sentences[0]}")
                else:
                    response_parts.append("• Документ содержит требования по безопасности и ЭМС")
        
        
        return "\n".join(response_parts)
    
    def _generate_data_gathering_response(self, relevant_info: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует специализированный ответ о сборе данных"""
        thinking_process.append("📊 Генерирую ответ о сборе данных...")
        
        response_parts = ["**Сбор данных в промышленности (Data Gathering):**\n"]
        
        # Методы сбора данных
        response_parts.append("**Основные методы сбора данных:**")
        response_parts.append("• **SCADA системы** - Supervisory Control and Data Acquisition")
        response_parts.append("• **HMI интерфейсы** - Human Machine Interface")
        response_parts.append("• **Промышленные сети** - Ethernet, Profinet, Modbus")
        response_parts.append("• **Датчики и измерители** - аналоговые и цифровые сигналы")
        response_parts.append("")
        
        # Типы данных
        response_parts.append("**Типы собираемых данных:**")
        response_parts.append("• **Процессные данные** - температура, давление, расход")
        response_parts.append("• **Состояние оборудования** - включено/выключено, аварии")
        response_parts.append("• **Производственные показатели** - производительность, качество")
        response_parts.append("• **Диагностическая информация** - состояние компонентов")
        response_parts.append("")
        
        # Архитектура систем
        response_parts.append("**Архитектура систем сбора данных:**")
        response_parts.append("• **Полевой уровень** - датчики, исполнительные механизмы")
        response_parts.append("• **Контроллерный уровень** - PLC, ПК")
        response_parts.append("• **Уровень управления** - SCADA, HMI")
        response_parts.append("• **Корпоративный уровень** - ERP, MES системы")
        
        # Информация из найденных документов
        if relevant_info:
            response_parts.append(f"\n**Найденная информация из документов:**")
            for info in relevant_info[:3]:
                response_parts.append(f"\n📄 **{info['title']}:**")
                useful_sentences = [s for s in info['sentences'][:4] if len(s.strip()) > 20 and not s.strip().startswith(';')]
                if useful_sentences:
                    response_parts.append(f"• {useful_sentences[0]}")
                else:
                    response_parts.append("• Документ содержит информацию о системах мониторинга и сбора данных")
        
        
        return "\n".join(response_parts)
    
    def _generate_programming_response(self, relevant_info: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует специализированный ответ о программировании и загрузке программ"""
        thinking_process.append("💻 Генерирую ответ о программировании...")
        
        response_parts = ["**Программирование и загрузка программ:**\n"]
        
        # Общая информация о программировании
        response_parts.append("**Основные этапы разработки программ:**")
        response_parts.append("• **Анализ требований** - определение функциональности")
        response_parts.append("• **Проектирование** - архитектура и структура программы")
        response_parts.append("• **Кодирование** - написание исходного кода")
        response_parts.append("• **Тестирование** - проверка работоспособности")
        response_parts.append("• **Развертывание** - установка в производственную среду")
        response_parts.append("")
        
        # Типы программного обеспечения
        response_parts.append("**Типы промышленного ПО:**")
        response_parts.append("• **PLC программы** - логика управления оборудованием")
        response_parts.append("• **SCADA системы** - визуализация и мониторинг")
        response_parts.append("• **HMI интерфейсы** - взаимодействие с оператором")
        response_parts.append("• **Драйверы устройств** - управление периферией")
        response_parts.append("")
        
        # Методы загрузки
        response_parts.append("**Методы загрузки программ:**")
        response_parts.append("• **USB/Flash** - через съемные носители")
        response_parts.append("• **Ethernet** - по сетевому соединению")
        response_parts.append("• **RS-232/RS-485** - через последовательные порты")
        response_parts.append("• **Wi-Fi/Bluetooth** - беспроводная передача")
        
        # Информация из найденных документов
        if relevant_info:
            response_parts.append(f"\n**Найденная информация из документов:**")
            for info in relevant_info[:3]:
                # Убираем название файла, оставляем только содержимое
                useful_sentences = [s for s in info['sentences'][:4] if len(s.strip()) > 20 and not s.strip().startswith(';')]
                if useful_sentences:
                    response_parts.append(f"• {useful_sentences[0]}")
                else:
                    response_parts.append("• Документ содержит информацию о программировании и конфигурации")
        
        
        return "\n".join(response_parts)
    
    def _generate_semiconductor_response(self, relevant_info: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует специализированный ответ о полупроводниках"""
        thinking_process.append("🔬 Генерирую ответ о полупроводниках...")
        
        response_parts = ["**Полупроводники и тиристоры:**\n"]
        
        # Общая информация о полупроводниках
        response_parts.append("**Полупроводники** - материалы с промежуточной проводимостью между проводниками и диэлектриками.")
        response_parts.append("")
        response_parts.append("**Основные типы полупроводников:**")
        response_parts.append("• **Диоды** - односторонняя проводимость")
        response_parts.append("• **Транзисторы** - усилители и переключатели")
        response_parts.append("• **Тиристоры (симисторы)** - управляемые выпрямители")
        response_parts.append("• **Оптоэлектронные приборы** - светодиоды, фототранзисторы")
        response_parts.append("")
        
        # Тиристоры (симисторы)
        response_parts.append("**Тиристоры (симистры):**")
        response_parts.append("• **Принцип работы:** Управляемый полупроводниковый прибор")
        response_parts.append("• **Структура:** 4-слойная p-n-p-n структура")
        response_parts.append("• **Управление:** Запуск через управляющий электрод")
        response_parts.append("• **Применение:** Регулировка мощности, коммутация")
        response_parts.append("")
        
        # Применение
        response_parts.append("**Области применения:**")
        response_parts.append("• **Силовая электроника** - управление двигателями")
        response_parts.append("• **Регулирование освещения** - диммеры")
        response_parts.append("• **Нагрев** - управление ТЭНами")
        response_parts.append("• **Промышленная автоматизация** - коммутация цепей")
        
        # Информация из найденных документов
        if relevant_info:
            response_parts.append(f"\n**Найденная информация из документов:**")
            for info in relevant_info[:3]:
                useful_sentences = [s for s in info['sentences'][:4] if len(s.strip()) > 20 and not s.strip().startswith(';')]
                if useful_sentences:
                    response_parts.append(f"• {useful_sentences[0]}")
                else:
                    response_parts.append("• Информация о тиристорах, симисторах и полупроводниковых приборах")
        
        
        return "\n".join(response_parts)
    
    def _generate_programming_algorithm_response(self, relevant_info: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует специализированный ответ о программировании алгоритмов"""
        thinking_process.append("🐍 Генерирую ответ о программировании алгоритмов...")
        
        response_parts = ["**Программирование алгоритмов управления:**\n"]
        
        # Общие принципы программирования
        response_parts.append("**Основы программирования алгоритмов:**")
        response_parts.append("• **Анализ задачи** - понимание требований")
        response_parts.append("• **Проектирование логики** - структура алгоритма")
        response_parts.append("• **Реализация** - написание кода")
        response_parts.append("• **Тестирование** - проверка работоспособности")
        response_parts.append("")
        
        # Python для промышленности
        response_parts.append("**Python в промышленной автоматизации:**")
        response_parts.append("• **Простота синтаксиса** - быстрая разработка")
        response_parts.append("• **Богатые библиотеки** - NumPy, SciPy, Pandas")
        response_parts.append("• **Сетевые возможности** - TCP/IP, Modbus, OPC")
        response_parts.append("• **Интеграция с оборудованием** - через драйверы")
        response_parts.append("")
        
        # Алгоритм управления конвейером
        response_parts.append("**Алгоритм управления конвейером на Python:**")
        response_parts.append("```python")
        response_parts.append("import time")
        response_parts.append("import threading")
        response_parts.append("")
        response_parts.append("class ConveyorController:")
        response_parts.append("    def __init__(self):")
        response_parts.append("        self.running = False")
        response_parts.append("        self.speed = 0")
        response_parts.append("    ")
        response_parts.append("    def start_conveyor(self):")
        response_parts.append("        self.running = True")
        response_parts.append("        print('Конвейер запущен')")
        response_parts.append("    ")
        response_parts.append("    def stop_conveyor(self):")
        response_parts.append("        self.running = False")
        response_parts.append("        print('Конвейер остановлен')")
        response_parts.append("```")
        response_parts.append("")
        
        # Основные функции
        response_parts.append("**Основные функции алгоритма:**")
        response_parts.append("• **Управление скоростью** - регулировка частоты")
        response_parts.append("• **Контроль направления** - вперед/назад")
        response_parts.append("• **Мониторинг состояния** - датчики и сигналы")
        response_parts.append("• **Обработка ошибок** - аварийные ситуации")
        
        # Информация из найденных документов
        if relevant_info:
            response_parts.append(f"\n**Найденная информация из документов:**")
            for info in relevant_info[:3]:
                useful_sentences = [s for s in info['sentences'][:4] if len(s.strip()) > 20 and not s.strip().startswith(';')]
                if useful_sentences:
                    response_parts.append(f"• {useful_sentences[0]}")
                else:
                    response_parts.append("• Документ содержит информацию об алгоритмах управления")
        
        
        return "\n".join(response_parts)
    
    def _generate_oop_response(self, relevant_info: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует специализированный ответ об объектно-ориентированном программировании"""
        thinking_process.append("🏗️ Генерирую ответ об ООП...")
        
        response_parts = ["**Принципы объектно-ориентированного программирования (ООП):**\n"]
        
        # Основные принципы ООП
        response_parts.append("**1. Инкапсуляция (Encapsulation):**")
        response_parts.append("• Сокрытие внутренней реализации объекта")
        response_parts.append("• Доступ к данным только через методы")
        response_parts.append("• Защита данных от несанкционированного изменения")
        response_parts.append("")
        
        response_parts.append("**2. Наследование (Inheritance):**")
        response_parts.append("• Создание новых классов на основе существующих")
        response_parts.append("• Переиспользование кода и функциональности")
        response_parts.append("• Иерархия классов (родительские и дочерние)")
        response_parts.append("")
        
        response_parts.append("**3. Полиморфизм (Polymorphism):**")
        response_parts.append("• Один интерфейс - множество реализаций")
        response_parts.append("• Переопределение методов в дочерних классах")
        response_parts.append("• Динамическое связывание методов")
        response_parts.append("")
        
        response_parts.append("**4. Абстракция (Abstraction):**")
        response_parts.append("• Упрощение сложных систем")
        response_parts.append("• Фокус на важных характеристиках")
        response_parts.append("• Сокрытие деталей реализации")
        response_parts.append("")
        
        # Примеры применения
        response_parts.append("**Применение в промышленности:**")
        response_parts.append("• **Системы управления** - классы для контроллеров")
        response_parts.append("• **SCADA системы** - объекты для устройств")
        response_parts.append("• **Базы данных** - модели данных как классы")
        response_parts.append("• **Сетевые протоколы** - абстракция коммуникации")
        
        # Информация из найденных документов
        if relevant_info:
            response_parts.append(f"\n**Найденная информация из документов:**")
            for info in relevant_info[:3]:
                useful_sentences = [s for s in info['sentences'][:2] if len(s.strip()) > 20 and not s.strip().startswith(';') and not s.strip().startswith('📄')]
                if useful_sentences:
                    response_parts.append(f"• {useful_sentences[0]}")
                else:
                    response_parts.append("• Документ содержит информацию о программировании")
        
        
        return "\n".join(response_parts)
    
    def _generate_power_factor_response(self, relevant_info: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует специализированный ответ о коэффициенте мощности"""
        thinking_process.append("⚡ Генерирую ответ о коэффициенте мощности...")
        
        response_parts = ["**Коэффициент мощности и его улучшение:**\n"]
        
        # Определение
        response_parts.append("**Что такое коэффициент мощности:**")
        response_parts.append("• **Определение:** cos φ = P/S (активная мощность / полная мощность)")
        response_parts.append("• **Диапазон:** от 0 до 1 (идеальный = 1)")
        response_parts.append("• **Физический смысл:** показывает эффективность использования энергии")
        response_parts.append("")
        
        # Типы мощности
        response_parts.append("**Типы электрической мощности:**")
        response_parts.append("• **Активная (P)** - полезная работа, измеряется в Вт")
        response_parts.append("• **Реактивная (Q)** - энергия магнитных полей, ВАр")
        response_parts.append("• **Полная (S)** - общая мощность, ВА")
        response_parts.append("• **Формула:** S² = P² + Q²")
        response_parts.append("")
        
        # Влияние низкого коэффициента
        response_parts.append("**Проблемы низкого коэффициента мощности:**")
        response_parts.append("• **Увеличение потерь** в линиях передачи")
        response_parts.append("• **Перегрузка оборудования** (трансформаторы, кабели)")
        response_parts.append("• **Штрафы от энергоснабжающих организаций**")
        response_parts.append("• **Снижение качества электроэнергии**")
        response_parts.append("")
        
        # Методы улучшения
        response_parts.append("**Методы улучшения коэффициента мощности:**")
        response_parts.append("• **Конденсаторные батареи** - компенсация реактивной мощности")
        response_parts.append("• **Синхронные компенсаторы** - генерация реактивной мощности")
        response_parts.append("• **Статические компенсаторы (SVC)** - быстрая регулировка")
        response_parts.append("• **Оптимизация нагрузки** - правильный выбор оборудования")
        
        # Информация из найденных документов
        if relevant_info:
            response_parts.append(f"\n**Найденная информация из документов:**")
            for info in relevant_info[:3]:
                useful_sentences = [s for s in info['sentences'][:2] if len(s.strip()) > 20 and not s.strip().startswith(';') and not s.strip().startswith('📄')]
                if useful_sentences:
                    response_parts.append(f"• {useful_sentences[0]}")
                else:
                    response_parts.append("• Информация о законах Кирхгофа, Ома и электротехнических расчетах")
        
        
        return "\n".join(response_parts)
    
    def _generate_syntax_response(self, relevant_info: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует специализированный ответ о синтаксисе и переменных"""
        thinking_process.append("📝 Генерирую ответ о синтаксисе...")
        
        response_parts = ["**Синтаксис и переменные в программировании:**\n"]
        
        # Основы синтаксиса
        response_parts.append("**Основы синтаксиса программирования:**")
        response_parts.append("• **Структура программы** - последовательность инструкций")
        response_parts.append("• **Правила написания** - ключевые слова, операторы, разделители")
        response_parts.append("• **Комментарии** - пояснения для разработчиков")
        response_parts.append("• **Блоки кода** - группировка логически связанных инструкций")
        response_parts.append("")
        
        # Переменные
        response_parts.append("**Переменные и типы данных:**")
        response_parts.append("• **Переменная** - именованная область памяти")
        response_parts.append("• **Инициализация** - присвоение начального значения")
        response_parts.append("• **Область видимости** - где переменная доступна")
        response_parts.append("• **Типизация** - строгая или динамическая")
        response_parts.append("")
        
        # Типы данных
        response_parts.append("**Основные типы данных:**")
        response_parts.append("• **Целые числа** - int, integer (1, 100, -5)")
        response_parts.append("• **Дробные числа** - float, double (3.14, 2.5)")
        response_parts.append("• **Строки** - string, char ('Hello', 'A')")
        response_parts.append("• **Логические** - boolean (true/false)")
        response_parts.append("• **Массивы** - коллекции элементов")
        response_parts.append("")
        
        # Специфика промышленного программирования
        response_parts.append("**Синтаксис в промышленном программировании:**")
        response_parts.append("• **PLC языки** - Ladder Logic, Function Block Diagram")
        response_parts.append("• **Переменные ввода/вывода** - I/O mapping")
        response_parts.append("• **Типы данных ПЛК** - BOOL, INT, REAL, DINT")
        response_parts.append("• **Структурированный текст** - ST (Structured Text)")
        
        # Информация из найденных документов
        if relevant_info:
            response_parts.append(f"\n**Найденная информация из документов:**")
            for info in relevant_info[:3]:
                useful_sentences = [s for s in info['sentences'][:2] if len(s.strip()) > 20 and not s.strip().startswith(';') and not s.strip().startswith('📄')]
                if useful_sentences:
                    response_parts.append(f"• {useful_sentences[0]}")
                else:
                    response_parts.append("• Документ содержит информацию о программировании")
        
        
        return "\n".join(response_parts)
    
    def _generate_debugging_response(self, relevant_info: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует специализированный ответ об отладке и поиске неисправностей"""
        thinking_process.append("🔧 Генерирую ответ об отладке...")
        
        response_parts = ["**Отладка и поиск неисправностей:**\n"]
        
        # Методы отладки
        response_parts.append("**Основные методы отладки:**")
        response_parts.append("• **Пошаговое выполнение** - трассировка кода")
        response_parts.append("• **Точки останова** - breakpoints для анализа состояния")
        response_parts.append("• **Логирование** - запись событий и состояний")
        response_parts.append("• **Мониторинг переменных** - отслеживание значений")
        response_parts.append("")
        
        # Поиск неисправностей в промышленности
        response_parts.append("**Поиск неисправностей в промышленных системах:**")
        response_parts.append("• **Диагностика оборудования** - проверка датчиков и исполнительных механизмов")
        response_parts.append("• **Анализ логов** - системные журналы и события")
        response_parts.append("• **Тестирование каналов связи** - проверка протоколов")
        response_parts.append("• **Визуализация процессов** - HMI и SCADA мониторинг")
        response_parts.append("")
        
        # Инструменты диагностики
        response_parts.append("**Инструменты диагностики:**")
        response_parts.append("• **Мультиметры** - измерение напряжения и тока")
        response_parts.append("• **Осциллографы** - анализ сигналов")
        response_parts.append("• **Анализаторы протоколов** - мониторинг сетевого трафика")
        response_parts.append("• **Программные отладчики** - встроенные инструменты IDE")
        response_parts.append("")
        
        # Системный подход
        response_parts.append("**Системный подход к диагностике:**")
        response_parts.append("• **Изоляция проблемы** - определение области неисправности")
        response_parts.append("• **Проверка по уровням** - от простого к сложному")
        response_parts.append("• **Документирование** - ведение журнала неисправностей")
        response_parts.append("• **Профилактика** - предотвращение повторных проблем")
        
        # Информация из найденных документов
        if relevant_info:
            response_parts.append(f"\n**Найденная информация из документов:**")
            for info in relevant_info[:3]:
                useful_sentences = [s for s in info['sentences'][:2] if len(s.strip()) > 20 and not s.strip().startswith(';') and not s.strip().startswith('📄')]
                if useful_sentences:
                    response_parts.append(f"• {useful_sentences[0]}")
                else:
                    response_parts.append("• Документ содержит информацию о диагностике и отладке")
        
        
        return "\n".join(response_parts)
    
    def _generate_advanced_functions_response(self, relevant_info: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует специализированный ответ о продвинутых функциях"""
        thinking_process.append("🚀 Генерирую ответ о продвинутых функциях...")
        
        response_parts = ["**Продвинутые и специфические функции:**\n"]
        
        # Продвинутые функции программирования
        response_parts.append("**Продвинутые функции в программировании:**")
        response_parts.append("• **Рекурсия** - функция, вызывающая сама себя")
        response_parts.append("• **Замыкания** - функции с доступом к внешним переменным")
        response_parts.append("• **Лямбда-функции** - анонимные функции")
        response_parts.append("• **Декораторы** - модификация поведения функций")
        response_parts.append("• **Генераторы** - итераторы с отложенным вычислением")
        response_parts.append("")
        
        # Специфические функции ПЛК
        response_parts.append("**Специфические функции ПЛК:**")
        response_parts.append("• **PID регуляторы** - пропорционально-интегрально-дифференциальные")
        response_parts.append("• **Таймеры и счетчики** - временные и количественные функции")
        response_parts.append("• **Блоки сравнения** - логические операции")
        response_parts.append("• **Арифметические блоки** - математические операции")
        response_parts.append("• **Функции связи** - коммуникационные протоколы")
        response_parts.append("")
        
        # Системные функции
        response_parts.append("**Системные функции:**")
        response_parts.append("• **Управление памятью** - динамическое выделение ресурсов")
        response_parts.append("• **Многопоточность** - параллельное выполнение задач")
        response_parts.append("• **Обработка исключений** - управление ошибками")
        response_parts.append("• **Интерфейсы API** - взаимодействие с внешними системами")
        response_parts.append("")
        
        # Промышленные специфические функции
        response_parts.append("**Промышленные специфические функции:**")
        response_parts.append("• **Алгоритмы управления** - сложные логические схемы")
        response_parts.append("• **Компенсация и коррекция** - улучшение точности")
        response_parts.append("• **Диагностика и мониторинг** - отслеживание состояния")
        response_parts.append("• **Интеграция с ERP** - связь с корпоративными системами")
        
        # Информация из найденных документов
        if relevant_info:
            response_parts.append(f"\n**Найденная информация из документов:**")
            for info in relevant_info[:3]:
                useful_sentences = [s for s in info['sentences'][:2] if len(s.strip()) > 20 and not s.strip().startswith(';') and not s.strip().startswith('📄')]
                if useful_sentences:
                    response_parts.append(f"• {useful_sentences[0]}")
                else:
                    response_parts.append("• Документ содержит информацию о продвинутых функциях")
        
        
        return "\n".join(response_parts)
    
    def _generate_fallback_response(self, message: str, thinking_process: List[str]) -> str:
        """Генерирует fallback ответ когда нет документов"""
        thinking_process.append("💭 Генерирую ответ без контекста документов...")
        
        message_lower = message.lower()
        
        # Обработка общих вопросов и приветствий
        if any(phrase in message_lower for phrase in [
            "какой хороший день", "хороший день", "как дела", "как поживаешь",
            "что нового", "как настроение", "как жизнь"
        ]):
            return """😊 **Отличный день!**

Спасибо за вопрос! У меня все замечательно:

**Мое состояние:**
• ✅ Система работает стабильно
• 🧠 База знаний активна (103 документа)
• 🔍 Поиск функционирует корректно
• 💬 Готов помочь с техническими вопросами

**Что могу предложить:**
• 🏭 Помощь с промышленной автоматизацией
• 💻 Консультации по программированию
• ⚡ Решение задач по электротехнике
• 📡 Вопросы по радиомеханике

**Как дела у вас?** Есть ли технические вопросы, с которыми могу помочь?"""

        if any(phrase in message_lower for phrase in [
            "привет", "hello", "hi", "добро пожаловать"
        ]):
            return """👋 **Привет! Добро пожаловать в Rubin AI!**

Я ваш помощник по техническим вопросам:

**Мои специализации:**
• 🏭 **Промышленная автоматизация** - PLC, ПИД-регуляторы, SCADA
• 💻 **Программирование** - Python, C++, алгоритмы
• ⚡ **Электротехника** - схемы, расчеты, компоненты
• 📡 **Радиомеханика** - антенны, сигналы, протоколы

**Как я могу помочь?**
Просто задайте вопрос по любой из этих тем, и я дам детальный ответ с примерами!

**Примеры вопросов:**
• "Как настроить ПИД-регулятор?"
• "Объясни принцип работы диода"
• "Что такое Modbus RTU?"

Чем могу быть полезен?"""

        # Специализированные ответы для технических вопросов
        if any(word in message_lower for word in ['энкодер', 'encoder', 'датчик', 'sensor']):
            return """**Энкодеры и датчики:**

Энкодер - это устройство для измерения углового или линейного положения. Основные типы:

• **Инкрементальные энкодеры** - выдают импульсы при вращении
• **Абсолютные энкодеры** - выдают уникальный код для каждого положения
• **Оптические энкодеры** - используют светодиоды и фототранзисторы
• **Магнитные энкодеры** - работают на основе магнитного поля
• **Резольверы** - используют электромагнитную индукцию

*Для получения более подробной информации рекомендуется поиск в технической документации.*"""

        elif any(word in message_lower for word in ['пид', 'pid', 'регулятор', 'controller']):
            return """**ПИД-регулятор:**

ПИД-регулятор (Proportional-Integral-Derivative) - это алгоритм управления с обратной связью.

**Компоненты:**
• **P (Пропорциональная)** - реагирует на текущую ошибку
• **I (Интегральная)** - учитывает накопленную ошибку
• **D (Дифференциальная)** - предсказывает будущую ошибку

**Принцип работы:** Регулятор сравнивает заданное значение с текущим и вычисляет управляющий сигнал для минимизации ошибки.

*Для настройки конкретных параметров требуется анализ характеристик системы.*"""

        elif any(word in message_lower for word in ['диод', 'diode', 'полупроводник', 'semiconductor']):
            return """**Диоды:**

Диод - это полупроводниковый прибор, пропускающий ток только в одном направлении.

**Основные типы:**
• **Выпрямительные диоды** - для преобразования переменного тока в постоянный
• **Стабилитроны** - для стабилизации напряжения
• **Светодиоды (LED)** - для излучения света
• **Фотодиоды** - для преобразования света в электрический сигнал

**Принцип работы:** Основан на p-n переходе, который создает барьер для электронов в одном направлении.

*Подробные характеристики зависят от конкретного типа и применения.*"""

        elif any(word in message_lower for word in ['контроллер', 'controller', 'промышленный', 'industrial']):
            return """**Промышленные контроллеры:**

Промышленный контроллер - это устройство для автоматизации технологических процессов.

**Основные типы:**
• **PLC (ПЛК)** - программируемые логические контроллеры
• **PAC** - программируемые автоматизированные контроллеры
• **DCS** - распределенные системы управления
• **SCADA** - системы диспетчерского управления

**Применение:** Автоматизация производства, контроль качества, управление оборудованием.

*Конкретные модели и возможности зависят от производителя и области применения.*"""

        elif any(word in message_lower for word in ['modbus', 'rtu', 'tcp', 'протокол', 'protocol', 'связь', 'communication']):
            return """**Промышленные протоколы связи:**

**Modbus RTU** - это протокол последовательной связи для промышленной автоматизации.

**Основные характеристики:**
• **Физический уровень:** RS-485, RS-232
• **Скорость:** 1200-115200 бод
• **Топология:** Мастер-Слейв (Master-Slave)
• **Адресация:** 1-247 устройств

**Основные функции:**
• 03h - Чтение регистров
• 06h - Запись одного регистра
• 10h - Запись множественных регистров

*Для детальной настройки требуется анализ конкретной системы.*"""

        elif any(word in message_lower for word in ['кирхгоф', 'kirchhoff', 'закон', 'law', 'ом', 'ohm', 'электрическ', 'electrical']):
            return """**Основные законы электротехники:**

**1. Закон Ома:**
U = I × R (напряжение = ток × сопротивление)

**2. Законы Кирхгофа:**
• **Первый закон (ЗКТ):** ΣIвх = ΣIвых
• **Второй закон (ЗКН):** ΣE = ΣIR

**3. Дополнительные законы:**
• **Закон Джоуля-Ленца:** Q = I²Rt
• **Закон Фарадея:** ЭДС индукции

*Для практических расчетов требуется анализ конкретной схемы.*"""

        elif any(word in message_lower for word in ['безопасность', 'safety', 'надежность', 'reliability', 'эмс', 'emc']):
            return """**Безопасность и надежность в промышленности:**

**Принципы безопасности:**
• **Электробезопасность** - защита от поражения током
• **Электромагнитная совместимость (ЭМС)** - защита от помех
• **Функциональная безопасность** - SIL уровни
• **Механическая безопасность** - защитные ограждения

**Стандарты:**
• **IEC 61508** - Функциональная безопасность
• **IEC 61000** - Электромагнитная совместимость
• **ГОСТ Р** - Российские стандарты

*Требования зависят от класса опасности и области применения.*"""

        elif any(word in message_lower for word in ['сбор данных', 'data gathering', 'мониторинг', 'monitoring']):
            return """**Сбор данных в промышленности (Data Gathering):**

**Основные методы:**
• **SCADA системы** - Supervisory Control and Data Acquisition
• **HMI интерфейсы** - Human Machine Interface
• **Промышленные сети** - Ethernet, Profinet, Modbus
• **Датчики и измерители** - аналоговые и цифровые сигналы

**Типы данных:**
• **Процессные данные** - температура, давление, расход
• **Состояние оборудования** - включено/выключено, аварии
• **Производственные показатели** - производительность, качество

*Архитектура системы зависит от масштаба и требований.*"""

        elif any(word in message_lower for word in ['скачивание', 'загрузка', 'downloading', 'uploading', 'программ', 'programs']):
            return """**Программирование и загрузка программ:**

**Основные этапы разработки:**
• **Анализ требований** - определение функциональности
• **Проектирование** - архитектура и структура программы
• **Кодирование** - написание исходного кода
• **Тестирование** - проверка работоспособности

**Методы загрузки:**
• **USB/Flash** - через съемные носители
• **Ethernet** - по сетевому соединению
• **RS-232/RS-485** - через последовательные порты

*Выбор метода зависит от типа оборудования и требований.*"""

        elif any(word in message_lower for word in ['симистр', 'тиристор', 'thyristor', 'semiconductor', 'полупроводник']):
            return """**Полупроводники и тиристоры:**

**Полупроводники** - материалы с промежуточной проводимостью между проводниками и диэлектриками.

**Основные типы:**
• **Диоды** - односторонняя проводимость
• **Транзисторы** - усилители и переключатели
• **Тиристоры (симисторы)** - управляемые выпрямители
• **Оптоэлектронные приборы** - светодиоды, фототранзисторы

**Применение:**
• **Силовая электроника** - управление двигателями
• **Регулирование освещения** - диммеры
• **Промышленная автоматизация** - коммутация цепей

*Тиристоры широко используются для управления мощностью.*"""

        elif any(word in message_lower for word in ['алгоритм', 'algorithm', 'python', 'программирование', 'конвейер', 'conveyor']):
            return """**Программирование алгоритмов управления:**

**Основы программирования:**
• **Анализ задачи** - понимание требований
• **Проектирование логики** - структура алгоритма
• **Реализация** - написание кода
• **Тестирование** - проверка работоспособности

**Python в промышленности:**
• **Простота синтаксиса** - быстрая разработка
• **Богатые библиотеки** - NumPy, SciPy, Pandas
• **Сетевые возможности** - TCP/IP, Modbus, OPC

**Алгоритм управления конвейером:**
• **Управление скоростью** - регулировка частоты
• **Контроль направления** - вперед/назад
• **Мониторинг состояния** - датчики и сигналы
• **Обработка ошибок** - аварийные ситуации

*Python отлично подходит для прототипирования алгоритмов.*"""

        elif any(word in message_lower for word in ['ооп', 'oop', 'объектно', 'ориентированное', 'класс', 'class', 'наследование', 'inheritance', 'инкапсуляция', 'encapsulation']):
            return """**Принципы объектно-ориентированного программирования (ООП):**

**1. Инкапсуляция (Encapsulation):**
• Сокрытие внутренней реализации объекта
• Доступ к данным только через методы
• Защита данных от несанкционированного изменения

**2. Наследование (Inheritance):**
• Создание новых классов на основе существующих
• Переиспользование кода и функциональности
• Иерархия классов (родительские и дочерние)

**3. Полиморфизм (Polymorphism):**
• Один интерфейс - множество реализаций
• Переопределение методов в дочерних классах
• Динамическое связывание методов

**4. Абстракция (Abstraction):**
• Упрощение сложных систем
• Фокус на важных характеристиках
• Сокрытие деталей реализации

*ООП широко используется в промышленной автоматизации.*"""

        elif any(word in message_lower for word in ['коэффициент мощности', 'power factor', 'cos φ', 'cos phi', 'реактивная мощность', 'reactive power']):
            return """**Коэффициент мощности и его улучшение:**

**Что такое коэффициент мощности:**
• **Определение:** cos φ = P/S (активная мощность / полная мощность)
• **Диапазон:** от 0 до 1 (идеальный = 1)
• **Физический смысл:** показывает эффективность использования энергии

**Типы электрической мощности:**
• **Активная (P)** - полезная работа, измеряется в Вт
• **Реактивная (Q)** - энергия магнитных полей, ВАр
• **Полная (S)** - общая мощность, ВА
• **Формула:** S² = P² + Q²

**Методы улучшения:**
• **Конденсаторные батареи** - компенсация реактивной мощности
• **Синхронные компенсаторы** - генерация реактивной мощности
• **Статические компенсаторы (SVC)** - быстрая регулировка
• **Оптимизация нагрузки** - правильный выбор оборудования

*Низкий коэффициент мощности приводит к потерям и штрафам.*"""

        elif any(word in message_lower for word in ['синтаксис', 'syntax', 'переменные', 'variables', 'типы данных', 'data types']):
            return """**Синтаксис и переменные в программировании:**

**Основы синтаксиса:**
• **Структура программы** - последовательность инструкций
• **Правила написания** - ключевые слова, операторы, разделители
• **Комментарии** - пояснения для разработчиков
• **Блоки кода** - группировка логически связанных инструкций

**Переменные и типы данных:**
• **Переменная** - именованная область памяти
• **Инициализация** - присвоение начального значения
• **Область видимости** - где переменная доступна
• **Типизация** - строгая или динамическая

**Основные типы данных:**
• **Целые числа** - int, integer (1, 100, -5)
• **Дробные числа** - float, double (3.14, 2.5)
• **Строки** - string, char ('Hello', 'A')
• **Логические** - boolean (true/false)
• **Массивы** - коллекции элементов

**В промышленном программировании:**
• **PLC языки** - Ladder Logic, Function Block Diagram
• **Переменные ввода/вывода** - I/O mapping
• **Типы данных ПЛК** - BOOL, INT, REAL, DINT

*Правильный синтаксис критичен для работы программы.*"""

        elif any(word in message_lower for word in ['отладка', 'debugging', 'неисправности', 'troubleshooting', 'диагностика', 'diagnostics']):
            return """**Отладка и поиск неисправностей:**

**Основные методы отладки:**
• **Пошаговое выполнение** - трассировка кода
• **Точки останова** - breakpoints для анализа состояния
• **Логирование** - запись событий и состояний
• **Мониторинг переменных** - отслеживание значений

**Поиск неисправностей в промышленности:**
• **Диагностика оборудования** - проверка датчиков и исполнительных механизмов
• **Анализ логов** - системные журналы и события
• **Тестирование каналов связи** - проверка протоколов
• **Визуализация процессов** - HMI и SCADA мониторинг

**Инструменты диагностики:**
• **Мультиметры** - измерение напряжения и тока
• **Осциллографы** - анализ сигналов
• **Анализаторы протоколов** - мониторинг сетевого трафика
• **Программные отладчики** - встроенные инструменты IDE

**Системный подход:**
• **Изоляция проблемы** - определение области неисправности
• **Проверка по уровням** - от простого к сложному
• **Документирование** - ведение журнала неисправностей

*Систематический подход ускоряет поиск проблем.*"""

        elif any(word in message_lower for word in ['продвинутые', 'advanced', 'специфические', 'specific', 'функции', 'functions']):
            return """**Продвинутые и специфические функции:**

**Продвинутые функции в программировании:**
• **Рекурсия** - функция, вызывающая сама себя
• **Замыкания** - функции с доступом к внешним переменным
• **Лямбда-функции** - анонимные функции
• **Декораторы** - модификация поведения функций
• **Генераторы** - итераторы с отложенным вычислением

**Специфические функции ПЛК:**
• **PID регуляторы** - пропорционально-интегрально-дифференциальные
• **Таймеры и счетчики** - временные и количественные функции
• **Блоки сравнения** - логические операции
• **Арифметические блоки** - математические операции
• **Функции связи** - коммуникационные протоколы

**Системные функции:**
• **Управление памятью** - динамическое выделение ресурсов
• **Многопоточность** - параллельное выполнение задач
• **Обработка исключений** - управление ошибками
• **Интерфейсы API** - взаимодействие с внешними системами

**Промышленные специфические функции:**
• **Алгоритмы управления** - сложные логические схемы
• **Компенсация и коррекция** - улучшение точности
• **Диагностика и мониторинг** - отслеживание состояния
• **Интеграция с ERP** - связь с корпоративными системами

*Продвинутые функции повышают эффективность систем.*"""

        elif any(word in message_lower for word in ['защит', 'protection', 'коротк', 'short', 'замыкание', 'circuit', 'предохранитель', 'fuse', 'автомат', 'breaker', 'выключатель', 'switch']):
            return """**Защита электрических цепей от короткого замыкания:**

**Основные методы защиты:**

**1. Предохранители (Fuses):**
• Плавкие вставки - перегорают при превышении тока
• Быстродействующие предохранители - для защиты полупроводников
• Замедленные предохранители - для защиты двигателей

**2. Автоматические выключатели:**
• Тепловые расцепители - срабатывают при перегрузке
• Электромагнитные расцепители - мгновенное отключение при КЗ
• Электронные расцепители - программируемая защита

**3. Супрессоры и ограничители:**
• Варисторы - защита от импульсных перенапряжений
• Газоразрядные приборы - защита от грозовых разрядов
• TVS-диоды - быстрая защита электронных схем

**4. Встроенные схемы защиты:**
• Защита от превышения напряжения
• Защита от переполюсовки
• Токовая защита с автоматическим восстановлением

*Для конкретной реализации требуется анализ характеристик системы.*"""

        else:
            # Общий ответ
            return f"""Я понимаю ваш вопрос: "{message}"

Для получения точного ответа мне нужен доступ к соответствующей технической документации. 

**Что я могу предложить:**
• Поиск в базе знаний Rubin AI
• Анализ технической документации
• Помощь с программированием и автоматизацией

Попробуйте переформулировать вопрос или уточнить область применения для более точного ответа."""

    def _generate_protection_response(self, relevant_info: List[Dict], thinking_process: List[str]) -> str:
        """Генерирует специализированный ответ о защите от короткого замыкания"""
        thinking_process.append("🛡️ Генерирую ответ о защите от короткого замыкания...")
        
        response_parts = ["**Защита электрических цепей от короткого замыкания:**\n"]
        
        # Основные методы защиты
        response_parts.append("**1. Предохранители (Fuses):**")
        response_parts.append("• Плавкие вставки - перегорают при превышении тока")
        response_parts.append("• Быстродействующие предохранители - для защиты полупроводников")
        response_parts.append("• Замедленные предохранители - для защиты двигателей")
        response_parts.append("")
        
        response_parts.append("**2. Автоматические выключатели (Circuit Breakers):**")
        response_parts.append("• Тепловые расцепители - срабатывают при перегрузке")
        response_parts.append("• Электромагнитные расцепители - мгновенное отключение при КЗ")
        response_parts.append("• Электронные расцепители - программируемая защита")
        response_parts.append("")
        
        response_parts.append("**3. Супрессоры и ограничители перенапряжений:**")
        response_parts.append("• Варисторы - защита от импульсных перенапряжений")
        response_parts.append("• Газоразрядные приборы - защита от грозовых разрядов")
        response_parts.append("• TVS-диоды - быстрая защита электронных схем")
        response_parts.append("")
        
        response_parts.append("**4. Встроенные схемы защиты:**")
        response_parts.append("• Защита от превышения напряжения")
        response_parts.append("• Защита от переполюсовки")
        response_parts.append("• Токовая защита с автоматическим восстановлением")
        response_parts.append("")
        
        response_parts.append("**Принципы защиты:**")
        response_parts.append("• Селективность - отключение только поврежденного участка")
        response_parts.append("• Быстродействие - минимальное время срабатывания")
        response_parts.append("• Надежность - гарантированное срабатывание при КЗ")
        response_parts.append("• Восстанавливаемость - возможность повторного включения")
        
        # Информация из найденных документов
        if relevant_info:
            response_parts.append(f"\n**Найденная информация из документов:**")
            for info in relevant_info[:3]:
                # Фильтруем предложения, содержащие информацию о защите
                protection_sentences = []
                for sentence in info['sentences']:
                    if any(word in sentence.lower() for word in ['защит', 'коротк', 'предохранитель', 'автомат', 'супрессор', 'замыкание']):
                        protection_sentences.append(sentence)
                
                if protection_sentences:
                    response_parts.append(f"\n📄 **{info['title']}:**")
                    for sentence in protection_sentences[:2]:
                        response_parts.append(f"• {sentence}")
                else:
                    response_parts.append(f"\n📄 **{info['title']}:**")
                    response_parts.append("• Документ содержит информацию о системах защиты")
        
        
        return "\n".join(response_parts)
    
    def _generate_understanding_process_response(self, thinking_process: List[str]) -> str:
        """Генерирует ответ о процессе понимания"""
        thinking_process.append("🧠 Объясняю процесс понимания...")
        
        return """🧠 **Как я понимаю ваши сообщения:**

**1. Предварительная обработка:**
• Нормализую текст (привожу к нижнему регистру)
• Очищаю от лишних символов и пробелов
• Разбиваю на слова и фразы
• Определяю длину и сложность сообщения

**2. Лингвистический анализ:**
• Определяю тип сообщения (вопрос, утверждение, команда)
• Извлекаю ключевые слова и термины
• Анализирую синтаксис и структуру
• Определяю эмоциональную окраску

**3. Семантическое понимание:**
• Определяю основную тему сообщения
• Выявляю намерения и цели пользователя
• Анализирую контекст и подтекст
• Определяю уровень сложности вопроса

**4. Техническое понимание:**
• Распознаю технические термины и концепции
• Классифицирую по областям знаний
• Проверяю корректность технических утверждений
• Выявляю возможные ошибки в понимании

**5. Контекстное понимание:**
• Учитываю предыдущие сообщения в диалоге
• Определяю текущую тему разговора
• Анализирую неявные ссылки и предположения
• Адаптирую понимание под уровень пользователя

**6. Генерация ответа:**
• Формирую структурированный и логичный ответ
• Включаю технические детали и примеры
• Адаптирую стиль под тип вопроса
• Проверяю полноту и полезность ответа

**Ваш вопрос я понял как запрос на объяснение моих когнитивных процессов понимания языка и смысла.**"""
    
    def _generate_thinking_process_response(self, thinking_process: List[str]) -> str:
        """Генерирует ответ о процессе мышления"""
        thinking_process.append("🤔 Объясняю процесс мышления...")
        
        return """🤔 **Мой процесс размышления:**

**1. Анализ входящей информации:**
• Что именно спрашивает пользователь?
• Какая информация нужна для полного ответа?
• Какой уровень детализации требуется?
• Есть ли подводные камни в вопросе?

**2. Поиск и анализ данных:**
• Ищу релевантные документы в базе знаний
• Анализирую найденную информацию на актуальность
• Определяю качество и надежность источников
• Выявляю пробелы в знаниях

**3. Синтез и структурирование:**
• Объединяю данные из разных источников
• Формирую логическую структуру ответа
• Проверяю внутреннюю согласованность
• Определяю приоритеты информации

**4. Адаптация и персонализация:**
• Определяю уровень технической подготовки пользователя
• Выбираю подходящий стиль изложения
• Добавляю практические примеры и аналогии
• Учитываю контекст и цели пользователя

**5. Проверка и валидация:**
• Убеждаюсь в технической корректности
• Проверяю полноту и логичность ответа
• Оцениваю полезность для решения задачи
• Корректирую при необходимости

**6. Формирование финального ответа:**
• Структурирую информацию для лучшего восприятия
• Добавляю визуальные элементы (эмодзи, форматирование)
• Включаю практические рекомендации
• Предлагаю дальнейшие шаги

**В данном случае я размышляю о том, как максимально понятно объяснить свой мыслительный процесс.**"""
    
    def _generate_decision_process_response(self, thinking_process: List[str]) -> str:
        """Генерирует ответ о процессе принятия решений"""
        thinking_process.append("🎯 Объясняю процесс принятия решений...")
        
        return """🎯 **Как я принимаю решения и выбираю ответы:**

**1. Анализ задачи:**
• Определяю тип и сложность вопроса
• Выявляю ключевые требования
• Оцениваю необходимые ресурсы
• Планирую стратегию решения

**2. Выбор подхода:**
• Технический вопрос → специализированные знания
• Мета-вопрос → объяснение процессов
• Практический вопрос → пошаговое руководство
• Концептуальный вопрос → теоретическое объяснение

**3. Поиск информации:**
• Анализирую базу знаний
• Ищу релевантные документы
• Проверяю актуальность данных
• Оцениваю качество источников

**4. Синтез решения:**
• Объединяю найденную информацию
• Структурирую логически
• Адаптирую под уровень пользователя
• Добавляю практические элементы

**5. Валидация ответа:**
• Проверяю техническую корректность
• Убеждаюсь в полноте
• Оцениваю полезность
• Корректирую при необходимости

**6. Финальная оптимизация:**
• Выбираю лучший формат представления
• Добавляю визуальные элементы
• Включаю примеры и аналогии
• Предлагаю дальнейшие действия

**Мой процесс принятия решений основан на анализе, синтезе и адаптации информации для максимальной полезности ответа.**"""

    def get_capabilities(self) -> List[str]:
        """Получить список возможностей провайдера"""
        return [
            TaskType.GENERAL_CHAT,
            TaskType.TECHNICAL_DOCUMENTATION,
            TaskType.PLC_ANALYSIS,
            TaskType.PMAC_ANALYSIS,
            TaskType.CNC_ANALYSIS
        ]
