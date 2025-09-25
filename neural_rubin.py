#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Rubin AI - Нейронная сеть для Rubin AI
Настоящий AI с обучением и развитием
"""

# Попытка импорта ML библиотек с fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Mock классы для работы без ML библиотек
    class torch:
        @staticmethod
        def device(name):
            return name
        
        @staticmethod
        def cuda_is_available():
            return False
        
        @staticmethod
        def FloatTensor(data):
            return data
        
        @staticmethod
        def randn(*args):
            return [0.1] * (args[0] * args[1] if len(args) > 1 else args[0])
        
        @staticmethod
        def argmax(tensor, dim=None):
            return 0  # Mock category index
        
        @staticmethod
        def max(tensor):
            return 0.85  # Mock confidence
        
        @staticmethod
        def no_grad():
            class NoGradContext:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return NoGradContext()
    
    class nn:
        class Module:
            def __init__(self): pass
            def to(self, device): return self
            def state_dict(self): return {}
            def load_state_dict(self, state_dict): pass

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
import json
import numpy as np
from datetime import datetime
import logging
import pickle
import os
from mathematical_problem_solver import MathematicalProblemSolver, ProblemSolution, ProblemType
from rubin_time_series_processor import RubinTimeSeriesProcessor, NPriceType
import re # Добавляем для парсинга параметров временных рядов
import csv # Добавляем для логирования истории обучения
from typing import Optional, Tuple, List, Dict, Any # Добавляем для Optional типизации
import matplotlib.pyplot as plt # Добавляем для визуализации
from rubin_data_preprocessor import RubinDataPreprocessor # Добавляем для нормализации данных

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Доступные функции активации
ACTIVATION_FUNCTIONS = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'Softsign': nn.Softsign,
    'Sigmoid': nn.Sigmoid, # Часто используется
    'ELU': nn.ELU,
    'LeakyReLU': nn.LeakyReLU, # Распространенный вариант ReLU
    # 'Arctg':, 'PReLU':, 'SoftPlus':, 'Sin':, 'Sinc':, 'Gaussian': - эти функции потребуют пользовательской реализации, так как их нет напрямую в nn
}

class RubinNeuralNetwork(nn.Module):
    """Нейронная сеть Rubin AI"""
    
    def __init__(self, input_size=384, hidden_sizes=[512, 256, 128], num_classes=10, activations=None, dropout_rates=None):
        super(RubinNeuralNetwork, self).__init__()
        
        if activations is None:
            activations = ['ReLU'] * len(hidden_sizes) # По умолчанию ReLU для всех скрытых слоев

        if dropout_rates is None:
            dropout_rates = [0.2] * (len(hidden_sizes) - 1) # Dropout после каждого скрытого слоя, кроме последнего
        
        # Проверка соответствия длин
        if len(activations) != len(hidden_sizes):
            raise ValueError("Длина списка активаций должна соответствовать количеству скрытых слоев.")
        if len(dropout_rates) != (len(hidden_sizes) - 1):
            logger.warning("⚠️ Количество dropout-слоев не соответствует количеству скрытых слоев - 1. Будут применены значения по умолчанию.")
            dropout_rates = [0.2] * (len(hidden_sizes) - 1)

        layers = []
        current_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(current_size, hidden_size))
            
            activation_name = activations[i]
            if activation_name not in ACTIVATION_FUNCTIONS:
                raise ValueError(f"Неизвестная функция активации: {activation_name}. Доступные: {list(ACTIVATION_FUNCTIONS.keys())}")
            layers.append(ACTIVATION_FUNCTIONS[activation_name]())

            if i < len(hidden_sizes) - 1 and dropout_rates[i] > 0:
                layers.append(nn.Dropout(dropout_rates[i]))
            
            current_size = hidden_size
            
        layers.append(nn.Linear(current_size, num_classes))
        
        self.encoder = nn.Sequential(*layers)
        
        # Классификатор типов вопросов
        self.classifier = nn.Softmax(dim=1)
        
    def forward(self, x):
        """Прямой проход через сеть"""
        encoded = self.encoder(x)
        classified = self.classifier(encoded)
        return classified

class NeuralRubinAI:
    """Главный класс нейронной сети Rubin AI"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Используется устройство: {self.device}")
        
        # Инициализация компонентов
        self.sentence_model = None
        self.neural_network = None
        self.tokenizer = None
        self.knowledge_base = {}
        self.conversation_history = []
        self.math_solver = MathematicalProblemSolver() # Инициализация математического решателя
        self.time_series_processor = RubinTimeSeriesProcessor() # Инициализация процессора временных рядов
        self.data_preprocessor = RubinDataPreprocessor() # Инициализация препроцессора данных
        
        # Категории вопросов
        self.categories = [
            'математика', 'физика', 'электротехника', 'программирование',
            'геометрия', 'химия', 'общие_вопросы', 'техника', 'наука', 'другое',
            'time_series', 'graph_analysis', 'data_visualization', 'formula_calculation'
        ]
        
        # Добавляем категорию physics, если ее нет
        if 'physics' not in self.categories:
            self.categories.insert(1, 'physics') # Вставляем после математики
        
        # Интеграция с улучшенными обработчиками
        self.enhanced_dispatcher = None
        self.programming_handler = None
        self.electrical_handler = None
        self.enhanced_categorizer = None
        
        self.initialize_models()
        self.initialize_enhanced_handlers()
    
    def initialize_models(self):
        """Инициализация моделей"""
        try:
            logger.info("🧠 Инициализация нейронной сети...")
            
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # Модель для создания эмбеддингов
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Sentence Transformer загружен")
            else:
                logger.warning("⚠️ SentenceTransformer недоступен, используем mock")
                self.sentence_model = None
            
            if ML_AVAILABLE:
                # Наша нейронная сеть
                self.neural_network = RubinNeuralNetwork(
                    input_size=384,  # Размер эмбеддингов
                    hidden_sizes=[1024, 512, 256, 128], # Сеть сделана глубже и шире
                    num_classes=len(self.categories)
                ).to(self.device)
                
                logger.info("✅ Нейронная сеть инициализирована")
                
                # Загружаем предобученную модель если есть (закомментировано для демонстрации с гибкими активациями)
                # self.load_model()
            else:
                logger.warning("⚠️ PyTorch недоступен, используем mock нейронную сеть")
                self.neural_network = None
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}")
            # Fallback к простой модели
            self.sentence_model = None
            self.neural_network = None
    
    def initialize_enhanced_handlers(self):
        """Инициализация улучшенных обработчиков"""
        try:
            logger.info("🔗 Интеграция с улучшенными обработчиками...")
            
            # Импортируем улучшенные компоненты
            from intelligent_dispatcher import get_intelligent_dispatcher
            from programming_knowledge_handler import get_programming_handler
            from electrical_knowledge_handler import get_electrical_handler
            from enhanced_request_categorizer import get_enhanced_categorizer
            
            self.enhanced_dispatcher = get_intelligent_dispatcher()
            self.programming_handler = get_programming_handler()
            self.electrical_handler = get_electrical_handler()
            self.enhanced_categorizer = get_enhanced_categorizer()
            
            logger.info("✅ Нейронная сеть интегрирована с улучшенными обработчиками!")
            logger.info(f"Диспетчер инициализирован с метриками модулей: {self.enhanced_dispatcher.module_metrics}") # Добавляем лог
            
        except ImportError as e:
            logger.warning(f"⚠️ Не удалось интегрировать улучшенные обработчики: {e}")
            self.enhanced_dispatcher = None
            self.programming_handler = None
            self.electrical_handler = None
            self.enhanced_categorizer = None
    
    def create_embedding(self, text):
        """Создает эмбеддинг для текста"""
        try:
            if self.sentence_model and SENTENCE_TRANSFORMERS_AVAILABLE:
                embedding = self.sentence_model.encode(text)
                if ML_AVAILABLE:
                    return torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
                else:
                    return embedding
            else:
                # Простой fallback эмбеддинг на основе длины текста и ключевых слов
                import random
                random.seed(len(text))  # Детерминированный эмбеддинг
                return [random.random() for _ in range(384)]
        except Exception as e:
            logger.error(f"Ошибка создания эмбеддинга: {e}")
            return [0.1] * 384
    
    def classify_question(self, text):
        """Классифицирует вопрос с помощью нейронной сети"""
        try:
            if not self.neural_network or not ML_AVAILABLE:
                return self._simple_classification(text)
            
            # Создаем эмбеддинг
            embedding = self.create_embedding(text)
            
            # Прогоняем через нейронную сеть
            with torch.no_grad():
                output = self.neural_network(embedding)
                predicted_class = torch.argmax(output, dim=1).item()
            
            category = self.categories[predicted_class]
            confidence = float(torch.max(output).item())
            
            logger.info(f"🎯 Нейронная классификация: {category} (уверенность: {confidence:.2f})")
            return category, confidence
            
        except Exception as e:
            logger.error(f"Ошибка нейронной классификации: {e}")
            return self._simple_classification(text)
    
    def _simple_classification(self, text):
        """Простая классификация без нейронной сети"""
        text_lower = text.lower()
        
        # Ключевые слова для категорий
        keywords = {
            'математика': ['сколько', '+', '-', '*', '/', 'вычисли', 'реши'],
            'физика': ['скорость', 'время', 'расстояние', 'сила', 'энергия'],
            'электротехника': ['транзистор', 'диод', 'ток', 'напряжение', 'сопротивление'],
            'программирование': ['код', 'python', 'c++', 'алгоритм', 'программа'],
            'геометрия': ['угол', 'треугольник', 'площадь', 'периметр']
        }
        
        for category, words in keywords.items():
            if any(word in text_lower for word in words):
                return category, 0.8
        
        return 'общие_вопросы', 0.5
    
    def generate_response(self, question):
        """Генерирует ответ на вопрос с использованием улучшенных обработчиков"""
        try:
            logger.info(f"🧠 Нейронная сеть обрабатывает: {question[:50]}...")
            
            # Если доступны улучшенные обработчики, используем их
            if self.enhanced_categorizer and self.enhanced_dispatcher:
                logger.info("🔗 Используем улучшенную категоризацию в нейронной сети")
                
                # Используем улучшенный категоризатор
                category = self.enhanced_categorizer.categorize(question)
                confidence = self.enhanced_categorizer.get_confidence(question, category)
                
                logger.info(f"🎯 Нейронная сеть + улучшенная категоризация: {category} (уверенность: {confidence:.2f})")
                
                # Используем улучшенные обработчики
                request_data = {'message': question}
                
                if category == 'programming' and self.programming_handler:
                    response_data = self.programming_handler.handle_request(question)
                    response = response_data['response']
                    provider = f"Neural + {response_data.get('provider', 'Programming Handler')}"
                    
                elif category == 'electrical' and self.electrical_handler:
                    response_data = self.electrical_handler.handle_request(question)
                    response = response_data['response']
                    provider = f"Neural + {response_data.get('provider', 'Electrical Handler')}"
                    
                elif category.startswith('mathematics'):
                    response_data = self.enhanced_dispatcher._handle_mathematical_request(request_data)
                    response = response_data['response']
                    provider = f"Neural + {response_data.get('provider', 'Mathematical Handler')}"
                    
                elif category == 'time_series' and self.time_series_processor:
                    # Заменяем некорректный вызов на правильный метод
                    response = self._solve_time_series_neural(question)
                    provider = "Neural + Time Series Handler"
                    
                elif category == 'physics' or category == 'formula_calculation':
                    # Обработка физических формул
                    response = self._solve_physics_neural(question)
                    provider = "Neural + Physics Handler"
                    
                elif category == 'chemistry':
                    # Обработка химических формул
                    response = self._solve_chemistry_neural(question)
                    provider = "Neural + Chemistry Handler"
                    
                elif category == 'graph_analysis':
                    # Анализ графиков
                    response = self._analyze_graph_neural(question)
                    provider = "Neural + Graph Analysis Handler"
                    
                elif category == 'data_visualization':
                    # Создание визуализаций
                    response = self._create_visualization_neural(question)
                    provider = "Neural + Data Visualization Handler"
                    
                else:
                    # Для других категорий используем общий обработчик
                    response_data = self.enhanced_dispatcher._handle_general_request(request_data)
                    response = response_data['response']
                    provider = f"Neural + {response_data.get('provider', 'General Handler')}"
                
                # Добавляем нейронную обработку к ответу
                neural_enhanced_response = f"""🧠 **Нейронная сеть Rubin AI + Улучшенные обработчики**

{response}

---
*Обработано нейронной сетью с категоризацией: {category} (уверенность: {confidence:.1%})*"""
                
            else:
                # Fallback к старой логике нейронной сети
                logger.info("⚠️ Улучшенные обработчики недоступны, используем базовую нейронную логику")
                category, confidence = self.classify_question(question)
                neural_enhanced_response = self._generate_category_response(question, category, confidence)
                provider = "Neural Rubin AI (Fallback)"
                
                # Добавляем информацию о нейронной сети
                neural_info = f"\n\n🧠 **Нейронная сеть:** Категория '{category}' (уверенность: {confidence:.1%})"
                neural_enhanced_response += neural_info
            
            # Сохраняем в историю
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'category': category,
                'confidence': confidence,
                'enhanced_processing': self.enhanced_categorizer is not None
            })
            
            return {
                'response': neural_enhanced_response,
                'category': category,
                'confidence': confidence,
                'neural_network': True,
                'enhanced_integration': self.enhanced_categorizer is not None,
                'provider': provider,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return {
                'response': f'Произошла ошибка в нейронной сети: {str(e)}',
                'category': 'error',
                'confidence': 0.0,
                'neural_network': False,
                'provider': 'Neural Error Handler'
            }
    
    def _generate_category_response(self, question, category, confidence):
        """Генерирует ответ для конкретной категории"""
        
        if category == 'математика':
            return self._solve_math_neural(question)
        elif category == 'физика':
            return self._solve_physics_neural(question)
        elif category == 'электротехника':
            return self._explain_electronics_neural(question)
        elif category == 'программирование':
            return self._explain_programming_neural(question)
        elif category == 'time_series':
            return self._solve_time_series_neural(question)
        else:
            return self._general_response_neural(question, category)
    
    def _solve_math_neural(self, question):
        """
        Решает математические задачи с помощью нейронной сети.
        Здесь будет интегрирована более сложная логика математических расчетов.
        """
        try:
            solution = self.math_solver.solve_problem(question)
            
            if solution.problem_type != ProblemType.UNKNOWN and solution.final_answer is not None:
                answer_str = str(solution.final_answer)
                if isinstance(solution.final_answer, dict):
                    answer_str = ', '.join([f'{k}={v:.2f}' for k, v in solution.final_answer.items()])
                elif isinstance(solution.final_answer, (float, int)):
                    answer_str = f'{solution.final_answer:.2f}'

                steps_str = "\n".join([f'    - {step}' for step in solution.solution_steps])
                
                return f"""🧮 **Нейронное решение математики:**\n\n**Задача:** {question}\n**Тип задачи:** {solution.problem_type.value}\n**Ответ:** {answer_str}\n\n**Пошаговое решение:**\n{steps_str}\n\n**Объяснение:** {solution.explanation}\n\n**Процесс нейронной сети:**\n1. Нейронная сеть классифицировала вопрос как математический.\n2. Встроенный математический модуль распознал тип задачи: {solution.problem_type.value}.\n3. Проведены расчеты и сформирован пошаговый ответ.\n---\n*Уверенность в решении: {solution.confidence:.1%} (Проверено: {'✓' if solution.verification else '✗'})*"""
            else:
                return f"🧮 Нейронная сеть анализирует математическую задачу... Не удалось найти точное решение. {solution.explanation}"
        except Exception as e:
            logger.error(f"Ошибка в математическом решателе: {e}")
            return f"🧮 Нейронная сеть столкнулась с ошибкой при решении математической задачи: {e}"
    
    def _solve_physics_neural(self, question):
        """Решает физические задачи"""
        return "⚡ Нейронная сеть анализирует физическую задачу..."
    
    def _explain_electronics_neural(self, question):
        """Объясняет электронику"""
        return "🔌 Нейронная сеть анализирует электротехнический вопрос..."
    
    def _explain_programming_neural(self, question):
        """Объясняет программирование"""
        return "💻 Нейронная сеть анализирует вопрос по программированию..."
    
    def _general_response_neural(self, question, category):
        """Общий ответ"""
        return f"🤖 Нейронная сеть классифицировала вопрос как '{category}' и анализирует ответ..."

    def _solve_time_series_neural(self, question):
        """
        Обрабатывает и прогнозирует временные ряды.
        Извлекает параметры из вопроса, использует RubinTimeSeriesProcessor и возвращает результат.
        """
        logger.info(f"📊 Обработка запроса на прогнозирование временных рядов: {question}")

        # Извлечение параметров из вопроса (упрощенный парсинг для примера)
        period = 1
        len_in = 1
        price_type = NPriceType.PriceClose # По умолчанию
        
        # Поиск периода
        period_match = re.search(r'период\s*(\d+)', question.lower())
        if period_match:
            period = int(period_match.group(1))

        # Поиск len_in
        len_in_match = re.search(r'входных\s*баров\s*(\d+)|len_in\s*(\d+)', question.lower())
        if len_in_match:
            len_in = int(len_in_match.group(1) or len_in_match.group(2))
            
        # Поиск price_type (можно расширить для более сложных запросов)
        if 'high' in question.lower() or 'максимум' in question.lower():
            price_type = NPriceType.PriceHigh
        elif 'low' in question.lower() or 'минимум' in question.lower():
            price_type = NPriceType.PriceLow
        elif 'close' in question.lower() or 'закрытие' in question.lower():
            price_type = NPriceType.PriceClose
            
        # Тестовые данные (аналогичные тем, что в rubin_time_series_processor.py)
        # В реальной системе эти данные будут загружаться из файла или базы данных
        raw_data = np.array([
            [100, 105, 98, 103, 1000],
            [103, 107, 101, 106, 1200],
            [106, 110, 104, 108, 1100],
            [108, 112, 106, 110, 1300],
            [110, 115, 108, 113, 1500],
            [113, 117, 111, 116, 1400],
            [116, 120, 114, 119, 1600],
            [119, 122, 117, 121, 1700],
            [121, 125, 119, 123, 1800],
            [123, 127, 121, 125, 1900],
        ], dtype=float)

        try:
            self.time_series_processor.set_parameters(
                period=period, 
                price_type=price_type, 
                len_in=len_in,
                koef_tg=10000.0, # Можно тоже парсить из вопроса, если нужно
                koef_price=1.0,
                koef_volume=1.0
            )
            
            processed_examples = self.time_series_processor.preprocess_data(raw_data)
            
            if processed_examples:
                # Временно возвращаем последнее известное 'output' из примеров для осмысленного ответа
                # В будущем здесь будет реальный прогноз от обученной нейронной сети
                predicted_output_raw = processed_examples[-1]['output'][0] 
                predicted_output = self.time_series_processor.postprocess_output(np.array([predicted_output_raw]))

                response = f"📊 **Прогнозирование временных рядов (временный ответ):**\n\n"
                response += f"**Параметры:** период={period}, входных баров={len_in}, тип цены={price_type}\n"
                response += f"**Предполагаемое следующее значение (временный прогноз):** {predicted_output:.4f}\n\n"
                response += "**Процесс:** Данные были агрегированы, преобразованы и подготовлены. " \
                            "(Временный прогноз на основе последнего известного значения, нейронная сеть пока не обучена для точного прогнозирования)"
            else:
                response = "⚠️ Не удалось обработать временные ряды с заданными параметрами или данными."

            return response

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке временных рядов: {e}")
            return f"❌ Произошла ошибка при прогнозировании временных рядов: {e}"
    
    def _solve_physics_neural(self, question):
        """Решение физических задач с помощью нейронной сети"""
        try:
            logger.info(f"⚡ Решение физической задачи: {question}")
            
            # Используем расширенный математический решатель
            result = self.math_solver.solve_problem(question)
            
            if result.problem_type.value == "физические_формулы":
                return f"""⚡ **ФИЗИЧЕСКИЙ РАСЧЕТ:**

**Задача:** {question}

**Пошаговое решение:**
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(result.solution_steps)])}

**Результат:** {result.final_answer}
**Уверенность:** {result.confidence:.1%}
**Объяснение:** {result.explanation}

*Расчет выполнен с помощью расширенного математического решателя Rubin AI*"""
            else:
                return f"""⚡ **ФИЗИЧЕСКАЯ ЗАДАЧА:**

**Задача:** {question}

**Статус:** Задача распознана как физическая, но требует дополнительной обработки
**Тип:** {result.problem_type.value}
**Уверенность:** {result.confidence:.1%}

*Используйте более конкретные формулировки для лучшего распознавания формул*"""
                
        except Exception as e:
            logger.error(f"Ошибка решения физической задачи: {e}")
            return f"Ошибка решения физической задачи: {e}"
    
    def _solve_chemistry_neural(self, question):
        """Решение химических задач с помощью нейронной сети"""
        try:
            logger.info(f"🧪 Решение химической задачи: {question}")
            
            # Используем расширенный математический решатель
            result = self.math_solver.solve_problem(question)
            
            if result.problem_type.value == "химические_расчеты":
                return f"""🧪 **ХИМИЧЕСКИЙ РАСЧЕТ:**

**Задача:** {question}

**Пошаговое решение:**
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(result.solution_steps)])}

**Результат:** {result.final_answer}
**Уверенность:** {result.confidence:.1%}
**Объяснение:** {result.explanation}

*Расчет выполнен с помощью расширенного математического решателя Rubin AI*"""
            else:
                return f"""🧪 **ХИМИЧЕСКАЯ ЗАДАЧА:**

**Задача:** {question}

**Статус:** Задача распознана как химическая, но требует дополнительной обработки
**Тип:** {result.problem_type.value}
**Уверенность:** {result.confidence:.1%}

*Используйте более конкретные формулировки для лучшего распознавания формул*"""
                
        except Exception as e:
            logger.error(f"Ошибка решения химической задачи: {e}")
            return f"Ошибка решения химической задачи: {e}"
    
    def _analyze_graph_neural(self, question):
        """Анализ графиков с помощью нейронной сети"""
        try:
            logger.info(f"📊 Анализ графика: {question}")
            
            # Используем расширенный математический решатель
            result = self.math_solver.solve_problem(question)
            
            if result.problem_type.value == "анализ_графиков":
                return f"""📊 **АНАЛИЗ ГРАФИКА:**

**Задача:** {question}

**Пошаговый анализ:**
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(result.solution_steps)])}

**Результат:** {result.final_answer}
**Уверенность:** {result.confidence:.1%}
**Объяснение:** {result.explanation}

*Анализ выполнен с помощью расширенного математического решателя Rubin AI*"""
            else:
                return f"""📊 **АНАЛИЗ ГРАФИКА:**

**Задача:** {question}

**Статус:** Задача распознана как анализ графика, но требует дополнительной обработки
**Тип:** {result.problem_type.value}
**Уверенность:** {result.confidence:.1%}

*Для полного анализа графиков требуется реализация OCR и компьютерного зрения*"""
                
        except Exception as e:
            logger.error(f"Ошибка анализа графика: {e}")
            return f"Ошибка анализа графика: {e}"
    
    def _create_visualization_neural(self, question):
        """Создание визуализаций с помощью нейронной сети"""
        try:
            logger.info(f"📈 Создание визуализации: {question}")
            
            # Используем расширенный математический решатель
            result = self.math_solver.solve_problem(question)
            
            if result.problem_type.value == "визуализация_данных":
                return f"""📈 **ВИЗУАЛИЗАЦИЯ ДАННЫХ:**

**Задача:** {question}

**Процесс создания:**
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(result.solution_steps)])}

**Результат:** {result.final_answer}
**Уверенность:** {result.confidence:.1%}
**Объяснение:** {result.explanation}

*Визуализация создана с помощью расширенного математического решателя Rubin AI*"""
            else:
                return f"""📈 **ВИЗУАЛИЗАЦИЯ ДАННЫХ:**

**Задача:** {question}

**Статус:** Задача распознана как визуализация данных, но требует дополнительной обработки
**Тип:** {result.problem_type.value}
**Уверенность:** {result.confidence:.1%}

*Для создания графиков требуется реализация модуля визуализации данных*"""
                
        except Exception as e:
            logger.error(f"Ошибка создания визуализации: {e}")
            return f"Ошибка создания визуализации: {e}"

    def _load_training_data(self, training_file: str) -> List[Dict[str, Any]]:
        """
        Загружает данные для обучения из JSONL-файла.
        """
        training_data = []
        if os.path.exists(training_file):
            with open(training_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        training_data.append(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Ошибка декодирования JSON в файле {training_file}: {e} - строка: {line.strip()}")
        else:
            logger.warning(f"⚠️ Файл обучения {training_file} не найден.")
        return training_data

    def learn_from_feedback(self, question, correct_category, user_rating):
        """Обучается на основе обратной связи"""
        try:
            logger.info(f"📚 Обучение: {question} -> {correct_category} (рейтинг: {user_rating})")
            
            # Здесь можно добавить логику обучения нейронной сети
            # Например, fine-tuning на основе обратной связи
            
            # Сохраняем данные для обучения
            training_data = {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'correct_category': correct_category,
                'user_rating': user_rating
            }
            
            # Добавляем в файл обучающих данных
            with open('neural_training_data.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(training_data, ensure_ascii=False) + '\n')
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обучения: {e}")
            return False
    
    def save_model(self):
        """Сохраняет модель"""
        try:
            if self.neural_network:
                torch.save(self.neural_network.state_dict(), 'rubin_neural_model.pth')
                logger.info("💾 Модель сохранена")
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")
    
    def load_model(self):
        """Загружает модель"""
        try:
            if os.path.exists('rubin_neural_model.pth') and self.neural_network:
                self.neural_network.load_state_dict(torch.load('rubin_neural_model.pth'))
                logger.info("📂 Модель загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")

    def _prepare_training_data(self, training_data, normalize_embeddings: bool = True) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Подготавливает данные для обучения нейронной сети.
        """
        try:
            questions = []
            categories = []
            for data in training_data:
                questions.append(data['question'])
                categories.append(data['correct_category'])

            if not questions:
                logger.warning("⚠️ Нет данных для обучения. Обучение невозможно.")
                return None, None

            # Преобразование категорий в числовые метки
            category_to_idx = {cat: i for i, cat in enumerate(self.categories)}
            numerical_labels = [category_to_idx.get(cat, len(self.categories) - 1) for cat in categories] # Fallback to 'другое'

            # Генерация эмбеддингов для всех вопросов
            logger.info("🔄 Создание эмбеддингов для обучающих данных...")
            embeddings = self.sentence_model.encode(questions, convert_to_numpy=True)
            embeddings_tensor = torch.FloatTensor(embeddings).to(self.device)
            labels_tensor = torch.LongTensor(numerical_labels).to(self.device)

            # Нормализация эмбеддингов перед подачей в нейронную сеть
            if normalize_embeddings:
                # Для простоты используем линейную нормализацию
                min_val = embeddings_tensor.min()
                max_val = embeddings_tensor.max()
                if min_val == max_val: # Избегаем деления на ноль, если все значения одинаковы
                    logger.warning("⚠️ Все эмбеддинги имеют одинаковое значение. Пропускаем нормализацию.")
                else:
                    # Убедимся, что нормализация происходит для тензора PyTorch, используя .cpu() для NumPy-операций
                    # и .to(self.device) обратно для PyTorch
                    normalized_np = self.data_preprocessor.linear_normalization(embeddings_tensor.cpu().numpy(), min_val.item(), max_val.item())
                    embeddings_tensor = torch.FloatTensor(normalized_np).to(self.device)

            return embeddings_tensor, labels_tensor
        
        except Exception as e:
            logger.error(f"❌ Ошибка при подготовке тренировочных данных: {e}")
            return None, None

    def train_neural_network(self, training_file='neural_training_data.jsonl', num_epochs=10, learning_rate=0.001, weight_decay=0.01, log_file_path: Optional[str] = "training_log.csv", normalize_embeddings: bool = True):
        """
        Обучает нейронную сеть на основе данных обратной связи.
        Включает L2-регуляризацию (weight_decay) и использует Dropout (уже в архитектуре).
        """
        if not ML_AVAILABLE or not self.neural_network or not self.sentence_model:
            logger.warning("⚠️ Нейронная сеть или SentenceTransformer недоступны для обучения.")
            return

        try:
            # 1. Загрузка данных для обучения
            training_data = self._load_training_data(training_file)
            if not training_data:
                logger.warning("⚠️ Нет данных для обучения. Обучение невозможно.")
                return

            # 2. Подготовка данных для обучения (вызываем новый метод)
            embeddings_tensor, labels_tensor = self._prepare_training_data(training_data, normalize_embeddings=normalize_embeddings)

            if embeddings_tensor is None or labels_tensor is None:
                logger.error("❌ Ошибка: не удалось подготовить данные для обучения.")
                return

            # 2. Определение функции потерь и оптимизатора
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.neural_network.parameters(), lr=learning_rate, weight_decay=weight_decay) # L2-регуляризация

            # 3. Цикл обучения
            self.neural_network.train() # Переводим сеть в режим обучения (активирует Dropout)
            
            # Инициализация CSV-лога
            if log_file_path:
                with open(log_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['Epoch', 'Loss'])
            
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = self.neural_network(embeddings_tensor)
                loss = criterion(outputs, labels_tensor)
                loss.backward()
                optimizer.step()

                logger.info(f'Эпоха [{epoch+1}/{num_epochs}], Потери: {loss.item():.4f}')
                
                # Запись в CSV-лог
                if log_file_path:
                    with open(log_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([epoch + 1, f'{loss.item():.4f}'])

            self.neural_network.eval() # Переводим сеть в режим оценки (отключает Dropout)
            self.save_model()
            logger.info("✅ Обучение нейронной сети завершено.")

        except Exception as e:
            logger.error(f"❌ Ошибка при обучении нейронной сети: {e}")

    def get_neural_stats(self):
        """Получает статистику нейронной сети"""
        return {
            'device': str(self.device),
            'neural_network_active': self.neural_network is not None,
            'sentence_model_active': self.sentence_model is not None,
            'categories': self.categories,
            'conversation_count': len(self.conversation_history),
            'model_parameters': sum(p.numel() for p in self.neural_network.parameters()) if self.neural_network else 0
        }

def plot_training_history(log_file_path: str = "training_log.csv"):
    """Строит график истории обучения из CSV-файла."""
    epochs = []
    losses = []
    
    if not os.path.exists(log_file_path):
        logger.warning(f"⚠️ Файл логов обучения {log_file_path} не найден для построения графика.")
        return

    try:
        with open(log_file_path, 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader) # Пропускаем заголовок
            for row in csv_reader:
                if len(row) == 2:
                    epochs.append(int(row[0]))
                    losses.append(float(row[1]))

        if not epochs:
            logger.warning("⚠️ Нет данных для построения графика в файле логов.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, marker='o', linestyle='-', color='blue')
        plt.title('История обучения нейронной сети')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери (Loss)')
        plt.grid(True)
        plt.show()
        logger.info("✅ График истории обучения построен.")

    except Exception as e:
        logger.error(f"❌ Ошибка при построении графика истории обучения: {e}")

# Глобальный экземпляр нейронной сети
neural_rubin = None

def get_neural_rubin():
    """Получает глобальный экземпляр нейронной сети"""
    global neural_rubin
    if neural_rubin is None:
        neural_rubin = NeuralRubinAI()
    return neural_rubin

if __name__ == "__main__":
    print("🚀 Запуск Rubin AI v2 в тестовом режиме")
    ai = get_neural_rubin()

    # Тестирование гибких функций активации
    print("\n🧠 ТЕСТИРОВАНИЕ ГИБКИХ ФУНКЦИЙ АКТИВАЦИИ")
    # Инициализируем нейронную сеть с другими параметрами для демонстрации
    # Переинициализация neural_network в ai для демонстрации
    ai.neural_network = RubinNeuralNetwork(
        input_size=384,
        hidden_sizes=[256, 128, 64],
        num_classes=len(ai.categories),
        activations=['Tanh', 'ReLU', 'Softsign'], # Пример разных активаций
        dropout_rates=[0.1, 0.1] # Пример других dropout rates
    ).to(ai.device)
    logger.info(f"✅ Нейронная сеть переинициализирована с гибкими активациями. Архитектура: {ai.neural_network.encoder}")

    # Проводим обучение с новыми параметрами
    print("\n📚 ТЕСТИРОВАНИЕ ОБУЧЕНИЯ С НОВЫМИ АКТИВАЦИЯМИ")
    training_file = 'neural_training_data.jsonl'
    if not os.path.exists(training_file):
        with open(training_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'question': '2+2', 'correct_category': 'математика', 'user_rating': 5}, ensure_ascii=False) + '\n')
            f.write(json.dumps({'question': 'Что такое транзистор?', 'correct_category': 'электротехника', 'user_rating': 5}, ensure_ascii=False) + '\n')
            f.write(json.dumps({'question': 'Реши 5*10', 'correct_category': 'математика', 'user_rating': 5}, ensure_ascii=False) + '\n')
            f.write(json.dumps({'question': 'Как работает for loop в python?', 'correct_category': 'программирование', 'user_rating': 5}, ensure_ascii=False) + '\n')

    ai.train_neural_network(num_epochs=5, learning_rate=0.01, log_file_path="training_log.csv") # Исправлено имя файла логов
    plot_training_history(log_file_path="training_log.csv") # Исправлено имя файла логов

    print("\n❓ ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ ОТВЕТОВ С ВЕКТОРНЫМ ПОИСКОМ И НОВЫМИ ВОЗМОЖНОСТЯМИ")
    test_questions = [
        "Что такое квантовая физика?",
        "Реши уравнение 2x + 5 = 11",
        "Напиши программу на Python, которая выводит Hello, World!",
        "Какое сопротивление резистора, если напряжение 12В и ток 2А?",
        "Дай прогноз временных рядов с периодом 3 и 2 входных бара для цены закрытия.", # Тест временных рядов
        "Какие бывают виды реакторов?"
    ]

    for question in test_questions:
        print(f"\n❓ Вопрос: {question}")
        response = ai.generate_response(question)
        print(f"🤖 Ответ: {response['response']}")
        print(f"📂 Категория: {response['category']}")
        print(f"📊 Уверенность: {response['confidence']:.1%}")
    
    # Статистика
    stats = ai.get_neural_stats()
    print(f"\n📊 СТАТИСТИКА НЕЙРОННОЙ СЕТИ:")
    print(f"• Устройство: {stats['device']}")
    print(f"• Нейронная сеть активна: {stats['neural_network_active']}")
    print(f"• Параметров в модели: {stats['model_parameters']:,}")
    print(f"• Диалогов: {stats['conversation_count']}")