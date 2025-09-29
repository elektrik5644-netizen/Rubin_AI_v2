#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенная система обучения Rubin AI с правильной векторизацией
"""

import json
import random
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple
from neural_rubin_v2 import EnhancedNeuralRubinAI

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedRubinAITraining:
    """Улучшенная система обучения Rubin AI"""
    
    def __init__(self):
        self.neural_rubin = EnhancedNeuralRubinAI()
        self.training_data = []
        self.validation_data = []
        self.test_data = []
        
        # Генерация обучающих данных
        self._generate_training_data()
        
        logger.info("🎓 Улучшенная система обучения Rubin AI инициализирована")
        logger.info(f"📊 Обучающих примеров: {len(self.training_data)}")
    
    def _generate_training_data(self):
        """Генерация обучающих данных с улучшенной разметкой"""
        
        # Математические задачи
        math_data = [
            {"question": "Реши уравнение x^2 + 5x + 6 = 0", "category": "mathematics"},
            {"question": "Найди производную функции f(x) = x^3 + 2x^2 + 1", "category": "mathematics"},
            {"question": "Вычисли интеграл от 0 до 1 функции x^2", "category": "mathematics"},
            {"question": "Найди напряжение при токе 2 А и сопротивлении 5 Ом", "category": "mathematics"},
            {"question": "Рассчитай кинетическую энергию тела массой 10 кг со скоростью 5 м/с", "category": "mathematics"},
            {"question": "Создай график функции y = sin(x)", "category": "mathematics"},
            {"question": "Найди корни уравнения 2x^2 - 8x + 6 = 0", "category": "mathematics"},
            {"question": "Вычисли площадь треугольника со сторонами 3, 4, 5", "category": "mathematics"},
            {"question": "Найди концентрацию раствора с 0.5 моль вещества в 2 л раствора", "category": "mathematics"},
            {"question": "Построй диаграмму для данных: Яблоки 25, Бананы 30, Апельсины 20", "category": "mathematics"},
            {"question": "Реши систему уравнений", "category": "mathematics"},
            {"question": "Найди предел функции", "category": "mathematics"},
            {"question": "Вычисли определитель матрицы", "category": "mathematics"},
            {"question": "Найди обратную матрицу", "category": "mathematics"},
            {"question": "Реши дифференциальное уравнение", "category": "mathematics"}
        ]
        
        # Программирование
        programming_data = [
            {"question": "Напиши код на Python для сортировки массива", "category": "programming"},
            {"question": "Объясни алгоритм быстрой сортировки", "category": "programming"},
            {"question": "Как работает рекурсия в программировании?", "category": "programming"},
            {"question": "Создай класс на Python для работы с файлами", "category": "programming"},
            {"question": "Напиши функцию для поиска элемента в списке", "category": "programming"},
            {"question": "Объясни принципы объектно-ориентированного программирования", "category": "programming"},
            {"question": "Как отладить ошибку в коде?", "category": "programming"},
            {"question": "Напиши код для работы с базой данных", "category": "programming"},
            {"question": "Объясни паттерн проектирования Singleton", "category": "programming"},
            {"question": "Создай REST API на Flask", "category": "programming"},
            {"question": "Напиши код на JavaScript", "category": "programming"},
            {"question": "Объясни принципы функционального программирования", "category": "programming"},
            {"question": "Создай веб-приложение", "category": "programming"},
            {"question": "Напиши тесты для кода", "category": "programming"},
            {"question": "Объясни принципы SOLID", "category": "programming"}
        ]
        
        # Электротехника
        electrical_data = [
            {"question": "Рассчитай ток в цепи с сопротивлением 10 Ом и напряжением 220 В", "category": "electrical"},
            {"question": "Проанализируй схему последовательного соединения резисторов", "category": "electrical"},
            {"question": "Рассчитай мощность электрической цепи", "category": "electrical"},
            {"question": "Объясни закон Кирхгофа для токов", "category": "electrical"},
            {"question": "Рассчитай сопротивление параллельного соединения", "category": "electrical"},
            {"question": "Проанализируй работу транзистора", "category": "electrical"},
            {"question": "Рассчитай параметры RC-цепи", "category": "electrical"},
            {"question": "Объясни принцип работы конденсатора", "category": "electrical"},
            {"question": "Рассчитай индуктивность катушки", "category": "electrical"},
            {"question": "Проанализируй схему мостового выпрямителя", "category": "electrical"},
            {"question": "Рассчитай параметры трансформатора", "category": "electrical"},
            {"question": "Объясни принцип работы диода", "category": "electrical"},
            {"question": "Проанализируй схему усилителя", "category": "electrical"},
            {"question": "Рассчитай параметры фильтра", "category": "electrical"},
            {"question": "Объясни принцип работы генератора", "category": "electrical"}
        ]
        
        # Контроллеры
        controllers_data = [
            {"question": "Создай программу PLC для управления двигателем", "category": "controllers"},
            {"question": "Напиши логику управления конвейером", "category": "controllers"},
            {"question": "Объясни принцип работы PID-регулятора", "category": "controllers"},
            {"question": "Создай программу для системы сигнализации", "category": "controllers"},
            {"question": "Настрой параметры контроллера для стабилизации температуры", "category": "controllers"},
            {"question": "Объясни протокол Modbus", "category": "controllers"},
            {"question": "Создай программу для управления освещением", "category": "controllers"},
            {"question": "Напиши логику для системы безопасности", "category": "controllers"},
            {"question": "Объясни принцип работы SCADA системы", "category": "controllers"},
            {"question": "Создай программу для управления насосом", "category": "controllers"},
            {"question": "Напиши программу для робота", "category": "controllers"},
            {"question": "Объясни принцип работы датчика", "category": "controllers"},
            {"question": "Создай программу для станка", "category": "controllers"},
            {"question": "Напиши логику для конвейера", "category": "controllers"},
            {"question": "Объясни принцип работы исполнительного механизма", "category": "controllers"}
        ]
        
        # Радиомеханика
        radiomechanics_data = [
            {"question": "Рассчитай параметры антенны для частоты 2.4 ГГц", "category": "radiomechanics"},
            {"question": "Объясни принцип работы радиопередатчика", "category": "radiomechanics"},
            {"question": "Рассчитай дальность радиосвязи", "category": "radiomechanics"},
            {"question": "Объясни модуляцию AM и FM", "category": "radiomechanics"},
            {"question": "Рассчитай параметры дипольной антенны", "category": "radiomechanics"},
            {"question": "Объясни принцип работы радиоприемника", "category": "radiomechanics"},
            {"question": "Рассчитай затухание сигнала в свободном пространстве", "category": "radiomechanics"},
            {"question": "Объясни принцип работы спутниковой связи", "category": "radiomechanics"},
            {"question": "Рассчитай параметры параболической антенны", "category": "radiomechanics"},
            {"question": "Объясни принцип работы радара", "category": "radiomechanics"},
            {"question": "Рассчитай параметры волновода", "category": "radiomechanics"},
            {"question": "Объясни принцип работы микроволновой печи", "category": "radiomechanics"},
            {"question": "Рассчитай параметры резонатора", "category": "radiomechanics"},
            {"question": "Объясни принцип работы лазера", "category": "radiomechanics"},
            {"question": "Рассчитай параметры оптического волокна", "category": "radiomechanics"}
        ]
        
        # Общие вопросы
        general_data = [
            {"question": "Привет! Как дела?", "category": "general"},
            {"question": "Спасибо за помощь", "category": "general"},
            {"question": "Что ты умеешь?", "category": "general"},
            {"question": "Расскажи о себе", "category": "general"},
            {"question": "Как работает искусственный интеллект?", "category": "general"},
            {"question": "Что такое машинное обучение?", "category": "general"},
            {"question": "Объясни принцип работы нейронных сетей", "category": "general"},
            {"question": "Что такое глубокое обучение?", "category": "general"},
            {"question": "Как работает компьютерное зрение?", "category": "general"},
            {"question": "Что такое обработка естественного языка?", "category": "general"},
            {"question": "Здравствуй!", "category": "general"},
            {"question": "До свидания", "category": "general"},
            {"question": "Помоги мне", "category": "general"},
            {"question": "Как дела?", "category": "general"},
            {"question": "Что нового?", "category": "general"}
        ]
        
        # Объединяем все данные
        all_data = math_data + programming_data + electrical_data + controllers_data + radiomechanics_data + general_data
        
        # Перемешиваем данные
        random.shuffle(all_data)
        
        # Разделяем на обучающую, валидационную и тестовую выборки
        total_size = len(all_data)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        
        self.training_data = all_data[:train_size]
        self.validation_data = all_data[train_size:train_size + val_size]
        self.test_data = all_data[train_size + val_size:]
        
        logger.info(f"📊 Данные разделены: обучение {len(self.training_data)}, валидация {len(self.validation_data)}, тест {len(self.test_data)}")
    
    def train_with_rule_based_fallback(self):
        """Обучение с использованием правило-основанной категоризации как fallback"""
        logger.info("🎓 Начинаем обучение с правило-основанной категоризацией")
        
        # Обучаем нейронную сеть
        self.neural_rubin.train_on_data(self.training_data)
        
        # Тестируем правило-основанную категоризацию
        logger.info("🧪 Тестируем правило-основанную категоризацию")
        
        test_questions = [
            "Реши уравнение x^2 + 5x + 6 = 0",
            "Напиши код на Python для сортировки массива", 
            "Рассчитай ток в цепи с сопротивлением 10 Ом и напряжением 220 В",
            "Создай программу PLC для управления двигателем",
            "Рассчитай параметры антенны для частоты 2.4 ГГц",
            "Привет! Как дела?"
        ]
        
        for question in test_questions:
            category, confidence = self.neural_rubin._rule_based_categorization(question)
            logger.info(f"📝 '{question}' → {category} (уверенность: {confidence:.3f})")
        
        logger.info("✅ Обучение завершено")
    
    def evaluate_with_rule_based(self) -> Dict[str, float]:
        """Оценка с использованием правило-основанной категоризации"""
        logger.info("📊 Оценка качества с правило-основанной категоризацией")
        
        correct_predictions = 0
        total_predictions = len(self.test_data)
        
        category_accuracy = {}
        
        for data in self.test_data:
            question = data['question']
            true_category = data['category']
            
            # Используем правило-основанную категоризацию
            predicted_category, confidence = self.neural_rubin._rule_based_categorization(question)
            
            # Проверяем правильность
            if predicted_category == true_category:
                correct_predictions += 1
                if true_category not in category_accuracy:
                    category_accuracy[true_category] = {'correct': 0, 'total': 0}
                category_accuracy[true_category]['correct'] += 1
            
            if true_category not in category_accuracy:
                category_accuracy[true_category] = {'correct': 0, 'total': 0}
            category_accuracy[true_category]['total'] += 1
        
        # Вычисляем метрики
        overall_accuracy = correct_predictions / total_predictions
        
        # Точность по категориям
        category_accuracies = {}
        for category, stats in category_accuracy.items():
            category_accuracies[category] = stats['correct'] / stats['total']
        
        results = {
            'overall_accuracy': overall_accuracy,
            'total_samples': total_predictions,
            'correct_predictions': correct_predictions,
            'category_accuracies': category_accuracies
        }
        
        logger.info(f"📊 Общая точность: {overall_accuracy:.3f}")
        logger.info(f"📊 Правильных предсказаний: {correct_predictions}/{total_predictions}")
        
        for category, accuracy in category_accuracies.items():
            logger.info(f"📊 Точность {category}: {accuracy:.3f}")
        
        return results
    
    def test_improved_categorization(self) -> List[Dict[str, Any]]:
        """Тестирование улучшенной категоризации"""
        test_questions = [
            "Реши уравнение x^2 + 5x + 6 = 0",
            "Напиши код на Python для сортировки массива",
            "Рассчитай ток в цепи с сопротивлением 10 Ом и напряжением 220 В",
            "Создай программу PLC для управления двигателем",
            "Рассчитай параметры антенны для частоты 2.4 ГГц",
            "Привет! Как дела?",
            "Объясни алгоритм быстрой сортировки",
            "Проанализируй схему последовательного соединения резисторов",
            "Напиши логику управления конвейером",
            "Объясни модуляцию AM и FM"
        ]
        
        results = []
        
        for question in test_questions:
            # Используем правило-основанную категоризацию
            category, confidence = self.neural_rubin._rule_based_categorization(question)
            
            # Генерируем ответ
            response_data = self.neural_rubin.category_handlers[category](question)
            
            results.append({
                'question': question,
                'category': category,
                'confidence': confidence,
                'response': response_data['response']
            })
        
        return results
    
    def generate_improved_report(self) -> str:
        """Генерация улучшенного отчета"""
        evaluation = self.evaluate_with_rule_based()
        
        report = f"""
# 📊 УЛУЧШЕННЫЙ ОТЧЕТ ОБ ОБУЧЕНИИ RUBIN AI

## 📈 Общая статистика
- **Общая точность**: {evaluation['overall_accuracy']:.3f} ({evaluation['correct_predictions']}/{evaluation['total_samples']})
- **Обучающих примеров**: {len(self.training_data)}
- **Валидационных примеров**: {len(self.validation_data)}
- **Тестовых примеров**: {len(self.test_data)}

## 🎯 Точность по категориям
"""
        
        for category, accuracy in evaluation['category_accuracies'].items():
            report += f"- **{category}**: {accuracy:.3f}\n"
        
        report += f"""
## 🧠 Состояние системы
- **ML библиотеки доступны**: {'✅' if self.neural_rubin.get_knowledge_summary()['neural_network_available'] else '❌'}
- **База знаний**: {self.neural_rubin.get_knowledge_summary()['total_categories']} категорий
- **Методы обработки**: {self.neural_rubin.get_knowledge_summary()['available_methods']}
- **Шаблоны ответов**: {self.neural_rubin.get_knowledge_summary()['response_templates']}

## 🔧 Используемые методы
- **Правило-основанная категоризация**: ✅ (fallback)
- **Нейронная сеть**: {'✅' if self.neural_rubin.get_knowledge_summary()['neural_network_available'] else '❌'}
- **Обработчики категорий**: ✅ (6 категорий)

## 📝 Рекомендации
"""
        
        if evaluation['overall_accuracy'] < 0.7:
            report += "- ⚠️ Точность ниже 70%, рекомендуется улучшение алгоритма категоризации\n"
        
        if evaluation['overall_accuracy'] >= 0.8:
            report += "- ✅ Отличная точность! Система готова к использованию\n"
        
        report += "- 🔄 Рекомендуется периодическое обновление базы знаний\n"
        report += "- 📊 Мониторинг качества ответов в реальном времени\n"
        report += "- 🧠 Интеграция с внешними ML моделями для улучшения категоризации\n"
        
        return report

def main():
    """Основная функция для улучшенного обучения и тестирования"""
    print("🎓 УЛУЧШЕННАЯ СИСТЕМА ОБУЧЕНИЯ RUBIN AI")
    print("=" * 50)
    
    # Создаем улучшенную систему обучения
    training_system = ImprovedRubinAITraining()
    
    # Обучаем с правило-основанной категоризацией
    training_system.train_with_rule_based_fallback()
    
    # Оцениваем модель
    print("\n📊 ОЦЕНКА МОДЕЛИ:")
    print("-" * 30)
    evaluation_results = training_system.evaluate_with_rule_based()
    
    # Тестируем на конкретных вопросах
    print("\n🧪 ТЕСТИРОВАНИЕ УЛУЧШЕННОЙ КАТЕГОРИЗАЦИИ:")
    print("-" * 45)
    test_results = training_system.test_improved_categorization()
    
    for result in test_results:
        print(f"\n📝 Вопрос: {result['question']}")
        print(f"🎯 Категория: {result['category']}")
        print(f"📊 Уверенность: {result['confidence']:.3f}")
        print(f"💡 Ответ: {result['response'][:100]}...")
    
    # Генерируем улучшенный отчет
    report = training_system.generate_improved_report()
    print(f"\n📋 УЛУЧШЕННЫЙ ОТЧЕТ:")
    print(report)
    
    # Сохраняем отчет
    with open("improved_training_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("\n✅ Улучшенное обучение и тестирование завершены!")
    print("📁 Отчет сохранен в improved_training_report.md")

if __name__ == "__main__":
    main()










