#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Network Training System для Rubin AI
Система обучения нейронной сети на реальных данных
"""

import json
import random
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
from neural_rubin_v2 import EnhancedNeuralRubinAI

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RubinAITrainingSystem:
    """Система обучения Rubin AI"""
    
    def __init__(self):
        self.neural_rubin = EnhancedNeuralRubinAI()
        self.training_data = []
        self.validation_data = []
        self.test_data = []
        
        # Генерация обучающих данных
        self._generate_training_data()
        
        logger.info("🎓 Система обучения Rubin AI инициализирована")
        logger.info(f"📊 Обучающих примеров: {len(self.training_data)}")
        logger.info(f"📊 Валидационных примеров: {len(self.validation_data)}")
        logger.info(f"📊 Тестовых примеров: {len(self.test_data)}")
    
    def _generate_training_data(self):
        """Генерация обучающих данных"""
        
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
            {"question": "Построй диаграмму для данных: Яблоки 25, Бананы 30, Апельсины 20", "category": "mathematics"}
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
            {"question": "Создай REST API на Flask", "category": "programming"}
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
            {"question": "Проанализируй схему мостового выпрямителя", "category": "electrical"}
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
            {"question": "Создай программу для управления насосом", "category": "controllers"}
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
            {"question": "Объясни принцип работы радара", "category": "radiomechanics"}
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
            {"question": "Что такое обработка естественного языка?", "category": "general"}
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
    
    def train_neural_network(self, epochs: int = 50):
        """Обучение нейронной сети"""
        logger.info(f"🎓 Начинаем обучение нейронной сети на {epochs} эпох")
        
        # Обучаем нейронную сеть
        self.neural_rubin.train_on_data(self.training_data)
        
        logger.info("✅ Обучение завершено")
    
    def evaluate_model(self) -> Dict[str, float]:
        """Оценка качества модели"""
        logger.info("📊 Оценка качества модели")
        
        correct_predictions = 0
        total_predictions = len(self.test_data)
        
        category_accuracy = {}
        
        for data in self.test_data:
            question = data['question']
            true_category = data['category']
            
            # Получаем предсказание
            response = self.neural_rubin.generate_response(question)
            predicted_category = response['category']
            
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
    
    def test_specific_questions(self) -> List[Dict[str, Any]]:
        """Тестирование на конкретных вопросах"""
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
            response = self.neural_rubin.generate_response(question)
            results.append({
                'question': question,
                'category': response['category'],
                'confidence': response['confidence'],
                'response': response['response']
            })
        
        return results
    
    def save_training_results(self, filename: str = "training_results.json"):
        """Сохранение результатов обучения"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'training_data_size': len(self.training_data),
            'validation_data_size': len(self.validation_data),
            'test_data_size': len(self.test_data),
            'evaluation_results': self.evaluate_model(),
            'test_questions_results': self.test_specific_questions()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Результаты обучения сохранены в {filename}")
    
    def generate_training_report(self) -> str:
        """Генерация отчета об обучении"""
        evaluation = self.evaluate_model()
        
        report = f"""
# 📊 ОТЧЕТ ОБ ОБУЧЕНИИ RUBIN AI

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
## 🧠 Состояние нейронной сети
- **ML библиотеки доступны**: {'✅' if self.neural_rubin.get_knowledge_summary()['neural_network_available'] else '❌'}
- **База знаний**: {self.neural_rubin.get_knowledge_summary()['total_categories']} категорий
- **Методы обработки**: {self.neural_rubin.get_knowledge_summary()['available_methods']}
- **Шаблоны ответов**: {self.neural_rubin.get_knowledge_summary()['response_templates']}

## 📝 Рекомендации
"""
        
        if evaluation['overall_accuracy'] < 0.7:
            report += "- ⚠️ Точность ниже 70%, рекомендуется дополнительное обучение\n"
        
        if evaluation['overall_accuracy'] >= 0.8:
            report += "- ✅ Отличная точность! Модель готова к использованию\n"
        
        report += "- 🔄 Рекомендуется периодическое переобучение на новых данных\n"
        report += "- 📊 Мониторинг качества ответов в реальном времени\n"
        
        return report

def main():
    """Основная функция для обучения и тестирования"""
    print("🎓 СИСТЕМА ОБУЧЕНИЯ RUBIN AI")
    print("=" * 50)
    
    # Создаем систему обучения
    training_system = RubinAITrainingSystem()
    
    # Обучаем нейронную сеть
    training_system.train_neural_network(epochs=20)
    
    # Оцениваем модель
    print("\n📊 ОЦЕНКА МОДЕЛИ:")
    print("-" * 30)
    evaluation_results = training_system.evaluate_model()
    
    # Тестируем на конкретных вопросах
    print("\n🧪 ТЕСТИРОВАНИЕ НА КОНКРЕТНЫХ ВОПРОСАХ:")
    print("-" * 40)
    test_results = training_system.test_specific_questions()
    
    for result in test_results:
        print(f"\n📝 Вопрос: {result['question']}")
        print(f"🎯 Категория: {result['category']}")
        print(f"📊 Уверенность: {result['confidence']:.3f}")
        print(f"💡 Ответ: {result['response'][:100]}...")
    
    # Сохраняем результаты
    training_system.save_training_results()
    
    # Генерируем отчет
    report = training_system.generate_training_report()
    print(f"\n📋 ОТЧЕТ ОБ ОБУЧЕНИИ:")
    print(report)
    
    # Сохраняем отчет
    with open("training_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("\n✅ Обучение и тестирование завершены!")
    print("📁 Результаты сохранены в training_results.json и training_report.md")

if __name__ == "__main__":
    main()





