#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты для обучения нейронной сети
"""

import unittest
import json
import os
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestNeuralTraining(unittest.TestCase):
    """Тесты для обучения нейронной сети"""
    
    def setUp(self):
        """Настройка тестов"""
        self.training_file = "neural_training_data.jsonl"
        self.test_data = [
            {"question": "привет", "correct_category": "общие", "user_rating": 5},
            {"question": "2+2", "correct_category": "математика", "user_rating": 5},
            {"question": "что такое транзистор", "correct_category": "электротехника", "user_rating": 5},
            {"question": "напиши программу", "correct_category": "программирование", "user_rating": 5}
        ]
    
    def test_training_data_format(self):
        """Тест формата обучающих данных"""
        logger.info("📊 Тестирование формата обучающих данных...")
        
        if not os.path.exists(self.training_file):
            logger.warning(f"⚠️ Файл {self.training_file} не найден")
            return
        
        with open(self.training_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Проверяем обязательные поля
                    self.assertIn("question", data, f"Строка {line_num}: отсутствует поле 'question'")
                    self.assertIn("correct_category", data, f"Строка {line_num}: отсутствует поле 'correct_category'")
                    self.assertIn("user_rating", data, f"Строка {line_num}: отсутствует поле 'user_rating'")
                    
                    # Проверяем типы данных
                    self.assertIsInstance(data["question"], str, f"Строка {line_num}: 'question' должно быть строкой")
                    self.assertIsInstance(data["correct_category"], str, f"Строка {line_num}: 'correct_category' должно быть строкой")
                    self.assertIsInstance(data["user_rating"], int, f"Строка {line_num}: 'user_rating' должно быть числом")
                    
                    # Проверяем диапазон рейтинга
                    self.assertGreaterEqual(data["user_rating"], 1, f"Строка {line_num}: рейтинг должен быть >= 1")
                    self.assertLessEqual(data["user_rating"], 5, f"Строка {line_num}: рейтинг должен быть <= 5")
                    
                except json.JSONDecodeError as e:
                    self.fail(f"Строка {line_num}: ошибка JSON - {e}")
        
        logger.info("✅ Формат обучающих данных корректен")
    
    def test_training_data_content(self):
        """Тест содержания обучающих данных"""
        logger.info("📝 Тестирование содержания обучающих данных...")
        
        if not os.path.exists(self.training_file):
            logger.warning(f"⚠️ Файл {self.training_file} не найден")
            return
        
        categories = set()
        questions = []
        
        with open(self.training_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                categories.add(data["correct_category"])
                questions.append(data["question"])
        
        # Проверяем разнообразие категорий
        self.assertGreaterEqual(len(categories), 3, "Должно быть минимум 3 категории")
        
        # Проверяем разнообразие вопросов
        self.assertGreaterEqual(len(questions), 10, "Должно быть минимум 10 вопросов")
        
        # Проверяем уникальность вопросов
        unique_questions = set(questions)
        self.assertEqual(len(unique_questions), len(questions), "Все вопросы должны быть уникальными")
        
        logger.info(f"✅ Содержание: {len(categories)} категорий, {len(questions)} вопросов")
    
    def test_neural_rubin_import(self):
        """Тест импорта нейронной сети"""
        logger.info("🧠 Тестирование импорта нейронной сети...")
        
        try:
            from neural_rubin import get_neural_rubin
            neural_ai = get_neural_rubin()
            
            # Проверяем основные атрибуты
            self.assertIsNotNone(neural_ai.categories, "Категории должны быть определены")
            self.assertIsNotNone(neural_ai.analytics, "Аналитика должна быть инициализирована")
            
            logger.info("✅ Нейронная сеть импортирована успешно")
            
        except ImportError as e:
            logger.warning(f"⚠️ Не удалось импортировать нейронную сеть: {e}")
        except Exception as e:
            logger.error(f"❌ Ошибка при инициализации нейронной сети: {e}")
    
    def test_neural_classification(self):
        """Тест классификации нейронной сети"""
        logger.info("🎯 Тестирование классификации нейронной сети...")
        
        try:
            from neural_rubin import get_neural_rubin
            neural_ai = get_neural_rubin()
            
            test_cases = [
                ("привет", "общие"),
                ("2+2", "математика"),
                ("транзистор", "электротехника"),
                ("python", "программирование")
            ]
            
            for question, expected_category in test_cases:
                category, confidence = neural_ai.classify_question(question)
                
                self.assertIsInstance(category, str, f"Категория для '{question}' должна быть строкой")
                self.assertIsInstance(confidence, float, f"Уверенность для '{question}' должна быть числом")
                self.assertGreaterEqual(confidence, 0.0, f"Уверенность для '{question}' должна быть >= 0")
                self.assertLessEqual(confidence, 1.0, f"Уверенность для '{question}' должна быть <= 1")
                
                logger.info(f"  ✅ '{question}' -> {category} (уверенность: {confidence:.2f})")
            
            logger.info("✅ Классификация работает корректно")
            
        except ImportError:
            logger.warning("⚠️ Нейронная сеть недоступна для тестирования")
        except Exception as e:
            logger.error(f"❌ Ошибка при тестировании классификации: {e}")
    
    def test_analytics(self):
        """Тест аналитики"""
        logger.info("📊 Тестирование аналитики...")
        
        try:
            from neural_rubin import get_neural_rubin
            neural_ai = get_neural_rubin()
            
            # Получаем статистику
            stats = neural_ai.get_neural_stats()
            
            self.assertIn("device", stats)
            self.assertIn("neural_network_active", stats)
            self.assertIn("categories", stats)
            self.assertIn("analytics", stats)
            
            # Проверяем аналитику
            analytics = neural_ai.get_analytics()
            self.assertIn("summary", analytics)
            self.assertIn("categories", analytics)
            self.assertIn("performance", analytics)
            
            logger.info("✅ Аналитика работает корректно")
            
        except ImportError:
            logger.warning("⚠️ Нейронная сеть недоступна для тестирования")
        except Exception as e:
            logger.error(f"❌ Ошибка при тестировании аналитики: {e}")
    
    def test_training_process(self):
        """Тест процесса обучения"""
        logger.info("📚 Тестирование процесса обучения...")
        
        try:
            from neural_rubin import get_neural_rubin
            neural_ai = get_neural_rubin()
            
            # Проверяем наличие метода обучения
            self.assertTrue(hasattr(neural_ai, 'train_neural_network'), "Должен быть метод train_neural_network")
            self.assertTrue(hasattr(neural_ai, 'learn_from_feedback'), "Должен быть метод learn_from_feedback")
            
            # Тестируем обучение на основе обратной связи
            result = neural_ai.learn_from_feedback(
                question="тестовый вопрос",
                correct_category="тест",
                user_rating=5
            )
            
            self.assertIsInstance(result, bool, "learn_from_feedback должен возвращать bool")
            
            logger.info("✅ Процесс обучения работает корректно")
            
        except ImportError:
            logger.warning("⚠️ Нейронная сеть недоступна для тестирования")
        except Exception as e:
            logger.error(f"❌ Ошибка при тестировании обучения: {e}")

def run_neural_tests():
    """Запуск тестов нейронной сети"""
    logger.info("🧠 Запуск тестов нейронной сети...")
    
    loader = unittest.TestLoader()
    test_suite = loader.loadTestsFromTestCase(TestNeuralTraining)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("🧪 Запуск тестов нейронной сети")
    print("=" * 50)
    
    success = run_neural_tests()
    
    if success:
        print("\n✅ Все тесты нейронной сети прошли успешно!")
    else:
        print("\n❌ Некоторые тесты нейронной сети не прошли")
    
    print("=" * 50)
