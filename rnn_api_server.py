#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API сервер для RNN (Recurrent Neural Networks) Rubin AI
Предоставляет endpoints для работы с рекуррентными нейронными сетями
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import logging
from datetime import datetime
import numpy as np

from rubin_rnn import RubinRNNManager, RubinRNNClassifier, RubinRNNRegressor

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Инициализация RNN Manager
rnn_manager = RubinRNNManager()

@app.route('/api/rnn/health', methods=['GET'])
def health_check():
    """Проверка состояния RNN сервера"""
    return jsonify({
        "status": "Rubin AI RNN Server ONLINE",
        "timestamp": datetime.now().isoformat(),
        "device": str(rnn_manager.device),
        "models_count": len(rnn_manager.models),
        "available_rnn_types": ["rnn", "lstm", "gru"],
        "available_tasks": ["classification", "regression"]
    }), 200

@app.route('/api/rnn/create_classifier', methods=['POST'])
def create_rnn_classifier():
    """Создание RNN классификатора"""
    try:
        data = request.get_json() or {}
        input_size = data.get('input_size', 10)
        hidden_size = data.get('hidden_size', 64)
        num_classes = data.get('num_classes', 5)
        rnn_type = data.get('rnn_type', 'lstm')
        num_layers = data.get('num_layers', 1)
        dropout = data.get('dropout', 0.0)
        bidirectional = data.get('bidirectional', False)
        model_name = data.get('model_name', f'rnn_classifier_{len(rnn_manager.models)}')
        
        logger.info(f"🎯 Создание RNN классификатора: {model_name}")
        
        model = rnn_manager.create_rnn_classifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            rnn_type=rnn_type,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            model_name=model_name
        )
        
        if model:
            model_info = rnn_manager.get_model_info(model_name)
            response_data = {
                "success": True,
                "model_name": model_name,
                "model_info": model_info,
                "message": f"RNN классификатор '{model_name}' создан успешно",
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(response_data), 201
        else:
            return jsonify({
                "success": False,
                "error": "Не удалось создать RNN классификатор"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Ошибка создания RNN классификатора: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка создания RNN классификатора: {str(e)}"
        }), 500

@app.route('/api/rnn/create_regressor', methods=['POST'])
def create_rnn_regressor():
    """Создание RNN регрессора"""
    try:
        data = request.get_json() or {}
        input_size = data.get('input_size', 10)
        hidden_size = data.get('hidden_size', 64)
        output_size = data.get('output_size', 1)
        rnn_type = data.get('rnn_type', 'lstm')
        num_layers = data.get('num_layers', 1)
        dropout = data.get('dropout', 0.0)
        bidirectional = data.get('bidirectional', False)
        model_name = data.get('model_name', f'rnn_regressor_{len(rnn_manager.models)}')
        
        logger.info(f"🎯 Создание RNN регрессора: {model_name}")
        
        model = rnn_manager.create_rnn_regressor(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            rnn_type=rnn_type,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            model_name=model_name
        )
        
        if model:
            model_info = rnn_manager.get_model_info(model_name)
            response_data = {
                "success": True,
                "model_name": model_name,
                "model_info": model_info,
                "message": f"RNN регрессор '{model_name}' создан успешно",
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(response_data), 201
        else:
            return jsonify({
                "success": False,
                "error": "Не удалось создать RNN регрессор"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Ошибка создания RNN регрессора: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка создания RNN регрессора: {str(e)}"
        }), 500

@app.route('/api/rnn/train', methods=['POST'])
def train_model():
    """Обучение модели"""
    try:
        data = request.get_json() or {}
        model_name = data.get('model_name')
        epochs = data.get('epochs', 10)
        learning_rate = data.get('learning_rate', 0.001)
        task_type = data.get('task_type', 'classification')
        
        if not model_name:
            return jsonify({
                "success": False,
                "error": "Не указано имя модели"
            }), 400
        
        logger.info(f"🚀 Обучение модели: {model_name}")
        
        start_time = time.time()
        success = rnn_manager.train_model(model_name, None, epochs, learning_rate, task_type)
        end_time = time.time()
        
        training_time = end_time - start_time
        
        if success:
            model_info = rnn_manager.get_model_info(model_name)
            response_data = {
                "success": True,
                "model_name": model_name,
                "task_type": task_type,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "training_time_seconds": f"{training_time:.2f}",
                "training_history": model_info.get("training_history", []),
                "message": f"Модель '{model_name}' обучена успешно",
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(response_data), 200
        else:
            return jsonify({
                "success": False,
                "error": f"Не удалось обучить модель '{model_name}'"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Ошибка обучения модели: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка обучения модели: {str(e)}"
        }), 500

@app.route('/api/rnn/predict', methods=['POST'])
def predict():
    """Предсказание с помощью модели"""
    try:
        data = request.get_json() or {}
        model_name = data.get('model_name')
        input_data = data.get('input_data')
        
        if not model_name:
            return jsonify({
                "success": False,
                "error": "Не указано имя модели"
            }), 400
        
        if not input_data:
            return jsonify({
                "success": False,
                "error": "Не предоставлены входные данные"
            }), 400
        
        logger.info(f"🔮 Предсказание модели: {model_name}")
        
        # Преобразуем данные в numpy array
        import numpy as np
        if isinstance(input_data, list):
            input_array = np.array(input_data)
        else:
            input_array = input_data
        
        # Добавляем batch dimension если нужно
        if input_array.ndim == 2:
            input_array = input_array.reshape(1, *input_array.shape)
        
        # Преобразуем в tensor
        import torch
        input_tensor = torch.FloatTensor(input_array)
        
        start_time = time.time()
        predictions = rnn_manager.predict(model_name, input_tensor)
        end_time = time.time()
        
        prediction_time = end_time - start_time
        
        if predictions is not None:
            # Определяем тип модели
            model_info = rnn_manager.get_model_info(model_name)
            is_classifier = "Classifier" in model_info["type"]
            
            if is_classifier:
                # Для классификатора
                predicted_class = np.argmax(predictions, axis=1)
                confidence = np.max(predictions, axis=1)
                
                response_data = {
                    "success": True,
                    "model_name": model_name,
                    "model_type": "classifier",
                    "predictions": predictions.tolist(),
                    "predicted_classes": predicted_class.tolist(),
                    "confidence_scores": confidence.tolist(),
                    "prediction_time_seconds": f"{prediction_time:.4f}",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Для регрессора
                response_data = {
                    "success": True,
                    "model_name": model_name,
                    "model_type": "regressor",
                    "predictions": predictions.tolist(),
                    "prediction_time_seconds": f"{prediction_time:.4f}",
                    "timestamp": datetime.now().isoformat()
                }
            
            return jsonify(response_data), 200
        else:
            return jsonify({
                "success": False,
                "error": f"Не удалось получить предсказание от модели '{model_name}'"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Ошибка предсказания: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка предсказания: {str(e)}"
        }), 500

@app.route('/api/rnn/models', methods=['GET'])
def get_models():
    """Получение списка всех моделей"""
    try:
        logger.info("📋 Получение списка моделей")
        
        all_models_info = rnn_manager.get_all_models_info()
        
        response_data = {
            "success": True,
            "models": all_models_info,
            "total_models": len(all_models_info),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения списка моделей: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения списка моделей: {str(e)}"
        }), 500

@app.route('/api/rnn/model/<model_name>', methods=['GET'])
def get_model_info(model_name):
    """Получение информации о конкретной модели"""
    try:
        logger.info(f"📊 Получение информации о модели: {model_name}")
        
        model_info = rnn_manager.get_model_info(model_name)
        
        if model_info:
            response_data = {
                "success": True,
                "model_info": model_info,
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(response_data), 200
        else:
            return jsonify({
                "success": False,
                "error": f"Модель '{model_name}' не найдена"
            }), 404
            
    except Exception as e:
        logger.error(f"❌ Ошибка получения информации о модели: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения информации о модели: {str(e)}"
        }), 500

@app.route('/api/rnn/save/<model_name>', methods=['POST'])
def save_model(model_name):
    """Сохранение модели"""
    try:
        data = request.get_json() or {}
        filepath = data.get('filepath', f'models/{model_name}.pth')
        
        logger.info(f"💾 Сохранение модели: {model_name}")
        
        success = rnn_manager.save_model(model_name, filepath)
        
        if success:
            response_data = {
                "success": True,
                "model_name": model_name,
                "filepath": filepath,
                "message": f"Модель '{model_name}' сохранена в {filepath}",
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(response_data), 200
        else:
            return jsonify({
                "success": False,
                "error": f"Не удалось сохранить модель '{model_name}'"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения модели: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка сохранения модели: {str(e)}"
        }), 500

@app.route('/api/rnn/load', methods=['POST'])
def load_model():
    """Загрузка модели"""
    try:
        data = request.get_json() or {}
        model_name = data.get('model_name')
        filepath = data.get('filepath')
        input_size = data.get('input_size')
        hidden_size = data.get('hidden_size')
        num_classes = data.get('num_classes')
        output_size = data.get('output_size')
        rnn_type = data.get('rnn_type', 'lstm')
        num_layers = data.get('num_layers', 1)
        dropout = data.get('dropout', 0.0)
        bidirectional = data.get('bidirectional', False)
        
        if not model_name or not filepath:
            return jsonify({
                "success": False,
                "error": "Не указаны имя модели или путь к файлу"
            }), 400
        
        logger.info(f"📂 Загрузка модели: {model_name}")
        
        success = rnn_manager.load_model(
            model_name, filepath, input_size, hidden_size, 
            num_classes, output_size, rnn_type, num_layers, dropout, bidirectional
        )
        
        if success:
            model_info = rnn_manager.get_model_info(model_name)
            response_data = {
                "success": True,
                "model_name": model_name,
                "filepath": filepath,
                "model_info": model_info,
                "message": f"Модель '{model_name}' загружена из {filepath}",
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(response_data), 200
        else:
            return jsonify({
                "success": False,
                "error": f"Не удалось загрузить модель '{model_name}'"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка загрузки модели: {str(e)}"
        }), 500

@app.route('/api/rnn/delete/<model_name>', methods=['DELETE'])
def delete_model(model_name):
    """Удаление модели"""
    try:
        logger.info(f"🗑️ Удаление модели: {model_name}")
        
        if model_name in rnn_manager.models:
            del rnn_manager.models[model_name]
            if model_name in rnn_manager.training_history:
                del rnn_manager.training_history[model_name]
            
            response_data = {
                "success": True,
                "model_name": model_name,
                "message": f"Модель '{model_name}' удалена",
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(response_data), 200
        else:
            return jsonify({
                "success": False,
                "error": f"Модель '{model_name}' не найдена"
            }), 404
            
    except Exception as e:
        logger.error(f"❌ Ошибка удаления модели: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка удаления модели: {str(e)}"
        }), 500

@app.route('/api/rnn/rnn_types', methods=['GET'])
def get_rnn_types():
    """Получение информации о типах RNN"""
    try:
        rnn_types = {
            "rnn": {
                "name": "Basic RNN",
                "description": "Базовая рекуррентная нейронная сеть",
                "advantages": ["Простота", "Быстрота обучения", "Низкие вычислительные требования"],
                "disadvantages": ["Проблема исчезающего градиента", "Ограниченная память"],
                "use_cases": ["Простые последовательности", "Короткие временные ряды"],
                "parameters": "~10K-100K"
            },
            "lstm": {
                "name": "Long Short-Term Memory",
                "description": "LSTM с ячейками памяти для долгосрочных зависимостей",
                "advantages": ["Долгосрочная память", "Решение проблемы исчезающего градиента", "Высокая точность"],
                "disadvantages": ["Больше параметров", "Медленнее обучения"],
                "use_cases": ["Долгосрочные зависимости", "Анализ текста", "Временные ряды"],
                "parameters": "~50K-500K"
            },
            "gru": {
                "name": "Gated Recurrent Unit",
                "description": "GRU с упрощенной архитектурой по сравнению с LSTM",
                "advantages": ["Баланс точности и скорости", "Меньше параметров чем LSTM", "Быстрое обучение"],
                "disadvantages": ["Меньше памяти чем LSTM", "Сложнее чем RNN"],
                "use_cases": ["Средние последовательности", "Реальное время", "Компромисс точности и скорости"],
                "parameters": "~30K-300K"
            }
        }
        
        response_data = {
            "success": True,
            "rnn_types": rnn_types,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения типов RNN: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения типов RNN: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = 8097  # Новый порт для RNN сервера
    logger.info(f"🧠 Запуск Rubin AI RNN сервера на порту {port}...")
    app.run(port=port, debug=True, use_reloader=False)










