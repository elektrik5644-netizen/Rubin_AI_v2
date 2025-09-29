#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API сервер для CNN (Convolutional Neural Networks) Rubin AI
Предоставляет endpoints для работы с сверточными нейронными сетями
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import logging
from datetime import datetime
import numpy as np

from rubin_cnn import RubinCNNManager, RubinCNN2D, RubinCNN1D

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Инициализация CNN Manager
cnn_manager = RubinCNNManager()

@app.route('/api/cnn/health', methods=['GET'])
def health_check():
    """Проверка состояния CNN сервера"""
    return jsonify({
        "status": "Rubin AI CNN Server ONLINE",
        "timestamp": datetime.now().isoformat(),
        "device": str(cnn_manager.device),
        "models_count": len(cnn_manager.models),
        "available_architectures": {
            "2d_cnn": ["standard", "deep", "lightweight"],
            "1d_cnn": ["standard", "deep", "temporal"]
        }
    }), 200

@app.route('/api/cnn/create_2d', methods=['POST'])
def create_cnn_2d():
    """Создание 2D CNN"""
    try:
        data = request.get_json() or {}
        input_channels = data.get('input_channels', 3)
        num_classes = data.get('num_classes', 10)
        architecture = data.get('architecture', 'standard')
        model_name = data.get('model_name', f'cnn_2d_{len(cnn_manager.models)}')
        
        logger.info(f"🎯 Создание 2D CNN: {model_name}")
        
        model = cnn_manager.create_cnn_2d(
            input_channels=input_channels,
            num_classes=num_classes,
            architecture=architecture,
            model_name=model_name
        )
        
        if model:
            model_info = cnn_manager.get_model_info(model_name)
            response_data = {
                "success": True,
                "model_name": model_name,
                "model_info": model_info,
                "message": f"2D CNN '{model_name}' создана успешно",
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(response_data), 201
        else:
            return jsonify({
                "success": False,
                "error": "Не удалось создать 2D CNN"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Ошибка создания 2D CNN: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка создания 2D CNN: {str(e)}"
        }), 500

@app.route('/api/cnn/create_1d', methods=['POST'])
def create_cnn_1d():
    """Создание 1D CNN"""
    try:
        data = request.get_json() or {}
        input_length = data.get('input_length', 100)
        num_classes = data.get('num_classes', 10)
        architecture = data.get('architecture', 'standard')
        model_name = data.get('model_name', f'cnn_1d_{len(cnn_manager.models)}')
        
        logger.info(f"🎯 Создание 1D CNN: {model_name}")
        
        model = cnn_manager.create_cnn_1d(
            input_length=input_length,
            num_classes=num_classes,
            architecture=architecture,
            model_name=model_name
        )
        
        if model:
            model_info = cnn_manager.get_model_info(model_name)
            response_data = {
                "success": True,
                "model_name": model_name,
                "model_info": model_info,
                "message": f"1D CNN '{model_name}' создана успешно",
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(response_data), 201
        else:
            return jsonify({
                "success": False,
                "error": "Не удалось создать 1D CNN"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Ошибка создания 1D CNN: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка создания 1D CNN: {str(e)}"
        }), 500

@app.route('/api/cnn/train', methods=['POST'])
def train_model():
    """Обучение модели"""
    try:
        data = request.get_json() or {}
        model_name = data.get('model_name')
        epochs = data.get('epochs', 10)
        learning_rate = data.get('learning_rate', 0.001)
        
        if not model_name:
            return jsonify({
                "success": False,
                "error": "Не указано имя модели"
            }), 400
        
        logger.info(f"🚀 Обучение модели: {model_name}")
        
        start_time = time.time()
        success = cnn_manager.train_model(model_name, None, epochs, learning_rate)
        end_time = time.time()
        
        training_time = end_time - start_time
        
        if success:
            model_info = cnn_manager.get_model_info(model_name)
            response_data = {
                "success": True,
                "model_name": model_name,
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

@app.route('/api/cnn/predict', methods=['POST'])
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
        elif input_array.ndim == 3:
            input_array = input_array.reshape(1, *input_array.shape)
        
        # Преобразуем в tensor
        import torch
        input_tensor = torch.FloatTensor(input_array)
        
        start_time = time.time()
        predictions = cnn_manager.predict(model_name, input_tensor)
        end_time = time.time()
        
        prediction_time = end_time - start_time
        
        if predictions is not None:
            # Находим класс с максимальной вероятностью
            predicted_class = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)
            
            response_data = {
                "success": True,
                "model_name": model_name,
                "predictions": predictions.tolist(),
                "predicted_classes": predicted_class.tolist(),
                "confidence_scores": confidence.tolist(),
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

@app.route('/api/cnn/models', methods=['GET'])
def get_models():
    """Получение списка всех моделей"""
    try:
        logger.info("📋 Получение списка моделей")
        
        all_models_info = cnn_manager.get_all_models_info()
        
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

@app.route('/api/cnn/model/<model_name>', methods=['GET'])
def get_model_info(model_name):
    """Получение информации о конкретной модели"""
    try:
        logger.info(f"📊 Получение информации о модели: {model_name}")
        
        model_info = cnn_manager.get_model_info(model_name)
        
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

@app.route('/api/cnn/save/<model_name>', methods=['POST'])
def save_model(model_name):
    """Сохранение модели"""
    try:
        data = request.get_json() or {}
        filepath = data.get('filepath', f'models/{model_name}.pth')
        
        logger.info(f"💾 Сохранение модели: {model_name}")
        
        success = cnn_manager.save_model(model_name, filepath)
        
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

@app.route('/api/cnn/load', methods=['POST'])
def load_model():
    """Загрузка модели"""
    try:
        data = request.get_json() or {}
        model_name = data.get('model_name')
        filepath = data.get('filepath')
        input_channels = data.get('input_channels')
        input_length = data.get('input_length')
        num_classes = data.get('num_classes', 10)
        architecture = data.get('architecture', 'standard')
        
        if not model_name or not filepath:
            return jsonify({
                "success": False,
                "error": "Не указаны имя модели или путь к файлу"
            }), 400
        
        logger.info(f"📂 Загрузка модели: {model_name}")
        
        success = cnn_manager.load_model(
            model_name, filepath, input_channels, 
            input_length, num_classes, architecture
        )
        
        if success:
            model_info = cnn_manager.get_model_info(model_name)
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

@app.route('/api/cnn/delete/<model_name>', methods=['DELETE'])
def delete_model(model_name):
    """Удаление модели"""
    try:
        logger.info(f"🗑️ Удаление модели: {model_name}")
        
        if model_name in cnn_manager.models:
            del cnn_manager.models[model_name]
            if model_name in cnn_manager.training_history:
                del cnn_manager.training_history[model_name]
            
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

@app.route('/api/cnn/architectures', methods=['GET'])
def get_architectures():
    """Получение доступных архитектур"""
    try:
        architectures = {
            "2d_cnn": {
                "standard": {
                    "description": "Стандартная архитектура для классификации изображений",
                    "layers": ["Conv2d(3→32)", "Conv2d(32→64)", "Conv2d(64→128)"],
                    "parameters": "~100K",
                    "use_cases": ["Классификация изображений", "Распознавание объектов"]
                },
                "deep": {
                    "description": "Глубокая архитектура для сложных задач",
                    "layers": ["Conv2d(3→64)", "Conv2d(64→128)", "Conv2d(128→256)", "Conv2d(256→512)"],
                    "parameters": "~1M",
                    "use_cases": ["Сложная классификация", "Детекция объектов"]
                },
                "lightweight": {
                    "description": "Легкая архитектура для мобильных устройств",
                    "layers": ["Conv2d(3→16)", "Conv2d(16→32)", "Conv2d(32→64)"],
                    "parameters": "~50K",
                    "use_cases": ["Мобильные приложения", "Реальное время"]
                }
            },
            "1d_cnn": {
                "standard": {
                    "description": "Стандартная архитектура для последовательностей",
                    "layers": ["Conv1d(1→32)", "Conv1d(32→64)", "Conv1d(64→128)"],
                    "parameters": "~10K",
                    "use_cases": ["Анализ временных рядов", "Классификация сигналов"]
                },
                "deep": {
                    "description": "Глубокая архитектура для сложных последовательностей",
                    "layers": ["Conv1d(1→64)", "Conv1d(64→128)", "Conv1d(128→256)"],
                    "parameters": "~100K",
                    "use_cases": ["Сложный анализ сигналов", "Распознавание речи"]
                },
                "temporal": {
                    "description": "Временная архитектура для анализа временных рядов",
                    "layers": ["Conv1d(1→16)", "Conv1d(16→32)", "Conv1d(32→64)"],
                    "parameters": "~5K",
                    "use_cases": ["Финансовые данные", "Сенсорные данные"]
                }
            }
        }
        
        response_data = {
            "success": True,
            "architectures": architectures,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения архитектур: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения архитектур: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = 8096  # Новый порт для CNN сервера
    logger.info(f"🧠 Запуск Rubin AI CNN сервера на порту {port}...")
    app.run(port=port, debug=True, use_reloader=False)










