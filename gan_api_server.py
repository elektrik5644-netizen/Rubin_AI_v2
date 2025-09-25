#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API сервер для GAN (Generative Adversarial Networks) Rubin AI
Предоставляет endpoints для работы с генеративными состязательными сетями
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import logging
from datetime import datetime
import numpy as np

from rubin_gan import RubinGANManager, RubinGAN

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Инициализация GAN Manager
gan_manager = RubinGANManager()

@app.route('/api/gan/health', methods=['GET'])
def health_check():
    """Проверка состояния GAN сервера"""
    return jsonify({
        "status": "Rubin AI GAN Server ONLINE",
        "timestamp": datetime.now().isoformat(),
        "device": str(gan_manager.device),
        "models_count": len(gan_manager.models),
        "available_architectures": ["standard", "deep", "lightweight"],
        "available_tasks": ["generation", "training", "evaluation"]
    }), 200

@app.route('/api/gan/create', methods=['POST'])
def create_gan():
    """Создание GAN модели"""
    try:
        data = request.get_json() or {}
        noise_dim = data.get('noise_dim', 100)
        output_channels = data.get('output_channels', 3)
        output_size = data.get('output_size', 64)
        generator_arch = data.get('generator_arch', 'standard')
        discriminator_arch = data.get('discriminator_arch', 'standard')
        model_name = data.get('model_name', f'gan_{len(gan_manager.models)}')
        
        logger.info(f"🎯 Создание GAN: {model_name}")
        
        model = gan_manager.create_gan(
            noise_dim=noise_dim,
            output_channels=output_channels,
            output_size=output_size,
            generator_arch=generator_arch,
            discriminator_arch=discriminator_arch,
            model_name=model_name
        )
        
        if model:
            model_info = gan_manager.get_model_info(model_name)
            response_data = {
                "success": True,
                "model_name": model_name,
                "model_info": model_info,
                "message": f"GAN '{model_name}' создана успешно",
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(response_data), 201
        else:
            return jsonify({
                "success": False,
                "error": "Не удалось создать GAN"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Ошибка создания GAN: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка создания GAN: {str(e)}"
        }), 500

@app.route('/api/gan/train', methods=['POST'])
def train_gan():
    """Обучение GAN"""
    try:
        data = request.get_json() or {}
        model_name = data.get('model_name')
        epochs = data.get('epochs', 100)
        batch_size = data.get('batch_size', 32)
        learning_rate_g = data.get('learning_rate_g', 0.0002)
        learning_rate_d = data.get('learning_rate_d', 0.0002)
        beta1 = data.get('beta1', 0.5)
        beta2 = data.get('beta2', 0.999)
        
        if not model_name:
            return jsonify({
                "success": False,
                "error": "Не указано имя модели"
            }), 400
        
        logger.info(f"🚀 Обучение GAN: {model_name}")
        
        start_time = time.time()
        success = gan_manager.train_gan(
            model_name, None, epochs, batch_size, 
            learning_rate_g, learning_rate_d, beta1, beta2
        )
        end_time = time.time()
        
        training_time = end_time - start_time
        
        if success:
            model_info = gan_manager.get_model_info(model_name)
            response_data = {
                "success": True,
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate_g": learning_rate_g,
                "learning_rate_d": learning_rate_d,
                "training_time_seconds": f"{training_time:.2f}",
                "training_history": model_info.get("training_history", {}),
                "message": f"GAN '{model_name}' обучена успешно",
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(response_data), 200
        else:
            return jsonify({
                "success": False,
                "error": f"Не удалось обучить GAN '{model_name}'"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Ошибка обучения GAN: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка обучения GAN: {str(e)}"
        }), 500

@app.route('/api/gan/generate', methods=['POST'])
def generate_samples():
    """Генерация образцов"""
    try:
        data = request.get_json() or {}
        model_name = data.get('model_name')
        num_samples = data.get('num_samples', 1)
        noise = data.get('noise')
        
        if not model_name:
            return jsonify({
                "success": False,
                "error": "Не указано имя модели"
            }), 400
        
        logger.info(f"🎨 Генерация образцов: {model_name}")
        
        # Преобразуем шум в tensor если предоставлен
        noise_tensor = None
        if noise is not None:
            import torch
            if isinstance(noise, list):
                noise_array = np.array(noise)
            else:
                noise_array = noise
            
            # Добавляем batch dimension если нужно
            if noise_array.ndim == 1:
                noise_array = noise_array.reshape(1, -1, 1, 1)
            elif noise_array.ndim == 2:
                noise_array = noise_array.reshape(noise_array.shape[0], noise_array.shape[1], 1, 1)
            
            noise_tensor = torch.FloatTensor(noise_array)
        
        start_time = time.time()
        samples = gan_manager.generate_samples(model_name, num_samples, noise_tensor)
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        if samples is not None:
            response_data = {
                "success": True,
                "model_name": model_name,
                "num_samples": len(samples),
                "sample_shape": list(samples.shape),
                "samples": samples.tolist(),
                "generation_time_seconds": f"{generation_time:.4f}",
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(response_data), 200
        else:
            return jsonify({
                "success": False,
                "error": f"Не удалось сгенерировать образцы с помощью GAN '{model_name}'"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Ошибка генерации образцов: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка генерации образцов: {str(e)}"
        }), 500

@app.route('/api/gan/models', methods=['GET'])
def get_models():
    """Получение списка всех моделей"""
    try:
        logger.info("📋 Получение списка моделей")
        
        all_models_info = gan_manager.get_all_models_info()
        
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

@app.route('/api/gan/model/<model_name>', methods=['GET'])
def get_model_info(model_name):
    """Получение информации о конкретной модели"""
    try:
        logger.info(f"📊 Получение информации о модели: {model_name}")
        
        model_info = gan_manager.get_model_info(model_name)
        
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

@app.route('/api/gan/save/<model_name>', methods=['POST'])
def save_model(model_name):
    """Сохранение модели"""
    try:
        data = request.get_json() or {}
        filepath = data.get('filepath', f'models/{model_name}.pth')
        
        logger.info(f"💾 Сохранение модели: {model_name}")
        
        success = gan_manager.save_model(model_name, filepath)
        
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

@app.route('/api/gan/load', methods=['POST'])
def load_model():
    """Загрузка модели"""
    try:
        data = request.get_json() or {}
        model_name = data.get('model_name')
        filepath = data.get('filepath')
        noise_dim = data.get('noise_dim', 100)
        output_channels = data.get('output_channels', 3)
        output_size = data.get('output_size', 64)
        generator_arch = data.get('generator_arch', 'standard')
        discriminator_arch = data.get('discriminator_arch', 'standard')
        
        if not model_name or not filepath:
            return jsonify({
                "success": False,
                "error": "Не указаны имя модели или путь к файлу"
            }), 400
        
        logger.info(f"📂 Загрузка модели: {model_name}")
        
        success = gan_manager.load_model(
            model_name, filepath, noise_dim, output_channels, 
            output_size, generator_arch, discriminator_arch
        )
        
        if success:
            model_info = gan_manager.get_model_info(model_name)
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

@app.route('/api/gan/delete/<model_name>', methods=['DELETE'])
def delete_model(model_name):
    """Удаление модели"""
    try:
        logger.info(f"🗑️ Удаление модели: {model_name}")
        
        if model_name in gan_manager.models:
            del gan_manager.models[model_name]
            if model_name in gan_manager.training_history:
                del gan_manager.training_history[model_name]
            
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

@app.route('/api/gan/architectures', methods=['GET'])
def get_architectures():
    """Получение информации об архитектурах"""
    try:
        architectures = {
            "generator": {
                "standard": {
                    "description": "Стандартная архитектура генератора",
                    "layers": ["ConvTranspose2d(100→512)", "ConvTranspose2d(512→256)", "ConvTranspose2d(256→128)", "ConvTranspose2d(128→64)", "ConvTranspose2d(64→3)"],
                    "parameters": "~3.5M",
                    "use_cases": ["Генерация изображений", "Создание синтетических данных"]
                },
                "deep": {
                    "description": "Глубокая архитектура генератора",
                    "layers": ["ConvTranspose2d(100→1024)", "ConvTranspose2d(1024→512)", "ConvTranspose2d(512→256)", "ConvTranspose2d(256→128)", "ConvTranspose2d(128→64)", "ConvTranspose2d(64→3)"],
                    "parameters": "~7M",
                    "use_cases": ["Высококачественная генерация", "Сложные изображения"]
                },
                "lightweight": {
                    "description": "Легкая архитектура генератора",
                    "layers": ["ConvTranspose2d(100→256)", "ConvTranspose2d(256→128)", "ConvTranspose2d(128→64)", "ConvTranspose2d(64→3)"],
                    "parameters": "~1.5M",
                    "use_cases": ["Быстрая генерация", "Мобильные устройства"]
                }
            },
            "discriminator": {
                "standard": {
                    "description": "Стандартная архитектура дискриминатора",
                    "layers": ["Conv2d(3→64)", "Conv2d(64→128)", "Conv2d(128→256)", "Conv2d(256→512)", "Conv2d(512→1)"],
                    "parameters": "~2.7M",
                    "use_cases": ["Классификация изображений", "Оценка качества генерации"]
                },
                "deep": {
                    "description": "Глубокая архитектура дискриминатора",
                    "layers": ["Conv2d(3→64)", "Conv2d(64→128)", "Conv2d(128→256)", "Conv2d(256→512)", "Conv2d(512→1024)", "Conv2d(1024→1)"],
                    "parameters": "~5.5M",
                    "use_cases": ["Высокая точность классификации", "Сложные изображения"]
                },
                "lightweight": {
                    "description": "Легкая архитектура дискриминатора",
                    "layers": ["Conv2d(3→64)", "Conv2d(64→128)", "Conv2d(128→256)", "Conv2d(256→1)"],
                    "parameters": "~1.2M",
                    "use_cases": ["Быстрая классификация", "Мобильные устройства"]
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

@app.route('/api/gan/applications', methods=['GET'])
def get_applications():
    """Получение информации о применениях GAN"""
    try:
        applications = {
            "image_generation": {
                "name": "Генерация изображений",
                "description": "Создание реалистичных изображений из случайного шума",
                "use_cases": ["Создание синтетических данных", "Арт-генерация", "Аугментация данных"],
                "examples": ["Генерация лиц", "Создание пейзажей", "Генерация технических схем"]
            },
            "data_augmentation": {
                "name": "Аугментация данных",
                "description": "Увеличение размера обучающих наборов данных",
                "use_cases": ["Увеличение разнообразия данных", "Улучшение качества обучения", "Балансировка классов"],
                "examples": ["Генерация дополнительных изображений", "Создание вариаций данных", "Увеличение размера датасета"]
            },
            "style_transfer": {
                "name": "Перенос стиля",
                "description": "Изменение стиля изображений с сохранением содержания",
                "use_cases": ["Художественная обработка", "Стилизация изображений", "Творческие приложения"],
                "examples": ["Перенос художественного стиля", "Стилизация фотографий", "Создание арт-работ"]
            },
            "super_resolution": {
                "name": "Супер-разрешение",
                "description": "Увеличение разрешения изображений с улучшением качества",
                "use_cases": ["Улучшение качества изображений", "Восстановление деталей", "Увеличение разрешения"],
                "examples": ["Увеличение разрешения фото", "Восстановление старых изображений", "Улучшение качества видео"]
            },
            "anomaly_detection": {
                "name": "Детекция аномалий",
                "description": "Обнаружение необычных или аномальных образцов в данных",
                "use_cases": ["Мониторинг качества", "Обнаружение дефектов", "Безопасность"],
                "examples": ["Детекция дефектов в продукции", "Обнаружение аномалий в системах", "Мониторинг безопасности"]
            }
        }
        
        response_data = {
            "success": True,
            "applications": applications,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения применений: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения применений: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = 8098  # Новый порт для GAN сервера
    logger.info(f"🧠 Запуск Rubin AI GAN сервера на порту {port}...")
    app.run(port=port, debug=True, use_reloader=False)





