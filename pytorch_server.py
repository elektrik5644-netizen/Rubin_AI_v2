#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔥 PYTORCH SERVER FOR RUBIN AI
==============================
Специализированный сервер для работы с PyTorch в Rubin AI
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
from datetime import datetime
from typing import Dict, List, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class PyTorchExpert:
    """Эксперт по PyTorch для Rubin AI"""
    
    def __init__(self):
        self.knowledge_base = {
            "device_management": {
                "description": "Управление устройствами (CPU/GPU)",
                "best_practices": [
                    "Всегда проверять доступность CUDA",
                    "Использовать torch.device() для явного указания устройства",
                    "Переносить модель и данные на одно устройство"
                ],
                "common_errors": [
                    "RuntimeError: Expected all tensors to be on the same device",
                    "Несоответствие устройств между моделью и данными"
                ],
                "solutions": [
                    "model = model.to(device)",
                    "data = data.to(device)",
                    "target = target.to(device)"
                ]
            },
            
            "model_saving": {
                "description": "Сохранение и загрузка моделей",
                "best_practices": [
                    "Сохранять только state_dict, не всю модель",
                    "Использовать .eval() при загрузке для предсказаний",
                    "Сохранять метаданные вместе с параметрами"
                ],
                "common_errors": [
                    "Missing key(s) in state_dict",
                    "Архитектура модели не совпадает с сохраненной"
                ],
                "solutions": [
                    "torch.save(model.state_dict(), 'model.pth')",
                    "model.load_state_dict(torch.load('model.pth'))",
                    "model.eval()"
                ]
            },
            
            "gradient_management": {
                "description": "Управление градиентами",
                "best_practices": [
                    "Всегда обнулять градиенты перед backward()",
                    "Использовать optimizer.zero_grad()",
                    "Не забывать про optimizer.step()"
                ],
                "common_errors": [
                    "Градиенты накапливаются между итерациями",
                    "Забыли обнулить градиенты"
                ],
                "solutions": [
                    "optimizer.zero_grad()",
                    "loss.backward()",
                    "optimizer.step()"
                ]
            },
            
            "model_modes": {
                "description": "Режимы модели (train/eval)",
                "best_practices": [
                    "Использовать model.train() для обучения",
                    "Использовать model.eval() для тестирования",
                    "Использовать torch.no_grad() для предсказаний"
                ],
                "common_errors": [
                    "Непредсказуемое поведение Dropout",
                    "Неправильная работа BatchNorm"
                ],
                "solutions": [
                    "model.train() - включает Dropout, BatchNorm",
                    "model.eval() - отключает Dropout, фиксирует BatchNorm",
                    "torch.no_grad() - отключает вычисление градиентов"
                ]
            },
            
            "tensor_debugging": {
                "description": "Отладка тензоров",
                "best_practices": [
                    "Всегда проверять формы тензоров",
                    "Использовать print() для отладки",
                    "Проверять типы данных тензоров"
                ],
                "common_errors": [
                    "RuntimeError: size mismatch",
                    "Неправильные размеры входных данных"
                ],
                "solutions": [
                    "print(f'Shape: {tensor.shape}')",
                    "print(f'Dtype: {tensor.dtype}')",
                    "print(f'Device: {tensor.device}')"
                ]
            }
        }
        
        self.learning_paths = {
            "beginner": [
                "Основы PyTorch",
                "Работа с тензорами",
                "Простая нейронная сеть",
                "Обучение модели"
            ],
            "intermediate": [
                "CNN для изображений",
                "RNN для последовательностей",
                "Трансферное обучение",
                "Регуляризация"
            ],
            "advanced": [
                "Трансформеры",
                "GAN (Generative Adversarial Networks)",
                "Reinforcement Learning",
                "Custom Layers"
            ]
        }
    
    def analyze_pytorch_question(self, question: str) -> Dict[str, Any]:
        """Анализ вопроса о PyTorch"""
        
        question_lower = question.lower()
        
        # Определяем категорию вопроса
        if any(word in question_lower for word in ['устройство', 'device', 'gpu', 'cuda', 'cpu']):
            category = "device_management"
        elif any(word in question_lower for word in ['сохран', 'загруз', 'save', 'load', 'model']):
            category = "model_saving"
        elif any(word in question_lower for word in ['градиент', 'gradient', 'backward', 'optimizer']):
            category = "gradient_management"
        elif any(word in question_lower for word in ['режим', 'mode', 'train', 'eval', 'test']):
            category = "model_modes"
        elif any(word in question_lower for word in ['форма', 'shape', 'размер', 'size', 'отлад']):
            category = "tensor_debugging"
        else:
            category = "general"
        
        return {
            "category": category,
            "confidence": 0.9 if category != "general" else 0.6,
            "question_type": "pytorch_expert"
        }
    
    def generate_pytorch_response(self, question: str, analysis: Dict[str, Any]) -> str:
        """Генерация ответа о PyTorch"""
        
        category = analysis["category"]
        
        if category == "device_management":
            return self._generate_device_response(question)
        elif category == "model_saving":
            return self._generate_saving_response(question)
        elif category == "gradient_management":
            return self._generate_gradient_response(question)
        elif category == "model_modes":
            return self._generate_modes_response(question)
        elif category == "tensor_debugging":
            return self._generate_debugging_response(question)
        else:
            return self._generate_general_response(question)
    
    def _generate_device_response(self, question: str) -> str:
        """Ответ о управлении устройствами"""
        return """🔥 **PYTORCH: УПРАВЛЕНИЕ УСТРОЙСТВАМИ**

**🎯 Основные принципы:**
1. **Проверка CUDA:** `torch.cuda.is_available()`
2. **Выбор устройства:** `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
3. **Перенос модели:** `model = model.to(device)`
4. **Перенос данных:** `data = data.to(device)`

**💡 Лучшие практики:**
- Всегда проверяйте доступность GPU перед использованием
- Переносите модель и данные на одно устройство
- Используйте явное указание устройства

**⚠️ Частые ошибки:**
- `RuntimeError: Expected all tensors to be on the same device`
- Несоответствие устройств между моделью и данными

**🔧 Решение:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
for data, target in train_loader:
    data = data.to(device)
    target = target.to(device)
    # ... обучение
```

**📊 Преимущества GPU:**
- До 100x ускорение для матричных операций
- Возможность работы с большими батчами
- Ускорение обучения сложных моделей"""
    
    def _generate_saving_response(self, question: str) -> str:
        """Ответ о сохранении моделей"""
        return """💾 **PYTORCH: СОХРАНЕНИЕ И ЗАГРУЗКА МОДЕЛЕЙ**

**🎯 Правильный подход:**
1. **Сохранение:** `torch.save(model.state_dict(), 'model.pth')`
2. **Загрузка:** `model.load_state_dict(torch.load('model.pth'))`
3. **Режим оценки:** `model.eval()`

**💡 Лучшие практики:**
- Сохраняйте только параметры (state_dict), не всю модель
- Используйте .eval() при загрузке для предсказаний
- Сохраняйте метаданные вместе с параметрами

**⚠️ Частые ошибки:**
- `Missing key(s) in state_dict`
- Архитектура модели не совпадает с сохраненной
- Сохранение всей модели (не рекомендуется)

**🔧 Полный пример:**
```python
# Сохранение
torch.save(model.state_dict(), 'model.pth')

# Загрузка
model = MNISTClassifier()  # Тот же класс!
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Важно для предсказаний

# Предсказание
with torch.no_grad():
    output = model(data)
    prediction = output.argmax().item()
```

**📋 Что такое state_dict?**
- Словарь с обучаемыми параметрами модели
- Содержит веса и смещения всех слоев
- Не включает архитектуру модели"""
    
    def _generate_gradient_response(self, question: str) -> str:
        """Ответ о управлении градиентами"""
        return """⚡ **PYTORCH: УПРАВЛЕНИЕ ГРАДИЕНТАМИ**

**🎯 Правильная последовательность:**
1. **Обнуление:** `optimizer.zero_grad()`
2. **Прямой проход:** `output = model(data)`
3. **Вычисление потерь:** `loss = criterion(output, target)`
4. **Обратный проход:** `loss.backward()`
5. **Обновление:** `optimizer.step()`

**💡 Лучшие практики:**
- Всегда обнуляйте градиенты перед backward()
- Не забывайте про optimizer.step()
- Используйте правильный порядок операций

**⚠️ Частые ошибки:**
- Градиенты накапливаются между итерациями
- Забыли обнулить градиенты
- Неправильный порядок операций

**🔧 Правильный цикл обучения:**
```python
for data, target in train_loader:
    optimizer.zero_grad()  # Обнуляем градиенты
    output = model(data)
    loss = criterion(output, target)
    loss.backward()        # Вычисляем градиенты
    optimizer.step()       # Обновляем параметры
```

**❌ Неправильно:**
```python
for data, target in train_loader:
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()  # Градиенты накапливаются!
```

**📊 Почему это важно:**
- Предотвращает накопление градиентов
- Обеспечивает правильное обучение
- Избегает нестабильности обучения"""
    
    def _generate_modes_response(self, question: str) -> str:
        """Ответ о режимах модели"""
        return """🔄 **PYTORCH: РЕЖИМЫ МОДЕЛИ (TRAIN/EVAL)**

**🎯 Два основных режима:**
1. **Обучение:** `model.train()`
2. **Оценка:** `model.eval()`

**💡 Когда использовать:**
- **model.train()** - во время обучения
- **model.eval()** - во время тестирования/предсказаний
- **torch.no_grad()** - для предсказаний (экономия памяти)

**⚠️ Что происходит в каждом режиме:**
- **train():** Включает Dropout, BatchNorm обучается
- **eval():** Отключает Dropout, фиксирует BatchNorm

**🔧 Правильное использование:**
```python
# Обучение
model.train()
for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# Тестирование
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        # Вычисляем метрики
```

**📊 Преимущества:**
- Правильное поведение Dropout и BatchNorm
- Экономия памяти при предсказаниях
- Стабильные результаты тестирования

**🎯 Важные моменты:**
- Всегда переключайте режим явно
- Используйте torch.no_grad() для предсказаний
- Проверяйте режим модели при отладке"""
    
    def _generate_debugging_response(self, question: str) -> str:
        """Ответ о отладке тензоров"""
        return """🐛 **PYTORCH: ОТЛАДКА ТЕНЗОРОВ**

**🎯 Основные проверки:**
1. **Форма:** `tensor.shape`
2. **Тип данных:** `tensor.dtype`
3. **Устройство:** `tensor.device`
4. **Значения:** `tensor.min()`, `tensor.max()`

**💡 Инструменты отладки:**
```python
print(f"Форма: {tensor.shape}")
print(f"Тип: {tensor.dtype}")
print(f"Устройство: {tensor.device}")
print(f"Минимум: {tensor.min()}")
print(f"Максимум: {tensor.max()}")
print(f"Среднее: {tensor.mean()}")
```

**⚠️ Частые проблемы:**
- `RuntimeError: size mismatch`
- Неправильные размеры входных данных
- Проблемы с типами данных

**🔧 Отладочный шаблон:**
```python
def debug_tensor(name, tensor):
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Min: {tensor.min()}")
    print(f"  Max: {tensor.max()}")
    print()

# Использование
debug_tensor("Input", data)
debug_tensor("Output", output)
debug_tensor("Target", target)
```

**📊 Полезные проверки:**
- Совпадают ли размеры батча?
- Правильный ли тип данных?
- На одном ли устройстве тензоры?
- Нормальные ли значения?

**🎯 Советы:**
- Добавляйте отладочные принты в критических местах
- Проверяйте формы тензоров при изменении архитектуры
- Используйте assert для проверки условий"""
    
    def _generate_general_response(self, question: str) -> str:
        """Общий ответ о PyTorch"""
        return """🔥 **PYTORCH: ОБЩИЕ ПРИНЦИПЫ**

**🎯 Основы PyTorch:**
- Динамические графы вычислений
- Интуитивный API
- Отличная производительность

**💡 Ключевые концепции:**
1. **Тензоры** - основа всех вычислений
2. **Автодифференциация** - автоматическое вычисление градиентов
3. **Модули** - строительные блоки нейронных сетей
4. **Оптимизаторы** - алгоритмы обучения

**📚 Пути изучения:**
- **Начинающий:** Основы, простые сети
- **Средний:** CNN, RNN, трансферное обучение
- **Продвинутый:** Трансформеры, GAN, RL

**🔧 Лучшие практики:**
- Начинайте с простых примеров
- Проверяйте формы тензоров
- Используйте правильные режимы модели
- Сохраняйте только параметры модели

**📖 Полезные ресурсы:**
- Официальная документация PyTorch
- PyTorch Tutorials
- Papers With Code

**🎯 Помните:** 80% работы с данными, 20% магии моделей!"""

# Создаем экземпляр эксперта
pytorch_expert = PyTorchExpert()

@app.route('/api/pytorch/chat', methods=['POST'])
def pytorch_chat():
    """Основной эндпоинт для вопросов о PyTorch"""
    try:
        data = request.get_json()
        question = data.get('message', '')
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'Вопрос не может быть пустым'
            }), 400
        
        logger.info(f"🔥 Получен вопрос о PyTorch: {question[:50]}...")
        
        # Анализируем вопрос
        analysis = pytorch_expert.analyze_pytorch_question(question)
        
        # Генерируем ответ
        response = pytorch_expert.generate_pytorch_response(question, analysis)
        
        logger.info("✅ Сгенерирован ответ о PyTorch")
        
        return jsonify({
            'success': True,
            'response': response,
            'category': analysis['category'],
            'confidence': analysis['confidence'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в pytorch_chat: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/pytorch/knowledge', methods=['GET'])
def get_pytorch_knowledge():
    """Получение базы знаний PyTorch"""
    try:
        return jsonify({
            'success': True,
            'knowledge_base': pytorch_expert.knowledge_base,
            'learning_paths': pytorch_expert.learning_paths,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Ошибка в get_pytorch_knowledge: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/pytorch/analyze', methods=['POST'])
def analyze_pytorch_code():
    """Анализ PyTorch кода"""
    try:
        data = request.get_json()
        code = data.get('code', '')
        
        if not code:
            return jsonify({
                'success': False,
                'error': 'Код не может быть пустым'
            }), 400
        
        # Простой анализ кода
        analysis = {
            'has_device_management': 'torch.device' in code or '.to(' in code,
            'has_gradient_management': 'zero_grad()' in code,
            'has_model_modes': 'model.train()' in code or 'model.eval()' in code,
            'has_saving': 'torch.save' in code or 'load_state_dict' in code,
            'potential_issues': []
        }
        
        # Проверяем на потенциальные проблемы
        if 'torch.save(model,' in code:
            analysis['potential_issues'].append("Рекомендуется сохранять только state_dict")
        
        if 'loss.backward()' in code and 'zero_grad()' not in code:
            analysis['potential_issues'].append("Отсутствует обнуление градиентов")
        
        if 'model.eval()' not in code and 'torch.no_grad()' not in code and 'with torch.no_grad' not in code:
            analysis['potential_issues'].append("Рекомендуется использовать eval() или no_grad() для предсказаний")
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в analyze_pytorch_code: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        'status': 'healthy',
        'service': 'PyTorch Expert Server',
        'timestamp': datetime.now().isoformat(),
        'knowledge_categories': len(pytorch_expert.knowledge_base)
    })

if __name__ == '__main__':
    print("🔥 PyTorch Expert Server запущен")
    print("URL: http://localhost:8092")
    print("Доступные эндпоинты:")
    print("  - POST /api/pytorch/chat - вопросы о PyTorch")
    print("  - GET  /api/pytorch/knowledge - база знаний")
    print("  - POST /api/pytorch/analyze - анализ кода")
    print("  - GET  /api/health - проверка здоровья")
    
    app.run(host='0.0.0.0', port=8092, debug=True)





