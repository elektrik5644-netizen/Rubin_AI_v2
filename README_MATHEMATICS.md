# 🧮 Rubin Mathematics Module

## 📖 Описание

Модуль математики для Rubin IDE - это мощная система математических вычислений, которая позволяет Рубину выполнять сложные математические операции, решать уравнения, рассчитывать площади геометрических фигур, проводить статистический анализ и обучать математическим концепциям.

## 🎯 Возможности

### 🔢 Основные операции
- **Арифметика**: сложение, вычитание, умножение, деление
- **Степени и корни**: возведение в степень, извлечение корня
- **Проценты**: расчет процентов от числа
- **Отношения**: вычисление отношений между числами

### 📐 Алгебра и уравнения
- **Линейные уравнения**: ax + b = 0 → x = -b/a
- **Квадратные уравнения**: ax² + bx + c = 0 с дискриминантом
- **Системы уравнений**: методы подстановки, сложения, определители

### 📏 Геометрия
- **Площади фигур**: треугольник, круг, прямоугольник, квадрат
- **Периметры**: расчет периметров различных фигур
- **Теорема Пифагора**: a² + b² = c² для прямоугольных треугольников

### 📊 Тригонометрия
- **Основные функции**: синус, косинус, тангенс
- **Тождества**: основное тригонометрическое тождество
- **Формулы сложения**: sin(α+β), cos(α+β)
- **Формулы двойного угла**: sin(2α), cos(2α)

### 📈 Математический анализ
- **Производные**: основные правила дифференцирования
- **Интегралы**: неопределенные интегралы
- **Цепное правило**: производная сложной функции
- **Интегрирование по частям**

### 📊 Статистика и вероятность
- **Описательная статистика**: среднее, медиана, мода
- **Дисперсия и стандартное отклонение**
- **Основы вероятности**: классическое определение
- **Условная вероятность и независимость событий**

## 🚀 Установка и запуск

### 1. Зависимости
```bash
pip install flask flask-cors
```

### 2. Запуск API сервера
```bash
python rubin_mathematics_api.py
```

Сервер будет доступен на порту 8088: `http://localhost:8088`

### 3. Проверка статуса
```bash
curl http://localhost:8088/api/mathematics/status
```

## 📚 Использование

### 🧮 Базовые вычисления

#### Простое выражение
```javascript
// В консоли браузера
window.mathematics.calculate('2 + 3 * 4')
    .then(response => console.log(response));
```

#### Сложные выражения
```javascript
window.mathematics.calculate('(5 + 3) * 2^3 - 10')
    .then(response => console.log(response));
```

### 📐 Решение уравнений

#### Линейное уравнение
```javascript
window.mathematics.solveEquation('линейное уравнение 2x + 5 = 13')
    .then(response => console.log(response));
```

#### Квадратное уравнение
```javascript
window.mathematics.solveEquation('квадратное уравнение x² - 5x + 6 = 0')
    .then(response => console.log(response));
```

### 📏 Геометрические расчеты

#### Площадь треугольника
```javascript
window.mathematics.calculateArea('треугольник', {
    base: 5,
    height: 3
}).then(response => console.log(response));
```

#### Площадь круга
```javascript
window.mathematics.calculateArea('круг', {
    radius: 4
}).then(response => console.log(response));
```

#### Площадь прямоугольника
```javascript
window.mathematics.calculateArea('прямоугольник', {
    length: 6,
    width: 8
}).then(response => console.log(response));
```

### 📊 Статистический анализ

#### Анализ данных
```javascript
window.mathematics.calculateStatistics([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    .then(response => console.log(response));
```

### 📚 Обучение математике

#### Начало урока
```javascript
window.mathematics.startLesson('lesson_1')
    .then(response => console.log(response));
```

#### Объяснение концепции
```javascript
window.mathematics.explainConcept('квадратные уравнения', 'basic')
    .then(response => console.log(response));
```

#### Получение упражнения
```javascript
window.mathematics.getExercise('уравнения', 'basic')
    .then(response => console.log(response));
```

### 🔧 Продвинутый калькулятор

#### Расчет процента
```javascript
window.mathematics.advancedCalculator('percentage', {
    number: 200,
    percent: 15
}).then(response => console.log(response));
```

#### Возведение в степень
```javascript
window.mathematics.advancedCalculator('power', {
    base: 2,
    exponent: 10
}).then(response => console.log(response));
```

#### Извлечение корня
```javascript
window.mathematics.advancedCalculator('root', {
    number: 16,
    degree: 4
}).then(response => console.log(response));
```

## 🌐 API Endpoints

### GET запросы
- `/api/mathematics/status` - Статус модуля
- `/api/mathematics/help` - Справка по модулю
- `/api/mathematics/plan` - План уроков
- `/api/mathematics/progress` - Прогресс обучения
- `/api/mathematics/history` - История вычислений

### POST запросы
- `/api/mathematics/calculate` - Вычисление выражения
- `/api/mathematics/solve` - Решение уравнения
- `/api/mathematics/area` - Расчет площади
- `/api/mathematics/statistics` - Статистический анализ
- `/api/mathematics/formulas` - Справка по формулам
- `/api/mathematics/lesson` - Начало урока
- `/api/mathematics/exercise` - Получение упражнения
- `/api/mathematics/check` - Проверка ответа
- `/api/mathematics/explain` - Объяснение концепции
- `/api/mathematics/calculator` - Продвинутый калькулятор

## 💡 Примеры использования

### 1. Вычисление сложного выражения
```javascript
// Вычисление выражения с тригонометрическими функциями
fetch('/api/mathematics/calculate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ expression: 'sin(π/2) + cos(0) * 2^3' })
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        console.log('Результат:', data.data.formatted_result);
    }
});
```

### 2. Решение системы уравнений
```javascript
// Решение системы линейных уравнений
fetch('/api/mathematics/solve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ equation: 'система уравнений 2x + y = 5, x - y = 1' })
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        console.log('Решение:', data.data);
    }
});
```

### 3. Расчет площади сложной фигуры
```javascript
// Расчет площади треугольника
fetch('/api/mathematics/area', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        shape: 'треугольник',
        dimensions: { base: 10, height: 6 }
    })
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        console.log(`Площадь треугольника: ${data.data.formatted_area}`);
    }
});
```

### 4. Статистический анализ данных
```javascript
// Анализ результатов тестирования
const testScores = [85, 92, 78, 96, 88, 91, 87, 94, 89, 93];

fetch('/api/mathematics/statistics', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data: testScores })
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        const stats = data.data;
        console.log(`Средний балл: ${stats.mean.toFixed(2)}`);
        console.log(`Медиана: ${stats.median}`);
        console.log(`Стандартное отклонение: ${stats.standard_deviation.toFixed(2)}`);
    }
});
```

## 🔧 Интеграция с Rubin IDE

### Автоматическое распознавание
Рубин автоматически распознает математические запросы и перенаправляет их в соответствующий модуль.

### Ключевые слова для активации
- математика, вычисление, калькулятор
- уравнение, решить, формула
- площадь, объем, геометрия
- статистика, среднее, процент
- синус, косинус, тригонометрия

### Примеры команд в чате
```
"Вычисли 2 + 3 * 4"
"Реши уравнение 2x + 5 = 13"
"Найди площадь треугольника с основанием 5 и высотой 3"
"Начни урок по математике"
"Покажи формулы для расчета площадей"
"Рассчитай среднее значение [1, 2, 3, 4, 5]"
"Что такое квадратное уравнение?"
```

## 📊 Мониторинг и аналитика

### Отслеживание прогресса
```javascript
// Получение отчета о прогрессе обучения
fetch('/api/mathematics/progress')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const progress = data.data;
            console.log(`📊 Прогресс: ${progress.completed_lessons}/${progress.total_lessons} уроков`);
        }
    });
```

### История вычислений
```javascript
// Получение истории всех вычислений
fetch('/api/mathematics/history')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('История вычислений:', data.data);
        }
    });
```

## 🛠️ Расширение функциональности

### Добавление новых математических операций
```python
# В rubin_mathematics.py
def calculate_new_operation(self, operation_type: str, values: Dict[str, float]) -> Dict[str, Any]:
    """Новая математическая операция"""
    if operation_type == "new_operation":
        # Логика новой операции
        result = self._perform_new_operation(values)
        return {
            "operation": "Новая операция",
            "result": result,
            "formula": "формула новой операции"
        }
    
    return {"error": f"Операция '{operation_type}' не поддерживается"}
```

### Создание новых типов уравнений
```python
def solve_new_equation_type(self, equation_description: str) -> Dict[str, Any]:
    """Решение нового типа уравнения"""
    # Анализ типа уравнения
    if "новый_тип" in equation_description.lower():
        return {
            "type": "Новый тип уравнения",
            "method": "Метод решения",
            "steps": ["Шаг 1", "Шаг 2", "Шаг 3"]
        }
    
    return {"error": "Тип уравнения не определен"}
```

## 🐛 Устранение неполадок

### Модуль недоступен
```
⚠️ Модуль математики недоступен
```
**Решение**: Убедитесь, что файл `rubin_mathematics.py` находится в той же папке.

### Ошибка вычисления
```
❌ Ошибка вычисления: invalid syntax
```
**Решение**: Проверьте правильность математического выражения. Используйте стандартные математические символы.

### Сервер не запускается
```
Address already in use
```
**Решение**: Измените порт в `rubin_mathematics_api.py` или остановите другие сервисы на порту 8088.

## 🔮 Планы развития

### Краткосрочные (1-2 месяца)
- [ ] Добавление графиков и визуализации
- [ ] Поддержка комплексных чисел
- [ ] Больше геометрических фигур

### Среднесрочные (3-6 месяцев)
- [ ] Матричные вычисления
- [ ] Дифференциальные уравнения
- [ ] Интеграция с графическими калькуляторами

### Долгосрочные (6+ месяцев)
- [ ] Машинное обучение для математических задач
- [ ] 3D геометрия и стереометрия
- [ ] Поддержка других математических дисциплин

## 📞 Поддержка

### Документация
- Основная документация: `README_MAIN.md`
- Модуль электротехники: `README_ELECTRICAL_ENGINEERING.md`
- Система обучения: `README_RUBIN_LEARNING.md`

### Тестирование
```bash
# Запуск тестов модуля
python rubin_mathematics.py

# Проверка API
curl http://localhost:8088/api/mathematics/status
```

### Обратная связь
Для вопросов и предложений по улучшению модуля математики используйте систему чата Rubin IDE или создавайте issues в репозитории.

---

**🧮 Модуль математики готов к использованию! Начните обучение командой "Начни урок по математике" в чате Rubin IDE.**






