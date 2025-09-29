# 📁 Система чтения папок других проектов для Rubin AI

## 🎯 Обзор

Система чтения папок других проектов позволяет Rubin AI анализировать, изучать и использовать знания из ваших проектов для более точных и контекстных ответов. Rubin может:

- 📊 **Анализировать структуру проектов** - понимать архитектуру и организацию кода
- 🔍 **Извлекать компоненты** - находить классы, функции, модули и их связи
- 🧩 **Распознавать паттерны** - выявлять паттерны проектирования и лучшие практики
- 💡 **Генерировать инсайты** - давать рекомендации по улучшению кода
- 🔗 **Использовать контекст** - отвечать на вопросы с учетом знаний из проектов

## 🚀 Быстрый старт

### 1. Анализ проекта

```python
from project_folder_reader import ProjectFolderReader

# Создаем анализатор
reader = ProjectFolderReader()

# Анализируем ваш проект
success = reader.analyze_project_folder("/path/to/your/project", "My Project")

if success:
    print("✅ Проект успешно проанализирован!")
```

### 2. Использование знаний

```python
from rubin_project_integration import RubinProjectIntegration

# Создаем интеграцию
integration = RubinProjectIntegration()

# Ищем информацию в проектах
results = integration.search_project_knowledge("class Calculator")

# Получаем ответ с контекстом
answer = integration.answer_with_project_context("Как работает калькулятор?")
print(answer['answer'])
```

### 3. Запуск API сервера

```bash
python project_api_server.py
```

API будет доступен на `http://localhost:8091`

## 📋 Поддерживаемые типы файлов

### Код
- **Python**: `.py`
- **JavaScript/TypeScript**: `.js`, `.ts`, `.jsx`, `.tsx`
- **Java**: `.java`
- **C/C++**: `.c`, `.cpp`, `.h`
- **C#**: `.cs`
- **PHP**: `.php`
- **Ruby**: `.rb`
- **Go**: `.go`
- **Rust**: `.rs`
- **Swift**: `.swift`
- **Kotlin**: `.kt`
- **Scala**: `.scala`
- **R**: `.r`
- **Shell**: `.sh`, `.bash`, `.ps1`

### Веб-технологии
- **HTML**: `.html`, `.htm`
- **CSS**: `.css`, `.scss`, `.sass`, `.less`
- **Vue**: `.vue`

### Конфигурация
- **JSON**: `.json`
- **XML**: `.xml`
- **YAML**: `.yaml`, `.yml`
- **TOML**: `.toml`
- **INI**: `.ini`, `.cfg`, `.conf`

### Документация
- **Markdown**: `.md`
- **reStructuredText**: `.rst`
- **Text**: `.txt`
- **PDF**: `.pdf`
- **Word**: `.doc`, `.docx`

### Данные
- **CSV**: `.csv`
- **SQL**: `.sql`
- **Database**: `.db`, `.sqlite`, `.sqlite3`

### Другие
- **Docker**: `Dockerfile`
- **Environment**: `.env`
- **Git**: `.gitignore`, `.gitattributes`

## 🏗️ Архитектура системы

```
┌─────────────────────────────────────────────────────────────┐
│                    Rubin AI Project Reader                 │
├─────────────────────────────────────────────────────────────┤
│  ProjectFolderReader                                        │
│  ├── Анализ структуры проекта                               │
│  ├── Извлечение файлов и метаданных                        │
│  ├── Определение языков и фреймворков                      │
│  └── Анализ сложности кода                                 │
├─────────────────────────────────────────────────────────────┤
│  RubinProjectIntegration                                    │
│  ├── Поиск знаний в проектах                               │
│  ├── Генерация инсайтов                                    │
│  ├── Извлечение архитектурных компонентов                 │
│  └── Распознавание паттернов проектирования               │
├─────────────────────────────────────────────────────────────┤
│  Project API Server                                         │
│  ├── REST API для интеграции                               │
│  ├── Анализ проектов через HTTP                            │
│  ├── Поиск и ответы с контекстом                           │
│  └── Экспорт знаний                                        │
└─────────────────────────────────────────────────────────────┘
```

## 📊 База данных

Система использует SQLite базу данных `rubin_project_knowledge.db` со следующими таблицами:

### `projects`
- Информация о проектах
- Статистика файлов и языков
- Время последнего анализа

### `project_files`
- Детальная информация о файлах
- Содержимое и метаданные
- Анализ сложности

### `project_components`
- Архитектурные компоненты
- Классы, функции, модули
- Связи и зависимости

### `project_knowledge`
- Паттерны проектирования
- Лучшие практики
- Анти-паттерны

## 🔧 API Endpoints

### Анализ проектов
```http
POST /api/projects/analyze
Content-Type: application/json

{
    "project_path": "/path/to/project",
    "project_name": "My Project"
}
```

### Получение проектов
```http
GET /api/projects
```

### Инсайты проекта
```http
GET /api/projects/{id}/insights
```

### Архитектура проекта
```http
GET /api/projects/{id}/architecture
```

### Паттерны проектирования
```http
GET /api/projects/{id}/patterns
```

### Поиск в проектах
```http
POST /api/projects/search
Content-Type: application/json

{
    "query": "class Calculator",
    "project_id": 1
}
```

### Ответы с контекстом
```http
POST /api/projects/answer
Content-Type: application/json

{
    "question": "Как работает калькулятор?",
    "project_id": 1
}
```

### Сравнение проектов
```http
POST /api/projects/compare
Content-Type: application/json

{
    "project_ids": [1, 2, 3]
}
```

### Экспорт знаний
```http
GET /api/projects/{id}/export?format=json
```

### Удаление проекта
```http
DELETE /api/projects/{id}
```

## 💡 Примеры использования

### 1. Анализ Python проекта

```python
# Анализируем Django проект
reader = ProjectFolderReader()
reader.analyze_project_folder("/path/to/django-app", "Django App")

# Получаем инсайты
integration = RubinProjectIntegration()
insights = integration.generate_project_insights(project_id)

print(f"Тип проекта: {insights['project_overview']['type']}")
print(f"Компонентов: {insights['architecture_analysis']['total_components']}")
print(f"Паттернов: {insights['design_patterns']['total_patterns']}")
```

### 2. Поиск решений

```python
# Ищем примеры использования паттерна Singleton
results = integration.search_project_knowledge("singleton pattern")

for result in results:
    print(f"Файл: {result['file_name']}")
    print(f"Проект: {result['project_name']}")
    print(f"Превью: {result['content_preview']}")
```

### 3. Получение рекомендаций

```python
# Получаем рекомендации по проекту
insights = integration.generate_project_insights(project_id)

for recommendation in insights['recommendations']:
    print(f"💡 {recommendation}")
```

### 4. Ответы с контекстом

```python
# Задаем вопрос с использованием знаний из проектов
answer = integration.answer_with_project_context(
    "Как реализовать паттерн Observer?",
    project_id
)

print(f"Ответ: {answer['answer']}")
print(f"Уверенность: {answer['confidence']:.1%}")
print(f"Источники: {len(answer['sources'])} файлов")
```

## 🎯 Интеграция с Rubin AI

### Через API

```python
import requests

# Анализируем проект
response = requests.post('http://localhost:8091/api/projects/analyze', json={
    'project_path': '/path/to/project',
    'project_name': 'My Project'
})

# Получаем ответ с контекстом
response = requests.post('http://localhost:8091/api/projects/answer', json={
    'question': 'Как работает этот проект?',
    'project_id': 1
})

answer = response.json()
print(answer['answer'])
```

### Прямая интеграция

```python
from rubin_project_integration import RubinProjectIntegration

# Инициализируем интеграцию
integration = RubinProjectIntegration()

# Анализируем проект
result = integration.analyze_user_project("/path/to/project")

if result['success']:
    project_id = result['project_id']
    
    # Используем знания для ответов
    answer = integration.answer_with_project_context(
        "Объясни архитектуру этого проекта",
        project_id
    )
    
    print(answer['answer'])
```

## 🔒 Безопасность и приватность

- **Локальное хранение**: Все данные хранятся локально в SQLite
- **Контроль доступа**: Только вы решаете, какие проекты анализировать
- **Исключения**: Система автоматически исключает конфиденциальные файлы
- **Хеширование**: Содержимое файлов хешируется для отслеживания изменений

## 🚫 Исключаемые файлы и папки

### Папки
- `__pycache__`, `.git`, `.svn`, `.hg`
- `node_modules`, `.vscode`, `.idea`
- `venv`, `env`, `.env`
- `build`, `dist`, `target`
- `.pytest_cache`, `.coverage`
- `logs`, `log`, `tmp`, `temp`

### Файлы
- `.gitignore`, `.gitattributes`
- `.DS_Store`, `Thumbs.db`
- `package-lock.json`, `yarn.lock`
- `Pipfile.lock`

## 📈 Производительность

- **Быстрый анализ**: Обработка 1000+ файлов за несколько минут
- **Инкрементальные обновления**: Анализ только измененных файлов
- **Оптимизированный поиск**: Индексация для быстрого поиска
- **Сжатие данных**: Эффективное хранение больших файлов

## 🛠️ Настройка и конфигурация

### Переменные окружения

```bash
export RUBIN_PROJECT_DB_PATH="/custom/path/projects.db"
export RUBIN_MAX_FILE_SIZE="50MB"
export RUBIN_SUPPORTED_LANGUAGES="python,javascript,java"
```

### Конфигурация исключений

```python
# Добавляем собственные исключения
reader = ProjectFolderReader()
reader.excluded_dirs.add("my_secret_folder")
reader.excluded_files.add("secret_config.json")
```

## 🔍 Отладка и логирование

```python
import logging

# Включаем подробное логирование
logging.basicConfig(level=logging.DEBUG)

# Анализируем проект с отладкой
reader = ProjectFolderReader()
reader.analyze_project_folder("/path/to/project")
```

## 📚 Дополнительные ресурсы

- **Документация API**: `http://localhost:8091/api/docs`
- **Проверка состояния**: `http://localhost:8091/api/health`
- **Примеры кода**: Смотрите файлы `demo_project_reader.py`
- **Тестирование**: Запустите `python demo_project_reader.py`

## 🤝 Поддержка

Если у вас есть вопросы или проблемы:

1. Проверьте логи в файле `project_reader.log`
2. Убедитесь, что все зависимости установлены
3. Проверьте права доступа к папкам проектов
4. Обратитесь к документации API

---

**Rubin AI теперь может читать и понимать ваши проекты! 🚀**










