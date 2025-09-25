#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Rubin AI - Контроллер редактора кода
Система управления редактором, исправления кода и создания документов
"""

import os
import json
import sqlite3
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import shutil

class EditorAction(Enum):
    """Действия редактора"""
    CREATE_FILE = "create_file"
    EDIT_FILE = "edit_file"
    DELETE_FILE = "delete_file"
    SAVE_FILE = "save_file"
    OPEN_FILE = "open_file"
    CREATE_FOLDER = "create_folder"
    DELETE_FOLDER = "delete_folder"
    SEARCH_REPLACE = "search_replace"
    INSERT_TEXT = "insert_text"
    DELETE_TEXT = "delete_text"
    FORMAT_CODE = "format_code"
    FIX_CODE = "fix_code"
    GENERATE_CODE = "generate_code"
    CREATE_DOCUMENT = "create_document"

class DocumentType(Enum):
    """Типы документов"""
    README = "readme"
    TECHNICAL_DOC = "technical_doc"
    API_DOC = "api_doc"
    USER_GUIDE = "user_guide"
    CODE_DOC = "code_doc"
    PROJECT_DOC = "project_doc"
    TROUBLESHOOTING = "troubleshooting"
    CHANGELOG = "changelog"

@dataclass
class EditorCommand:
    """Команда для редактора"""
    id: str
    action: EditorAction
    target: str  # файл или папка
    content: Optional[str] = None
    position: Optional[Tuple[int, int]] = None  # строка, колонка
    parameters: Optional[Dict[str, Any]] = None
    timestamp: datetime.datetime = None

@dataclass
class DocumentTemplate:
    """Шаблон документа"""
    type: DocumentType
    title: str
    content: str
    variables: List[str]
    sections: List[str]

class RubinEditorController:
    """Основной класс управления редактором"""
    
    def __init__(self, workspace_path: str = ".", db_path: str = "rubin_editor.db"):
        self.workspace_path = workspace_path
        self.db_path = db_path
        self.document_templates = self.load_document_templates()
        self.code_templates = self.load_code_templates()
        self.init_database()
    
    def init_database(self):
        """Инициализация базы данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS editor_commands (
                    id TEXT PRIMARY KEY,
                    action TEXT,
                    target TEXT,
                    content TEXT,
                    position TEXT,
                    parameters TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT FALSE
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS created_documents (
                    id TEXT PRIMARY KEY,
                    document_type TEXT,
                    title TEXT,
                    file_path TEXT,
                    content TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    modified_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_operations (
                    id TEXT PRIMARY KEY,
                    operation_type TEXT,
                    file_path TEXT,
                    backup_path TEXT,
                    content_before TEXT,
                    content_after TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Ошибка инициализации БД: {e}")
    
    def load_document_templates(self) -> Dict[DocumentType, DocumentTemplate]:
        """Загружает шаблоны документов"""
        return {
            DocumentType.README: DocumentTemplate(
                type=DocumentType.README,
                title="README.md",
                content="""# {project_name}

## Описание
{description}

## Возможности
{features}

## Установка
{installation}

## Использование
{usage}

## Примеры
{examples}

## API
{api_docs}

## Лицензия
{license}

## Автор
{author}""",
                variables=["project_name", "description", "features", "installation", "usage", "examples", "api_docs", "license", "author"],
                sections=["Описание", "Возможности", "Установка", "Использование", "Примеры", "API", "Лицензия", "Автор"]
            ),
            
            DocumentType.TECHNICAL_DOC: DocumentTemplate(
                type=DocumentType.TECHNICAL_DOC,
                title="TECHNICAL_DOCUMENTATION.md",
                content="""# Техническая документация: {project_name}

## Архитектура системы
{architecture}

## Компоненты
{components}

## API Reference
{api_reference}

## Конфигурация
{configuration}

## Развертывание
{deployment}

## Мониторинг
{monitoring}

## Безопасность
{security}

## Производительность
{performance}

## Устранение неполадок
{troubleshooting}""",
                variables=["project_name", "architecture", "components", "api_reference", "configuration", "deployment", "monitoring", "security", "performance", "troubleshooting"],
                sections=["Архитектура системы", "Компоненты", "API Reference", "Конфигурация", "Развертывание", "Мониторинг", "Безопасность", "Производительность", "Устранение неполадок"]
            ),
            
            DocumentType.API_DOC: DocumentTemplate(
                type=DocumentType.API_DOC,
                title="API_DOCUMENTATION.md",
                content="""# API Documentation: {api_name}

## Обзор
{overview}

## Аутентификация
{authentication}

## Endpoints

### {endpoint_1}
**{method_1}** `{url_1}`

{description_1}

**Параметры:**
{parameters_1}

**Ответ:**
{response_1}

### {endpoint_2}
**{method_2}** `{url_2}`

{description_2}

**Параметры:**
{parameters_2}

**Ответ:**
{response_2}

## Коды ошибок
{error_codes}

## Примеры использования
{usage_examples}""",
                variables=["api_name", "overview", "authentication", "endpoint_1", "method_1", "url_1", "description_1", "parameters_1", "response_1", "endpoint_2", "method_2", "url_2", "description_2", "parameters_2", "response_2", "error_codes", "usage_examples"],
                sections=["Обзор", "Аутентификация", "Endpoints", "Коды ошибок", "Примеры использования"]
            ),
            
            DocumentType.USER_GUIDE: DocumentTemplate(
                type=DocumentType.USER_GUIDE,
                title="USER_GUIDE.md",
                content="""# Руководство пользователя: {product_name}

## Введение
{introduction}

## Быстрый старт
{quick_start}

## Основные функции
{main_features}

## Пошаговые инструкции
{step_by_step}

## Настройки
{settings}

## Горячие клавиши
{hotkeys}

## Часто задаваемые вопросы
{faq}

## Поддержка
{support}""",
                variables=["product_name", "introduction", "quick_start", "main_features", "step_by_step", "settings", "hotkeys", "faq", "support"],
                sections=["Введение", "Быстрый старт", "Основные функции", "Пошаговые инструкции", "Настройки", "Горячие клавиши", "Часто задаваемые вопросы", "Поддержка"]
            ),
            
            DocumentType.CODE_DOC: DocumentTemplate(
                type=DocumentType.CODE_DOC,
                title="CODE_DOCUMENTATION.md",
                content="""# Документация кода: {project_name}

## Структура проекта
{project_structure}

## Основные модули
{main_modules}

## Классы и функции
{classes_functions}

## Алгоритмы
{algorithms}

## Конфигурация
{configuration}

## Тестирование
{testing}

## Стиль кода
{code_style}

## Вклад в проект
{contributing}""",
                variables=["project_name", "project_structure", "main_modules", "classes_functions", "algorithms", "configuration", "testing", "code_style", "contributing"],
                sections=["Структура проекта", "Основные модули", "Классы и функции", "Алгоритмы", "Конфигурация", "Тестирование", "Стиль кода", "Вклад в проект"]
            )
        }
    
    def load_code_templates(self) -> Dict[str, Dict]:
        """Загружает шаблоны кода"""
        return {
            'python': {
                'class': '''class {class_name}:
    """
    {description}
    """
    
    def __init__(self{init_params}):
        """
        Инициализация {class_name}
        
        Args:
            {init_docs}
        """
        {init_body}
    
    def {method_name}(self{method_params}):
        """
        {method_description}
        
        Args:
            {method_docs}
        
        Returns:
            {return_doc}
        """
        {method_body}''',
                
                'function': '''def {function_name}({parameters}):
    """
    {description}
    
    Args:
        {args_doc}
    
    Returns:
        {return_doc}
    """
    {body}''',
                
                'module': '''"""
{module_name}
{description}
"""

import {imports}

{constants}

{classes}

{functions}

if __name__ == "__main__":
    {main_code}'''
            },
            
            'cpp': {
                'class': '''class {class_name} {{
public:
    {class_name}({constructor_params});
    ~{class_name}();
    
    {methods}
    
private:
    {private_members}
}};''',
                
                'function': '''{return_type} {function_name}({parameters}) {{
    {body}
}}''',
                
                'header': '''#ifndef {header_guard}
#define {header_guard}

{includes}

{namespace}

{class_declarations}

{function_declarations}

#endif // {header_guard}'''
            },
            
            'pmac': {
                'program': '''OPEN PROG {program_number}
CLEAR
// {description}

// Инициализация переменных
{variables}

// Основная логика
{logic}

// Завершение программы
CLOSE''',
                
                'motion': '''// {description}
IF ({safety_condition}) THEN
    {motion_command}
ELSE
    // Ошибка безопасности
    M{error_flag} = 1
ENDIF''',
                
                'plc': '''PLC {plc_number}
// {description}

{safety_checks}

{main_logic}

{error_handling}

DISABLE PLC {plc_number}'''
            }
        }
    
    def execute_command(self, command: EditorCommand) -> bool:
        """Выполняет команду редактора"""
        try:
            success = False
            
            if command.action == EditorAction.CREATE_FILE:
                success = self.create_file(command.target, command.content)
            elif command.action == EditorAction.EDIT_FILE:
                success = self.edit_file(command.target, command.content, command.position)
            elif command.action == EditorAction.DELETE_FILE:
                success = self.delete_file(command.target)
            elif command.action == EditorAction.SAVE_FILE:
                success = self.save_file(command.target, command.content)
            elif command.action == EditorAction.OPEN_FILE:
                success = self.open_file(command.target)
            elif command.action == EditorAction.CREATE_FOLDER:
                success = self.create_folder(command.target)
            elif command.action == EditorAction.DELETE_FOLDER:
                success = self.delete_folder(command.target)
            elif command.action == EditorAction.SEARCH_REPLACE:
                success = self.search_replace(command.target, command.parameters)
            elif command.action == EditorAction.INSERT_TEXT:
                success = self.insert_text(command.target, command.content, command.position)
            elif command.action == EditorAction.DELETE_TEXT:
                success = self.delete_text(command.target, command.position, command.parameters)
            elif command.action == EditorAction.FORMAT_CODE:
                success = self.format_code(command.target)
            elif command.action == EditorAction.FIX_CODE:
                success = self.fix_code(command.target)
            elif command.action == EditorAction.GENERATE_CODE:
                success = self.generate_code(command.target, command.parameters)
            elif command.action == EditorAction.CREATE_DOCUMENT:
                success = self.create_document(command.target, command.parameters)
            
            # Сохраняем команду в базу данных
            self.save_command_to_db(command, success)
            
            return success
            
        except Exception as e:
            print(f"Ошибка выполнения команды {command.action.value}: {e}")
            self.save_command_to_db(command, False)
            return False
    
    def create_file(self, file_path: str, content: str = "") -> bool:
        """Создает новый файл"""
        try:
            full_path = os.path.join(self.workspace_path, file_path)
            
            # Создаем директории если нужно
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Создаем файл
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Создан файл: {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка создания файла {file_path}: {e}")
            return False
    
    def edit_file(self, file_path: str, content: str, position: Optional[Tuple[int, int]] = None) -> bool:
        """Редактирует файл"""
        try:
            full_path = os.path.join(self.workspace_path, file_path)
            
            if not os.path.exists(full_path):
                return self.create_file(file_path, content)
            
            # Создаем резервную копию
            backup_path = f"{full_path}.backup.{int(datetime.datetime.now().timestamp())}"
            shutil.copy2(full_path, backup_path)
            
            # Читаем текущее содержимое
            with open(full_path, 'r', encoding='utf-8') as f:
                current_content = f.read()
            
            # Применяем изменения
            if position:
                lines = current_content.split('\n')
                line_idx, col_idx = position
                
                if 0 <= line_idx < len(lines):
                    line = lines[line_idx]
                    new_line = line[:col_idx] + content + line[col_idx:]
                    lines[line_idx] = new_line
                    new_content = '\n'.join(lines)
                else:
                    new_content = current_content + '\n' + content
            else:
                new_content = content
            
            # Сохраняем изменения
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # Сохраняем информацию об операции
            self.save_file_operation("edit", file_path, backup_path, current_content, new_content)
            
            print(f"✅ Отредактирован файл: {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка редактирования файла {file_path}: {e}")
            return False
    
    def delete_file(self, file_path: str) -> bool:
        """Удаляет файл"""
        try:
            full_path = os.path.join(self.workspace_path, file_path)
            
            if not os.path.exists(full_path):
                print(f"⚠️ Файл не существует: {file_path}")
                return False
            
            # Создаем резервную копию
            backup_path = f"{full_path}.deleted.{int(datetime.datetime.now().timestamp())}"
            shutil.copy2(full_path, backup_path)
            
            # Читаем содержимое для записи в БД
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Удаляем файл
            os.remove(full_path)
            
            # Сохраняем информацию об операции
            self.save_file_operation("delete", file_path, backup_path, content, "")
            
            print(f"✅ Удален файл: {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка удаления файла {file_path}: {e}")
            return False
    
    def save_file(self, file_path: str, content: str) -> bool:
        """Сохраняет файл"""
        try:
            full_path = os.path.join(self.workspace_path, file_path)
            
            # Создаем директории если нужно
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Создаем резервную копию если файл существует
            if os.path.exists(full_path):
                backup_path = f"{full_path}.backup.{int(datetime.datetime.now().timestamp())}"
                shutil.copy2(full_path, backup_path)
                
                with open(full_path, 'r', encoding='utf-8') as f:
                    old_content = f.read()
            else:
                old_content = ""
            
            # Сохраняем файл
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Сохраняем информацию об операции
            self.save_file_operation("save", file_path, "", old_content, content)
            
            print(f"✅ Сохранен файл: {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка сохранения файла {file_path}: {e}")
            return False
    
    def open_file(self, file_path: str) -> Optional[str]:
        """Открывает файл и возвращает содержимое"""
        try:
            full_path = os.path.join(self.workspace_path, file_path)
            
            if not os.path.exists(full_path):
                print(f"❌ Файл не существует: {file_path}")
                return None
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"✅ Открыт файл: {file_path}")
            return content
            
        except Exception as e:
            print(f"❌ Ошибка открытия файла {file_path}: {e}")
            return None
    
    def create_folder(self, folder_path: str) -> bool:
        """Создает папку"""
        try:
            full_path = os.path.join(self.workspace_path, folder_path)
            os.makedirs(full_path, exist_ok=True)
            
            print(f"✅ Создана папка: {folder_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка создания папки {folder_path}: {e}")
            return False
    
    def delete_folder(self, folder_path: str) -> bool:
        """Удаляет папку"""
        try:
            full_path = os.path.join(self.workspace_path, folder_path)
            
            if not os.path.exists(full_path):
                print(f"⚠️ Папка не существует: {folder_path}")
                return False
            
            shutil.rmtree(full_path)
            
            print(f"✅ Удалена папка: {folder_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка удаления папки {folder_path}: {e}")
            return False
    
    def search_replace(self, file_path: str, parameters: Dict[str, Any]) -> bool:
        """Поиск и замена в файле"""
        try:
            search_text = parameters.get('search', '')
            replace_text = parameters.get('replace', '')
            case_sensitive = parameters.get('case_sensitive', False)
            
            if not search_text:
                return False
            
            content = self.open_file(file_path)
            if content is None:
                return False
            
            # Выполняем поиск и замену
            if case_sensitive:
                new_content = content.replace(search_text, replace_text)
            else:
                new_content = re.sub(re.escape(search_text), replace_text, content, flags=re.IGNORECASE)
            
            if new_content != content:
                return self.save_file(file_path, new_content)
            else:
                print(f"⚠️ Текст не найден в файле: {file_path}")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка поиска и замены в файле {file_path}: {e}")
            return False
    
    def insert_text(self, file_path: str, text: str, position: Optional[Tuple[int, int]] = None) -> bool:
        """Вставляет текст в файл"""
        try:
            content = self.open_file(file_path)
            if content is None:
                return self.create_file(file_path, text)
            
            if position:
                lines = content.split('\n')
                line_idx, col_idx = position
                
                if 0 <= line_idx < len(lines):
                    line = lines[line_idx]
                    new_line = line[:col_idx] + text + line[col_idx:]
                    lines[line_idx] = new_line
                    new_content = '\n'.join(lines)
                else:
                    new_content = content + '\n' + text
            else:
                new_content = content + '\n' + text
            
            return self.save_file(file_path, new_content)
            
        except Exception as e:
            print(f"❌ Ошибка вставки текста в файл {file_path}: {e}")
            return False
    
    def delete_text(self, file_path: str, position: Optional[Tuple[int, int]] = None, parameters: Optional[Dict[str, Any]] = None) -> bool:
        """Удаляет текст из файла"""
        try:
            content = self.open_file(file_path)
            if content is None:
                return False
            
            if position and parameters:
                lines = content.split('\n')
                line_idx, col_idx = position
                length = parameters.get('length', 1)
                
                if 0 <= line_idx < len(lines):
                    line = lines[line_idx]
                    new_line = line[:col_idx] + line[col_idx + length:]
                    lines[line_idx] = new_line
                    new_content = '\n'.join(lines)
                else:
                    return False
            else:
                return False
            
            return self.save_file(file_path, new_content)
            
        except Exception as e:
            print(f"❌ Ошибка удаления текста из файла {file_path}: {e}")
            return False
    
    def format_code(self, file_path: str) -> bool:
        """Форматирует код в файле"""
        try:
            content = self.open_file(file_path)
            if content is None:
                return False
            
            # Простое форматирование (в реальности здесь будет интеграция с форматтерами)
            formatted_content = self.simple_format_code(content, file_path)
            
            return self.save_file(file_path, formatted_content)
            
        except Exception as e:
            print(f"❌ Ошибка форматирования кода в файле {file_path}: {e}")
            return False
    
    def simple_format_code(self, content: str, file_path: str) -> str:
        """Простое форматирование кода"""
        # Определяем язык по расширению
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.py':
            # Простое форматирование Python
            lines = content.split('\n')
            formatted_lines = []
            indent_level = 0
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    formatted_lines.append('')
                    continue
                
                # Уменьшаем отступ для закрывающих блоков
                if stripped.startswith(('else:', 'elif ', 'except:', 'finally:', 'except ')):
                    indent_level = max(0, indent_level - 1)
                
                # Добавляем отступ
                formatted_line = '    ' * indent_level + stripped
                formatted_lines.append(formatted_line)
                
                # Увеличиваем отступ для открывающих блоков
                if stripped.endswith(':'):
                    indent_level += 1
            
            return '\n'.join(formatted_lines)
        
        elif ext in ['.cpp', '.c', '.h', '.hpp']:
            # Простое форматирование C++
            lines = content.split('\n')
            formatted_lines = []
            indent_level = 0
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    formatted_lines.append('')
                    continue
                
                # Уменьшаем отступ для закрывающих блоков
                if stripped in ('}', '};'):
                    indent_level = max(0, indent_level - 1)
                
                # Добавляем отступ
                formatted_line = '    ' * indent_level + stripped
                formatted_lines.append(formatted_line)
                
                # Увеличиваем отступ для открывающих блоков
                if stripped.endswith('{'):
                    indent_level += 1
            
            return '\n'.join(formatted_lines)
        
        else:
            # Для других языков возвращаем как есть
            return content
    
    def fix_code(self, file_path: str) -> bool:
        """Исправляет код в файле"""
        try:
            content = self.open_file(file_path)
            if content is None:
                return False
            
            # Здесь будет интеграция с системой автоматического исправления
            # Пока простое исправление
            fixed_content = self.simple_fix_code(content, file_path)
            
            return self.save_file(file_path, fixed_content)
            
        except Exception as e:
            print(f"❌ Ошибка исправления кода в файле {file_path}: {e}")
            return False
    
    def simple_fix_code(self, content: str, file_path: str) -> str:
        """Простое исправление кода"""
        # Простые исправления
        fixes = [
            ('import *', 'import specific_function'),
            ('except:', 'except Exception as e:'),
            ('eval(', 'ast.literal_eval('),
        ]
        
        fixed_content = content
        for old, new in fixes:
            fixed_content = fixed_content.replace(old, new)
        
        return fixed_content
    
    def generate_code(self, file_path: str, parameters: Dict[str, Any]) -> bool:
        """Генерирует код в файле"""
        try:
            language = parameters.get('language', 'python')
            code_type = parameters.get('type', 'function')
            template_params = parameters.get('parameters', {})
            
            # Получаем шаблон
            if language in self.code_templates and code_type in self.code_templates[language]:
                template = self.code_templates[language][code_type]
                
                # Заполняем шаблон
                generated_code = template.format(**template_params)
                
                return self.save_file(file_path, generated_code)
            else:
                print(f"❌ Неизвестный тип кода: {language}.{code_type}")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка генерации кода в файле {file_path}: {e}")
            return False
    
    def create_document(self, file_path: str, parameters: Dict[str, Any]) -> bool:
        """Создает документ"""
        try:
            doc_type = DocumentType(parameters.get('type', 'readme'))
            doc_params = parameters.get('parameters', {})
            
            # Получаем шаблон
            if doc_type in self.document_templates:
                template = self.document_templates[doc_type]
                
                # Заполняем шаблон
                content = template.content.format(**doc_params)
                
                # Создаем файл
                success = self.save_file(file_path, content)
                
                if success:
                    # Сохраняем информацию о документе
                    self.save_document_to_db(doc_type, parameters.get('title', ''), file_path, content)
                
                return success
            else:
                print(f"❌ Неизвестный тип документа: {doc_type}")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка создания документа {file_path}: {e}")
            return False
    
    def save_command_to_db(self, command: EditorCommand, success: bool):
        """Сохраняет команду в базу данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO editor_commands 
                (id, action, target, content, position, parameters, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                command.id,
                command.action.value,
                command.target,
                command.content,
                json.dumps(command.position) if command.position else None,
                json.dumps(command.parameters) if command.parameters else None,
                success
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Ошибка сохранения команды в БД: {e}")
    
    def save_file_operation(self, operation_type: str, file_path: str, backup_path: str, content_before: str, content_after: str):
        """Сохраняет информацию об операции с файлом"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            operation_id = f"{operation_type}_{int(datetime.datetime.now().timestamp())}"
            
            cursor.execute('''
                INSERT INTO file_operations 
                (id, operation_type, file_path, backup_path, content_before, content_after)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                operation_id,
                operation_type,
                file_path,
                backup_path,
                content_before,
                content_after
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Ошибка сохранения операции с файлом: {e}")
    
    def save_document_to_db(self, doc_type: DocumentType, title: str, file_path: str, content: str):
        """Сохраняет информацию о документе в базу данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            doc_id = f"doc_{int(datetime.datetime.now().timestamp())}"
            
            cursor.execute('''
                INSERT INTO created_documents 
                (id, document_type, title, file_path, content)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                doc_id,
                doc_type.value,
                title,
                file_path,
                content
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Ошибка сохранения документа в БД: {e}")
    
    def get_workspace_status(self) -> Dict[str, Any]:
        """Получает статус рабочего пространства"""
        try:
            files = []
            folders = []
            
            for root, dirs, filenames in os.walk(self.workspace_path):
                for dirname in dirs:
                    rel_path = os.path.relpath(os.path.join(root, dirname), self.workspace_path)
                    folders.append(rel_path)
                
                for filename in filenames:
                    rel_path = os.path.relpath(os.path.join(root, filename), self.workspace_path)
                    files.append(rel_path)
            
            return {
                'workspace_path': self.workspace_path,
                'files_count': len(files),
                'folders_count': len(folders),
                'files': files[:10],  # Первые 10 файлов
                'folders': folders[:10]  # Первые 10 папок
            }
            
        except Exception as e:
            print(f"Ошибка получения статуса рабочего пространства: {e}")
            return {}

def main():
    """Основная функция для тестирования"""
    print("🎮 SMART RUBIN AI - КОНТРОЛЛЕР РЕДАКТОРА")
    print("=" * 60)
    
    # Создаем контроллер редактора
    controller = RubinEditorController()
    
    # Тестируем создание файла
    print("\n📝 Тестируем создание файла...")
    command = EditorCommand(
        id="test_1",
        action=EditorAction.CREATE_FILE,
        target="test_file.py",
        content="""# Тестовый файл
def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()"""
    )
    
    success = controller.execute_command(command)
    print(f"Результат: {'✅ Успешно' if success else '❌ Ошибка'}")
    
    # Тестируем редактирование файла
    print("\n✏️ Тестируем редактирование файла...")
    command = EditorCommand(
        id="test_2",
        action=EditorAction.EDIT_FILE,
        target="test_file.py",
        content="\n# Добавленный комментарий",
        position=(0, 0)
    )
    
    success = controller.execute_command(command)
    print(f"Результат: {'✅ Успешно' if success else '❌ Ошибка'}")
    
    # Тестируем создание документа
    print("\n📄 Тестируем создание документа...")
    command = EditorCommand(
        id="test_3",
        action=EditorAction.CREATE_DOCUMENT,
        target="README.md",
        parameters={
            'type': 'readme',
            'title': 'Test Project',
            'parameters': {
                'project_name': 'Test Project',
                'description': 'Это тестовый проект для демонстрации возможностей Smart Rubin AI',
                'features': '- Автоматическое создание файлов\n- Редактирование кода\n- Создание документов',
                'installation': 'pip install test-project',
                'usage': 'python main.py',
                'examples': 'Примеры использования...',
                'api_docs': 'API документация...',
                'license': 'MIT',
                'author': 'Smart Rubin AI'
            }
        }
    )
    
    success = controller.execute_command(command)
    print(f"Результат: {'✅ Успешно' if success else '❌ Ошибка'}")
    
    # Тестируем генерацию кода
    print("\n🎯 Тестируем генерацию кода...")
    command = EditorCommand(
        id="test_4",
        action=EditorAction.GENERATE_CODE,
        target="generated_class.py",
        parameters={
            'language': 'python',
            'type': 'class',
            'parameters': {
                'class_name': 'TestClass',
                'description': 'Тестовый класс',
                'init_params': ', value=0',
                'init_docs': 'value: начальное значение',
                'init_body': 'self.value = value',
                'method_name': 'get_value',
                'method_params': '',
                'method_description': 'Получить значение',
                'method_docs': '',
                'return_doc': 'int: текущее значение',
                'method_body': 'return self.value'
            }
        }
    )
    
    success = controller.execute_command(command)
    print(f"Результат: {'✅ Успешно' if success else '❌ Ошибка'}")
    
    # Показываем статус рабочего пространства
    print("\n📊 Статус рабочего пространства:")
    status = controller.get_workspace_status()
    print(f"   Файлов: {status.get('files_count', 0)}")
    print(f"   Папок: {status.get('folders_count', 0)}")
    print(f"   Путь: {status.get('workspace_path', 'N/A')}")
    
    print("\n🎉 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")

if __name__ == "__main__":
    main()
