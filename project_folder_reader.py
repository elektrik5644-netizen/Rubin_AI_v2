#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Система чтения папок других проектов для Rubin AI
Позволяет Rubin AI изучать и анализировать код и документацию других проектов
"""

import os
import sys
import sqlite3
import hashlib
import mimetypes
import json
import re
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import shutil

class ProjectFolderReader:
    """Система чтения и анализа папок других проектов"""
    
    def __init__(self, db_path="rubin_project_knowledge.db"):
        self.db_path = db_path
        self.setup_logging()
        self.setup_database()
        
        # Поддерживаемые типы файлов для анализа
        self.supported_extensions = {
            # Код
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go', '.rs',
            '.swift', '.kt', '.scala', '.r', '.m', '.pl', '.sh', '.bash', '.ps1',
            # Веб
            '.html', '.htm', '.css', '.scss', '.sass', '.less', '.vue', '.jsx', '.tsx',
            # Конфигурация
            '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
            # Документация
            '.md', '.rst', '.txt', '.doc', '.docx', '.pdf', '.rtf',
            # Данные
            '.csv', '.sql', '.db', '.sqlite', '.sqlite3',
            # Другие
            '.dockerfile', '.gitignore', '.env', '.gitattributes'
        }
        
        # Исключаемые папки
        self.excluded_dirs = {
            '__pycache__', '.git', '.svn', '.hg', 'node_modules', '.vscode', 
            '.idea', 'venv', 'env', '.env', 'build', 'dist', 'target',
            '.pytest_cache', '.coverage', 'coverage', '.nyc_output',
            'logs', 'log', 'tmp', 'temp', '.DS_Store', 'Thumbs.db'
        }
        
        # Исключаемые файлы
        self.excluded_files = {
            '.gitignore', '.gitattributes', '.DS_Store', 'Thumbs.db',
            'package-lock.json', 'yarn.lock', 'Pipfile.lock'
        }
        
    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('project_reader.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_database(self):
        """Создание таблиц базы данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Таблица проектов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_name TEXT NOT NULL,
                    project_path TEXT UNIQUE NOT NULL,
                    project_type TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_analyzed TIMESTAMP,
                    total_files INTEGER DEFAULT 0,
                    total_size INTEGER DEFAULT 0,
                    language_stats TEXT,  -- JSON статистика языков
                    framework_stats TEXT, -- JSON статистика фреймворков
                    UNIQUE(project_path)
                )
            ''')
            
            # Таблица файлов проекта
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS project_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_extension TEXT,
                    file_size INTEGER,
                    file_type TEXT,
                    content_hash TEXT,
                    content_preview TEXT,  -- Первые 1000 символов
                    full_content TEXT,     -- Полное содержимое для небольших файлов
                    language TEXT,
                    framework TEXT,
                    complexity_score INTEGER DEFAULT 0,
                    line_count INTEGER DEFAULT 0,
                    function_count INTEGER DEFAULT 0,
                    class_count INTEGER DEFAULT 0,
                    import_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id),
                    UNIQUE(project_id, file_path)
                )
            ''')
            
            # Таблица архитектурных компонентов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS project_components (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    component_type TEXT NOT NULL,  -- 'class', 'function', 'module', 'api', 'database'
                    component_name TEXT NOT NULL,
                    file_id INTEGER,
                    line_start INTEGER,
                    line_end INTEGER,
                    description TEXT,
                    dependencies TEXT,  -- JSON список зависимостей
                    complexity_score INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id),
                    FOREIGN KEY (file_id) REFERENCES project_files (id)
                )
            ''')
            
            # Таблица знаний и паттернов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS project_knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    knowledge_type TEXT NOT NULL,  -- 'pattern', 'architecture', 'best_practice', 'anti_pattern'
                    title TEXT NOT NULL,
                    description TEXT,
                    code_example TEXT,
                    file_id INTEGER,
                    line_start INTEGER,
                    line_end INTEGER,
                    tags TEXT,  -- JSON список тегов
                    confidence_score REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id),
                    FOREIGN KEY (file_id) REFERENCES project_files (id)
                )
            ''')
            
            # Индекс для быстрого поиска
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_project_files_path 
                ON project_files(project_id, file_path)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_project_components_type 
                ON project_components(project_id, component_type)
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("✅ База данных проектов инициализирована")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации базы данных: {e}")
            
    def analyze_project_folder(self, project_path: str, project_name: str = None) -> bool:
        """Анализ папки проекта"""
        try:
            project_path = Path(project_path)
            
            if not project_path.exists():
                self.logger.error(f"❌ Путь проекта не найден: {project_path}")
                return False
                
            if not project_path.is_dir():
                self.logger.error(f"❌ Путь не является папкой: {project_path}")
                return False
                
            # Определяем имя проекта
            if not project_name:
                project_name = project_path.name
                
            self.logger.info(f"🔍 Начинаем анализ проекта: {project_name}")
            self.logger.info(f"📁 Путь: {project_path}")
            
            # Проверяем, не анализировали ли мы уже этот проект
            project_id = self.get_project_id(project_path)
            if project_id:
                self.logger.info(f"📋 Проект уже существует в базе (ID: {project_id})")
                # Обновляем время последнего анализа
                self.update_project_analysis_time(project_id)
            else:
                # Создаем новый проект
                project_id = self.create_project_record(project_path, project_name)
                if not project_id:
                    return False
                    
            # Анализируем файлы проекта
            analysis_result = self.analyze_project_files(project_id, project_path)
            
            # Обновляем статистику проекта
            self.update_project_stats(project_id, analysis_result)
            
            # Извлекаем архитектурные компоненты
            self.extract_architectural_components(project_id)
            
            # Извлекаем знания и паттерны
            self.extract_project_knowledge(project_id)
            
            self.logger.info(f"✅ Анализ проекта {project_name} завершен")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка анализа проекта: {e}")
            return False
            
    def get_project_id(self, project_path: Path) -> Optional[int]:
        """Получение ID проекта по пути"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT id FROM projects WHERE project_path = ?",
                (str(project_path),)
            )
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            self.logger.error(f"Ошибка получения ID проекта: {e}")
            return None
            
    def create_project_record(self, project_path: Path, project_name: str) -> Optional[int]:
        """Создание записи проекта"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Определяем тип проекта
            project_type = self.detect_project_type(project_path)
            
            cursor.execute('''
                INSERT INTO projects (project_name, project_path, project_type, description)
                VALUES (?, ?, ?, ?)
            ''', (project_name, str(project_path), project_type, f"Проект {project_name}"))
            
            project_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            self.logger.info(f"📋 Создана запись проекта: {project_name} (ID: {project_id})")
            return project_id
            
        except Exception as e:
            self.logger.error(f"Ошибка создания проекта: {e}")
            return None
            
    def detect_project_type(self, project_path: Path) -> str:
        """Определение типа проекта по структуре и файлам"""
        indicators = {
            'Python': ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile', '__init__.py'],
            'Node.js': ['package.json', 'package-lock.json', 'yarn.lock', 'node_modules'],
            'Java': ['pom.xml', 'build.gradle', 'Maven', 'src/main/java'],
            'C++': ['CMakeLists.txt', 'Makefile', '*.cpp', '*.h'],
            'C#': ['*.csproj', '*.sln', 'Program.cs'],
            'PHP': ['composer.json', 'index.php', '*.php'],
            'Ruby': ['Gemfile', 'Rakefile', '*.rb'],
            'Go': ['go.mod', 'go.sum', 'main.go'],
            'Rust': ['Cargo.toml', 'Cargo.lock', 'src/main.rs'],
            'Docker': ['Dockerfile', 'docker-compose.yml'],
            'Web': ['index.html', 'style.css', 'app.js'],
            'React': ['src/App.js', 'src/App.jsx', 'public/index.html'],
            'Vue': ['src/App.vue', 'vue.config.js'],
            'Angular': ['angular.json', 'src/app/app.component.ts']
        }
        
        scores = {}
        
        # Проверяем наличие индикаторных файлов
        for project_type, indicators_list in indicators.items():
            score = 0
            for indicator in indicators_list:
                if '*' in indicator:
                    # Паттерн файлов
                    pattern = indicator.replace('*', '')
                    for file_path in project_path.rglob(f'*{pattern}'):
                        if file_path.is_file():
                            score += 1
                else:
                    # Конкретный файл или папка
                    indicator_path = project_path / indicator
                    if indicator_path.exists():
                        score += 1
                        
            scores[project_type] = score
            
        # Возвращаем тип с наибольшим счетом
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] > 0:
                return best_type
                
        return 'Unknown'
        
    def analyze_project_files(self, project_id: int, project_path: Path) -> Dict:
        """Анализ файлов проекта"""
        self.logger.info(f"📁 Анализируем файлы проекта (ID: {project_id})")
        
        total_files = 0
        analyzed_files = 0
        total_size = 0
        language_stats = {}
        framework_stats = {}
        
        # Рекурсивный обход файлов
        for file_path in project_path.rglob('*'):
            if file_path.is_file():
                # Проверяем исключения
                if self.should_skip_file(file_path):
                    continue
                    
                total_files += 1
                
                try:
                    # Анализируем файл
                    file_info = self.analyze_single_file(project_id, file_path)
                    
                    if file_info:
                        analyzed_files += 1
                        total_size += file_info.get('file_size', 0)
                        
                        # Обновляем статистику языков
                        language = file_info.get('language', 'Unknown')
                        language_stats[language] = language_stats.get(language, 0) + 1
                        
                        # Обновляем статистику фреймворков
                        framework = file_info.get('framework', 'None')
                        if framework != 'None':
                            framework_stats[framework] = framework_stats.get(framework, 0) + 1
                            
                except Exception as e:
                    self.logger.error(f"Ошибка анализа файла {file_path}: {e}")
                    
        self.logger.info(f"📊 Проанализировано {analyzed_files} из {total_files} файлов")
        
        return {
            'total_files': total_files,
            'analyzed_files': analyzed_files,
            'total_size': total_size,
            'language_stats': language_stats,
            'framework_stats': framework_stats
        }
        
    def should_skip_file(self, file_path: Path) -> bool:
        """Проверка, нужно ли пропустить файл"""
        # Проверяем расширение
        if file_path.suffix.lower() not in self.supported_extensions:
            return True
            
        # Проверяем исключаемые папки
        for part in file_path.parts:
            if part in self.excluded_dirs:
                return True
                
        # Проверяем исключаемые файлы
        if file_path.name in self.excluded_files:
            return True
            
        # Проверяем размер файла (пропускаем файлы больше 10MB)
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:
                return True
        except:
            pass
            
        return False
        
    def analyze_single_file(self, project_id: int, file_path: Path) -> Optional[Dict]:
        """Анализ одного файла"""
        try:
            # Получаем информацию о файле
            file_size = file_path.stat().st_size
            file_extension = file_path.suffix.lower()
            file_type = mimetypes.guess_type(str(file_path))[0] or 'unknown'
            
            # Читаем содержимое файла
            content = self.read_file_content(file_path)
            if content is None:
                return None
                
            # Создаем хеш содержимого
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            # Определяем язык программирования
            language = self.detect_programming_language(file_extension, content)
            
            # Определяем фреймворк
            framework = self.detect_framework(content, language)
            
            # Анализируем сложность кода
            complexity_analysis = self.analyze_code_complexity(content, language)
            
            # Подготавливаем превью содержимого
            content_preview = content[:1000] if len(content) > 1000 else content
            
            # Сохраняем полное содержимое только для небольших файлов
            full_content = content if len(content) < 50000 else None
            
            # Сохраняем в базу данных
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO project_files (
                    project_id, file_path, file_name, file_extension, file_size,
                    file_type, content_hash, content_preview, full_content,
                    language, framework, complexity_score, line_count,
                    function_count, class_count, import_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                project_id, str(file_path), file_path.name, file_extension,
                file_size, file_type, content_hash, content_preview, full_content,
                language, framework, complexity_analysis['complexity_score'],
                complexity_analysis['line_count'], complexity_analysis['function_count'],
                complexity_analysis['class_count'], complexity_analysis['import_count']
            ))
            
            conn.commit()
            conn.close()
            
            return {
                'file_size': file_size,
                'language': language,
                'framework': framework,
                'complexity_score': complexity_analysis['complexity_score'],
                'line_count': complexity_analysis['line_count']
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа файла {file_path}: {e}")
            return None
            
    def read_file_content(self, file_path: Path) -> Optional[str]:
        """Чтение содержимого файла с обработкой различных кодировок"""
        try:
            # Пробуем разные кодировки
            encodings = ['utf-8', 'utf-8-sig', 'cp1251', 'latin1', 'ascii']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        content = f.read()
                        return content
                except UnicodeDecodeError:
                    continue
                    
            # Если ничего не сработало, читаем как бинарный файл
            with open(file_path, 'rb') as f:
                content = f.read()
                return content.decode('utf-8', errors='ignore')
                
        except Exception as e:
            self.logger.error(f"Ошибка чтения файла {file_path}: {e}")
            return None
            
    def detect_programming_language(self, file_extension: str, content: str) -> str:
        """Определение языка программирования"""
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.cs': 'C#',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.go': 'Go',
            '.rs': 'Rust',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.r': 'R',
            '.m': 'Objective-C',
            '.pl': 'Perl',
            '.sh': 'Shell',
            '.bash': 'Bash',
            '.ps1': 'PowerShell',
            '.html': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.sass': 'Sass',
            '.less': 'Less',
            '.vue': 'Vue',
            '.jsx': 'JSX',
            '.tsx': 'TSX',
            '.json': 'JSON',
            '.xml': 'XML',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.toml': 'TOML',
            '.ini': 'INI',
            '.cfg': 'Config',
            '.conf': 'Config',
            '.md': 'Markdown',
            '.rst': 'reStructuredText',
            '.txt': 'Text',
            '.sql': 'SQL',
            '.dockerfile': 'Dockerfile',
            '.env': 'Environment'
        }
        
        return language_map.get(file_extension, 'Unknown')
        
    def detect_framework(self, content: str, language: str) -> str:
        """Определение фреймворка по содержимому"""
        frameworks = {
            'Python': {
                'Django': ['django', 'from django', 'import django'],
                'Flask': ['flask', 'from flask', 'import flask'],
                'FastAPI': ['fastapi', 'from fastapi', 'import fastapi'],
                'PyTorch': ['torch', 'import torch', 'from torch'],
                'TensorFlow': ['tensorflow', 'import tensorflow', 'from tensorflow'],
                'Pandas': ['pandas', 'import pandas', 'from pandas'],
                'NumPy': ['numpy', 'import numpy', 'from numpy']
            },
            'JavaScript': {
                'React': ['react', 'import React', 'from react'],
                'Vue': ['vue', 'import Vue', 'from vue'],
                'Angular': ['angular', 'import angular', 'from angular'],
                'Express': ['express', 'import express', 'from express'],
                'Node.js': ['node', 'require(', 'module.exports']
            },
            'Java': {
                'Spring': ['spring', 'import org.springframework'],
                'Hibernate': ['hibernate', 'import org.hibernate'],
                'Maven': ['maven', 'pom.xml']
            }
        }
        
        if language in frameworks:
            content_lower = content.lower()
            for framework, indicators in frameworks[language].items():
                for indicator in indicators:
                    if indicator in content_lower:
                        return framework
                        
        return 'None'
        
    def analyze_code_complexity(self, content: str, language: str) -> Dict:
        """Анализ сложности кода"""
        lines = content.split('\n')
        line_count = len(lines)
        
        # Подсчет функций, классов и импортов
        function_count = 0
        class_count = 0
        import_count = 0
        
        if language == 'Python':
            function_count = len(re.findall(r'^\s*def\s+\w+', content, re.MULTILINE))
            class_count = len(re.findall(r'^\s*class\s+\w+', content, re.MULTILINE))
            import_count = len(re.findall(r'^\s*(import|from)\s+', content, re.MULTILINE))
        elif language in ['JavaScript', 'TypeScript']:
            function_count = len(re.findall(r'function\s+\w+|const\s+\w+\s*=\s*\(|let\s+\w+\s*=\s*\(', content))
            class_count = len(re.findall(r'class\s+\w+', content))
            import_count = len(re.findall(r'import\s+', content))
        elif language == 'Java':
            function_count = len(re.findall(r'public\s+\w+\s+\w+\s*\(|private\s+\w+\s+\w+\s*\(', content))
            class_count = len(re.findall(r'public\s+class\s+\w+|private\s+class\s+\w+', content))
            import_count = len(re.findall(r'import\s+', content))
            
        # Простая оценка сложности
        complexity_score = min(100, (function_count * 2 + class_count * 3 + import_count) // 10)
        
        return {
            'line_count': line_count,
            'function_count': function_count,
            'class_count': class_count,
            'import_count': import_count,
            'complexity_score': complexity_score
        }
        
    def extract_architectural_components(self, project_id: int):
        """Извлечение архитектурных компонентов проекта"""
        self.logger.info(f"🏗️ Извлекаем архитектурные компоненты проекта {project_id}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Получаем все файлы проекта
            cursor.execute('''
                SELECT id, file_path, language, full_content 
                FROM project_files 
                WHERE project_id = ? AND language IN ('Python', 'JavaScript', 'TypeScript', 'Java', 'C++', 'C#')
            ''', (project_id,))
            
            files = cursor.fetchall()
            
            for file_id, file_path, language, content in files:
                if not content:
                    continue
                    
                # Извлекаем компоненты в зависимости от языка
                components = self.extract_components_from_file(content, language, file_path)
                
                for component in components:
                    cursor.execute('''
                        INSERT INTO project_components (
                            project_id, component_type, component_name, file_id,
                            line_start, line_end, description, complexity_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        project_id, component['type'], component['name'], file_id,
                        component['line_start'], component['line_end'], 
                        component['description'], component['complexity']
                    ))
                    
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Извлечено архитектурных компонентов для проекта {project_id}")
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения компонентов: {e}")
            
    def extract_components_from_file(self, content: str, language: str, file_path: str) -> List[Dict]:
        """Извлечение компонентов из файла"""
        components = []
        lines = content.split('\n')
        
        if language == 'Python':
            # Классы
            for i, line in enumerate(lines):
                class_match = re.match(r'^\s*class\s+(\w+)', line)
                if class_match:
                    class_name = class_match.group(1)
                    components.append({
                        'type': 'class',
                        'name': class_name,
                        'line_start': i + 1,
                        'line_end': self.find_class_end(lines, i),
                        'description': f"Класс {class_name}",
                        'complexity': self.calculate_class_complexity(lines, i)
                    })
                    
            # Функции
            for i, line in enumerate(lines):
                func_match = re.match(r'^\s*def\s+(\w+)', line)
                if func_match:
                    func_name = func_match.group(1)
                    components.append({
                        'type': 'function',
                        'name': func_name,
                        'line_start': i + 1,
                        'line_end': self.find_function_end(lines, i),
                        'description': f"Функция {func_name}",
                        'complexity': self.calculate_function_complexity(lines, i)
                    })
                    
        elif language in ['JavaScript', 'TypeScript']:
            # Классы
            for i, line in enumerate(lines):
                class_match = re.match(r'^\s*class\s+(\w+)', line)
                if class_match:
                    class_name = class_match.group(1)
                    components.append({
                        'type': 'class',
                        'name': class_name,
                        'line_start': i + 1,
                        'line_end': self.find_js_class_end(lines, i),
                        'description': f"Класс {class_name}",
                        'complexity': 5  # Простая оценка
                    })
                    
        return components
        
    def find_class_end(self, lines: List[str], start_line: int) -> int:
        """Поиск конца класса в Python"""
        indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
        
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip() == '':
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level:
                return i
                
        return len(lines)
        
    def find_function_end(self, lines: List[str], start_line: int) -> int:
        """Поиск конца функции в Python"""
        return self.find_class_end(lines, start_line)  # Аналогичная логика
        
    def find_js_class_end(self, lines: List[str], start_line: int) -> int:
        """Поиск конца класса в JavaScript"""
        brace_count = 0
        in_class = False
        
        for i in range(start_line, len(lines)):
            line = lines[i]
            if '{' in line:
                brace_count += line.count('{')
                in_class = True
            if '}' in line:
                brace_count -= line.count('}')
                if in_class and brace_count == 0:
                    return i + 1
                    
        return len(lines)
        
    def calculate_class_complexity(self, lines: List[str], start_line: int) -> int:
        """Расчет сложности класса"""
        # Простая эвристика: количество методов и атрибутов
        complexity = 0
        indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
        
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip() == '':
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level:
                break
            if re.match(r'^\s*def\s+', line):
                complexity += 2
            elif re.match(r'^\s*self\.\w+', line):
                complexity += 1
                
        return min(complexity, 20)
        
    def calculate_function_complexity(self, lines: List[str], start_line: int) -> int:
        """Расчет сложности функции"""
        # Простая эвристика: количество условий и циклов
        complexity = 1  # Базовая сложность
        indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
        
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip() == '':
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level:
                break
            if re.search(r'\b(if|elif|else|for|while|try|except|with)\b', line):
                complexity += 1
                
        return min(complexity, 10)
        
    def extract_project_knowledge(self, project_id: int):
        """Извлечение знаний и паттернов из проекта"""
        self.logger.info(f"🧠 Извлекаем знания из проекта {project_id}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Получаем все файлы проекта
            cursor.execute('''
                SELECT id, file_path, language, full_content 
                FROM project_files 
                WHERE project_id = ? AND full_content IS NOT NULL
            ''', (project_id,))
            
            files = cursor.fetchall()
            
            for file_id, file_path, language, content in files:
                # Извлекаем паттерны проектирования
                patterns = self.extract_design_patterns(content, language)
                
                for pattern in patterns:
                    cursor.execute('''
                        INSERT INTO project_knowledge (
                            project_id, knowledge_type, title, description,
                            code_example, file_id, line_start, line_end,
                            tags, confidence_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        project_id, 'pattern', pattern['title'], pattern['description'],
                        pattern['code_example'], file_id, pattern['line_start'],
                        pattern['line_end'], json.dumps(pattern['tags']), pattern['confidence']
                    ))
                    
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Извлечено знаний из проекта {project_id}")
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения знаний: {e}")
            
    def extract_design_patterns(self, content: str, language: str) -> List[Dict]:
        """Извлечение паттернов проектирования"""
        patterns = []
        lines = content.split('\n')
        
        # Singleton паттерн
        if language == 'Python':
            singleton_pattern = re.search(r'class\s+(\w+).*?__new__.*?if.*?instance.*?return.*?instance', content, re.DOTALL)
            if singleton_pattern:
                patterns.append({
                    'title': 'Singleton Pattern',
                    'description': 'Реализация паттерна Singleton',
                    'code_example': singleton_pattern.group(0),
                    'line_start': 1,
                    'line_end': len(lines),
                    'tags': ['design_pattern', 'singleton', 'creational'],
                    'confidence': 0.8
                })
                
        # Factory паттерн
        factory_pattern = re.search(r'class\s+(\w+Factory)', content)
        if factory_pattern:
            patterns.append({
                'title': 'Factory Pattern',
                'description': 'Реализация паттерна Factory',
                'code_example': factory_pattern.group(0),
                'line_start': 1,
                'line_end': len(lines),
                'tags': ['design_pattern', 'factory', 'creational'],
                'confidence': 0.7
            })
            
        return patterns
        
    def update_project_stats(self, project_id: int, analysis_result: Dict):
        """Обновление статистики проекта"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE projects SET 
                    total_files = ?, 
                    total_size = ?,
                    language_stats = ?,
                    framework_stats = ?,
                    last_analyzed = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (
                analysis_result['analyzed_files'],
                analysis_result['total_size'],
                json.dumps(analysis_result['language_stats']),
                json.dumps(analysis_result['framework_stats']),
                project_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления статистики: {e}")
            
    def update_project_analysis_time(self, project_id: int):
        """Обновление времени последнего анализа"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE projects SET last_analyzed = CURRENT_TIMESTAMP WHERE id = ?
            ''', (project_id,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления времени анализа: {e}")
            
    def get_project_summary(self, project_id: int) -> Optional[Dict]:
        """Получение сводки по проекту"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Информация о проекте
            cursor.execute('SELECT * FROM projects WHERE id = ?', (project_id,))
            project = cursor.fetchone()
            
            if not project:
                return None
                
            # Статистика файлов
            cursor.execute('''
                SELECT COUNT(*), SUM(file_size), AVG(complexity_score)
                FROM project_files WHERE project_id = ?
            ''', (project_id,))
            file_stats = cursor.fetchone()
            
            # Статистика компонентов
            cursor.execute('''
                SELECT component_type, COUNT(*) 
                FROM project_components 
                WHERE project_id = ? 
                GROUP BY component_type
            ''', (project_id,))
            component_stats = cursor.fetchall()
            
            # Статистика знаний
            cursor.execute('''
                SELECT knowledge_type, COUNT(*) 
                FROM project_knowledge 
                WHERE project_id = ? 
                GROUP BY knowledge_type
            ''', (project_id,))
            knowledge_stats = cursor.fetchall()
            
            conn.close()
            
            return {
                'project': project,
                'file_stats': file_stats,
                'component_stats': component_stats,
                'knowledge_stats': knowledge_stats
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка получения сводки проекта: {e}")
            return None
            
    def search_in_projects(self, query: str, project_ids: List[int] = None) -> List[Dict]:
        """Поиск по проектам"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Поиск в файлах
            if project_ids:
                placeholders = ','.join('?' * len(project_ids))
                cursor.execute(f'''
                    SELECT pf.*, p.project_name 
                    FROM project_files pf
                    JOIN projects p ON pf.project_id = p.id
                    WHERE pf.project_id IN ({placeholders})
                    AND (pf.content_preview LIKE ? OR pf.file_name LIKE ?)
                ''', project_ids + [f'%{query}%', f'%{query}%'])
            else:
                cursor.execute('''
                    SELECT pf.*, p.project_name 
                    FROM project_files pf
                    JOIN projects p ON pf.project_id = p.id
                    WHERE pf.content_preview LIKE ? OR pf.file_name LIKE ?
                ''', (f'%{query}%', f'%{query}%'))
                
            results = cursor.fetchall()
            conn.close()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Ошибка поиска: {e}")
            return []
            
    def get_all_projects(self) -> List[Dict]:
        """Получение списка всех проектов"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT p.*, 
                       COUNT(pf.id) as file_count,
                       SUM(pf.file_size) as total_size
                FROM projects p
                LEFT JOIN project_files pf ON p.id = pf.project_id
                GROUP BY p.id
                ORDER BY p.last_analyzed DESC
            ''')
            
            projects = cursor.fetchall()
            conn.close()
            
            return projects
            
        except Exception as e:
            self.logger.error(f"Ошибка получения проектов: {e}")
            return []


def main():
    """Главная функция для тестирования"""
    print("🔍 СИСТЕМА ЧТЕНИЯ ПАПОК ДРУГИХ ПРОЕКТОВ ДЛЯ RUBIN AI")
    print("=" * 60)
    
    reader = ProjectFolderReader()
    
    # Пример использования
    test_project_path = input("Введите путь к проекту для анализа: ").strip()
    
    if not test_project_path:
        print("❌ Путь не указан")
        return
        
    if not os.path.exists(test_project_path):
        print(f"❌ Путь не найден: {test_project_path}")
        return
        
    # Анализируем проект
    success = reader.analyze_project_folder(test_project_path)
    
    if success:
        print("✅ Проект успешно проанализирован!")
        
        # Получаем список всех проектов
        projects = reader.get_all_projects()
        print(f"\n📋 Всего проектов в базе: {len(projects)}")
        
        for project in projects:
            print(f"  - {project[1]} ({project[2]}) - {project[8]} файлов")
    else:
        print("❌ Ошибка анализа проекта")


if __name__ == "__main__":
    main()





