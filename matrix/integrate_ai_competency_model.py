#!/usr/bin/env python3
"""
Интеграция компетентностной модели ИИ от ИТМО в Rubin AI
"""

import os
import json
import re
from pathlib import Path

class AICompetencyIntegrator:
    def __init__(self, competency_model_path):
        self.competency_model_path = Path(competency_model_path)
        self.ai_roles = {}
        self.ai_competencies = {}
        self.ai_learning_paths = {}
        
    def extract_ai_roles(self):
        """Извлечение профессиональных ролей в области ИИ"""
        print("🔍 Извлечение профессиональных ролей ИИ...")
        
        jobs_dir = self.competency_model_path / "Jobs"
        if not jobs_dir.exists():
            print("❌ Папка Jobs не найдена")
            return
            
        for job_file in jobs_dir.glob("*.md"):
            role_name = job_file.stem.replace("job", "")
            print(f"📋 Обработка роли: {role_name}")
            
            with open(job_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Извлекаем компетенции
            competencies = self._extract_competencies_from_content(content)
            
            self.ai_roles[role_name] = {
                "name": role_name,
                "file": str(job_file),
                "competencies": competencies,
                "description": self._extract_description(content)
            }
            
        print(f"✅ Извлечено {len(self.ai_roles)} ролей")
        
    def _extract_competencies_from_content(self, content):
        """Извлечение компетенций из содержимого файла"""
        competencies = []
        
        # Ищем заголовки компетенций
        competency_pattern = r'### (\d+)\. (.+?)(?=###|\Z)'
        matches = re.findall(competency_pattern, content, re.DOTALL)
        
        for i, (num, competency_text) in enumerate(matches):
            competency = {
                "number": int(num),
                "title": self._extract_competency_title(competency_text),
                "description": self._extract_competency_description(competency_text),
                "indicators": self._extract_indicators(competency_text)
            }
            competencies.append(competency)
            
        return competencies
        
    def _extract_competency_title(self, text):
        """Извлечение заголовка компетенции"""
        lines = text.strip().split('\n')
        if lines:
            return lines[0].strip()
        return ""
        
    def _extract_competency_description(self, text):
        """Извлечение описания компетенции"""
        # Ищем секцию "Трудовая функция"
        function_match = re.search(r'#### "Трудовая функция"\s*\n(.+?)(?=####|\Z)', text, re.DOTALL)
        if function_match:
            return function_match.group(1).strip()
        return ""
        
    def _extract_indicators(self, text):
        """Извлечение индикаторов компетенции"""
        indicators = []
        
        # Ищем секцию "Индикаторы"
        indicators_match = re.search(r'#### Индикаторы\s*\n(.+?)(?=####|\Z)', text, re.DOTALL)
        if indicators_match:
            indicators_text = indicators_match.group(1)
            # Разделяем по маркерам списка
            indicator_lines = re.findall(r'\* (.+?)(?=\n\*|\Z)', indicators_text, re.DOTALL)
            indicators = [line.strip() for line in indicator_lines if line.strip()]
            
        return indicators
        
    def _extract_description(self, content):
        """Извлечение общего описания роли"""
        lines = content.split('\n')
        for line in lines:
            if line.strip() and not line.startswith('#'):
                return line.strip()
        return ""
        
    def extract_competencies_from_roles(self):
        """Извлечение компетенций из файла ролей"""
        print("🔍 Извлечение компетенций из ролей...")
        
        roles_file = self.competency_model_path / "RolesCompetencies.md"
        if not roles_file.exists():
            print("❌ Файл RolesCompetencies.md не найден")
            return
            
        with open(roles_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Парсим таблицу ролей и компетенций
        self._parse_roles_table(content)
        
    def _parse_roles_table(self, content):
        """Парсинг таблицы ролей и компетенций"""
        lines = content.split('\n')
        
        for line in lines:
            if '|' in line and not line.startswith('|:') and not line.startswith('| **'):
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 4:
                    role = parts[1]
                    function = parts[2]
                    competency = parts[3]
                    indicator = parts[4] if len(parts) > 4 else ""
                    
                    if role and competency:
                        if role not in self.ai_competencies:
                            self.ai_competencies[role] = []
                            
                        self.ai_competencies[role].append({
                            "function": function,
                            "competency": competency,
                            "indicator": indicator
                        })
                        
    def create_ai_knowledge_base(self):
        """Создание базы знаний ИИ для Rubin AI"""
        print("🧠 Создание базы знаний ИИ...")
        
        knowledge_base = {
            "ai_roles": self.ai_roles,
            "ai_competencies": self.ai_competencies,
            "learning_paths": self._create_learning_paths(),
            "competency_categories": self._categorize_competencies()
        }
        
        # Сохраняем в JSON
        output_file = Path("ai_competency_knowledge_base.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
            
        print(f"✅ База знаний сохранена в {output_file}")
        return knowledge_base
        
    def _create_learning_paths(self):
        """Создание образовательных траекторий"""
        learning_paths = {
            "data_engineer": {
                "name": "Data Engineer",
                "description": "Специалист по работе с данными",
                "prerequisites": ["Python", "SQL", "Статистика"],
                "core_skills": ["ETL", "Data Pipeline", "Data Quality"],
                "advanced_skills": ["Big Data", "DataOps", "ML Pipeline"]
            },
            "ai_architect": {
                "name": "AI Architect", 
                "description": "Архитектор систем ИИ",
                "prerequisites": ["ML", "Software Architecture", "Cloud"],
                "core_skills": ["System Design", "MLOps", "AI Strategy"],
                "advanced_skills": ["AI Governance", "Ethics", "Risk Management"]
            },
            "ml_engineer": {
                "name": "ML Engineer",
                "description": "Инженер машинного обучения",
                "prerequisites": ["Python", "ML", "DevOps"],
                "core_skills": ["Model Deployment", "MLOps", "Model Monitoring"],
                "advanced_skills": ["Distributed ML", "Model Optimization", "A/B Testing"]
            }
        }
        
        return learning_paths
        
    def _categorize_competencies(self):
        """Категоризация компетенций"""
        categories = {
            "technical": ["Python", "ML", "Deep Learning", "Data Engineering"],
            "business": ["AI Strategy", "Business Analysis", "Project Management"],
            "mathematical": ["Statistics", "Linear Algebra", "Optimization"],
            "soft_skills": ["Communication", "Leadership", "Problem Solving"]
        }
        
        return categories
        
    def generate_rubin_ai_responses(self):
        """Генерация ответов Rubin AI для ИИ компетенций"""
        print("🤖 Генерация ответов Rubin AI...")
        
        responses = {}
        
        # Ответы по ролям
        for role_name, role_data in self.ai_roles.items():
            responses[f"что такое {role_name.lower()}"] = self._generate_role_response(role_name, role_data)
            responses[f"роль {role_name.lower()}"] = self._generate_role_response(role_name, role_data)
            
        # Ответы по компетенциям
        for role, competencies in self.ai_competencies.items():
            for comp in competencies:
                key = f"компетенция {comp['competency'][:50].lower()}"
                responses[key] = self._generate_competency_response(comp)
                
        # Общие ответы
        responses["ии компетенции"] = self._generate_general_ai_response()
        responses["профессии в ии"] = self._generate_professions_response()
        
        return responses
        
    def _generate_role_response(self, role_name, role_data):
        """Генерация ответа о роли"""
        response = f"**{role_name.upper()}** - профессиональная роль в области ИИ.\n\n"
        
        if role_data.get("description"):
            response += f"**Описание:** {role_data['description']}\n\n"
            
        if role_data.get("competencies"):
            response += f"**Основные компетенции:**\n"
            for comp in role_data["competencies"][:3]:  # Показываем первые 3
                response += f"• {comp['title']}\n"
                
        response += f"\n**Всего компетенций:** {len(role_data.get('competencies', []))}"
        
        return response
        
    def _generate_competency_response(self, competency):
        """Генерация ответа о компетенции"""
        response = f"**Компетенция:** {competency['competency']}\n\n"
        
        if competency.get("function"):
            response += f"**Трудовая функция:** {competency['function']}\n\n"
            
        if competency.get("indicator"):
            response += f"**Индикатор:** {competency['indicator']}"
            
        return response
        
    def _generate_general_ai_response(self):
        """Генерация общего ответа об ИИ компетенциях"""
        response = "**КОМПЕТЕНТНОСТНАЯ МОДЕЛЬ ИИ** (ИТМО)\n\n"
        response += "Структурированная система профессиональных ролей и компетенций в области искусственного интеллекта.\n\n"
        
        response += "**Основные роли:**\n"
        for role_name in self.ai_roles.keys():
            response += f"• {role_name}\n"
            
        response += f"\n**Всего ролей:** {len(self.ai_roles)}"
        response += f"\n**Всего компетенций:** {sum(len(comps) for comps in self.ai_competencies.values())}"
        
        return response
        
    def _generate_professions_response(self):
        """Генерация ответа о профессиях в ИИ"""
        response = "**ПРОФЕССИИ В ОБЛАСТИ ИИ**\n\n"
        
        for role_name, role_data in self.ai_roles.items():
            response += f"**{role_name}**\n"
            if role_data.get("description"):
                response += f"{role_data['description']}\n"
            response += f"Компетенций: {len(role_data.get('competencies', []))}\n\n"
            
        return response
        
    def run_integration(self):
        """Запуск полной интеграции"""
        print("🚀 Начинаем интеграцию компетентностной модели ИИ...")
        
        # Извлекаем данные
        self.extract_ai_roles()
        self.extract_competencies_from_roles()
        
        # Создаем базу знаний
        knowledge_base = self.create_ai_knowledge_base()
        
        # Генерируем ответы
        responses = self.generate_rubin_ai_responses()
        
        # Сохраняем ответы
        responses_file = Path("ai_competency_responses.json")
        with open(responses_file, 'w', encoding='utf-8') as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Ответы Rubin AI сохранены в {responses_file}")
        
        # Статистика
        print(f"\n📊 СТАТИСТИКА ИНТЕГРАЦИИ:")
        print(f"• Ролей ИИ: {len(self.ai_roles)}")
        print(f"• Компетенций: {sum(len(comps) for comps in self.ai_competencies.values())}")
        print(f"• Ответов Rubin AI: {len(responses)}")
        
        return knowledge_base, responses

def main():
    """Основная функция"""
    competency_model_path = r"C:\Users\elekt\OneDrive\Desktop\ai-competency-model-main"
    
    if not os.path.exists(competency_model_path):
        print(f"❌ Путь не найден: {competency_model_path}")
        return
        
    integrator = AICompetencyIntegrator(competency_model_path)
    knowledge_base, responses = integrator.run_integration()
    
    print("\n🎉 Интеграция завершена успешно!")

if __name__ == "__main__":
    main()
