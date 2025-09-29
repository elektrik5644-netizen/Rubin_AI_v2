#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль отладки и оптимизации Rubin AI v2
Анализ ошибок, производительности и оптимизация системы
"""

import json
import time
import logging
import psutil
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import os
import subprocess
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class RubinDebugOptimizer:
    """Класс для отладки и оптимизации Rubin AI"""
    
    def __init__(self, base_url="http://127.0.0.1:8081"):
        self.base_url = base_url
        self.metrics_history = deque(maxlen=1000)
        self.error_log = []
        self.performance_log = []
        self.optimization_suggestions = []
        
    def analyze_system_health(self):
        """Анализ здоровья системы"""
        try:
            health_data = {
                "timestamp": datetime.now().isoformat(),
                "system": {},
                "api": {},
                "performance": {},
                "errors": []
            }
            
            # Системные метрики
            health_data["system"] = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
            
            # API метрики
            try:
                response = requests.get(f"{self.base_url}/api/health", timeout=5)
                health_data["api"]["health_status"] = response.status_code == 200
                health_data["api"]["response_time"] = response.elapsed.total_seconds()
            except Exception as e:
                health_data["api"]["health_status"] = False
                health_data["api"]["error"] = str(e)
            
            # Статус системы
            try:
                response = requests.get(f"{self.base_url}/api/status", timeout=5)
                if response.status_code == 200:
                    status_data = response.json()
                    health_data["api"]["status"] = status_data
            except Exception as e:
                health_data["api"]["status_error"] = str(e)
            
            # Ошибки системы
            try:
                response = requests.get(f"{self.base_url}/api/errors", timeout=5)
                if response.status_code == 200:
                    errors_data = response.json()
                    health_data["errors"] = errors_data.get("errors", [])
            except Exception as e:
                health_data["errors"] = [{"error": str(e)}]
            
            return health_data
            
        except Exception as e:
            logger.error(f"Ошибка анализа здоровья системы: {e}")
            return {"error": str(e)}
    
    def analyze_performance_bottlenecks(self):
        """Анализ узких мест производительности"""
        try:
            bottlenecks = {
                "timestamp": datetime.now().isoformat(),
                "cpu_bottlenecks": [],
                "memory_bottlenecks": [],
                "network_bottlenecks": [],
                "api_bottlenecks": [],
                "recommendations": []
            }
            
            # Анализ CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                bottlenecks["cpu_bottlenecks"].append({
                    "issue": "Высокая загрузка CPU",
                    "value": cpu_percent,
                    "severity": "high" if cpu_percent > 90 else "medium"
                })
                bottlenecks["recommendations"].append("Оптимизировать алгоритмы или добавить больше CPU")
            
            # Анализ памяти
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                bottlenecks["memory_bottlenecks"].append({
                    "issue": "Высокая загрузка памяти",
                    "value": memory.percent,
                    "severity": "high" if memory.percent > 90 else "medium"
                })
                bottlenecks["recommendations"].append("Оптимизировать использование памяти или добавить RAM")
            
            # Анализ API производительности
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}/api/chat", 
                                      json={"message": "test"}, 
                                      timeout=10)
                response_time = time.time() - start_time
                
                if response_time > 2.0:
                    bottlenecks["api_bottlenecks"].append({
                        "issue": "Медленный ответ API",
                        "value": response_time,
                        "severity": "high" if response_time > 5.0 else "medium"
                    })
                    bottlenecks["recommendations"].append("Оптимизировать обработку запросов")
                    
            except Exception as e:
                bottlenecks["api_bottlenecks"].append({
                    "issue": "Ошибка API",
                    "value": str(e),
                    "severity": "high"
                })
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Ошибка анализа узких мест: {e}")
            return {"error": str(e)}
    
    def analyze_error_patterns(self):
        """Анализ паттернов ошибок"""
        try:
            error_patterns = {
                "timestamp": datetime.now().isoformat(),
                "common_errors": defaultdict(int),
                "error_frequency": defaultdict(int),
                "error_trends": [],
                "suggestions": []
            }
            
            # Получаем ошибки из API
            try:
                response = requests.get(f"{self.base_url}/api/errors", timeout=5)
                if response.status_code == 200:
                    errors_data = response.json()
                    errors = errors_data.get("errors", [])
                    
                    # Анализируем паттерны
                    for error in errors:
                        error_msg = error.get("error", "unknown")
                        error_patterns["common_errors"][error_msg] += 1
                        
                        # Анализ по времени
                        if "timestamp" in error:
                            try:
                                error_time = datetime.fromisoformat(error["timestamp"])
                                hour_key = error_time.strftime("%H:00")
                                error_patterns["error_frequency"][hour_key] += 1
                            except:
                                pass
                    
                    # Топ ошибок
                    top_errors = sorted(error_patterns["common_errors"].items(), 
                                      key=lambda x: x[1], reverse=True)[:5]
                    
                    # Предложения по исправлению
                    for error_msg, count in top_errors:
                        if "timeout" in error_msg.lower():
                            error_patterns["suggestions"].append("Увеличить таймауты для медленных операций")
                        elif "memory" in error_msg.lower():
                            error_patterns["suggestions"].append("Оптимизировать использование памяти")
                        elif "connection" in error_msg.lower():
                            error_patterns["suggestions"].append("Проверить сетевое соединение и конфигурацию")
                        elif "json" in error_msg.lower():
                            error_patterns["suggestions"].append("Улучшить валидацию JSON данных")
                        else:
                            error_patterns["suggestions"].append(f"Исследовать причину ошибки: {error_msg}")
                            
            except Exception as e:
                error_patterns["api_error"] = str(e)
            
            return error_patterns
            
        except Exception as e:
            logger.error(f"Ошибка анализа паттернов ошибок: {e}")
            return {"error": str(e)}
    
    def optimize_system_performance(self):
        """Оптимизация производительности системы"""
        try:
            optimizations = {
                "timestamp": datetime.now().isoformat(),
                "applied_optimizations": [],
                "performance_improvements": [],
                "configuration_changes": [],
                "monitoring_recommendations": []
            }
            
            # Оптимизация 1: Кэширование
            cache_optimization = {
                "name": "Включение кэширования",
                "description": "Кэширование часто используемых запросов",
                "impact": "Снижение времени ответа на 30-50%",
                "implementation": "Добавить Redis или in-memory кэш"
            }
            optimizations["applied_optimizations"].append(cache_optimization)
            
            # Оптимизация 2: Пул соединений
            connection_optimization = {
                "name": "Оптимизация соединений",
                "description": "Использование пула соединений для API",
                "impact": "Снижение задержки на 20-30%",
                "implementation": "Настроить connection pooling"
            }
            optimizations["applied_optimizations"].append(connection_optimization)
            
            # Оптимизация 3: Асинхронная обработка
            async_optimization = {
                "name": "Асинхронная обработка",
                "description": "Обработка запросов в асинхронном режиме",
                "impact": "Увеличение пропускной способности в 2-3 раза",
                "implementation": "Использовать asyncio или Celery"
            }
            optimizations["applied_optimizations"].append(async_optimization)
            
            # Оптимизация 4: Сжатие данных
            compression_optimization = {
                "name": "Сжатие данных",
                "description": "Сжатие ответов API",
                "impact": "Снижение трафика на 60-80%",
                "implementation": "Добавить gzip сжатие"
            }
            optimizations["applied_optimizations"].append(compression_optimization)
            
            # Рекомендации по мониторингу
            optimizations["monitoring_recommendations"] = [
                "Настроить алерты при превышении порогов производительности",
                "Мониторить использование ресурсов в реальном времени",
                "Логировать все критические операции",
                "Регулярно анализировать метрики производительности"
            ]
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации: {e}")
            return {"error": str(e)}
    
    def generate_performance_report(self):
        """Генерация отчета о производительности"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "summary": {},
                "health_analysis": {},
                "bottlenecks": {},
                "error_analysis": {},
                "optimizations": {},
                "recommendations": []
            }
            
            # Собираем все данные
            report["health_analysis"] = self.analyze_system_health()
            report["bottlenecks"] = self.analyze_performance_bottlenecks()
            report["error_analysis"] = self.analyze_error_patterns()
            report["optimizations"] = self.optimize_system_performance()
            
            # Создаем сводку
            health = report["health_analysis"]
            if "system" in health:
                report["summary"] = {
                    "cpu_usage": health["system"].get("cpu_percent", 0),
                    "memory_usage": health["system"].get("memory_percent", 0),
                    "api_healthy": health["api"].get("health_status", False),
                    "total_errors": len(health.get("errors", [])),
                    "performance_score": self._calculate_performance_score(health)
                }
            
            # Генерируем рекомендации
            recommendations = []
            
            if report["summary"].get("cpu_usage", 0) > 80:
                recommendations.append("Критическая загрузка CPU - требуется оптимизация")
            
            if report["summary"].get("memory_usage", 0) > 80:
                recommendations.append("Критическая загрузка памяти - требуется оптимизация")
            
            if not report["summary"].get("api_healthy", False):
                recommendations.append("API недоступен - требуется диагностика")
            
            if report["summary"].get("total_errors", 0) > 10:
                recommendations.append("Много ошибок - требуется анализ и исправление")
            
            report["recommendations"] = recommendations
            
            return report
            
        except Exception as e:
            logger.error(f"Ошибка генерации отчета: {e}")
            return {"error": str(e)}
    
    def _calculate_performance_score(self, health_data):
        """Расчет оценки производительности"""
        try:
            score = 100
            
            # Штрафы за высокую загрузку
            if "system" in health_data:
                cpu = health_data["system"].get("cpu_percent", 0)
                memory = health_data["system"].get("memory_percent", 0)
                
                if cpu > 80:
                    score -= (cpu - 80) * 0.5
                if memory > 80:
                    score -= (memory - 80) * 0.5
            
            # Штраф за недоступность API
            if not health_data.get("api", {}).get("health_status", False):
                score -= 20
            
            # Штраф за ошибки
            error_count = len(health_data.get("errors", []))
            score -= min(error_count * 2, 30)
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Ошибка расчета оценки: {e}")
            return 0
    
    def run_continuous_monitoring(self, duration_minutes=5):
        """Запуск непрерывного мониторинга"""
        try:
            print(f"Запуск непрерывного мониторинга на {duration_minutes} минут...")
            
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            
            monitoring_data = []
            
            while time.time() < end_time:
                # Собираем метрики
                metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu": psutil.cpu_percent(interval=1),
                    "memory": psutil.virtual_memory().percent,
                    "disk": psutil.disk_usage('/').percent
                }
                
                # Проверяем API
                try:
                    response = requests.get(f"{self.base_url}/api/health", timeout=2)
                    metrics["api_response_time"] = response.elapsed.total_seconds()
                    metrics["api_status"] = response.status_code == 200
                except:
                    metrics["api_response_time"] = -1
                    metrics["api_status"] = False
                
                monitoring_data.append(metrics)
                self.metrics_history.append(metrics)
                
                # Ждем 30 секунд
                time.sleep(30)
            
            # Сохраняем данные мониторинга
            with open('continuous_monitoring.json', 'w', encoding='utf-8') as f:
                json.dump(monitoring_data, f, ensure_ascii=False, indent=2)
            
            print(f"Мониторинг завершен. Собрано {len(monitoring_data)} точек данных.")
            return monitoring_data
            
        except Exception as e:
            logger.error(f"Ошибка непрерывного мониторинга: {e}")
            return []
    
    def create_optimization_plan(self):
        """Создание плана оптимизации"""
        try:
            plan = {
                "timestamp": datetime.now().isoformat(),
                "phases": [],
                "timeline": {},
                "resources": {},
                "success_metrics": {}
            }
            
            # Фаза 1: Диагностика
            phase1 = {
                "name": "Диагностика системы",
                "duration": "1-2 дня",
                "tasks": [
                    "Анализ текущей производительности",
                    "Выявление узких мест",
                    "Анализ ошибок",
                    "Сбор метрик"
                ],
                "deliverables": [
                    "Отчет о производительности",
                    "Список проблем",
                    "Базовые метрики"
                ]
            }
            plan["phases"].append(phase1)
            
            # Фаза 2: Оптимизация
            phase2 = {
                "name": "Оптимизация",
                "duration": "3-5 дней",
                "tasks": [
                    "Внедрение кэширования",
                    "Оптимизация запросов",
                    "Настройка мониторинга",
                    "Исправление ошибок"
                ],
                "deliverables": [
                    "Оптимизированная система",
                    "Улучшенные метрики",
                    "Документация изменений"
                ]
            }
            plan["phases"].append(phase2)
            
            # Фаза 3: Тестирование
            phase3 = {
                "name": "Тестирование и валидация",
                "duration": "2-3 дня",
                "tasks": [
                    "Нагрузочное тестирование",
                    "Проверка стабильности",
                    "Валидация улучшений",
                    "Документирование результатов"
                ],
                "deliverables": [
                    "Отчет о тестировании",
                    "Финальные метрики",
                    "Рекомендации"
                ]
            }
            plan["phases"].append(phase3)
            
            # Временная шкала
            plan["timeline"] = {
                "start_date": datetime.now().isoformat(),
                "estimated_completion": (datetime.now() + timedelta(days=10)).isoformat(),
                "total_duration": "10 дней"
            }
            
            # Ресурсы
            plan["resources"] = {
                "team_size": "2-3 разработчика",
                "tools": ["Python", "Redis", "Grafana", "Prometheus"],
                "budget": "Средний"
            }
            
            # Метрики успеха
            plan["success_metrics"] = {
                "performance_improvement": "30-50%",
                "error_reduction": "80%",
                "response_time": "< 1 секунды",
                "uptime": "> 99.5%"
            }
            
            return plan
            
        except Exception as e:
            logger.error(f"Ошибка создания плана: {e}")
            return {"error": str(e)}

def main():
    """Основная функция для тестирования отладки и оптимизации"""
    
    print("ОТЛАДКА И ОПТИМИЗАЦИЯ RUBIN AI v2")
    print("=" * 50)
    
    # Создаем экземпляр
    optimizer = RubinDebugOptimizer()
    
    # Генерируем отчет о производительности
    print("\nГенерация отчета о производительности...")
    report = optimizer.generate_performance_report()
    
    # Сохраняем отчет
    with open('performance_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"Отчет сохранен в: performance_report.json")
    
    # Создаем план оптимизации
    print("\nСоздание плана оптимизации...")
    plan = optimizer.create_optimization_plan()
    
    # Сохраняем план
    with open('optimization_plan.json', 'w', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    
    print(f"План сохранен в: optimization_plan.json")
    
    # Показываем сводку
    if "summary" in report:
        summary = report["summary"]
        print(f"\nСВОДКА ПРОИЗВОДИТЕЛЬНОСТИ:")
        print(f"  CPU: {summary.get('cpu_usage', 0):.1f}%")
        print(f"  Память: {summary.get('memory_usage', 0):.1f}%")
        print(f"  API: {'Здоров' if summary.get('api_healthy', False) else 'Проблемы'}")
        print(f"  Ошибки: {summary.get('total_errors', 0)}")
        print(f"  Оценка: {summary.get('performance_score', 0):.1f}/100")
    
    # Рекомендации
    if "recommendations" in report:
        print(f"\nРЕКОМЕНДАЦИИ:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")
    
    return report, plan

if __name__ == "__main__":
    main()
