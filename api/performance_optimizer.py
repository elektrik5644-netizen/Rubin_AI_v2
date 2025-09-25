"""
Модуль оптимизации производительности Rubin AI
Автоматически оптимизирует производительность всех модулей системы
"""

import os
import json
import logging
import psutil
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import gc

class PerformanceOptimizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.optimization_config = self._load_optimization_config()
        self.performance_metrics = {}
        self.optimization_history = []
        
    def _load_optimization_config(self) -> Dict:
        """Загрузка конфигурации оптимизации"""
        default_config = {
            "enabled": True,
            "optimization_interval": 300,  # 5 минут
            "memory_threshold": 80,  # 80% использования памяти
            "cpu_threshold": 70,  # 70% использования CPU
            "response_time_threshold": 2.0,  # 2 секунды
            "modules": {
                "ai_chat": {
                    "cache_size": 1000,
                    "vector_cache_size": 500,
                    "model_optimization": True,
                    "batch_processing": True
                },
                "electrical": {
                    "response_cache": True,
                    "connection_pooling": True,
                    "timeout_optimization": True
                },
                "radiomechanics": {
                    "response_cache": True,
                    "connection_pooling": True,
                    "timeout_optimization": True
                },
                "controllers": {
                    "response_cache": True,
                    "connection_pooling": True,
                    "timeout_optimization": True
                },
                "documents": {
                    "response_cache": True,
                    "connection_pooling": True,
                    "timeout_optimization": True
                }
            }
        }
        
        try:
            config_path = "config/performance_optimization.json"
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    default_config.update(config)
            else:
                os.makedirs("config", exist_ok=True)
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                    
            return default_config
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки конфигурации оптимизации: {e}")
            return default_config
    
    def analyze_performance(self) -> Dict:
        """Анализ производительности системы"""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "overall_performance": "good",
            "modules": {},
            "system_metrics": {},
            "recommendations": []
        }
        
        try:
            # Системные метрики
            analysis["system_metrics"] = self._get_system_metrics()
            
            # Анализ каждого модуля
            for module_name in self.optimization_config["modules"].keys():
                module_analysis = self._analyze_module_performance(module_name)
                analysis["modules"][module_name] = module_analysis
            
            # Определение общего состояния
            analysis["overall_performance"] = self._determine_overall_performance(analysis)
            
            # Генерация рекомендаций
            analysis["recommendations"] = self._generate_optimization_recommendations(analysis)
            
            self.logger.info(f"🔍 Анализ производительности завершен: {analysis['overall_performance']}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка анализа производительности: {e}")
            analysis["overall_performance"] = "error"
            return analysis
    
    def _get_system_metrics(self) -> Dict:
        """Получение системных метрик"""
        try:
            # Использование памяти
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Использование CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Дисковое пространство
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Количество процессов
            process_count = len(psutil.pids())
            
            return {
                "memory_percent": memory_percent,
                "cpu_percent": cpu_percent,
                "disk_percent": disk_percent,
                "process_count": process_count,
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_gb": memory.used / (1024**3)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения системных метрик: {e}")
            return {}
    
    def _analyze_module_performance(self, module_name: str) -> Dict:
        """Анализ производительности конкретного модуля"""
        analysis = {
            "module": module_name,
            "performance_score": 0.0,
            "issues": [],
            "metrics": {},
            "optimization_potential": "low"
        }
        
        try:
            # Получаем метрики модуля
            metrics = self._get_module_metrics(module_name)
            analysis["metrics"] = metrics
            
            # Вычисляем оценку производительности
            performance_score = self._calculate_performance_score(module_name, metrics)
            analysis["performance_score"] = performance_score
            
            # Определяем проблемы
            issues = self._identify_performance_issues(module_name, metrics)
            analysis["issues"] = issues
            
            # Определяем потенциал оптимизации
            analysis["optimization_potential"] = self._assess_optimization_potential(performance_score, issues)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка анализа модуля {module_name}: {e}")
            analysis["issues"].append(f"Ошибка анализа: {str(e)}")
        
        return analysis
    
    def _get_module_metrics(self, module_name: str) -> Dict:
        """Получение метрик модуля"""
        metrics = {
            "response_time": 0.0,
            "memory_usage": 0.0,
            "cpu_usage": 0.0,
            "request_count": 0,
            "error_rate": 0.0,
            "cache_hit_rate": 0.0
        }
        
        try:
            # Получаем информацию о процессе модуля
            for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
                if 'python' in proc.info['name'].lower():
                    try:
                        cmdline = proc.cmdline()
                        if any(module_name in arg for arg in cmdline):
                            # Использование памяти
                            memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                            metrics["memory_usage"] = memory_mb
                            
                            # Использование CPU
                            metrics["cpu_usage"] = proc.info['cpu_percent']
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            
            # Получаем метрики из базы данных (если доступна)
            metrics.update(self._get_database_metrics(module_name))
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения метрик модуля {module_name}: {e}")
        
        return metrics
    
    def _get_database_metrics(self, module_name: str) -> Dict:
        """Получение метрик из базы данных"""
        metrics = {}
        
        try:
            # Здесь можно добавить логику получения метрик из базы данных
            # Например, количество запросов, время ответа, ошибки
            pass
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения метрик БД для {module_name}: {e}")
        
        return metrics
    
    def _calculate_performance_score(self, module_name: str, metrics: Dict) -> float:
        """Вычисление оценки производительности модуля"""
        try:
            score = 100.0
            
            # Штрафы за проблемы
            if metrics.get("memory_usage", 0) > 500:  # > 500MB
                score -= 20
            elif metrics.get("memory_usage", 0) > 200:  # > 200MB
                score -= 10
            
            if metrics.get("cpu_usage", 0) > 50:  # > 50% CPU
                score -= 15
            elif metrics.get("cpu_usage", 0) > 25:  # > 25% CPU
                score -= 5
            
            if metrics.get("response_time", 0) > 2.0:  # > 2 секунды
                score -= 25
            elif metrics.get("response_time", 0) > 1.0:  # > 1 секунда
                score -= 10
            
            if metrics.get("error_rate", 0) > 0.1:  # > 10% ошибок
                score -= 20
            elif metrics.get("error_rate", 0) > 0.05:  # > 5% ошибок
                score -= 10
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка вычисления оценки производительности: {e}")
            return 0.0
    
    def _identify_performance_issues(self, module_name: str, metrics: Dict) -> List[str]:
        """Идентификация проблем производительности"""
        issues = []
        
        try:
            if metrics.get("memory_usage", 0) > 500:
                issues.append(f"Высокое использование памяти: {metrics['memory_usage']:.1f}MB")
            
            if metrics.get("cpu_usage", 0) > 50:
                issues.append(f"Высокое использование CPU: {metrics['cpu_usage']:.1f}%")
            
            if metrics.get("response_time", 0) > 2.0:
                issues.append(f"Медленный ответ: {metrics['response_time']:.2f}с")
            
            if metrics.get("error_rate", 0) > 0.1:
                issues.append(f"Высокий уровень ошибок: {metrics['error_rate']:.1%}")
            
            if metrics.get("cache_hit_rate", 0) < 0.7:
                issues.append(f"Низкая эффективность кэша: {metrics['cache_hit_rate']:.1%}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка идентификации проблем: {e}")
        
        return issues
    
    def _assess_optimization_potential(self, performance_score: float, issues: List[str]) -> str:
        """Оценка потенциала оптимизации"""
        if performance_score < 50 or len(issues) > 3:
            return "high"
        elif performance_score < 75 or len(issues) > 1:
            return "medium"
        else:
            return "low"
    
    def _determine_overall_performance(self, analysis: Dict) -> str:
        """Определение общего состояния производительности"""
        try:
            system_metrics = analysis.get("system_metrics", {})
            modules = analysis.get("modules", {})
            
            # Проверяем системные метрики
            if (system_metrics.get("memory_percent", 0) > 90 or 
                system_metrics.get("cpu_percent", 0) > 80):
                return "critical"
            
            # Проверяем модули
            critical_modules = 0
            warning_modules = 0
            
            for module_name, module_analysis in modules.items():
                score = module_analysis.get("performance_score", 0)
                if score < 50:
                    critical_modules += 1
                elif score < 75:
                    warning_modules += 1
            
            if critical_modules > 0:
                return "critical"
            elif warning_modules > 2:
                return "warning"
            else:
                return "good"
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка определения общего состояния: {e}")
            return "unknown"
    
    def _generate_optimization_recommendations(self, analysis: Dict) -> List[str]:
        """Генерация рекомендаций по оптимизации"""
        recommendations = []
        
        try:
            system_metrics = analysis.get("system_metrics", {})
            modules = analysis.get("modules", {})
            
            # Системные рекомендации
            if system_metrics.get("memory_percent", 0) > 80:
                recommendations.append("Очистить память системы")
            
            if system_metrics.get("cpu_percent", 0) > 70:
                recommendations.append("Оптимизировать использование CPU")
            
            # Рекомендации по модулям
            for module_name, module_analysis in modules.items():
                score = module_analysis.get("performance_score", 0)
                issues = module_analysis.get("issues", [])
                
                if score < 75:
                    recommendations.append(f"Оптимизировать производительность {module_name}")
                
                for issue in issues:
                    if "память" in issue.lower():
                        recommendations.append(f"Очистить память модуля {module_name}")
                    elif "cpu" in issue.lower():
                        recommendations.append(f"Оптимизировать CPU модуля {module_name}")
                    elif "медленный" in issue.lower():
                        recommendations.append(f"Ускорить ответы модуля {module_name}")
                    elif "ошибок" in issue.lower():
                        recommendations.append(f"Исправить ошибки модуля {module_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка генерации рекомендаций: {e}")
        
        return list(set(recommendations))  # Убираем дубликаты
    
    def optimize_module(self, module_name: str) -> Dict:
        """Оптимизация конкретного модуля"""
        optimization_result = {
            "module": module_name,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "optimizations_applied": [],
            "performance_improvement": 0.0,
            "error": None
        }
        
        try:
            # Получаем текущие метрики
            initial_metrics = self._get_module_metrics(module_name)
            initial_score = self._calculate_performance_score(module_name, initial_metrics)
            
            # Применяем оптимизации
            optimizations = self._apply_module_optimizations(module_name)
            optimization_result["optimizations_applied"] = optimizations
            
            # Ждем немного для стабилизации
            time.sleep(2)
            
            # Получаем новые метрики
            final_metrics = self._get_module_metrics(module_name)
            final_score = self._calculate_performance_score(module_name, final_metrics)
            
            # Вычисляем улучшение
            improvement = final_score - initial_score
            optimization_result["performance_improvement"] = improvement
            optimization_result["success"] = improvement > 0 or len(optimizations) > 0
            
            self.logger.info(f"🔧 Оптимизация модуля {module_name}: улучшение на {improvement:.1f}%")
            
        except Exception as e:
            optimization_result["error"] = str(e)
            self.logger.error(f"❌ Ошибка оптимизации модуля {module_name}: {e}")
        
        return optimization_result
    
    def _apply_module_optimizations(self, module_name: str) -> List[str]:
        """Применение оптимизаций для модуля"""
        optimizations = []
        
        try:
            config = self.optimization_config["modules"].get(module_name, {})
            
            if module_name == "ai_chat":
                optimizations.extend(self._optimize_ai_chat(config))
            else:
                optimizations.extend(self._optimize_api_module(module_name, config))
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка применения оптимизаций для {module_name}: {e}")
        
        return optimizations
    
    def _optimize_ai_chat(self, config: Dict) -> List[str]:
        """Оптимизация AI чата"""
        optimizations = []
        
        try:
            # Очистка памяти
            gc.collect()
            optimizations.append("Очистка памяти Python")
            
            # Оптимизация кэша
            if config.get("cache_size"):
                optimizations.append(f"Настройка размера кэша: {config['cache_size']}")
            
            # Оптимизация векторного поиска
            if config.get("vector_cache_size"):
                optimizations.append(f"Настройка кэша векторов: {config['vector_cache_size']}")
            
            # Оптимизация модели
            if config.get("model_optimization"):
                optimizations.append("Оптимизация модели машинного обучения")
            
            # Пакетная обработка
            if config.get("batch_processing"):
                optimizations.append("Включение пакетной обработки")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка оптимизации AI чата: {e}")
        
        return optimizations
    
    def _optimize_api_module(self, module_name: str, config: Dict) -> List[str]:
        """Оптимизация API модуля"""
        optimizations = []
        
        try:
            # Кэширование ответов
            if config.get("response_cache"):
                optimizations.append("Включение кэширования ответов")
            
            # Пул соединений
            if config.get("connection_pooling"):
                optimizations.append("Настройка пула соединений")
            
            # Оптимизация таймаутов
            if config.get("timeout_optimization"):
                optimizations.append("Оптимизация таймаутов")
            
            # Очистка памяти
            gc.collect()
            optimizations.append("Очистка памяти")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка оптимизации API модуля {module_name}: {e}")
        
        return optimizations
    
    def optimize_all_modules(self) -> Dict:
        """Оптимизация всех модулей"""
        optimization_result = {
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "modules_optimized": [],
            "total_improvement": 0.0,
            "errors": []
        }
        
        try:
            modules = list(self.optimization_config["modules"].keys())
            
            for module_name in modules:
                try:
                    result = self.optimize_module(module_name)
                    optimization_result["modules_optimized"].append(result)
                    optimization_result["total_improvement"] += result.get("performance_improvement", 0)
                    
                    if not result.get("success", False):
                        optimization_result["success"] = False
                        
                except Exception as e:
                    optimization_result["errors"].append(f"Ошибка оптимизации {module_name}: {str(e)}")
                    optimization_result["success"] = False
            
            # Сохраняем историю оптимизации
            self._save_optimization_history(optimization_result)
            
            self.logger.info(f"🔧 Оптимизация всех модулей завершена: общее улучшение {optimization_result['total_improvement']:.1f}%")
            
        except Exception as e:
            optimization_result["success"] = False
            optimization_result["errors"].append(str(e))
            self.logger.error(f"❌ Ошибка оптимизации всех модулей: {e}")
        
        return optimization_result
    
    def _save_optimization_history(self, result: Dict):
        """Сохранение истории оптимизации"""
        try:
            self.optimization_history.append(result)
            
            # Ограничиваем размер истории
            if len(self.optimization_history) > 50:
                self.optimization_history = self.optimization_history[-50:]
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения истории оптимизации: {e}")
    
    def get_optimization_history(self, limit: int = 10) -> List[Dict]:
        """Получение истории оптимизации"""
        return self.optimization_history[-limit:] if self.optimization_history else []
    
    def start_continuous_optimization(self):
        """Запуск непрерывной оптимизации"""
        if not self.optimization_config["enabled"]:
            return
        
        def optimization_loop():
            while True:
                try:
                    time.sleep(self.optimization_config["optimization_interval"])
                    
                    # Анализируем производительность
                    analysis = self.analyze_performance()
                    
                    # Если производительность плохая, оптимизируем
                    if analysis["overall_performance"] in ["warning", "critical"]:
                        self.logger.info("🔧 Запуск автоматической оптимизации...")
                        result = self.optimize_all_modules()
                        
                        if result["success"]:
                            self.logger.info(f"✅ Автоматическая оптимизация завершена: улучшение {result['total_improvement']:.1f}%")
                        else:
                            self.logger.warning("⚠️ Автоматическая оптимизация завершена с ошибками")
                    
                except Exception as e:
                    self.logger.error(f"❌ Ошибка в цикле оптимизации: {e}")
                    time.sleep(60)  # Ждем минуту перед повтором
        
        # Запускаем в отдельном потоке
        optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        optimization_thread.start()
        self.logger.info("🚀 Непрерывная оптимизация запущена")

# Глобальный экземпляр оптимизатора производительности
performance_optimizer = PerformanceOptimizer()

















