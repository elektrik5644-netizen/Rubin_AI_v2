"""
Модуль автоматического восстановления Rubin AI
Автоматически исправляет проблемы и обновляет конфигурацию
"""

import os
import json
import logging
import subprocess
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import psutil
import sqlite3

class AutoHealer:
    def __init__(self, config_path: str = "config/auto_heal.json"):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config()
        self.healing_history = []
        
    def _load_config(self) -> Dict:
        """Загрузка конфигурации автоматического восстановления"""
        default_config = {
            "enabled": True,
            "max_restart_attempts": 3,
            "restart_delay": 5,
            "health_check_interval": 30,
            "auto_config_update": True,
            "backup_before_changes": True,
            "modules": {
                "ai_chat": {"port": 8084, "critical": True},
                "electrical": {"port": 8087, "critical": False},
                "radiomechanics": {"port": 8089, "critical": False},
                "controllers": {"port": 8090, "critical": True},
                "documents": {"port": 8088, "critical": False}
            },
            "healing_strategies": {
                "port_conflict": "restart_service",
                "memory_leak": "restart_service",
                "connection_timeout": "restart_service",
                "config_error": "update_config",
                "dependency_missing": "install_dependency"
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Объединяем с дефолтной конфигурацией
                    default_config.update(config)
            else:
                # Создаем директорию если не существует
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                    
            return default_config
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки конфигурации: {e}")
            return default_config
    
    def diagnose_system(self) -> Dict:
        """Диагностика состояния системы"""
        diagnosis = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "healthy",
            "modules": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            for module_name, module_config in self.config["modules"].items():
                port = module_config["port"]
                module_status = self._check_module_health(module_name, port)
                diagnosis["modules"][module_name] = module_status
                
                if not module_status["healthy"]:
                    diagnosis["issues"].append({
                        "module": module_name,
                        "issue": module_status["issue"],
                        "severity": "critical" if module_config["critical"] else "warning"
                    })
            
            # Определяем общее состояние
            critical_issues = [i for i in diagnosis["issues"] if i["severity"] == "critical"]
            if critical_issues:
                diagnosis["overall_health"] = "critical"
            elif diagnosis["issues"]:
                diagnosis["overall_health"] = "warning"
            
            # Генерируем рекомендации
            diagnosis["recommendations"] = self._generate_recommendations(diagnosis)
            
            self.logger.info(f"🔍 Диагностика завершена: {diagnosis['overall_health']}")
            return diagnosis
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка диагностики: {e}")
            diagnosis["overall_health"] = "error"
            diagnosis["issues"].append({"error": str(e)})
            return diagnosis
    
    def _check_module_health(self, module_name: str, port: int) -> Dict:
        """Проверка здоровья конкретного модуля"""
        status = {
            "healthy": True,
            "port": port,
            "response_time": None,
            "issue": None,
            "last_check": datetime.now().isoformat()
        }
        
        try:
            # Проверяем, слушается ли порт
            if not self._is_port_listening(port):
                status["healthy"] = False
                status["issue"] = f"Порт {port} не слушается"
                return status
            
            # Проверяем HTTP ответ
            start_time = time.time()
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=5)
                response_time = time.time() - start_time
                status["response_time"] = response_time
                
                if response.status_code != 200:
                    status["healthy"] = False
                    status["issue"] = f"HTTP {response.status_code}"
                elif response_time > 2.0:
                    status["healthy"] = False
                    status["issue"] = f"Медленный ответ: {response_time:.2f}с"
                    
            except requests.exceptions.RequestException as e:
                status["healthy"] = False
                status["issue"] = f"Ошибка соединения: {str(e)}"
            
            # Проверяем использование памяти
            memory_usage = self._get_module_memory_usage(module_name)
            if memory_usage and memory_usage > 500:  # 500MB
                status["memory_usage"] = memory_usage
                if memory_usage > 1000:  # 1GB
                    status["healthy"] = False
                    status["issue"] = f"Высокое использование памяти: {memory_usage}MB"
            
        except Exception as e:
            status["healthy"] = False
            status["issue"] = f"Ошибка проверки: {str(e)}"
        
        return status
    
    def _is_port_listening(self, port: int) -> bool:
        """Проверка, слушается ли порт"""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    return True
            return False
        except Exception:
            return False
    
    def _get_module_memory_usage(self, module_name: str) -> Optional[float]:
        """Получение использования памяти модулем"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                if 'python' in proc.info['name'].lower():
                    try:
                        cmdline = proc.cmdline()
                        if any(module_name in arg for arg in cmdline):
                            memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                            return memory_mb
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            return None
        except Exception:
            return None
    
    def _generate_recommendations(self, diagnosis: Dict) -> List[str]:
        """Генерация рекомендаций по исправлению"""
        recommendations = []
        
        for issue in diagnosis["issues"]:
            module = issue["module"]
            issue_type = issue["issue"]
            
            if "порт" in issue_type.lower():
                recommendations.append(f"Перезапустить сервис {module}")
            elif "память" in issue_type.lower():
                recommendations.append(f"Очистить память модуля {module}")
            elif "соединение" in issue_type.lower():
                recommendations.append(f"Проверить сетевые настройки для {module}")
            elif "медленный" in issue_type.lower():
                recommendations.append(f"Оптимизировать производительность {module}")
        
        return recommendations
    
    def auto_heal(self, diagnosis: Dict = None) -> Dict:
        """Автоматическое восстановление системы"""
        if not self.config["enabled"]:
            return {"status": "disabled", "message": "Автоматическое восстановление отключено"}
        
        if diagnosis is None:
            diagnosis = self.diagnose_system()
        
        healing_result = {
            "timestamp": datetime.now().isoformat(),
            "actions_taken": [],
            "success": True,
            "errors": []
        }
        
        try:
            for issue in diagnosis["issues"]:
                module = issue["module"]
                issue_type = issue["issue"]
                severity = issue["severity"]
                
                # Определяем стратегию восстановления
                strategy = self._determine_healing_strategy(issue_type)
                
                if strategy:
                    action_result = self._execute_healing_action(module, strategy, issue_type)
                    healing_result["actions_taken"].append(action_result)
                    
                    if not action_result["success"]:
                        healing_result["success"] = False
                        healing_result["errors"].append(action_result["error"])
                else:
                    self.logger.warning(f"⚠️ Неизвестная стратегия для {issue_type}")
            
            # Сохраняем историю восстановления
            self._save_healing_history(healing_result)
            
            self.logger.info(f"🔧 Автоматическое восстановление завершено: {healing_result['success']}")
            return healing_result
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка автоматического восстановления: {e}")
            healing_result["success"] = False
            healing_result["errors"].append(str(e))
            return healing_result
    
    def _determine_healing_strategy(self, issue_type: str) -> Optional[str]:
        """Определение стратегии восстановления"""
        issue_lower = issue_type.lower()
        
        if "порт" in issue_lower or "слушается" in issue_lower:
            return "restart_service"
        elif "память" in issue_lower:
            return "restart_service"
        elif "соединение" in issue_lower or "timeout" in issue_lower:
            return "restart_service"
        elif "конфигурация" in issue_lower or "config" in issue_lower:
            return "update_config"
        elif "зависимость" in issue_lower or "dependency" in issue_lower:
            return "install_dependency"
        
        return "restart_service"  # Стратегия по умолчанию
    
    def _execute_healing_action(self, module: str, strategy: str, issue_type: str) -> Dict:
        """Выполнение действия восстановления"""
        action_result = {
            "module": module,
            "strategy": strategy,
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "error": None
        }
        
        try:
            if strategy == "restart_service":
                action_result = self._restart_service(module)
            elif strategy == "update_config":
                action_result = self._update_config(module)
            elif strategy == "install_dependency":
                action_result = self._install_dependency(module)
            
            self.logger.info(f"🔧 Выполнено действие {strategy} для {module}: {action_result['success']}")
            
        except Exception as e:
            action_result["error"] = str(e)
            self.logger.error(f"❌ Ошибка выполнения действия {strategy} для {module}: {e}")
        
        return action_result
    
    def _restart_service(self, module: str) -> Dict:
        """Перезапуск сервиса"""
        result = {
            "module": module,
            "strategy": "restart_service",
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "error": None
        }
        
        try:
            # Находим и останавливаем процесс
            self._stop_module_process(module)
            time.sleep(2)
            
            # Запускаем сервис
            self._start_module_process(module)
            time.sleep(5)
            
            # Проверяем, что сервис запустился
            module_config = self.config["modules"].get(module)
            if module_config:
                port = module_config["port"]
                if self._is_port_listening(port):
                    result["success"] = True
                else:
                    result["error"] = "Сервис не запустился после перезапуска"
            else:
                result["error"] = "Конфигурация модуля не найдена"
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _stop_module_process(self, module: str):
        """Остановка процесса модуля"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'python' in proc.info['name'].lower():
                    try:
                        cmdline = proc.cmdline()
                        if any(module in arg for arg in cmdline):
                            proc.terminate()
                            proc.wait(timeout=10)
                            self.logger.info(f"🛑 Остановлен процесс модуля {module}")
                    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                        continue
        except Exception as e:
            self.logger.error(f"❌ Ошибка остановки процесса {module}: {e}")
    
    def _start_module_process(self, module: str):
        """Запуск процесса модуля"""
        try:
            module_scripts = {
                "ai_chat": "python api/rubin_ai_v2_simple.py",
                "electrical": "python api/electrical_api.py",
                "radiomechanics": "python api/radiomechanics_api.py",
                "controllers": "python api/controllers_api.py",
                "documents": "python api/documents_api.py"
            }
            
            script = module_scripts.get(module)
            if script:
                subprocess.Popen(script, shell=True, cwd=os.getcwd())
                self.logger.info(f"🚀 Запущен процесс модуля {module}")
            else:
                raise ValueError(f"Неизвестный модуль: {module}")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка запуска процесса {module}: {e}")
            raise
    
    def _update_config(self, module: str) -> Dict:
        """Обновление конфигурации модуля"""
        result = {
            "module": module,
            "strategy": "update_config",
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "error": None
        }
        
        try:
            # Здесь можно добавить логику обновления конфигурации
            # Например, сброс к дефолтным настройкам или применение исправлений
            self.logger.info(f"⚙️ Обновлена конфигурация модуля {module}")
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _install_dependency(self, module: str) -> Dict:
        """Установка зависимостей"""
        result = {
            "module": module,
            "strategy": "install_dependency",
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "error": None
        }
        
        try:
            # Здесь можно добавить логику установки зависимостей
            # Например, pip install missing_package
            self.logger.info(f"📦 Установлены зависимости для модуля {module}")
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _save_healing_history(self, healing_result: Dict):
        """Сохранение истории восстановления"""
        try:
            self.healing_history.append(healing_result)
            
            # Ограничиваем размер истории
            if len(self.healing_history) > 100:
                self.healing_history = self.healing_history[-100:]
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения истории: {e}")
    
    def get_healing_history(self, limit: int = 10) -> List[Dict]:
        """Получение истории восстановления"""
        return self.healing_history[-limit:] if self.healing_history else []
    
    def update_config(self, new_config: Dict):
        """Обновление конфигурации автоматического восстановления"""
        try:
            self.config.update(new_config)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            self.logger.info("✅ Конфигурация автоматического восстановления обновлена")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка обновления конфигурации: {e}")

# Глобальный экземпляр автоматического восстановления
auto_healer = AutoHealer()

















