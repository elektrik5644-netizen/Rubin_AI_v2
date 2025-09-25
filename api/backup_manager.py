"""
Модуль управления резервными копиями Rubin AI
Создает и восстанавливает резервные копии состояния системы
"""

import os
import json
import shutil
import sqlite3
import logging
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hashlib

class BackupManager:
    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = backup_dir
        self.logger = logging.getLogger(__name__)
        self._init_backup_directory()
    
    def _init_backup_directory(self):
        """Инициализация директории для резервных копий"""
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
            self.logger.info(f"✅ Директория резервных копий: {self.backup_dir}")
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания директории резервных копий: {e}")
    
    def create_backup(self, backup_name: str = None, include_data: bool = True, 
                     include_config: bool = True, include_logs: bool = False) -> Dict:
        """Создание резервной копии системы"""
        if backup_name is None:
            backup_name = f"rubin_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_result = {
            "backup_name": backup_name,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "files_backed_up": [],
            "size_mb": 0,
            "error": None
        }
        
        try:
            backup_path = os.path.join(self.backup_dir, backup_name)
            os.makedirs(backup_path, exist_ok=True)
            
            # Создаем метаданные резервной копии
            metadata = {
                "backup_name": backup_name,
                "timestamp": backup_result["timestamp"],
                "rubin_version": "2.0",
                "includes": {
                    "data": include_data,
                    "config": include_config,
                    "logs": include_logs
                }
            }
            
            # Резервное копирование базы данных
            if include_data:
                db_backup = self._backup_database(backup_path)
                if db_backup["success"]:
                    backup_result["files_backed_up"].extend(db_backup["files"])
                else:
                    backup_result["error"] = db_backup["error"]
                    return backup_result
            
            # Резервное копирование конфигурации
            if include_config:
                config_backup = self._backup_configuration(backup_path)
                if config_backup["success"]:
                    backup_result["files_backed_up"].extend(config_backup["files"])
                else:
                    backup_result["error"] = config_backup["error"]
                    return backup_result
            
            # Резервное копирование логов
            if include_logs:
                logs_backup = self._backup_logs(backup_path)
                if logs_backup["success"]:
                    backup_result["files_backed_up"].extend(logs_backup["files"])
            
            # Сохраняем метаданные
            metadata_path = os.path.join(backup_path, "backup_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Создаем архив
            archive_path = f"{backup_path}.zip"
            self._create_archive(backup_path, archive_path)
            
            # Вычисляем размер
            backup_result["size_mb"] = os.path.getsize(archive_path) / (1024 * 1024)
            backup_result["success"] = True
            
            # Удаляем временную директорию
            shutil.rmtree(backup_path)
            
            self.logger.info(f"✅ Резервная копия создана: {backup_name} ({backup_result['size_mb']:.2f} MB)")
            
        except Exception as e:
            backup_result["error"] = str(e)
            self.logger.error(f"❌ Ошибка создания резервной копии: {e}")
        
        return backup_result
    
    def _backup_database(self, backup_path: str) -> Dict:
        """Резервное копирование базы данных"""
        result = {"success": False, "files": [], "error": None}
        
        try:
            db_dir = os.path.join(backup_path, "database")
            os.makedirs(db_dir, exist_ok=True)
            
            # Копируем основные базы данных
            db_files = [
                "rubin_ai.db",
                "rubin_errors.db",
                "documents.db"
            ]
            
            for db_file in db_files:
                if os.path.exists(db_file):
                    backup_file = os.path.join(db_dir, db_file)
                    shutil.copy2(db_file, backup_file)
                    result["files"].append(backup_file)
                    self.logger.info(f"📊 Скопирована база данных: {db_file}")
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"❌ Ошибка резервного копирования БД: {e}")
        
        return result
    
    def _backup_configuration(self, backup_path: str) -> Dict:
        """Резервное копирование конфигурации"""
        result = {"success": False, "files": [], "error": None}
        
        try:
            config_dir = os.path.join(backup_path, "config")
            os.makedirs(config_dir, exist_ok=True)
            
            # Копируем конфигурационные файлы
            config_files = [
                "config/auto_heal.json",
                "config/modules.json",
                "config/settings.json"
            ]
            
            for config_file in config_files:
                if os.path.exists(config_file):
                    backup_file = os.path.join(config_dir, os.path.basename(config_file))
                    shutil.copy2(config_file, backup_file)
                    result["files"].append(backup_file)
                    self.logger.info(f"⚙️ Скопирован конфиг: {config_file}")
            
            # Копируем API конфигурации
            api_dir = os.path.join(backup_path, "api_config")
            os.makedirs(api_dir, exist_ok=True)
            
            api_files = [
                "api/rubin_ai_v2_simple.py",
                "api/electrical_api.py",
                "api/radiomechanics_api.py",
                "api/controllers_api.py",
                "api/documents_api.py"
            ]
            
            for api_file in api_files:
                if os.path.exists(api_file):
                    backup_file = os.path.join(api_dir, os.path.basename(api_file))
                    shutil.copy2(api_file, backup_file)
                    result["files"].append(backup_file)
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"❌ Ошибка резервного копирования конфигурации: {e}")
        
        return result
    
    def _backup_logs(self, backup_path: str) -> Dict:
        """Резервное копирование логов"""
        result = {"success": False, "files": [], "error": None}
        
        try:
            logs_dir = os.path.join(backup_path, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            # Копируем логи за последние 7 дней
            log_files = []
            for root, dirs, files in os.walk("."):
                for file in files:
                    if file.endswith(('.log', '.txt')) and 'log' in file.lower():
                        file_path = os.path.join(root, file)
                        if os.path.getmtime(file_path) > (datetime.now() - timedelta(days=7)).timestamp():
                            log_files.append(file_path)
            
            for log_file in log_files:
                backup_file = os.path.join(logs_dir, os.path.basename(log_file))
                shutil.copy2(log_file, backup_file)
                result["files"].append(backup_file)
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"❌ Ошибка резервного копирования логов: {e}")
        
        return result
    
    def _create_archive(self, source_dir: str, archive_path: str):
        """Создание архива резервной копии"""
        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(source_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, source_dir)
                        zipf.write(file_path, arcname)
            
            self.logger.info(f"📦 Создан архив: {archive_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания архива: {e}")
            raise
    
    def restore_backup(self, backup_name: str, restore_data: bool = True, 
                      restore_config: bool = True, restore_logs: bool = False) -> Dict:
        """Восстановление из резервной копии"""
        restore_result = {
            "backup_name": backup_name,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "files_restored": [],
            "error": None
        }
        
        try:
            archive_path = os.path.join(self.backup_dir, f"{backup_name}.zip")
            
            if not os.path.exists(archive_path):
                restore_result["error"] = f"Резервная копия не найдена: {backup_name}"
                return restore_result
            
            # Создаем временную директорию для извлечения
            temp_dir = os.path.join(self.backup_dir, f"temp_{backup_name}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Извлекаем архив
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Читаем метаданные
            metadata_path = os.path.join(temp_dir, "backup_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    restore_result["backup_info"] = metadata
            
            # Восстанавливаем базу данных
            if restore_data:
                db_restore = self._restore_database(temp_dir)
                if db_restore["success"]:
                    restore_result["files_restored"].extend(db_restore["files"])
                else:
                    restore_result["error"] = db_restore["error"]
                    return restore_result
            
            # Восстанавливаем конфигурацию
            if restore_config:
                config_restore = self._restore_configuration(temp_dir)
                if config_restore["success"]:
                    restore_result["files_restored"].extend(config_restore["files"])
                else:
                    restore_result["error"] = config_restore["error"]
                    return restore_result
            
            # Восстанавливаем логи
            if restore_logs:
                logs_restore = self._restore_logs(temp_dir)
                if logs_restore["success"]:
                    restore_result["files_restored"].extend(logs_restore["files"])
            
            # Удаляем временную директорию
            shutil.rmtree(temp_dir)
            
            restore_result["success"] = True
            self.logger.info(f"✅ Восстановление завершено: {backup_name}")
            
        except Exception as e:
            restore_result["error"] = str(e)
            self.logger.error(f"❌ Ошибка восстановления: {e}")
        
        return restore_result
    
    def _restore_database(self, temp_dir: str) -> Dict:
        """Восстановление базы данных"""
        result = {"success": False, "files": [], "error": None}
        
        try:
            db_dir = os.path.join(temp_dir, "database")
            
            if os.path.exists(db_dir):
                for db_file in os.listdir(db_dir):
                    source_file = os.path.join(db_dir, db_file)
                    target_file = db_file
                    
                    # Создаем резервную копию существующей БД
                    if os.path.exists(target_file):
                        backup_name = f"{target_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        shutil.copy2(target_file, backup_name)
                    
                    shutil.copy2(source_file, target_file)
                    result["files"].append(target_file)
                    self.logger.info(f"📊 Восстановлена база данных: {db_file}")
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"❌ Ошибка восстановления БД: {e}")
        
        return result
    
    def _restore_configuration(self, temp_dir: str) -> Dict:
        """Восстановление конфигурации"""
        result = {"success": False, "files": [], "error": None}
        
        try:
            config_dir = os.path.join(temp_dir, "config")
            
            if os.path.exists(config_dir):
                os.makedirs("config", exist_ok=True)
                
                for config_file in os.listdir(config_dir):
                    source_file = os.path.join(config_dir, config_file)
                    target_file = os.path.join("config", config_file)
                    
                    # Создаем резервную копию существующего конфига
                    if os.path.exists(target_file):
                        backup_name = f"{target_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        shutil.copy2(target_file, backup_name)
                    
                    shutil.copy2(source_file, target_file)
                    result["files"].append(target_file)
                    self.logger.info(f"⚙️ Восстановлен конфиг: {config_file}")
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"❌ Ошибка восстановления конфигурации: {e}")
        
        return result
    
    def _restore_logs(self, temp_dir: str) -> Dict:
        """Восстановление логов"""
        result = {"success": False, "files": [], "error": None}
        
        try:
            logs_dir = os.path.join(temp_dir, "logs")
            
            if os.path.exists(logs_dir):
                for log_file in os.listdir(logs_dir):
                    source_file = os.path.join(logs_dir, log_file)
                    target_file = log_file
                    
                    shutil.copy2(source_file, target_file)
                    result["files"].append(target_file)
                    self.logger.info(f"📝 Восстановлен лог: {log_file}")
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"❌ Ошибка восстановления логов: {e}")
        
        return result
    
    def list_backups(self) -> List[Dict]:
        """Получение списка резервных копий"""
        backups = []
        
        try:
            for file in os.listdir(self.backup_dir):
                if file.endswith('.zip'):
                    backup_name = file[:-4]  # Убираем .zip
                    file_path = os.path.join(self.backup_dir, file)
                    
                    # Получаем информацию о файле
                    stat = os.stat(file_path)
                    size_mb = stat.st_size / (1024 * 1024)
                    created_time = datetime.fromtimestamp(stat.st_ctime)
                    
                    # Пытаемся прочитать метаданные
                    metadata = None
                    try:
                        with zipfile.ZipFile(file_path, 'r') as zipf:
                            if 'backup_metadata.json' in zipf.namelist():
                                with zipf.open('backup_metadata.json') as f:
                                    metadata = json.load(f)
                    except Exception:
                        pass
                    
                    backups.append({
                        "name": backup_name,
                        "size_mb": round(size_mb, 2),
                        "created": created_time.isoformat(),
                        "metadata": metadata
                    })
            
            # Сортируем по дате создания (новые сначала)
            backups.sort(key=lambda x: x["created"], reverse=True)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения списка резервных копий: {e}")
        
        return backups
    
    def delete_backup(self, backup_name: str) -> Dict:
        """Удаление резервной копии"""
        result = {
            "backup_name": backup_name,
            "success": False,
            "error": None
        }
        
        try:
            archive_path = os.path.join(self.backup_dir, f"{backup_name}.zip")
            
            if os.path.exists(archive_path):
                os.remove(archive_path)
                result["success"] = True
                self.logger.info(f"🗑️ Удалена резервная копия: {backup_name}")
            else:
                result["error"] = f"Резервная копия не найдена: {backup_name}"
                
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"❌ Ошибка удаления резервной копии: {e}")
        
        return result
    
    def cleanup_old_backups(self, days_to_keep: int = 30) -> Dict:
        """Очистка старых резервных копий"""
        result = {
            "deleted_count": 0,
            "deleted_backups": [],
            "success": False,
            "error": None
        }
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            for file in os.listdir(self.backup_dir):
                if file.endswith('.zip'):
                    file_path = os.path.join(self.backup_dir, file)
                    created_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    
                    if created_time < cutoff_date:
                        backup_name = file[:-4]
                        delete_result = self.delete_backup(backup_name)
                        
                        if delete_result["success"]:
                            result["deleted_count"] += 1
                            result["deleted_backups"].append(backup_name)
            
            result["success"] = True
            self.logger.info(f"🧹 Очищено старых резервных копий: {result['deleted_count']}")
            
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"❌ Ошибка очистки резервных копий: {e}")
        
        return result

# Глобальный экземпляр менеджера резервных копий
backup_manager = BackupManager()

















