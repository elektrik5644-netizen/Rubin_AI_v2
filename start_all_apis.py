import subprocess
import os
import sys
import time
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Полный путь к исполняемому файлу Python
# Используйте sys.executable для получения текущего интерпретатора Python
PYTHON_EXECUTABLE = sys.executable

# Пути к API файлам
API_FILES = {
    "controllers_api": {"path": "api/controllers_api.py", "port": 8090},
    "electrical_api": {"path": "api/electrical_api.py", "port": 8087},
    "radiomechanics_api": {"path": "api/radiomechanics_api.py", "port": 8089},
    "documents_api": {"path": "api/documents_api.py", "port": 8088},
    "rubin_ai_v2_server": {"path": "api/rubin_ai_v2_server.py", "port": 8084} # Основной сервер для общих запросов
}

# Список процессов, чтобы мы могли их остановить позже
running_processes = {}

def start_api(name, file_path, port):
    """Запускает один API-сервер в фоновом режиме."""
    try:
        # Убедимся, что путь к файлу корректен относительно текущей рабочей директории
        full_path = os.path.join(os.getcwd(), file_path)
        
        command = [PYTHON_EXECUTABLE, full_path]
        
        logger.info(f"🚀 Запуск {name} на порту {port} командой: {' '.join(command)}")
        
        # Запускаем процесс в фоновом режиме
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0 # Для Windows
        )
        running_processes[name] = process
        logger.info(f"✅ {name} запущен с PID: {process.pid}")
        return process
    except Exception as e:
        logger.error(f"❌ Ошибка при запуске {name}: {e}")
        return None

def stop_all_apis():
    """Останавливает все запущенные API-серверы."""
    logger.info("⛔ Остановка всех API-серверов...")
    for name, process in running_processes.items():
        try:
            if process.poll() is None:  # Если процесс еще запущен
                if sys.platform == "win32":
                    subprocess.run(f"TASKKILL /F /PID {process.pid} /T", shell=True) # Принудительное завершение для Windows
                else:
                    process.terminate()
                    process.wait(timeout=5)
                    if process.poll() is None:
                        process.kill()
                logger.info(f"✅ {name} (PID: {process.pid}) остановлен.")
            else:
                logger.info(f"ℹ️ {name} (PID: {process.pid}) уже был остановлен.")
        except Exception as e:
            logger.error(f"❌ Ошибка при остановке {name} (PID: {process.pid}): {e}")
    running_processes.clear()
    logger.info("Все API-серверы остановлены.")

if __name__ == "__main__":
    logger.info("🛠️ Запуск всех API-серверов Rubin AI v2.0...")
    
    # Запускаем каждый API
    for name, config in API_FILES.items():
        start_api(name, config["path"], config["port"])
        time.sleep(1) # Даем время на запуск
    
    logger.info("✅ Все API-серверы запущены. Ожидание завершения работы...")
    logger.info("Для остановки всех серверов нажмите Ctrl+C (возможно, потребуется несколько нажатий).")
    logger.info(f"Если Python-интерпретатор установлен в Microsoft Store, используйте полный путь: {PYTHON_EXECUTABLE}")

    try:
        # Держим главный скрипт запущенным, чтобы фоновые процессы не завершились
        # и чтобы можно было остановить их по Ctrl+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nReceived KeyboardInterrupt. Stopping all APIs...")
        stop_all_apis()
    except Exception as e:
        logger.error(f"Неожиданная ошибка в основном цикле: {e}")
        stop_all_apis()









