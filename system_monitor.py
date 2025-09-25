import psutil
import time

def monitor_system():
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')
        
        print(f"CPU Usage: {cpu_usage}%")
        print(f"Memory Usage: {memory_info.percent}%")
        print(f"Disk Usage: {disk_usage.percent}%")
        print("-" * 30)
        
        time.sleep(5)  # Интервал между проверками

if __name__ == "__main__":
    monitor_system()

