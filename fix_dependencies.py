#!/usr/bin/env python3
"""
Исправление зависимостей Rubin AI
"""

import subprocess
import sys
import os

def fix_dependencies():
    """Исправление проблем с зависимостями"""
    
    print("🔧 ИСПРАВЛЕНИЕ ЗАВИСИМОСТЕЙ RUBIN AI")
    print("=" * 50)
    
    # Обновление Hugging Face Hub
    print("📦 Обновление Hugging Face Hub...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "huggingface_hub"], check=True)
        print("✅ Hugging Face Hub обновлен")
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка обновления Hugging Face Hub: {e}")
    
    # Установка совместимой версии
    print("📦 Установка совместимой версии...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub==0.19.4"], check=True)
        print("✅ Совместимая версия установлена")
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки совместимой версии: {e}")
    
    # Установка Sentence Transformers
    print("📦 Установка Sentence Transformers...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "sentence-transformers"], check=True)
        print("✅ Sentence Transformers установлен")
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки Sentence Transformers: {e}")
    
    # Установка PyTorch
    print("📦 Установка PyTorch...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"], check=True)
        print("✅ PyTorch установлен")
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки PyTorch: {e}")
    
    # Установка Transformers
    print("📦 Установка Transformers...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "transformers"], check=True)
        print("✅ Transformers установлен")
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки Transformers: {e}")
    
    print("\n🎯 ЗАВИСИМОСТИ ИСПРАВЛЕНЫ!")
    print("Теперь запустите: python check_system_status.py")

if __name__ == "__main__":
    fix_dependencies()












