#!/usr/bin/env python3
"""
Проверка статуса системы Rubin AI
"""

import sys
import os
sys.path.append('.')

def check_system_status():
    """Проверка статуса всех компонентов системы"""
    
    print("🔍 ПРОВЕРКА СТАТУСА СИСТЕМЫ RUBIN AI")
    print("=" * 50)
    
    # Проверка Intelligent Dispatcher
    try:
        from intelligent_dispatcher import get_intelligent_dispatcher
        dispatcher = get_intelligent_dispatcher()
        print("✅ Intelligent Dispatcher: OK")
    except Exception as e:
        print(f"❌ Intelligent Dispatcher: {e}")
    
    # Проверка Neural Network
    try:
        from neural_rubin import get_neural_rubin
        neural = get_neural_rubin()
        stats = neural.get_neural_stats()
        print(f"✅ Neural Network: {stats['neural_network_active']}")
        print(f"   - Device: {stats['device']}")
        print(f"   - Parameters: {stats['model_parameters']:,}")
    except Exception as e:
        print(f"❌ Neural Network: {e}")
    
    # Проверка Provider Selector
    try:
        from providers.smart_provider_selector import SmartProviderSelector
        selector = SmartProviderSelector()
        print("✅ Provider Selector: OK")
    except Exception as e:
        print(f"❌ Provider Selector: {e}")
    
    # Проверка Hugging Face Provider
    try:
        from providers.huggingface_provider import HuggingFaceProvider
        hf = HuggingFaceProvider()
        print(f"✅ Hugging Face Provider: {hf.is_available}")
    except Exception as e:
        print(f"❌ Hugging Face Provider: {e}")
    
    # Проверка Google Cloud Provider
    try:
        from providers.google_cloud_provider import GoogleCloudProvider
        gc = GoogleCloudProvider()
        print(f"✅ Google Cloud Provider: {gc.is_available}")
    except Exception as e:
        print(f"❌ Google Cloud Provider: {e}")
    
    # Проверка базы данных
    try:
        import sqlite3
        if os.path.exists('rubin_ai_v2.db'):
            conn = sqlite3.connect('rubin_ai_v2.db')
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"✅ Database: {len(tables)} tables")
            conn.close()
        else:
            print("❌ Database: Not found")
    except Exception as e:
        print(f"❌ Database: {e}")
    
    # Проверка документов
    try:
        import pickle
        if os.path.exists('documents_storage_v2.pkl'):
            with open('documents_storage_v2.pkl', 'rb') as f:
                docs = pickle.load(f)
            print(f"✅ Documents: {len(docs)} documents")
        else:
            print("❌ Documents: Not found")
    except Exception as e:
        print(f"❌ Documents: {e}")
    
    print("\n🎯 РЕКОМЕНДАЦИИ:")
    print("1. Убедитесь, что все AI провайдеры инициализированы")
    print("2. Проверьте наличие API ключей в .env файле")
    print("3. Запустите систему заново: python start_rubin_stable_v2.py")
    print("4. Проверьте логи: tail -f rubin_ai_v2.log")

if __name__ == "__main__":
    check_system_status()