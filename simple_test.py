#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔧 ПРОСТОЙ ТЕСТ HTTP 500 ИСПРАВЛЕНИЯ
===================================
"""

import requests
import json

def test_simple():
    """Простой тест исправления"""
    print("🔧 ПРОСТОЙ ТЕСТ HTTP 500 ИСПРАВЛЕНИЯ")
    print("=" * 50)
    
    # Тест запроса к Smart Dispatcher
    message = "C:\\Users\\elekt\\OneDrive\\Desktop\\VMB630_v_005_019_000\\out\\plc_18_background_ctrl.plc прочти и найди ошибку"
    
    try:
        response = requests.post('http://localhost:8080/api/chat', 
                               json={'message': message})
        
        print(f"📊 Статус ответа: {response.status_code}")
        print(f"📋 Содержимое ответа:")
        print(response.text)
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ JSON ответ получен:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            # Проверяем структуру ответа
            if 'response' in data:
                if 'explanation' in data['response']:
                    print(f"\n🎉 УСПЕХ! Ответ содержит объяснение")
                    return True
                else:
                    print(f"\n⚠️ Ответ не содержит объяснения")
                    print(f"Доступные ключи: {list(data['response'].keys())}")
            else:
                print(f"\n⚠️ Ответ не содержит 'response'")
                print(f"Доступные ключи: {list(data.keys())}")
        else:
            print(f"\n❌ HTTP ошибка: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    return False

if __name__ == "__main__":
    test_simple()





