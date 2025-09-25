#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Подсчет нейронов в Rubin AI
"""

def count_neurons_in_rubin():
    """Подсчитывает количество нейронов в нейронной сети Rubin AI"""
    
    print("=" * 60)
    print("ПОДСЧЕТ НЕЙРОНОВ В RUBIN AI")
    print("=" * 60)
    
    # Архитектура нейронной сети Rubin AI
    input_size = 384  # Размер входного слоя (embeddings)
    hidden_sizes = [512, 256, 128]  # Скрытые слои
    num_classes = 10  # Выходной слой (категории)
    
    print("АРХИТЕКТУРА НЕЙРОННОЙ СЕТИ:")
    print("-" * 40)
    print(f"Входной слой:     {input_size:4d} нейронов")
    
    total_neurons = input_size
    
    for i, hidden_size in enumerate(hidden_sizes):
        print(f"Скрытый слой {i+1}:   {hidden_size:4d} нейронов")
        total_neurons += hidden_size
    
    print(f"Выходной слой:     {num_classes:4d} нейронов")
    total_neurons += num_classes
    
    print("-" * 40)
    print(f"ОБЩЕЕ КОЛИЧЕСТВО НЕЙРОНОВ: {total_neurons}")
    print()
    
    # Подсчет связей (весов)
    print("ПОДСЧЕТ СВЯЗЕЙ (ВЕСОВ):")
    print("-" * 40)
    
    total_connections = 0
    current_size = input_size
    
    for i, hidden_size in enumerate(hidden_sizes):
        connections = current_size * hidden_size
        print(f"Слой {i+1}: {current_size:4d} x {hidden_size:4d} = {connections:6d} связей")
        total_connections += connections
        current_size = hidden_size
    
    # Выходной слой
    output_connections = current_size * num_classes
    print(f"Выход: {current_size:4d} x {num_classes:4d} = {output_connections:6d} связей")
    total_connections += output_connections
    
    print("-" * 40)
    print(f"ОБЩЕЕ КОЛИЧЕСТВО СВЯЗЕЙ: {total_connections}")
    print()
    
    # Дополнительная информация
    print("ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:")
    print("-" * 40)
    print(f"Размер embeddings: {input_size} (Sentence Transformers)")
    print(f"Количество категорий: {num_classes}")
    print(f"Функции активации: ReLU для скрытых слоев")
    print(f"Dropout: 0.2 (20%)")
    print()
    
    # Сравнение с другими сетями
    print("СРАВНЕНИЕ С ДРУГИМИ СЕТЯМИ:")
    print("-" * 40)
    print(f"Rubin AI:           {total_neurons:6d} нейронов")
    print(f"LeNet-5:            {2274:6d} нейронов")
    print(f"AlexNet:            {62378:6d} нейронов")
    print(f"VGG-16:             {138357544:6d} нейронов")
    print(f"ResNet-50:          {25557032:6d} нейронов")
    print()
    
    print("=" * 60)
    print(f"ИТОГО: В RUBIN AI {total_neurons} НЕЙРОНОВ")
    print("=" * 60)
    
    return total_neurons, total_connections

if __name__ == "__main__":
    try:
        neurons, connections = count_neurons_in_rubin()
    except Exception as e:
        print(f"Ошибка: {e}")





