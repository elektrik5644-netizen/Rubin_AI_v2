# test_model_loader.py
import logging
import traceback

# Устанавливаем базовую конфигурацию логирования, чтобы видеть все сообщения
logging.basicConfig(level=logging.INFO)
print("--- Запуск теста загрузки нейросетевой модели ---")

try:
    # Пытаемся импортировать основной класс
    from neural_rubin import NeuralRubinAI
    print("✅ Файл neural_rubin.py успешно импортирован.")

    # Пытаемся создать экземпляр класса, что вызовет загрузку моделей
    print("⏳ Попытка инициализации NeuralRubinAI (это может занять время)...")
    ai = NeuralRubinAI()

    # Проверяем, что модели действительно загрузились
    print("--- Результаты инициализации ---")
    print(f"Устройство для вычислений: {ai.device}")
    if ai.sentence_model:
        print("✅ Модель векторизации (SentenceTransformer) успешно загружена.")
    else:
        print("❌ Модель векторизации (SentenceTransformer) НЕ загружена.")

    if ai.neural_network:
        print("✅ Основная нейросеть (PyTorch) успешно создана.")
    else:
        print("❌ Основная нейросеть (PyTorch) НЕ создана.")

    print("\n--- Тест завершен успешно ---")

except Exception as e:
    print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: Тест провалился на этапе импорта или инициализации.")
    print("--- Текст ошибки ---")
    traceback.print_exc()
    print("--- Конец ошибки ---")
