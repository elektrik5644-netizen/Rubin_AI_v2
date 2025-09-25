import numpy as np
import math
import logging
from typing import Union, Tuple, List

logger = logging.getLogger(__name__)

class RubinDataPreprocessor:
    """Класс для различных методов нормализации числовых данных."""

    def __init__(self):
        pass

    def linear_normalization(self, data: Union[np.ndarray, List[float]], min_val: float, max_val: float) -> Union[np.ndarray, List[float]]:
        """Линейная нормализация данных в диапазон [0, 1] или [min_out, max_out]."""
        data_arr = np.array(data, dtype=float)
        
        if max_val - min_val == 0:
            logger.warning("⚠️ Диапазон min_val и max_val равен нулю, линейная нормализация невозможна. Возвращаю исходные данные.")
            return data_arr
            
        normalized_data = (data_arr - min_val) / (max_val - min_val)
        return normalized_data

    def softstep_normalization(self, data: Union[np.ndarray, List[float]], koef: float = 1.0) -> Union[np.ndarray, List[float]]:
        """Нормализация данных с использованием логистической (SoftStep) функции в пределах (0, 1)."""
        data_arr = np.array(data, dtype=float)
        # Логистическая функция: 1 / (1 + exp(-x))
        normalized_data = 1 / (1 + np.exp(-koef * data_arr))
        return normalized_data

    def arctg_normalization(self, data: Union[np.ndarray, List[float]], koef: float = 1.0) -> Union[np.ndarray, List[float]]:
        """Нормализация данных с использованием функции арктангенса в пределах (-pi/2, pi/2)."""
        data_arr = np.array(data, dtype=float)
        normalized_data = np.arctan(koef * data_arr)
        return normalized_data

    def gaussian_normalization(self, data: Union[np.ndarray, List[float]], mean: float, std_dev: float) -> Union[np.ndarray, List[float]]:
        """Нормализация данных по Гауссу (Z-score нормализация)."""
        data_arr = np.array(data, dtype=float)

        if std_dev == 0:
            logger.warning("⚠️ Стандартное отклонение равно нулю, Гауссова нормализация невозможна. Возвращаю исходные данные.")
            return data_arr
            
        normalized_data = (data_arr - mean) / std_dev
        return normalized_data

if __name__ == "__main__":
    logger.info("🧪 ТЕСТИРОВАНИЕ RubinDataPreprocessor")
    preprocessor = RubinDataPreprocessor()

    test_data = np.array([-10, -5, 0, 5, 10], dtype=float)
    print(f"\nИсходные данные: {test_data}")

    # Линейная нормализация
    linear_norm = preprocessor.linear_normalization(test_data, min_val=-10, max_val=10)
    print(f"Линейная нормализация ([-10,10] -> [0,1]): {linear_norm}")
    assert np.allclose(linear_norm, [0.0, 0.25, 0.5, 0.75, 1.0])

    # SoftStep нормализация
    softstep_norm = preprocessor.softstep_normalization(test_data)
    print(f"SoftStep нормализация: {softstep_norm}")
    # Проверим, что значения находятся в диапазоне (0, 1)
    assert np.all(softstep_norm > 0) and np.all(softstep_norm < 1)

    # Arctan нормализация
    arctg_norm = preprocessor.arctg_normalization(test_data)
    print(f"Arctg нормализация: {arctg_norm}")
    # Проверим, что значения находятся в диапазоне (-pi/2, pi/2)
    assert np.all(arctg_norm > -math.pi/2) and np.all(arctg_norm < math.pi/2)

    # Гауссова нормализация (Z-score)
    mean_test = np.mean(test_data)
    std_dev_test = np.std(test_data)
    gaussian_norm = preprocessor.gaussian_normalization(test_data, mean=mean_test, std_dev=std_dev_test)
    print(f"Гауссова нормализация (mean={mean_test:.2f}, std={std_dev_test:.2f}): {gaussian_norm}")
    assert np.isclose(np.mean(gaussian_norm), 0.0) and np.isclose(np.std(gaussian_norm), 1.0)

    logger.info("✅ Все тесты RubinDataPreprocessor пройдены успешно!")









