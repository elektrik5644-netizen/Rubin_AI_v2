import numpy as np
import math
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class NPriceType:
    PriceHigh = 0
    PriceLow = 1
    PriceClose = 2

class RubinTimeSeriesProcessor:
    """Адаптированный класс для обработки временных рядов (модуль 'Trade' из NeuroRepository)."""

    def __init__(self, period: int = 1, price_type: int = NPriceType.PriceClose, len_in: int = 0,
                 koef_tg: float = 10000.0, koef_price: float = 1.0, koef_volume: float = 1.0):
        self.period = period
        self.price_type = price_type
        self.len_in = len_in
        self.koef_tg = koef_tg
        self.koef_price = koef_price
        self.koef_volume = koef_volume

        self.data_matrix: Optional[np.ndarray] = None
        self.data_period: Optional[np.ndarray] = None
        self.data_tg: Optional[np.ndarray] = None
        self.examples: List[Dict[str, np.ndarray]] = []

    def set_parameters(self, period: int, price_type: int, len_in: int,
                       koef_tg: float, koef_price: float, koef_volume: float):
        self.period = period
        self.price_type = price_type
        self.len_in = len_in
        self.koef_tg = koef_tg
        self.koef_price = koef_price
        self.koef_volume = koef_volume

    def preprocess_data(self, raw_data: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """Основной метод предобработки данных временных рядов."""
        self.data_matrix = raw_data
        self._calc_period()
        self._calc_tg()
        self._calc_example()
        return self.examples

    def postprocess_output(self, nn_output: np.ndarray) -> float:
        """Постобработка выхода нейронной сети."""
        if nn_output.size == 0:
            return 0.0 # или другое значение по умолчанию

        # Предполагаем, что nn_output - это одномерный массив или скаляр
        value = math.tan(nn_output.flatten()[0]) / (self.koef_tg * self.koef_price)
        return value

    def _calc_period(self):
        """Агрегация исходных данных по заданному периоду."""
        if self.data_matrix is None or self.period <= 0:
            logger.warning("⚠️ Исходные данные отсутствуют или период некорректен для _calc_period.")
            return

        num_rows, num_cols = self.data_matrix.shape
        new_num_rows = num_rows // self.period

        if new_num_rows == 0: # Если данных меньше периода
            logger.warning("⚠️ Недостаточно данных для агрегации по периоду.")
            self.data_period = np.array([])
            return

        self.data_period = np.zeros((new_num_rows, 5), dtype=float) # O, H, L, C, V

        for knd in range(new_num_rows):
            start_row = knd * self.period
            end_row = min((knd + 1) * self.period, num_rows)
            period_data = self.data_matrix[start_row:end_row, :]

            if period_data.size == 0:
                continue

            v_open = period_data[0, 0] # Open
            v_close = period_data[-1, 3] # Close
            v_high = np.max(period_data[:, 1]) # High
            v_low = np.min(period_data[:, 2]) # Low
            v_volume = np.sum(period_data[:, 4]) # Volume

            self.data_period[knd, 0] = v_open
            self.data_period[knd, 1] = v_high
            self.data_period[knd, 2] = v_low
            self.data_period[knd, 3] = v_close
            self.data_period[knd, 4] = v_volume

        logger.info(f"✅ Данные агрегированы по периоду {self.period}. Новая размерность: {self.data_period.shape}")

    def _calc_tg(self):
        """Выполнение преобразования данных с atan() и коэффициентами."""
        if self.data_period is None or self.data_period.size == 0 or self.data_period.shape[0] < 2:
            logger.warning("⚠️ Агрегированные данные отсутствуют или недостаточны для _calc_tg.")
            self.data_tg = np.array([])
            return

        num_rows, num_cols = self.data_period.shape
        self.data_tg = np.zeros((num_rows - 1, num_cols), dtype=float)

        for ind in range(1, num_rows):
            for jnd in range(num_cols):
                d_value = self.data_period[ind, jnd] - self.data_period[ind - 1, jnd]
                if jnd == num_cols - 1: # Volume
                    d_value = self.koef_volume * d_value
                else: # Price
                    d_value = self.koef_tg * self.koef_price * d_value
                
                # Избегаем atan() от очень больших значений, которые могут привести к Inf/NaN
                d_value = np.clip(d_value, -1e10, 1e10) # Ограничиваем значение перед atan
                self.data_tg[ind - 1, jnd] = math.atan(d_value)

        logger.info(f"✅ Данные преобразованы с atan(). Новая размерность: {self.data_tg.shape}")

    def _calc_example(self):
        """Формирование обучающих примеров (входных и выходных векторов)."""
        self.examples = []
        if self.data_tg is None or self.data_tg.size == 0 or self.len_in <= 0:
            logger.warning("⚠️ Преобразованные данные или len_in некорректны для _calc_example.")
            return

        num_rows, num_cols = self.data_tg.shape
        effective_rows = num_rows - self.len_in # +1 для output, -1 так как начинаем с 0

        if effective_rows <= 0:
            logger.warning("⚠️ Недостаточно данных для формирования примеров с текущим len_in.")
            return

        for ind in range(effective_rows):
            input_vector = []
            for knd in range(self.len_in):
                # Пропускаем Open (столбец 0) как в NTradeTg::calcExample
                input_vector.extend(self.data_tg[ind + knd, 1:num_cols]) 

            output_value = 0.0
            # Определение выходного значения в зависимости от price_type
            # Аналогично NTradeTg::calcExample
            if self.price_type == NPriceType.PriceHigh:
                output_value = self.data_tg[ind + self.len_in, 1] # High
            elif self.price_type == NPriceType.PriceLow:
                output_value = self.data_tg[ind + self.len_in, 2] # Low
            elif self.price_type == NPriceType.PriceClose:
                output_value = self.data_tg[ind + self.len_in, 3] # Close
            
            self.examples.append({
                'input': np.array(input_vector, dtype=float),
                'output': np.array([output_value], dtype=float)
            })
        logger.info(f"✅ Сформировано {len(self.examples)} обучающих примеров.")

if __name__ == "__main__":
    # Пример использования и тестирование
    logger.info("🧪 ТЕСТИРОВАНИЕ RubinTimeSeriesProcessor")
    
    # Пример сырых данных (Open, High, Low, Close, Volume)
    # 5 столбцов, как ожидается в NTradeTg
    raw_data = np.array([
        [100, 105, 98, 103, 1000],
        [103, 107, 101, 106, 1200],
        [106, 110, 104, 108, 1100],
        [108, 112, 106, 110, 1300],
        [110, 115, 108, 113, 1500],
        [113, 117, 111, 116, 1400],
        [116, 120, 114, 119, 1600],
        [119, 122, 117, 121, 1700],
        [121, 125, 119, 123, 1800],
        [123, 127, 121, 125, 1900],
    ])
    
    # Тестирование с периодом 2, предсказание Close, len_in 2
    processor = RubinTimeSeriesProcessor(
        period=2, 
        price_type=NPriceType.PriceClose, 
        len_in=2, 
        koef_tg=10000.0, 
        koef_price=1.0, 
        koef_volume=1.0
    )
    
    processed_examples = processor.preprocess_data(raw_data)
    
    if processed_examples:
        print("\nСформированные примеры (первые 2):")
        for i, example in enumerate(processed_examples[:2]):
            print(f"  Пример {i+1}:")
            print(f"    Input: {example['input']}")
            print(f"    Output: {example['output']}")

        # Тестирование postprocess_output
        test_nn_output = np.array([math.atan(0.0001 * 10000 * 1)]) # Пример выхода нейросети
        postprocessed_val = processor.postprocess_output(test_nn_output)
        print(f"\nТестовый postprocess_output для {test_nn_output}: {postprocessed_val}")
    else:
        logger.warning("⚠️ Не удалось сформировать примеры для тестирования.")

    # Тестирование с len_in > num_rows
    logger.info("\n🧪 Тестирование с некорректным len_in (больше доступных данных)")
    processor_invalid = RubinTimeSeriesProcessor(period=1, price_type=NPriceType.PriceClose, len_in=100)
    processed_examples_invalid = processor_invalid.preprocess_data(raw_data)
    assert not processed_examples_invalid, "Должно быть пустым"
    logger.info("✅ Тест с некорректным len_in пройден.")

    logger.info("\n🧪 Тестирование с другим period и len_in")
    processor_diff = RubinTimeSeriesProcessor(period=3, price_type=NPriceType.PriceHigh, len_in=1)
    processed_examples_diff = processor_diff.preprocess_data(raw_data)
    if processed_examples_diff:
        print("\nСформированные примеры (period=3, len_in=1, PriceHigh):")
        for i, example in enumerate(processed_examples_diff[:2]):
            print(f"  Пример {i+1}:")
            print(f"    Input: {example['input']}")
            print(f"    Output: {example['output']}")
    else:
        logger.warning("⚠️ Не удалось сформировать примеры для второго теста.")









