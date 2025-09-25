#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rubin AI v2 - Electrical Engineering API Server
Сервер для обработки вопросов по электротехнике
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
from datetime import datetime
import os
import re
import xml.etree.ElementTree as ET
from io import BytesIO

# Опциональные зависимости для анализа изображений
try:
    from PIL import Image
    import numpy as np
    PIL_NUMPY_AVAILABLE = True
except Exception:
    PIL_NUMPY_AVAILABLE = False

# Попытка импорта электротехнических утилит
try:
    from electrical_utils import ElectricalUtils
    from electrical_knowledge_handler import ElectricalKnowledgeHandler
    ELECTRICAL_UTILS_AVAILABLE = True
    electrical_utils = ElectricalUtils()
    electrical_handler = ElectricalKnowledgeHandler()
except ImportError:
    ELECTRICAL_UTILS_AVAILABLE = False
    electrical_utils = None
    electrical_handler = None

app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

# Обработка CORS preflight запросов
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization")
        response.headers.add('Access-Control-Allow-Methods', "GET,POST,OPTIONS")
        response.headers.add('Content-Type', 'application/json; charset=utf-8')
        return response

# Установка правильных заголовков для всех ответов
@app.after_request
def after_request(response):
    if response.content_type == 'application/json':
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

# База знаний по электротехнике
ELECTRICAL_KNOWLEDGE = {
    "диод": {
        "title": "Диод",
        "description": "Полупроводниковый прибор, пропускающий ток только в одном направлении",
        "explanation": """
**Диод - это полупроводниковый прибор с двумя электродами:**

**Принцип работы:**
• Пропускает ток только в одном направлении (прямое включение)
• Блокирует ток в обратном направлении (обратное включение)
• Основан на p-n переходе

**Типы диодов:**
• Выпрямительные диоды - для выпрямления переменного тока
• Стабилитроны - для стабилизации напряжения
• Светодиоды (LED) - излучают свет при прохождении тока
• Фотодиоды - преобразуют свет в электрический ток

**Характеристики:**
• Прямое напряжение: 0.6-0.7В (кремний), 0.2-0.3В (германий)
• Обратное напряжение - максимальное напряжение в обратном направлении
• Максимальный прямой ток

**Применение:**
• Выпрямление переменного тока
• Защита от обратной полярности
• Стабилизация напряжения
• Индикация (светодиоды)
• Детектирование сигналов
        """,
        "keywords": ["диод", "полупроводник", "p-n переход", "выпрямитель", "стабилитрон", "светодиод", "led"]
    },
    "закон ома": {
        "description": "Основной закон электротехники, связывающий напряжение, ток и сопротивление",
        "formula": "U = I × R",
        "explanation": """
**Закон Ома для участка цепи:**
• U = I × R (напряжение = ток × сопротивление)
• I = U / R (ток = напряжение / сопротивление)  
• R = U / I (сопротивление = напряжение / ток)

**Закон Ома для полной цепи:**
• I = E / (R + r)
• E - ЭДС источника
• R - внешнее сопротивление
• r - внутреннее сопротивление источника

**Применение:**
• Расчет токов в цепях
• Выбор резисторов
• Анализ цепей постоянного тока
        """,
        "examples": [
            "При напряжении 12В и сопротивлении 4Ом ток = 12/4 = 3А",
            "При токе 2А и сопротивлении 6Ом напряжение = 2×6 = 12В"
        ]
    },
    
    "закон кирхгофа": {
        "title": "Законы Кирхгофа",
        "description": "Фундаментальные законы для анализа электрических цепей",
        "explanation": """
**Первый закон Кирхгофа (закон токов):**
• Сумма токов, входящих в узел = сумме токов, выходящих из узла
• ΣIвх = ΣIвых

**Второй закон Кирхгофа (закон напряжений):**
• Сумма ЭДС в замкнутом контуре = сумме падений напряжений
• ΣE = Σ(I×R)

**Применение:**
• Анализ сложных цепей
• Расчет токов в разветвленных цепях
• Проверка правильности расчетов
        """,
        "examples": [
            "В узле: I1 + I2 = I3 + I4",
            "В контуре: E1 - E2 = I1×R1 + I2×R2"
        ]
    },
    
    "мощность": {
        "title": "Электрическая мощность",
        "description": "Мощность в электрических цепях",
        "formula": "P = U × I = I² × R = U² / R",
        "explanation": """
**Формулы мощности (однофазная цепь):**
• P = U × I (мощность = напряжение × ток)
• P = I² × R (мощность = ток² × сопротивление)
• P = U² / R (мощность = напряжение² / сопротивление)

**Единицы измерения:**
• Ватт (Вт) - основная единица
• Киловатт (кВт) = 1000 Вт
• Мегаватт (МВт) = 1000000 Вт

**Применение:**
• Расчет потребляемой мощности
• Выбор проводов и защитных устройств
• Энергетические расчеты
        """
    },
    
    "трехфазная мощность": {
        "title": "Мощность в трехфазной системе",
        "description": "Расчет активной, реактивной и полной мощности в трехфазных цепях",
        "explanation": """
**Основные формулы:**
*   **Uл** - Линейное напряжение (между двумя фазами).
*   **Uф** - Фазное напряжение (между фазой и нейтралью).
*   **Iл** - Линейный ток (в линии).
*   **Iф** - Фазный ток (в фазе нагрузки).
*   **cos(φ)** - Коэффициент мощности.

**1. Активная мощность (P, измеряется в Ваттах, Вт):**
Это полезная мощность, которая выполняет работу.
*   `P = 3 * Uф * Iф * cos(φ)`
*   `P = √3 * Uл * Iл * cos(φ)` (наиболее частая формула)

**2. Реактивная мощность (Q, измеряется в Вольт-Амперах реактивных, вар):**
Это мощность, необходимая для создания магнитных полей в двигателях и трансформаторах.
*   `Q = 3 * Uф * Iф * sin(φ)`
*   `Q = √3 * Uл * Iл * sin(φ)`

**3. Полная мощность (S, измеряется в Вольт-Амперах, ВА):**
Это геометрическая сумма активной и реактивной мощностей.
*   `S = 3 * Uф * Iф`
*   `S = √3 * Uл * Iл`
*   `S = √(P² + Q²)`
        """
    },
    
    "резистор": {
        "title": "Резисторы",
        "description": "Пассивные элементы, ограничивающие ток в цепи",
        "explanation": """
**Типы резисторов:**
• Постоянные - фиксированное сопротивление
• Переменные (потенциометры) - регулируемое сопротивление
• Термисторы - сопротивление зависит от температуры

**Маркировка:**
• Цветовая кодировка (4-6 полос)
• Цифровая маркировка
• SMD маркировка

**Соединения:**
• Последовательное: Rобщ = R1 + R2 + R3
• Параллельное: 1/Rобщ = 1/R1 + 1/R2 + 1/R3
        """
    },
    
    "конденсатор": {
        "title": "Конденсаторы",
        "description": "Элементы, накапливающие электрический заряд",
        "explanation": """
**Принцип работы:**
• Накопление заряда на обкладках
• Емкость C = Q / U
• Энергия W = C × U² / 2

**Типы конденсаторов:**
• Керамические - малые размеры, стабильность
• Электролитические - большая емкость
• Пленочные - высокое качество

**Применение:**
• Фильтрация сигналов
• Развязка цепей
• Временные задержки
        """
    },
    
    "modbus": {
        "title": "Протокол Modbus",
        "description": "Промышленный протокол связи для автоматизации",
        "explanation": """
**Modbus RTU:**
• Последовательная связь (RS-485, RS-232)
• Бинарный формат данных
• CRC контрольная сумма
• Адресация устройств (1-247)

**Функции Modbus:**
• 01 - Чтение дискретных выходов
• 02 - Чтение дискретных входов  
• 03 - Чтение регистров хранения
• 04 - Чтение входных регистров
• 05 - Запись одного выхода
• 06 - Запись одного регистра

**Структура кадра:**
• Адрес устройства (1 байт)
• Код функции (1 байт)
• Данные (N байт)
• CRC (2 байта)

**Применение:**
• PLC системы
• SCADA системы
• Промышленные сети
        """
    },
    "шим": {
        "title": "ШИМ (Широтно-импульсная модуляция)",
        "description": "Метод управления мощностью и скоростью электродвигателей",
        "explanation": """
**ШИМ (Широтно-импульсная модуляция):**

**Принцип работы:**
• **Импульсы** - прямоугольные сигналы переменной ширины
• **Частота** - постоянная частота переключения
• **Скважность** - отношение времени включения к периоду
• **Среднее значение** - пропорционально скважности

**Параметры ШИМ:**
• **Частота** - количество импульсов в секунду (Гц)
• **Скважность** - отношение времени включения к периоду (%)
• **Амплитуда** - максимальное значение напряжения
• **Разрешение** - точность установки скважности

**Типы ШИМ:**
• **Аналоговая** - плавное изменение скважности
• **Цифровая** - дискретные значения скважности
• **Синхронная** - синхронизация с внешним сигналом
• **Асинхронная** - независимая частота

**Применение:**
• **Управление двигателями** - регулирование скорости
• **Стабилизаторы напряжения** - поддержание постоянного напряжения
• **Инверторы** - преобразование постоянного тока в переменный
• **Зарядные устройства** - управление током зарядки

**Преимущества ШИМ:**
• **Высокий КПД** - минимальные потери мощности
• **Плавное регулирование** - точное управление параметрами
• **Компактность** - малые размеры устройств
• **Надежность** - отсутствие механических элементов

**Недостатки:**
• **Электромагнитные помехи** - высокочастотные переключения
• **Сложность фильтрации** - необходимость сглаживания
• **Нагрев ключей** - потери при переключении
• **Стоимость** - сложная электроника

**Плата ШИМ:**
• **Микроконтроллер** - генерация ШИМ сигналов
• **Драйверы** - усиление сигналов управления
• **Ключи** - транзисторы или MOSFET
• **Фильтры** - сглаживание выходного сигнала
• **Защита** - от перегрузок и коротких замыканий

**Настройка ШИМ:**
• **Частота** - выбор оптимальной частоты переключения
• **Скважность** - установка требуемой мощности
• **Защита** - настройка токовой защиты
• **Фильтрация** - подбор параметров фильтра
        """
    },
    "трансформатор": {
        "title": "Трансформатор",
        "description": "Электромагнитное устройство для преобразования переменного напряжения",
        "formula": "U1/U2 = N1/N2",
        "explanation": """
**Принцип работы трансформатора:**

**Основные элементы:**
• **Первичная обмотка** - подключена к источнику питания
• **Вторичная обмотка** - подключена к нагрузке
• **Магнитопровод** - сердечник из ферромагнитного материала
• **Изоляция** - между обмотками и сердечником

**Принцип действия:**
• **Переменный ток** в первичной обмотке создает переменное магнитное поле
• **Магнитное поле** пронизывает вторичную обмотку
• **ЭДС индукции** возникает во вторичной обмотке
• **Напряжение** пропорционально количеству витков

**Основные формулы:**
• **Коэффициент трансформации:** k = U1/U2 = N1/N2
• **Мощность:** P1 ≈ P2 (идеальный трансформатор)
• **Ток:** I1/I2 = N2/N1 = 1/k

**Типы трансформаторов:**
• **Повышающий** - U2 > U1, N2 > N1
• **Понижающий** - U2 < U1, N2 < N1
• **Разделительный** - U1 = U2, N1 = N2
• **Автотрансформатор** - общая обмотка

**Применение:**
• **Электроснабжение** - передача электроэнергии
• **Электроника** - источники питания
• **Измерения** - измерительные трансформаторы
• **Сварка** - сварочные трансформаторы

**Характеристики:**
• **Номинальная мощность** - максимальная мощность
• **Коэффициент трансформации** - отношение напряжений
• **КПД** - отношение выходной мощности к входной
• **Напряжение короткого замыкания** - характеристика защиты
        """
    }
}


# === Анализ изображений графиков ===
@app.route('/api/graph/import_xml', methods=['POST'])
def import_graph_from_xml():
    """Импорт графика из XML-файла.

    Принимает:
      - multipart/form-data с ключом 'file' (загрузка XML)
      - или JSON с полем 'file_path' (путь к XML на диске)

    Возвращает извлечённые точки (x,y) и базовые метрики. Если явной оси X нет,
    используется индекс точки в качестве X.
    """
    try:
        raw_xml = None

        # 1) Попытка прочитать из multipart
        if 'file' in request.files:
            f = request.files['file']
            raw_xml = f.read()
        else:
            # 2) Попытка прочитать из JSON по file_path
            data = request.get_json(silent=True) or {}
            file_path = data.get('file_path') or request.form.get('file_path')
            if file_path:
                if not os.path.isfile(file_path):
                    return jsonify({'success': False, 'error': f'Файл не найден: {file_path}'}), 400
                with open(file_path, 'rb') as fh:
                    raw_xml = fh.read()

        if not raw_xml:
            return jsonify({'success': False, 'error': 'Не передан файл XML ни через form-data (file), ни через file_path'}), 400

        # Парсим XML
        try:
            root = ET.fromstring(raw_xml)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Некорректный XML: {e}'}), 400

        # Универсальный извлекатель чисел
        def extract_floats(text: str):
            if not text:
                return []
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
            try:
                return [float(x) for x in nums]
            except Exception:
                return []

        points = []  # Список [x, y]
        xs, ys = [], []

        # Стратегии извлечения:
        # A) Ищем узлы с атрибутами x/y
        for elem in root.iter():
            x_attr = elem.attrib.get('x')
            y_attr = elem.attrib.get('y')
            if x_attr is not None and y_attr is not None:
                try:
                    x_val = float(x_attr)
                    y_val = float(y_attr)
                    points.append([x_val, y_val])
                except Exception:
                    pass

        # B) Если нет явных точек, пробуем собрать массивы X/Y из тегов
        if not points:
            # Часто встречающиеся названия
            x_tags = ['x', 'xs', 'time', 't']
            y_tags = ['y', 'ys', 'value', 'val', 'position', 'pos', 'z', 'data']

            # Соберём кандидатов по тексту
            def collect_series_by_tags(tag_names):
                series = []
                for tn in tag_names:
                    for elem in root.iter():
                        if elem.tag.lower().endswith(tn):
                            vals = extract_floats((elem.text or '').strip())
                            if vals:
                                series.append(vals)
                return series

            x_series = collect_series_by_tags(x_tags)
            y_series = collect_series_by_tags(y_tags)

            # Выбираем самую длинную серию X и Y
            best_x = max(x_series, key=len) if x_series else []
            best_y = max(y_series, key=len) if y_series else []

            if best_y and best_x and len(best_x) == len(best_y):
                points = [[best_x[i], best_y[i]] for i in range(len(best_y))]
            elif best_y:
                points = [[i, best_y[i]] for i in range(len(best_y))]

        # C) Если всё ещё пусто – пробуем по узлам <point>число число</point>
        if not points:
            for elem in root.iter():
                vals = extract_floats((elem.text or '').strip())
                if len(vals) >= 2:
                    # Берём попарно
                    for i in range(0, len(vals) - 1, 2):
                        points.append([vals[i], vals[i+1]])
            # Если собрали нечётно/мало, fallback по одному массиву
            if len(points) < 5:
                flat_vals = []
                for elem in root.iter():
                    flat_vals.extend(extract_floats((elem.text or '').strip()))
                if len(flat_vals) >= 5:
                    points = [[i, v] for i, v in enumerate(flat_vals)]

        if not points:
            return jsonify({'success': False, 'error': 'Не удалось извлечь точки из XML (неизвестный формат)'}), 400

        # Метрики
        import numpy as _np
        arr = _np.array(points, dtype=_np.float64)
        xs = arr[:, 0]
        ys = arr[:, 1]
        n = len(points)
        try:
            # Линейная регрессия y ~ a*x + b
            A = _np.vstack([xs, _np.ones_like(xs)]).T
            a, b = _np.linalg.lstsq(A, ys, rcond=None)[0]
        except Exception:
            a, b = 0.0, float(_np.mean(ys))

        summary = {
            'points_count': int(n),
            'x_min': float(_np.min(xs)), 'x_max': float(_np.max(xs)),
            'y_min': float(_np.min(ys)), 'y_max': float(_np.max(ys)),
            'y_mean': float(_np.mean(ys)), 'y_std': float(_np.std(ys)),
            'trend_slope': float(a), 'trend_intercept': float(b)
        }

        # Ограничим выдачу точек (до 5000)
        max_pts = 5000
        pts_out = points if n <= max_pts else points[:: max(1, n // max_pts)]

        return jsonify({
            'success': True,
            'service': 'Electrical API',
            'source': 'xml',
            'points': pts_out,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logging.exception('Ошибка импорта XML графика')
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/graph/analyze', methods=['POST'])
def analyze_graph_image():
    """Принимает PNG/JPG графика, извлекает базовые метрики и рекомендации."""
    try:
        if not PIL_NUMPY_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Отсутствуют зависимости Pillow/Numpy для анализа изображений.'
            }), 500

        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Загрузите файл с ключом form-data "file"'
            }), 400

        file = request.files['file']
        raw = file.read()
        if not raw:
            return jsonify({'success': False, 'error': 'Пустой файл'}), 400

        # Загрузка изображения в градациях серого
        img = Image.open(BytesIO(raw)).convert('L')
        arr = np.asarray(img, dtype=np.float32)
        h, w = arr.shape

        # Базовые метрики
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))

        # Градиенты по осям для оценки "резкости"/наличия линий
        gx = np.abs(np.diff(arr, axis=1))
        gy = np.abs(np.diff(arr, axis=0))
        edge_strength = float(np.mean(gx) + np.mean(gy))

        # Оценка тренда по средней яркости колонок (простая линейная регрессия)
        col_mean = np.mean(arr, axis=0)
        x = np.arange(col_mean.shape[0], dtype=np.float32)
        # np.polyfit может бросить исключение при NaN — защитимся
        try:
            slope = float(np.polyfit(x, col_mean, 1)[0])
        except Exception:
            slope = 0.0

        # Эвристики рекомендаций
        recs = []
        if edge_strength < 1.0:
            recs.append('Низкая контрастность линий — увеличьте толщину/контраст кривых на графике.')
        else:
            recs.append('Контуры графика читаемы — можно извлекать точки для численного анализа.')

        if abs(slope) > 0.01:
            recs.append('Наблюдается выраженный тренд — оцените наклон, добавьте скользящее среднее для сглаживания.')
        else:
            recs.append('Тренд слабый — проверьте периодичность/сезонность, примените спектральный анализ.')

        if std_val > 20:
            recs.append('Высокий разброс значений — используйте медианный фильтр/сглаживание и увеличьте частоту измерений.')
        else:
            recs.append('Разброс умеренный — текущего шага дискретизации достаточно для базовых выводов.')

        return jsonify({
            'success': True,
            'service': 'Electrical API',
            'analysis': {
                'image_width': int(w),
                'image_height': int(h),
                'mean_intensity': round(mean_val, 2),
                'std_intensity': round(std_val, 2),
                'edge_strength': round(edge_strength, 3),
                'trend_slope': round(slope, 5)
            },
            'recommendations': recs,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logging.exception('Ошибка анализа изображения')
        return jsonify({'success': False, 'error': str(e)}), 500


# === Оцифровка кривой графика (MVP) ===
@app.route('/api/graph/digitize', methods=['POST'])
def digitize_graph_image():
    """Принимает PNG/JPG, извлекает одну тёмную кривую на светлом фоне и возвращает точки."""
    try:
        if not PIL_NUMPY_AVAILABLE:
            return jsonify({'success': False, 'error': 'Pillow/Numpy не установлены'}), 500

        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Загрузите файл (form-data key=file)'}), 400

        raw = request.files['file'].read()
        if not raw:
            return jsonify({'success': False, 'error': 'Пустой файл'}), 400

        # Готовим изображение
        img = Image.open(BytesIO(raw)).convert('L')  # grayscale
        arr = np.asarray(img, dtype=np.uint8)
        h, w = arr.shape

        # Нормализация и простая бинаризация (линия тёмная)
        # Порог выберем адаптивно по 20-му перцентилю яркости
        thr = int(np.percentile(arr, 20))
        mask = arr <= thr  # True там, где тёмная линия/сетка

        # Уберём разреженный шум: игнорируем пиксели, где в 3x3 окне мало тёмных
        # (простая морфология без scipy)
        pad = np.pad(mask.astype(np.uint8), ((1,1),(1,1)), 'constant', constant_values=0)
        dens = (
            pad[0:-2,0:-2] + pad[0:-2,1:-1] + pad[0:-2,2:] +
            pad[1:-1,0:-2] + pad[1:-1,1:-1] + pad[1:-1,2:] +
            pad[2:  ,0:-2] + pad[2:  ,1:-1] + pad[2:  ,2:]
        )
        mask = (mask & (dens >= 3))

        # По каждому X-столбцу вычислим взвешенное среднее Y для тёмных пикселей
        points = []  # в пикселях (x,y)
        ys = np.arange(h, dtype=np.float32)
        for x in range(w):
            col = mask[:, x]
            cnt = int(col.sum())
            if cnt == 0:
                continue
            y_mean = float((ys[col]).mean())
            points.append([x, y_mean])

        if len(points) < max(20, w * 0.02):
            return jsonify({'success': False, 'error': 'Не удалось выделить кривую (слишком мало точек)'}), 400

        # Нормализуем координаты в [0,1] (ось Y инвертируем – ноль внизу)
        norm_points = []
        for x, y in points:
            nx = x / (w - 1) if w > 1 else 0.0
            ny = 1.0 - (y / (h - 1) if h > 1 else 0.0)
            norm_points.append([round(nx, 6), round(ny, 6)])

        # Простейшие метрики по нормализованной кривой
        npx = np.array([p[0] for p in norm_points], dtype=np.float32)
        npy = np.array([p[1] for p in norm_points], dtype=np.float32)
        # Линейная регрессия наклона: y ~ a*x + b
        try:
            A = np.vstack([npx, np.ones_like(npx)]).T
            a, b = np.linalg.lstsq(A, npy, rcond=None)[0]
        except Exception:
            a, b = 0.0, float(npy.mean())

        # Поиск экстремумов (наивно): локальные макс/мин по окну 3
        peaks = []
        troughs = []
        for i in range(1, len(npy)-1):
            if npy[i] > npy[i-1] and npy[i] > npy[i+1]:
                peaks.append([float(npx[i]), float(npy[i])])
            if npy[i] < npy[i-1] and npy[i] < npy[i+1]:
                troughs.append([float(npx[i]), float(npy[i])])

        summary = {
            'trend_slope': round(float(a), 6),
            'mean': round(float(npy.mean()), 6),
            'std': round(float(npy.std()), 6),
            'points_extracted': len(points),
            'peaks': peaks[:20],
            'troughs': troughs[:20]
        }

        # Ограничим объём точек в ответе (например, до 2000)
        max_pts = 2000
        pts_out = points if len(points) <= max_pts else points[::len(points)//max_pts]
        npts_out = norm_points if len(norm_points) <= max_pts else norm_points[::len(norm_points)//max_pts]

        return jsonify({
            'success': True,
            'service': 'Electrical API',
            'image': {'width': int(w), 'height': int(h)},
            'points': pts_out,
            'normalized_points': npts_out,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logging.exception('Ошибка оцифровки графика')
        return jsonify({'success': False, 'error': str(e)}), 500

def find_best_match(query):
    """Поиск наиболее подходящего ответа по запросу"""
    query_lower = query.lower()
    
    # Прямое совпадение
    for key, data in ELECTRICAL_KNOWLEDGE.items():
        if key in query_lower:
            return data
    
    # Поиск по ключевым словам
    keywords = {
        "ом": "закон ома",
        "кирхгоф": "закон кирхгофа", 
        "мощность": "мощность",
        "резистор": "резистор",
        "конденсатор": "конденсатор",
        "modbus": "modbus",
        "напряжение": "закон ома",
        "ток": "закон ома",
        "сопротивление": "закон ома",
        "шим": "шим",
        "плата": "шим",
        "модуляция": "шим",
        "импульсная": "шим",
        "широтно": "шим",
        "скважность": "шим",
        "переключение": "шим",
        "трансформатор": "трансформатор",
        "преобразование": "трансформатор",
        "напряжение": "трансформатор"
    }
    
    for keyword, topic in keywords.items():
        if keyword in query_lower:
            return ELECTRICAL_KNOWLEDGE[topic]
    
    return None

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка состояния сервера"""
    return jsonify({
        "status": "healthy",
        "service": "Electrical Engineering API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/electrical/status', methods=['GET'])
def get_status():
    """Получение статуса модуля электротехники"""
    return jsonify({
        "status": "online",
        "module": "Электротехника",
        "port": 8087,
        "description": "Расчеты электрических цепей, схемы, закон Ома, Кирхгофа",
        "topics_available": list(ELECTRICAL_KNOWLEDGE.keys()),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/electrical/explain', methods=['GET', 'POST'])
def explain_concept():
    """Объяснение концепций электротехники"""
    try:
        # Обработка как GET, так и POST запросов
        if request.method == 'POST':
            data = request.get_json()
            concept = data.get('concept', '').strip() if data else ''
            level = data.get('level', 'detailed') if data else 'detailed'
        else:  # GET запрос
            concept = request.args.get('concept', '').strip()
            level = request.args.get('level', 'detailed')
        
        if not concept:
            return jsonify({
                "error": "Не указана концепция для объяснения"
            }), 400
        
        # Поиск подходящего ответа
        knowledge = find_best_match(concept)
        
        if knowledge:
            response = {
                "success": True,
                "concept": concept,
                "title": knowledge["title"],
                "description": knowledge["description"],
                "explanation": knowledge["explanation"]
            }
            
            if "formula" in knowledge:
                response["formula"] = knowledge["formula"]
            
            if "examples" in knowledge:
                response["examples"] = knowledge["examples"]
            
            return jsonify(response)
        else:
            return jsonify({
                "success": False,
                "message": f"Концепция '{concept}' не найдена в базе знаний электротехники",
                "available_topics": list(ELECTRICAL_KNOWLEDGE.keys())
            })
    
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        return jsonify({
            "error": "Внутренняя ошибка сервера",
            "details": str(e)
        }), 500

@app.route('/api/electrical/calculate', methods=['POST'])
def calculate():
    """Выполнение расчетов по электротехнике"""
    try:
        data = request.get_json()
        calculation_type = data.get('type', '')
        parameters = data.get('parameters', {})
        
        if calculation_type == 'ohm_law':
            # Расчет по закону Ома
            if 'voltage' in parameters and 'resistance' in parameters:
                current = parameters['voltage'] / parameters['resistance']
                power = parameters['voltage'] * current
                return jsonify({
                    "success": True,
                    "calculation": "Закон Ома",
                    "input": parameters,
                    "result": {
                        "current": round(current, 3),
                        "power": round(power, 3)
                    }
                })
            elif 'current' in parameters and 'resistance' in parameters:
                voltage = parameters['current'] * parameters['resistance']
                power = voltage * parameters['current']
                return jsonify({
                    "success": True,
                    "calculation": "Закон Ома",
                    "input": parameters,
                    "result": {
                        "voltage": round(voltage, 3),
                        "power": round(power, 3)
                    }
                })
            else:
                return jsonify({
                    "error": "Недостаточно параметров для расчета по закону Ома"
                }), 400
        
        return jsonify({
            "error": f"Тип расчета '{calculation_type}' не поддерживается"
        }), 400
    
    except Exception as e:
        logger.error(f"Ошибка при расчете: {str(e)}")
        return jsonify({
            "error": "Ошибка при выполнении расчета",
            "details": str(e)
        }), 500

@app.route('/api/electrical/advanced_calculate', methods=['POST'])
def advanced_calculate():
    """Продвинутые расчеты с использованием утилит"""
    try:
        data = request.get_json()
        calculation_type = data.get('type', '')
        parameters = data.get('parameters', {})
        
        if not calculation_type:
            return jsonify({'error': 'Тип расчета не указан'}), 400
        
        result = {}
        
        if ELECTRICAL_UTILS_AVAILABLE:
            if calculation_type == 'ohm_law':
                voltage = parameters.get('voltage')
                current = parameters.get('current')
                resistance = parameters.get('resistance')
                
                if voltage and current:
                    result = electrical_utils.calculate_ohm_law(voltage=voltage, current=current)
                elif voltage and resistance:
                    result = electrical_utils.calculate_ohm_law(voltage=voltage, resistance=resistance)
                elif current and resistance:
                    result = electrical_utils.calculate_ohm_law(current=current, resistance=resistance)
                else:
                    return jsonify({'error': 'Недостаточно параметров для расчета'}), 400
            
            elif calculation_type == 'power':
                voltage = parameters.get('voltage')
                current = parameters.get('current')
                resistance = parameters.get('resistance')
                
                if voltage and current:
                    result = electrical_utils.calculate_power(voltage=voltage, current=current)
                elif current and resistance:
                    result = electrical_utils.calculate_power(current=current, resistance=resistance)
                elif voltage and resistance:
                    result = electrical_utils.calculate_power(voltage=voltage, resistance=resistance)
                else:
                    return jsonify({'error': 'Недостаточно параметров для расчета мощности'}), 400
            
            elif calculation_type == 'circuit_analysis':
                circuit_data = parameters.get('circuit_data', {})
                result = electrical_utils.analyze_circuit(circuit_data)
            
            else:
                return jsonify({'error': f'Неизвестный тип расчета: {calculation_type}'}), 400
        else:
            return jsonify({'error': 'Электротехнические утилиты недоступны'}), 503
        
        return jsonify({
            'success': True,
            'calculation_type': calculation_type,
            'parameters': parameters,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка продвинутого расчета: {e}")
        return jsonify({'error': f'Ошибка расчета: {str(e)}'}), 500

@app.route('/api/electrical/knowledge_advanced', methods=['POST'])
def get_advanced_knowledge():
    """Получение продвинутых знаний по электротехнике"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        knowledge_type = data.get('type', 'general')
        
        if not query:
            return jsonify({'error': 'Запрос не может быть пустым'}), 400
        
        result = {}
        
        if ELECTRICAL_UTILS_AVAILABLE and electrical_handler:
            if knowledge_type == 'advanced':
                result = electrical_handler.get_advanced_knowledge(query)
            elif knowledge_type == 'formulas':
                result = electrical_handler.get_formulas(query)
            elif knowledge_type == 'components':
                result = electrical_handler.get_components_info(query)
            else:
                result = electrical_handler.get_general_knowledge(query)
        else:
            # Fallback поиск в базе знаний
            result = find_best_match(query)
        
        return jsonify({
            'success': True,
            'query': query,
            'knowledge_type': knowledge_type,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения продвинутых знаний: {e}")
        return jsonify({'error': f'Ошибка получения знаний: {str(e)}'}), 500

@app.route('/api/electrical/topics', methods=['GET'])

@app.route('/api/chat', methods=['POST'])
def chat():
    """Универсальный endpoint для чата"""
    try:
        data = request.get_json() or {}
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({
                "success": False,
                "error": "Сообщение не может быть пустым"
            }), 400
        
        # Поиск подходящего ответа
        knowledge = find_best_match(message)
        
        if knowledge:
            response = {
                "success": True,
                "response": knowledge["explanation"],
                "concept": knowledge["title"],
                "server_type": "electrical",
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(response)
        else:
            return jsonify({
                "success": False,
                "error": f"Не найдена информация по запросу: {message}",
                "suggestions": list(ELECTRICAL_KNOWLEDGE.keys())[:5]
            }), 404
            
    except Exception as e:
        print(f"Ошибка в chat endpoint: {e}")
        return jsonify({
            "success": False,
            "error": f"Внутренняя ошибка сервера: {str(e)}"
        }), 500

if __name__ == '__main__':
    print("🚀 Запуск сервера электротехники на порту 8087...")
    app.run(host='0.0.0.0', port=8087, debug=False)

