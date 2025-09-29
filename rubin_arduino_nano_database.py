#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arduino Nano Programming Database для Rubin AI v2
База данных знаний по программированию Arduino Nano
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArduinoNanoDatabase:
    """База данных знаний по Arduino Nano"""
    
    def __init__(self, db_path: str = "rubin_arduino_nano.db"):
        self.db_path = db_path
        self.connection = None
        self._initialize_database()
        
        logger.info(f"🔧 Arduino Nano Database инициализирована: {db_path}")
    
    def _initialize_database(self):
        """Инициализация базы данных"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            
            # Создаем таблицы
            self._create_tables()
            
            # Заполняем начальными данными
            self._populate_arduino_data()
            
            logger.info("✅ База данных Arduino Nano инициализирована успешно")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации базы данных: {e}")
            raise
    
    def _create_tables(self):
        """Создание таблиц базы данных"""
        cursor = self.connection.cursor()
        
        # Таблица категорий Arduino
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS arduino_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                keywords TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Таблица знаний по Arduino
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS arduino_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_id INTEGER,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                code_example TEXT,
                pinout_info TEXT,
                keywords TEXT,
                difficulty_level INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (category_id) REFERENCES arduino_categories (id)
            )
        """)
        
        # Таблица функций Arduino
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS arduino_functions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                function_name TEXT NOT NULL,
                description TEXT NOT NULL,
                syntax TEXT NOT NULL,
                parameters TEXT,
                return_value TEXT,
                example_code TEXT,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Таблица библиотек Arduino
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS arduino_libraries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                library_name TEXT NOT NULL,
                description TEXT NOT NULL,
                installation TEXT,
                usage_example TEXT,
                functions_list TEXT,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Таблица проектов Arduino
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS arduino_projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL,
                description TEXT NOT NULL,
                components TEXT,
                code TEXT NOT NULL,
                circuit_diagram TEXT,
                difficulty_level INTEGER DEFAULT 1,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Таблица пинов Arduino Nano
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS arduino_pins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pin_number INTEGER NOT NULL,
                pin_type TEXT NOT NULL,
                description TEXT,
                voltage_level TEXT,
                max_current TEXT,
                special_functions TEXT,
                usage_examples TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Таблица ошибок и решений
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS arduino_troubleshooting (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_description TEXT NOT NULL,
                possible_causes TEXT,
                solutions TEXT,
                prevention_tips TEXT,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
        logger.info("📊 Таблицы базы данных Arduino Nano созданы")
    
    def _populate_arduino_data(self):
        """Заполнение базы данных знаниями по Arduino Nano"""
        cursor = self.connection.cursor()
        
        # Проверяем, есть ли уже данные
        cursor.execute("SELECT COUNT(*) FROM arduino_categories")
        if cursor.fetchone()[0] > 0:
            logger.info("📊 База данных Arduino уже содержит данные")
            return
        
        # Категории Arduino
        categories = [
            ("basics", "Основы Arduino Nano", "основы,настройка,первый,скетч,программа"),
            ("digital_io", "Цифровые входы и выходы", "digital,pinMode,digitalWrite,digitalRead,пин,вход,выход"),
            ("analog_io", "Аналоговые входы и выходы", "analog,analogRead,analogWrite,PWM,аналог,сигнал"),
            ("communication", "Коммуникация", "serial,SPI,I2C,Wire,коммуникация,передача,данных"),
            ("sensors", "Датчики", "датчик,sensor,температура,влажность,расстояние,движение"),
            ("actuators", "Исполнительные устройства", "мотор,серво,реле,светодиод,звук,привод"),
            ("timing", "Таймеры и задержки", "delay,millis,micros,таймер,время,задержка"),
            ("interrupts", "Прерывания", "interrupt,attachInterrupt,detachInterrupt,прерывание"),
            ("memory", "Память и хранение", "EEPROM,память,хранение,данные,сохранение"),
            ("power", "Питание и энергопотребление", "питание,power,энергия,батарея,USB,питание"),
            ("libraries", "Библиотеки", "библиотека,library,подключение,установка,функции"),
            ("projects", "Проекты", "проект,пример,схема,сборка,готовый,код")
        ]
        
        for name, description, keywords in categories:
            cursor.execute("""
                INSERT INTO arduino_categories (name, description, keywords)
                VALUES (?, ?, ?)
            """, (name, description, keywords))
        
        # Основные знания по Arduino Nano
        knowledge_data = [
            # Основы
            (1, "Arduino Nano - обзор", 
             """Arduino Nano - это компактная плата микроконтроллера на базе ATmega328P.

**Основные характеристики:**
- Микроконтроллер: ATmega328P
- Рабочее напряжение: 5V
- Питание: USB или внешний источник 7-12V
- Цифровые пины: 14 (6 с PWM)
- Аналоговые пины: 8
- Flash память: 32KB
- SRAM: 2KB
- EEPROM: 1KB
- Тактовая частота: 16MHz

**Преимущества:**
- Компактный размер (18x45mm)
- Низкая стоимость
- Простота использования
- Большое сообщество
- Множество библиотек""",
             """// Базовый скетч - мигание светодиода
void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  digitalWrite(LED_BUILTIN, HIGH);
  delay(1000);
  digitalWrite(LED_BUILTIN, LOW);
  delay(1000);
}""",
             "D1-D13: цифровые пины, A0-A7: аналоговые пины, VCC: 5V, GND: земля",
             "arduino,nano,микроконтроллер,ATmega328P,плата,характеристики",
             1),
            
            # Цифровые пины
            (2, "Цифровые пины Arduino Nano",
             """Arduino Nano имеет 14 цифровых пинов (D0-D13).

**Функции цифровых пинов:**
- Вход/выход: все пины могут работать как входы или выходы
- PWM: пины D3, D5, D6, D9, D10, D11 поддерживают ШИМ
- Прерывания: пины D2 и D3 поддерживают внешние прерывания
- Serial: пины D0 (RX) и D1 (TX) для последовательной связи

**Основные функции:**
- pinMode(pin, mode) - настройка режима пина
- digitalWrite(pin, value) - запись цифрового значения
- digitalRead(pin) - чтение цифрового значения""",
             """// Пример работы с цифровыми пинами
void setup() {
  pinMode(13, OUTPUT);    // Пин 13 как выход
  pinMode(2, INPUT);      // Пин 2 как вход
}

void loop() {
  // Читаем состояние кнопки на пине 2
  int buttonState = digitalRead(2);
  
  // Если кнопка нажата, включаем светодиод
  if (buttonState == HIGH) {
    digitalWrite(13, HIGH);
  } else {
    digitalWrite(13, LOW);
  }
}""",
             "D0-D13: цифровые пины, D3,D5,D6,D9,D10,D11: PWM, D2,D3: прерывания",
             "цифровой,пин,pinMode,digitalWrite,digitalRead,PWM,прерывание",
             1),
            
            # Аналоговые пины
            (3, "Аналоговые пины Arduino Nano",
             """Arduino Nano имеет 8 аналоговых пинов (A0-A7).

**Характеристики аналоговых пинов:**
- Разрешение: 10 бит (0-1023)
- Напряжение: 0-5V
- Внутреннее сопротивление: 100 МОм
- Время преобразования: ~100 мкс

**Функции:**
- analogRead(pin) - чтение аналогового значения
- analogReference(type) - настройка опорного напряжения

**Примеры использования:**
- Потенциометры
- Датчики температуры
- Датчики освещенности
- Датчики расстояния""",
             """// Пример работы с аналоговыми пинами
void setup() {
  Serial.begin(9600);
}

void loop() {
  // Читаем значение с потенциометра на пине A0
  int sensorValue = analogRead(A0);
  
  // Преобразуем в напряжение (0-5V)
  float voltage = sensorValue * (5.0 / 1023.0);
  
  // Выводим в Serial Monitor
  Serial.print("Значение: ");
  Serial.print(sensorValue);
  Serial.print(", Напряжение: ");
  Serial.println(voltage);
  
  delay(100);
}""",
             "A0-A7: аналоговые пины, 10 бит, 0-1023, 0-5V",
             "аналоговый,analogRead,потенциометр,датчик,напряжение,разрешение",
             1),
            
            # PWM
            (4, "ШИМ (PWM) на Arduino Nano",
             """PWM (Pulse Width Modulation) - широтно-импульсная модуляция.

**Пины с PWM на Arduino Nano:**
- D3, D5, D6, D9, D10, D11
- Частота: ~490 Гц (D5, D6: ~980 Гц)
- Разрешение: 8 бит (0-255)

**Функция analogWrite():**
- analogWrite(pin, value) - где value от 0 до 255
- 0 = 0% заполнения (0V)
- 127 = 50% заполнения (~2.5V)
- 255 = 100% заполнения (5V)

**Применения:**
- Управление яркостью светодиодов
- Управление скоростью моторов
- Генерация звуковых сигналов
- Управление сервоприводами""",
             """// Пример использования PWM
void setup() {
  pinMode(9, OUTPUT);  // PWM пин
}

void loop() {
  // Плавное изменение яркости светодиода
  for (int brightness = 0; brightness <= 255; brightness++) {
    analogWrite(9, brightness);
    delay(10);
  }
  
  for (int brightness = 255; brightness >= 0; brightness--) {
    analogWrite(9, brightness);
    delay(10);
  }
}""",
             "D3,D5,D6,D9,D10,D11: PWM пины, 8 бит, 0-255, ~490 Гц",
             "PWM,analogWrite,ШИМ,модуляция,яркость,мотор,серво",
             2),
            
            # Serial
            (5, "Последовательная связь (Serial)",
             """Arduino Nano поддерживает последовательную связь через USB.

**Основные функции:**
- Serial.begin(speed) - инициализация
- Serial.print() - вывод данных
- Serial.println() - вывод с переводом строки
- Serial.available() - проверка наличия данных
- Serial.read() - чтение данных

**Скорости передачи:**
- 300, 600, 1200, 2400, 4800, 9600, 14400, 19200, 28800, 38400, 57600, 115200

**Применения:**
- Отладка программ
- Передача данных на компьютер
- Управление Arduino с компьютера
- Логирование данных""",
             """// Пример работы с Serial
void setup() {
  Serial.begin(9600);
  Serial.println("Arduino Nano готов к работе!");
}

void loop() {
  // Отправляем данные каждую секунду
  Serial.print("Время работы: ");
  Serial.print(millis() / 1000);
  Serial.println(" секунд");
  
  // Проверяем входящие данные
  if (Serial.available() > 0) {
    String command = Serial.readString();
    Serial.print("Получена команда: ");
    Serial.println(command);
  }
  
  delay(1000);
}""",
             "D0: RX, D1: TX, USB, 9600, 115200",
             "Serial,последовательная,связь,USB,отладка,данные,команда",
             1)
        ]
        
        for category_id, title, content, code_example, pinout_info, keywords, difficulty in knowledge_data:
            cursor.execute("""
                INSERT INTO arduino_knowledge 
                (category_id, title, content, code_example, pinout_info, keywords, difficulty_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (category_id, title, content, code_example, pinout_info, keywords, difficulty))
        
        # Функции Arduino
        functions_data = [
            ("pinMode", "Настройка режима пина", "pinMode(pin, mode)", 
             "pin: номер пина, mode: INPUT, OUTPUT, INPUT_PULLUP",
             "void", 
             """void setup() {
  pinMode(13, OUTPUT);
  pinMode(2, INPUT_PULLUP);
}""", "basics"),
            
            ("digitalWrite", "Запись цифрового значения", "digitalWrite(pin, value)",
             "pin: номер пина, value: HIGH или LOW",
             "void",
             """void loop() {
  digitalWrite(13, HIGH);
  delay(1000);
  digitalWrite(13, LOW);
  delay(1000);
}""", "digital_io"),
            
            ("digitalRead", "Чтение цифрового значения", "digitalRead(pin)",
             "pin: номер пина",
             "int",
             """void loop() {
  int buttonState = digitalRead(2);
  if (buttonState == HIGH) {
    // Кнопка нажата
  }
}""", "digital_io"),
            
            ("analogRead", "Чтение аналогового значения", "analogRead(pin)",
             "pin: номер аналогового пина (A0-A7)",
             "int",
             """void loop() {
  int sensorValue = analogRead(A0);
  Serial.println(sensorValue);
}""", "analog_io"),
            
            ("analogWrite", "Запись PWM значения", "analogWrite(pin, value)",
             "pin: PWM пин, value: 0-255",
             "void",
             """void loop() {
  analogWrite(9, 128);  // 50% заполнения
}""", "analog_io"),
            
            ("delay", "Задержка в миллисекундах", "delay(ms)",
             "ms: время задержки в миллисекундах",
             "void",
             """void loop() {
  digitalWrite(13, HIGH);
  delay(1000);  // 1 секунда
  digitalWrite(13, LOW);
  delay(1000);
}""", "timing"),
            
            ("millis", "Время работы в миллисекундах", "millis()",
             "Без параметров",
             "unsigned long",
             """void loop() {
  unsigned long currentTime = millis();
  Serial.println(currentTime);
}""", "timing"),
            
            ("Serial.begin", "Инициализация последовательной связи", "Serial.begin(speed)",
             "speed: скорость передачи (9600, 115200 и т.д.)",
             "void",
             """void setup() {
  Serial.begin(9600);
}""", "communication"),
            
            ("Serial.print", "Вывод данных", "Serial.print(data)",
             "data: данные для вывода",
             "size_t",
             """void loop() {
  Serial.print("Hello ");
  Serial.print(42);
  Serial.println(" World!");
}""", "communication")
        ]
        
        for func_name, description, syntax, parameters, return_value, example_code, category in functions_data:
            cursor.execute("""
                INSERT INTO arduino_functions 
                (function_name, description, syntax, parameters, return_value, example_code, category)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (func_name, description, syntax, parameters, return_value, example_code, category))
        
        # Библиотеки Arduino
        libraries_data = [
            ("Servo", "Управление сервоприводами", 
             "Скачать через Library Manager или добавить вручную",
             """#include <Servo.h>
Servo myServo;

void setup() {
  myServo.attach(9);
}

void loop() {
  myServo.write(90);
  delay(1000);
  myServo.write(0);
  delay(1000);
}""",
             "attach(), write(), read(), detach()", "actuators"),
            
            ("Wire", "I2C коммуникация",
             "Встроенная библиотека",
             """#include <Wire.h>

void setup() {
  Wire.begin();
  Wire.beginTransmission(0x48);
  Wire.write(0x00);
  Wire.endTransmission();
}

void loop() {
  Wire.requestFrom(0x48, 2);
  while (Wire.available()) {
    byte data = Wire.read();
    Serial.println(data);
  }
  delay(1000);
}""",
             "begin(), beginTransmission(), write(), read(), endTransmission()", "communication"),
            
            ("EEPROM", "Работа с энергонезависимой памятью",
             "Встроенная библиотека",
             """#include <EEPROM.h>

void setup() {
  EEPROM.write(0, 42);
  int value = EEPROM.read(0);
  Serial.println(value);
}

void loop() {
  // Код программы
}""",
             "read(), write(), update(), get(), put()", "memory"),
            
            ("SPI", "SPI коммуникация",
             "Встроенная библиотека",
             """#include <SPI.h>

void setup() {
  SPI.begin();
  digitalWrite(SS, HIGH);
}

void loop() {
  digitalWrite(SS, LOW);
  SPI.transfer(0x01);
  digitalWrite(SS, HIGH);
  delay(1000);
}""",
             "begin(), transfer(), setBitOrder(), setDataMode(), setClockDivider()", "communication")
        ]
        
        for lib_name, description, installation, usage_example, functions_list, category in libraries_data:
            cursor.execute("""
                INSERT INTO arduino_libraries 
                (library_name, description, installation, usage_example, functions_list, category)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (lib_name, description, installation, usage_example, functions_list, category))
        
        # Проекты Arduino
        projects_data = [
            ("Мигающий светодиод", "Базовый проект - мигание встроенного светодиода",
             "Arduino Nano, USB кабель",
             """void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  digitalWrite(LED_BUILTIN, HIGH);
  delay(1000);
  digitalWrite(LED_BUILTIN, LOW);
  delay(1000);
}""",
             "Подключите Arduino Nano к компьютеру через USB", 1, "basics"),
            
            ("Управление яркостью светодиода", "Плавное изменение яркости с помощью PWM",
             "Arduino Nano, светодиод, резистор 220 Ом, провода",
             """void setup() {
  pinMode(9, OUTPUT);
}

void loop() {
  for (int brightness = 0; brightness <= 255; brightness++) {
    analogWrite(9, brightness);
    delay(10);
  }
  for (int brightness = 255; brightness >= 0; brightness--) {
    analogWrite(9, brightness);
    delay(10);
  }
}""",
             "Светодиод к пину 9 через резистор 220 Ом", 2, "analog_io"),
            
            ("Датчик температуры LM35", "Измерение температуры с помощью LM35",
             "Arduino Nano, датчик LM35, провода",
             """void setup() {
  Serial.begin(9600);
}

void loop() {
  int sensorValue = analogRead(A0);
  float voltage = sensorValue * (5.0 / 1023.0);
  float temperature = voltage * 100;  // LM35: 10mV/°C
  
  Serial.print("Температура: ");
  Serial.print(temperature);
  Serial.println(" °C");
  
  delay(1000);
}""",
             "LM35: VCC к 5V, GND к GND, OUT к A0", 2, "sensors"),
            
            ("Сервопривод", "Управление сервоприводом",
             "Arduino Nano, сервопривод SG90, провода",
             """#include <Servo.h>
Servo myServo;

void setup() {
  myServo.attach(9);
}

void loop() {
  myServo.write(0);
  delay(1000);
  myServo.write(90);
  delay(1000);
  myServo.write(180);
  delay(1000);
}""",
             "Сервопривод: VCC к 5V, GND к GND, Signal к пину 9", 2, "actuators")
        ]
        
        for project_name, description, components, code, circuit_diagram, difficulty_level, category in projects_data:
            cursor.execute("""
                INSERT INTO arduino_projects 
                (project_name, description, components, code, circuit_diagram, difficulty_level, category)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (project_name, description, components, code, circuit_diagram, difficulty_level, category))
        
        # Пины Arduino Nano
        pins_data = [
            (0, "Digital", "RX - прием данных", "0-5V", "40mA", "Serial RX", "Последовательная связь"),
            (1, "Digital", "TX - передача данных", "0-5V", "40mA", "Serial TX", "Последовательная связь"),
            (2, "Digital", "Цифровой пин", "0-5V", "40mA", "External Interrupt", "Кнопки, датчики"),
            (3, "Digital", "Цифровой пин с PWM", "0-5V", "40mA", "PWM, External Interrupt", "Светодиоды, моторы"),
            (4, "Digital", "Цифровой пин", "0-5V", "40mA", "None", "Общие цифровые функции"),
            (5, "Digital", "Цифровой пин с PWM", "0-5V", "40mA", "PWM", "Светодиоды, моторы"),
            (6, "Digital", "Цифровой пин с PWM", "0-5V", "40mA", "PWM", "Светодиоды, моторы"),
            (7, "Digital", "Цифровой пин", "0-5V", "40mA", "None", "Общие цифровые функции"),
            (8, "Digital", "Цифровой пин", "0-5V", "40mA", "None", "Общие цифровые функции"),
            (9, "Digital", "Цифровой пин с PWM", "0-5V", "40mA", "PWM", "Светодиоды, моторы, серво"),
            (10, "Digital", "Цифровой пин с PWM", "0-5V", "40mA", "PWM", "Светодиоды, моторы"),
            (11, "Digital", "Цифровой пин с PWM", "0-5V", "40mA", "PWM", "Светодиоды, моторы"),
            (12, "Digital", "Цифровой пин", "0-5V", "40mA", "None", "Общие цифровые функции"),
            (13, "Digital", "Цифровой пин", "0-5V", "40mA", "Built-in LED", "Встроенный светодиод"),
            (14, "Analog", "Аналоговый пин A0", "0-5V", "100 МОм", "ADC", "Потенциометры, датчики"),
            (15, "Analog", "Аналоговый пин A1", "0-5V", "100 МОм", "ADC", "Потенциометры, датчики"),
            (16, "Analog", "Аналоговый пин A2", "0-5V", "100 МОм", "ADC", "Потенциометры, датчики"),
            (17, "Analog", "Аналоговый пин A3", "0-5V", "100 МОм", "ADC", "Потенциометры, датчики"),
            (18, "Analog", "Аналоговый пин A4", "0-5V", "100 МОм", "ADC, I2C SDA", "Датчики, I2C"),
            (19, "Analog", "Аналоговый пин A5", "0-5V", "100 МОм", "ADC, I2C SCL", "Датчики, I2C"),
            (20, "Analog", "Аналоговый пин A6", "0-5V", "100 МОм", "ADC", "Потенциометры, датчики"),
            (21, "Analog", "Аналоговый пин A7", "0-5V", "100 МОм", "ADC", "Потенциометры, датчики")
        ]
        
        for pin_number, pin_type, description, voltage_level, max_current, special_functions, usage_examples in pins_data:
            cursor.execute("""
                INSERT INTO arduino_pins 
                (pin_number, pin_type, description, voltage_level, max_current, special_functions, usage_examples)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (pin_number, pin_type, description, voltage_level, max_current, special_functions, usage_examples))
        
        # Решение проблем
        troubleshooting_data = [
            ("Arduino не определяется компьютером",
             "Проблемы с драйверами, неисправный USB кабель, неправильный порт",
             "1. Установите драйверы CH340 или CP2102\n2. Попробуйте другой USB кабель\n3. Проверьте настройки порта в IDE\n4. Перезагрузите Arduino",
             "Используйте качественные USB кабели, обновляйте драйверы", "basics"),
            
            ("Скетч не загружается",
             "Неправильный порт, проблемы с загрузчиком, несовместимость",
             "1. Выберите правильный порт (Tools > Port)\n2. Выберите правильную плату (Tools > Board)\n3. Нажмите кнопку Reset перед загрузкой\n4. Проверьте соединения",
             "Всегда проверяйте настройки порта и платы", "basics"),
            
            ("Светодиод не мигает",
             "Неправильное подключение, неисправный светодиод, проблемы с кодом",
             "1. Проверьте полярность светодиода\n2. Убедитесь в правильности пина\n3. Добавьте резистор 220 Ом\n4. Проверьте код на ошибки",
             "Используйте резистор для ограничения тока", "digital_io"),
            
            ("Аналоговые значения нестабильны",
             "Помехи, неправильное питание, плохие соединения",
             "1. Добавьте конденсатор 100nF к аналоговому пину\n2. Используйте стабилизированное питание\n3. Проверьте качество соединений\n4. Добавьте фильтрацию в коде",
             "Используйте экранированные провода для аналоговых сигналов", "analog_io"),
            
            ("PWM не работает",
             "Неправильный пин, проблемы с кодом, неисправность",
             "1. Убедитесь, что используете PWM пин (3,5,6,9,10,11)\n2. Проверьте синтаксис analogWrite()\n3. Проверьте диапазон значений (0-255)\n4. Проверьте подключение нагрузки",
             "Всегда проверяйте, что пин поддерживает PWM", "analog_io")
        ]
        
        for error_description, possible_causes, solutions, prevention_tips, category in troubleshooting_data:
            cursor.execute("""
                INSERT INTO arduino_troubleshooting 
                (error_description, possible_causes, solutions, prevention_tips, category)
                VALUES (?, ?, ?, ?, ?)
            """, (error_description, possible_causes, solutions, prevention_tips, category))
        
        self.connection.commit()
        logger.info("✅ База данных Arduino Nano заполнена данными")
    
    def search_knowledge(self, query: str, category: Optional[str] = None) -> List[Dict]:
        """Поиск знаний по запросу"""
        cursor = self.connection.cursor()
        
        if category:
            cursor.execute("""
                SELECT ak.*, ac.name as category_name
                FROM arduino_knowledge ak
                JOIN arduino_categories ac ON ak.category_id = ac.id
                WHERE ac.name = ? AND (
                    ak.title LIKE ? OR 
                    ak.content LIKE ? OR 
                    ak.keywords LIKE ?
                )
                ORDER BY ak.difficulty_level
            """, (category, f"%{query}%", f"%{query}%", f"%{query}%"))
        else:
            cursor.execute("""
                SELECT ak.*, ac.name as category_name
                FROM arduino_knowledge ak
                JOIN arduino_categories ac ON ak.category_id = ac.id
                WHERE ak.title LIKE ? OR 
                      ak.content LIKE ? OR 
                      ak.keywords LIKE ?
                ORDER BY ak.difficulty_level
            """, (f"%{query}%", f"%{query}%", f"%{query}%"))
        
        results = []
        for row in cursor.fetchall():
            results.append(dict(row))
        
        return results
    
    def get_function_info(self, function_name: str) -> Optional[Dict]:
        """Получение информации о функции"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM arduino_functions 
            WHERE function_name = ?
        """, (function_name,))
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_library_info(self, library_name: str) -> Optional[Dict]:
        """Получение информации о библиотеке"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM arduino_libraries 
            WHERE library_name = ?
        """, (library_name,))
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_project_info(self, project_name: str) -> Optional[Dict]:
        """Получение информации о проекте"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM arduino_projects 
            WHERE project_name LIKE ?
        """, (f"%{project_name}%",))
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_pin_info(self, pin_number: int) -> Optional[Dict]:
        """Получение информации о пине"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM arduino_pins 
            WHERE pin_number = ?
        """, (pin_number,))
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_troubleshooting(self, error_keywords: str) -> List[Dict]:
        """Поиск решений проблем"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM arduino_troubleshooting 
            WHERE error_description LIKE ? OR 
                  possible_causes LIKE ? OR 
                  solutions LIKE ?
        """, (f"%{error_keywords}%", f"%{error_keywords}%", f"%{error_keywords}%"))
        
        results = []
        for row in cursor.fetchall():
            results.append(dict(row))
        
        return results
    
    def get_categories(self) -> List[Dict]:
        """Получение списка категорий"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM arduino_categories ORDER BY name")
        
        results = []
        for row in cursor.fetchall():
            results.append(dict(row))
        
        return results
    
    def close(self):
        """Закрытие соединения с базой данных"""
        if self.connection:
            self.connection.close()
            logger.info("🔒 Соединение с базой данных Arduino Nano закрыто")

def main():
    """Тестирование базы данных"""
    print("🔧 Тестирование базы данных Arduino Nano")
    print("=" * 50)
    
    # Создаем базу данных
    db = ArduinoNanoDatabase()
    
    # Тестируем поиск
    print("\n📚 Поиск знаний по 'светодиод':")
    results = db.search_knowledge("светодиод")
    for result in results[:3]:  # Показываем первые 3 результата
        print(f"- {result['title']} ({result['category_name']})")
    
    # Тестируем функции
    print("\n⚙️ Информация о функции 'digitalWrite':")
    func_info = db.get_function_info("digitalWrite")
    if func_info:
        print(f"Синтаксис: {func_info['syntax']}")
        print(f"Описание: {func_info['description']}")
    
    # Тестируем библиотеки
    print("\n📦 Информация о библиотеке 'Servo':")
    lib_info = db.get_library_info("Servo")
    if lib_info:
        print(f"Описание: {lib_info['description']}")
        print(f"Функции: {lib_info['functions_list']}")
    
    # Тестируем проекты
    print("\n🚀 Проекты Arduino:")
    projects = db.get_project_info("светодиод")
    if projects:
        print(f"Проект: {projects['project_name']}")
        print(f"Компоненты: {projects['components']}")
    
    # Тестируем пины
    print("\n📌 Информация о пине 13:")
    pin_info = db.get_pin_info(13)
    if pin_info:
        print(f"Тип: {pin_info['pin_type']}")
        print(f"Описание: {pin_info['description']}")
        print(f"Особые функции: {pin_info['special_functions']}")
    
    # Тестируем решение проблем
    print("\n🔧 Решение проблем:")
    troubleshooting = db.get_troubleshooting("не работает")
    for item in troubleshooting[:2]:  # Показываем первые 2 результата
        print(f"- {item['error_description']}")
    
    # Закрываем соединение
    db.close()
    
    print("\n✅ Тестирование завершено!")

if __name__ == "__main__":
    main()





