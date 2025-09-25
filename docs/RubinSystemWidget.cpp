// =================================================================================
//                 RubinSystemWidget.cpp - Виджет диагностики системы "Рубин"
//
// Описание:
// Этот файл представляет собой визуальный компонент (виджет) для отображения
// состояния ключевых модулей системы "Рубин" в реальном времени.
// Код написан с использованием стилистики библиотеки ImGui (Immediate Mode GUI).
//
// Разработчик: Rubin AI
// Дата создания: 12.09.2025
// =================================================================================

#include "imgui.h" // Подключаем заголовочный файл ImGui

// =================================================================================
//     Функция RenderRubinSystemWidget()
//
//     Отрисовывает виджет с полной диагностической информацией о системе.
// =================================================================================
void RenderRubinSystemWidget()
{
    // --- Начало определения окна виджета ---
    // Задаем заголовок окна "Панель Диагностики Rubin"
    ImGui::Begin("Панель Диагностики Rubin");

    // --- Секция 1: Статус API Серверов ---
    // Используем разделитель с заголовком для визуальной группировки
    ImGui::SeparatorText("Состояние API Серверов");

    // Метка: "Основной API (8083)"
    ImGui::Text("Основной API (8083):      ");
    ImGui::SameLine(); // Располагаем следующий элемент на той же строке
    // Устанавливаем цвет текста в зависимости от статуса (зеленый для "OK")
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
    ImGui::Text("[OK]");
    ImGui::PopStyleColor(); // Возвращаем стандартный цвет

    // Метка: "AI Чат API (8086)"
    ImGui::Text("AI Чат API (8086):        ");
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
    ImGui::Text("[OK]");
    ImGui::PopStyleColor();

    // Метка: "Yandex.Disk API (8094)"
    ImGui::Text("Yandex.Disk API (8094):   ");
    ImGui::SameLine();
    // Устанавливаем красный цвет для статуса "ОШИБКА"
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
    ImGui::Text("[ОШИБКА]");
    ImGui::PopStyleColor();


    // --- Секция 2: Ядро системы (C++) ---
    ImGui::Spacing(); // Добавляем небольшой отступ
    ImGui::SeparatorText("Ядро Системы (C++)");

    ImGui::Text("Статус ядра:              ");
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
    ImGui::Text("[ПОДКЛЮЧЕНО]");
    ImGui::PopStyleColor();

    // Метка: "Загрузка ЦП ядра"
    ImGui::Text("Загрузка ЦП ядра:");
    // Визуализируем загрузку в виде ProgressBar
    static float core_cpu_usage = 0.75f; // Примерное значение загрузки
    ImGui::ProgressBar(core_cpu_usage, ImVec2(-1.0f, 0.0f), "75%");


    // --- Секция 3: Контроллер PMAC ---
    ImGui::Spacing();
    ImGui::SeparatorText("Контроллер PMAC");

    ImGui::Text("Соединение с PMAC:       ");
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
    ImGui::Text("[УСТАНОВЛЕНО]");
    ImGui::PopStyleColor();

    // Метка: "Статус осей"
    ImGui::Text("Статус Оси 1:             ");
    ImGui::SameLine();
    ImGui::Text("[ОЖИДАНИЕ]");

    ImGui::Text("Статус Оси 2:             ");
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.6f, 0.0f, 1.0f)); // Оранжевый цвет
    ImGui::Text("[В ДВИЖЕНИИ]");
    ImGui::PopStyleColor();


    // --- Секция 4: База Знаний ---
    ImGui::Spacing();
    ImGui::SeparatorText("База Знаний");

    // Метка: "Количество документов"
    ImGui::Text("Документов в базе:       ");
    ImGui::SameLine();
    ImGui::Text("1,254");

    // Метка: "Статус векторной базы"
    ImGui::Text("Векторная база (Qdrant):  ");
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
    ImGui::Text("[ONLINE]");
    ImGui::PopStyleColor();


    // --- Секция 5: Управляющие кнопки ---
    ImGui::Spacing();
    ImGui::Separator(); // Обычный разделитель
    ImGui::Spacing();

    // Кнопка для запуска полной диагностики
    if (ImGui::Button("Запустить полную диагностику", ImVec2(-1.0f, 0.0f)))
    {
        // Здесь будет код, который вызывается при нажатии на кнопку
        // Например, запуск функции RunFullDiagnostics();
    }

    // Кнопка для перезапуска сервисов
    if (ImGui::Button("Перезапустить все сервисы", ImVec2(-1.0f, 0.0f)))
    {
        // Код для перезапуска всех API серверов
    }

    // --- Конец определения окна виджета ---
    ImGui::End();
}

// =================================================================================
//     Конец файла RubinSystemWidget.cpp
// =================================================================================
