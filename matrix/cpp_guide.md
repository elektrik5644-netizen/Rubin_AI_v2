
# Руководство по языку программирования C++

## Введение в C++

C++ - это мощный язык программирования общего назначения, созданный Бьярном Страуструпом в 1985 году как расширение языка C.

### Основные особенности C++:
- Объектно-ориентированное программирование
- Обобщенное программирование (шаблоны)
- Управление памятью
- Высокая производительность
- Кроссплатформенность

## Основы синтаксиса

### Переменные и типы данных
```cpp
#include <iostream>
using namespace std;

int main() {
    int age = 25;
    float price = 19.99f;
    string name = "Иван";
    bool isStudent = true;
    return 0;
}
```

### Условные операторы
```cpp
if (age >= 18) {
    cout << "Совершеннолетний" << endl;
} else {
    cout << "Несовершеннолетний" << endl;
}
```

### Циклы
```cpp
for (int i = 0; i < 10; i++) {
    cout << i << " ";
}
```

## Классы и объекты

```cpp
class Person {
private:
    string name;
    int age;
    
public:
    Person(string n, int a) : name(n), age(a) {}
    
    void introduce() {
        cout << "Меня зовут " << name << ", мне " << age << " лет" << endl;
    }
};
```

## Наследование

```cpp
class Animal {
protected:
    string species;
    
public:
    Animal(string s) : species(s) {}
    
    virtual void makeSound() {
        cout << "Животное издает звук" << endl;
    }
};

class Dog : public Animal {
public:
    Dog() : Animal("Собака") {}
    
    void makeSound() override {
        cout << "Гав-гав!" << endl;
    }
};
```

## Шаблоны

```cpp
template <typename T>
T getMax(T a, T b) {
    return (a > b) ? a : b;
}
```

## Управление памятью

```cpp
int* numbers = new int[10];
// Использование памяти
delete[] numbers;
```

## Стандартная библиотека

```cpp
#include <vector>
#include <string>
#include <algorithm>

vector<int> numbers = {5, 2, 8, 1, 9};
sort(numbers.begin(), numbers.end());
```

## Обработка исключений

```cpp
try {
    double result = divide(10, 0);
} catch (const invalid_argument& e) {
    cout << "Ошибка: " << e.what() << endl;
}
```

## Лучшие практики

1. Используйте const для неизменяемых значений
2. Предпочитайте ссылки указателям
3. Используйте умные указатели для управления памятью
4. Следуйте RAII принципу
5. Используйте auto для автоматического вывода типов
