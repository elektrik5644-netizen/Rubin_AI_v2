
// Примеры кода на C++

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

int main() {
    // Hello World
    cout << "Привет, мир!" << endl;
    
    // Работа с векторами
    vector<int> numbers = {1, 2, 3, 4, 5};
    for (int num : numbers) {
        cout << num << " ";
    }
    cout << endl;
    
    // Сортировка
    sort(numbers.begin(), numbers.end());
    
    // Класс
    class Rectangle {
    private:
        double width, height;
        
    public:
        Rectangle(double w, double h) : width(w), height(h) {}
        
        double area() const {
            return width * height;
        }
    };
    
    Rectangle rect(5.0, 3.0);
    cout << "Площадь: " << rect.area() << endl;
    
    return 0;
}
