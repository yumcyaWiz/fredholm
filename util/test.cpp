#include <cmath>
#include <iostream>

inline float repeat(float x) { return x - std::floor(x); }

int main()
{
    std::cout << repeat(1.1f) << std::endl;

    return 0;
}