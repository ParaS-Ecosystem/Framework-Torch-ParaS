// repro_b.cpp
#include "/storage/Goutam/Framework-Torch-ParaS/repro_a.h"
int main() {
    repro::Queue q;
    q.init();
    int* a = static_cast<int*>(q.alloc(sizeof(int) * 4));
    int* b = static_cast<int*>(q.alloc(sizeof(int) * 4));
    for (int i = 0; i < 4; ++i) a[i] = i;
    q.parallel_for(4, [a](std::size_t i) { a[i] *= 2; });
    q.copy(b, a, sizeof(int) * 4);
    return 0;
}