#pragma once

#include <fstream>
#include <vector>

#define BARK_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "BARK_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

static const size_t MB = 4*1024*1024;

template<typename T>
static void read_safe(std::ifstream& fin, T& dest) {
    fin.read((char*)& dest, sizeof(T));
}

template<typename T>
static void write_safe(std::ofstream& fout, T& dest) {
    fout.write((char*)& dest, sizeof(T));
}


static size_t utf8_len(char src) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

template <typename E,typename X>
void unroll(const std::vector<E>& v,std::vector<X>& out){
    std::cout << "unroll vector\n";
    out.insert(out.end(), v.begin(), v.end());
}

template <typename V,typename X>
void unroll(const std::vector<std::vector<V>>& v,std::vector<X>& out) {
    std::cout << "unroll vector of vectors\n";
    for (const auto& e : v) unroll(e,out);
}