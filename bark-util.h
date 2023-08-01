#pragma once

#include <cmath>
#include <fstream>

#define BARK_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "BARK_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

static const size_t MB = 4*1024*1024;

template<typename T>
static void read_safe(std::ifstream& infile, T& dest) {
    infile.read((char*)& dest, sizeof(T));
}

static size_t utf8_len(char src) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

void softmax(std::vector<float> & logits) {
    // for numerical stability
    float maxl = -INFINITY;
    for (const auto & l : logits)
        maxl = std::max(maxl, l);

    // softmax
    float sum = 0.0;
    for (auto & l : logits) {
        l = exp(l - maxl);
        sum += l;
    }

    for (auto & l : logits)
        l /= sum;
}
