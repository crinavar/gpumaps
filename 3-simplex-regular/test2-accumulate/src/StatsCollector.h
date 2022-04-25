#pragma once

#include <cinttypes>
#include <cmath>
#include <numeric>
#include <vector>
#define NOT_CALCULATED HUGE_VAL

template <typename T>
class StatsCollector {
    std::vector<T> runs;
    float average;
    float standardDeviation;
    float variance;

public:
    StatsCollector();

    void add(T val);
    float getAverage();
    float getStandardDeviation();
    float getVariance();

    bool isInvalid(float var);
};