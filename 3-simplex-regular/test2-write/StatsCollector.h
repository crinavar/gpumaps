#pragma once

#include <cinttypes>
#include <cmath>
#include <numeric>
#include <vector>
#define NOT_CALCULATED HUGE_VAL

template <typename T>
class StatsCollector {
    std::vector<T> runs;
    T average = NOT_CALCULATED;
    T standardDeviation = NOT_CALCULATED;
    T variance = NOT_CALCULATED;

    void add(T val);
    T getAverage();
    T getStandardDeviation();
    T getVariance();

    bool isInvalid(double var);
};