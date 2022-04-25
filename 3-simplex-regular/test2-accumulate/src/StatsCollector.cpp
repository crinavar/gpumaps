#include "StatsCollector.h"

template <typename T>
StatsCollector<T>::StatsCollector() {
    this->average = NOT_CALCULATED;
    this->standardDeviation = NOT_CALCULATED;
    this->variance = NOT_CALCULATED;
}

template <typename T>
void StatsCollector<T>::add(T val) {
    if (val > 0) {
        this->runs.push_back(val);
    }
    this->average = NOT_CALCULATED;
    this->variance = NOT_CALCULATED;
    this->standardDeviation = NOT_CALCULATED;
}

template <typename T>
float StatsCollector<T>::getAverage() {
    if (isInvalid(this->average) && this->runs.size() != 0) {
        this->average = std::reduce(this->runs.begin(), this->runs.end(), 0.0) / this->runs.size();
    }
    return this->average;
}

template <typename T>
float StatsCollector<T>::getStandardDeviation() {
    if (isInvalid(this->standardDeviation) && this->runs.size() != 0) {
        const float variance = getVariance();
        this->standarDeviation = sqrt(variance);
    }
    return this->standardDeviation;
}

template <typename T>
float StatsCollector<T>::getVariance() {
    if (isInvalid(this->variance) && this->runs.size() != 0) {
        const float mean = getAverage();
        auto sz = runs.size();
        auto variance_func = [&mean, &sz](T accumulator, const T& val) {
            return accumulator + ((val - mean) * (val - mean) / (sz - 1));
        };

        this->variance = std::accumulate(runs.begin(), runs.end(), 0.0, variance_func);
    }
    return this->variance;
}
template <typename T>
bool StatsCollector<T>::isInvalid(float var) {
    return var == NOT_CALCULATED;
}