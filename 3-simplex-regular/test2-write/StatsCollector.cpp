#include <StatsCollector.h>

template <typename T>
void StatsCollector::add(T val) {
    if (val > 0) {
        this->runs.push_back(val);
    }
    this->average = NOT_CALCULATED;
    this->variance = NOT_CALCULATED;
    this->standardDeviation = NOT_CALCULATED;
}

template <typename T>
double StatsCollector::getAverage() {
    if (isInvalid(this->average)) {
        if (this->runs.size() != 0) {
            return std::reduce(this->runs.begin(), this->runs.end(), 0.0) / this->runs.size();
        }
    } else {
        return this->average;
    }
    return NOT_CALCULATED;
}

template <typename T>
double StatsCollector::getStandardDeviation() {
    if (isInvalid(this->standardDeviation)) {
        if (this->runs.size() != 0) {

        } else {
        }
    } else {
        return this->standardDeviation;
    }

    return NOT_CALCULATED;
}

template <typename T>
T StatsCollector::getVariance() {
    if (isInvalid(this->variance)) {
        if (this->runs.size() != 0) {
            const T mean = getAverage();
            // Now calculate the variance
            auto variance_func = [&mean, &sz](T accumulator, const T& val) {
                return accumulator + ((val - mean) * (val - mean) / (sz - 1));
            };

            return std::accumulate(vec.begin(), vec.end(), 0.0, variance_func);
        }
    } else {
        return this->averageTime;
    }
    return NOT_CALCULATED;
}

bool StatsCollector::isInvalid(double var) {
    return var == NOT_CALCULATED;
}