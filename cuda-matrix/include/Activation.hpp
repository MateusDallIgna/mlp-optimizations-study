#pragma once
#include <algorithm>
#include <cmath>

struct Sigmoid {
  template <typename T> T operator()(T x) const {
    return static_cast<T>(1) / (1 + exp(-x));
  };
  template <typename T> T derivative(T x) const {
    T s = (*this)(x);
    return s * (static_cast<T>(1.0) - s);
  }
};

struct Relu {
  template <typename T>

  T operator()(T x) const {
    return std::max(static_cast<T>(0), x);
  };
  template <typename T> T derivative(T x) const {
    return x > static_cast<T>(0) ? static_cast<T>(1.0) : static_cast<T>(0.0);
  };
};
