#pragma once
#include <stdexcept>
#include <vector>

template <typename T>

class Matrix {
private:
  int rows, columns;
  std::vector<T> data;

public:
  Matrix() : rows(0), columns(0), data(0) {}
  Matrix(int r, int c) : rows(r), columns(c), data(r * c, static_cast<T>(0)) {}

  T &operator()(int r, int c) { return data[r * columns + c]; }

  T operator()(int r, int c) const { return data[r * columns + c]; }

  int getRows() const { return rows; }
  int getColumns() const { return columns; }

  template <typename ACTIVATION_FUNC> void map(ACTIVATION_FUNC f) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        (*this)(i, j) = f((*this)(i, j));
      }
    }
  }

  template <typename GENERATOR> void fill(GENERATOR g) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        (*this)(i, j) = g();
      }
    }
  }

  Matrix<T> operator*(const Matrix<T> &other) {

    if (this->columns != other.rows) {
      throw std::invalid_argument("Incompatible dimensions to multiply");
    }

    Matrix<T> result(this->rows, other.columns);

    for (int i = 0; i < this->rows; i++) {
      for (int j = 0; j < other.columns; j++) {

        T sum = 0;

        for (int k = 0; k < this->columns; k++) {

          sum += (*this)(i, k) * const_cast<Matrix<T> &>(other)(k, j);
        }
        result(i, j) = sum;
      }
    }
    return result;
  }

  Matrix<T> operator*(T number) const {

    Matrix<T> result(this->rows, this->columns);

    for (int i = 0; i < this->rows; i++) {
      for (int j = 0; j < this->columns; j++) {

        result(i, j) = (*this)(i, j) * number;
      }
    }
    return result;
  }

  Matrix<T> operator-(const Matrix<T> &other) const {
    if (this->rows != other.rows || this->columns != other.columns) {
      throw std::invalid_argument("Incompatible dimensions to sum");
    }

    Matrix<T> result(this->rows, this->columns);

    for (int i = 0; i < this->rows; i++) {
      for (int j = 0; j < this->columns; j++) {
        result(i, j) = (*this)(i, j) - other(i, j);
      }
    }
    return result;
  }

  Matrix<T> operator+(const Matrix<T> &other) const {
    if (this->rows != other.rows || this->columns != other.columns) {
      throw std::invalid_argument("Incompatible dimensions to sum");
    }

    Matrix<T> result(this->rows, this->columns);

    for (int i = 0; i < this->rows; i++) {
      for (int j = 0; j < this->columns; j++) {
        result(i, j) = (*this)(i, j) + other(i, j);
      }
    }
    return result;
  }

  Matrix<T> hadamard(const Matrix<T> &other) const {

    if (this->rows != other.rows || this->columns != other.columns) {
      throw std::invalid_argument("Incompatible dimensions to sum");
    }

    Matrix<T> result(this->rows, this->columns);

    for (int i = 0; i < this->rows; i++) {
      for (int j = 0; j < this->columns; j++) {
        result(i, j) = (*this)(i, j) * other(i, j);
      }
    }
    return result;
  }

  Matrix<T> Transpose() const {
    Matrix<T> result(columns, rows);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        result(j, i) = (*this)(i, j);
      }
    }
    return result;
  }
};
