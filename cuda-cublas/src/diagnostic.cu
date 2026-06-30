#include "../include/cuda_wrappers.hpp"
#include <iostream>
#include <cmath>

bool check(float got, float expected, const char* name) {
    if (std::abs(got - expected) > 0.001f) {
        std::cout << "FAIL " << name << ": got " << got << " expected " << expected << std::endl;
        return false;
    }
    std::cout << "PASS " << name << ": " << got << std::endl;
    return true;
}

int main() {
    // 1. Test matmul: A(2×3) * B(3×2) = C(2×2)
    Matrix<float> A(2, 3), B(3, 2), C(2, 2);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;  A(1,0)=4; A(1,1)=5; A(1,2)=6;
    B(0,0)=7; B(0,1)=8;  B(1,0)=9; B(1,1)=10; B(2,0)=11; B(2,1)=12;
    matmulGpu(A, B, C);
    check(C(0,0), 1*7+2*9+3*11, "matmul[0][0]");
    check(C(0,1), 1*8+2*10+3*12, "matmul[0][1]");
    check(C(1,0), 4*7+5*9+6*11, "matmul[1][0]");
    check(C(1,1), 4*8+5*10+6*12, "matmul[1][1]");

    // 2. Test sumColumns: D(2×3) → E(2×1)
    Matrix<float> D(2, 3), E(2, 1);
    D(0,0)=1; D(0,1)=2; D(0,2)=3;  D(1,0)=4; D(1,1)=5; D(1,2)=6;
    sumColumnsGpu(D, E);
    check(E(0,0), 6, "sumColumns[0]");
    check(E(1,0), 15, "sumColumns[1]");

    // 3. Test biasBroadcast: F(2×3), bias(2×1) → F += bias
    Matrix<float> F(2, 3), bias(2, 1);
    F(0,0)=1; F(0,1)=2; F(0,2)=3;  F(1,0)=4; F(1,1)=5; F(1,2)=6;
    bias(0,0)=10; bias(1,0)=20;
    biasBroadcastGpu(F, bias);
    check(F(0,0), 11, "biasBroadcast[0][0]");
    check(F(0,1), 12, "biasBroadcast[0][1]");
    check(F(1,0), 24, "biasBroadcast[1][0]");
    check(F(1,2), 26, "biasBroadcast[1][2]");

    // 4. Test transpose: G(2×3) → H(3×2)
    Matrix<float> G(2, 3), H(3, 2);
    G(0,0)=1; G(0,1)=2; G(0,2)=3;  G(1,0)=4; G(1,1)=5; G(1,2)=6;
    transposeGpu(G, H);
    check(H(0,0), 1, "transpose[0][0]");
    check(H(0,1), 4, "transpose[0][1]");
    check(H(2,0), 3, "transpose[2][0]");
    check(H(2,1), 6, "transpose[2][1]");

    // 5. Test element-wise ops: ADD, SUB, HADAMARD, SCALAR
    Matrix<float> X(2, 2), Y(2, 2), Z(2, 2);
    X(0,0)=1; X(0,1)=2; X(1,0)=3; X(1,1)=4;
    Y(0,0)=5; Y(0,1)=6; Y(1,0)=7; Y(1,1)=8;
    addGpu(X, Y, Z);
    check(Z(0,0), 6, "add[0][0]");
    check(Z(1,1), 12, "add[1][1]");
    subGpu(X, Y, Z);
    check(Z(0,0), -4, "sub[0][0]");
    hadamardGpu(X, Y, Z);
    check(Z(0,0), 5, "hadamard[0][0]");
    check(Z(1,1), 32, "hadamard[1][1]");
    scalarGpu(X, Z, 3.0f);
    check(Z(0,0), 3, "scalar[0][0]");
    check(Z(1,1), 12, "scalar[1][1]");

    // 6. Test map functions
    Matrix<float> M(2, 2);
    M(0,0)=-1; M(0,1)=0; M(1,0)=2; M(1,1)=3;
    mapReluGpu(M);
    check(M(0,0), 0, "relu[-1]");
    check(M(0,1), 0, "relu[0]");
    check(M(1,0), 2, "relu[2]");
    check(M(1,1), 3, "relu[3]");

    M(0,0)=0; M(0,1)=1; M(1,0)=2; M(1,1)=3;
    mapReluDerivGpu(M);
    check(M(0,0), 0, "reluDeriv[0]");
    check(M(0,1), 1, "reluDeriv[1]");

    M(0,0)=0; M(0,1)=1; M(1,0)=2; M(1,1)=3;
    mapSigmoidGpu(M);
    check(M(0,0), 0.5f, "sigmoid[0]");
    check(M(1,1), 0.952574f, "sigmoid[3]");

    std::cout << "Todos os testes passaram!" << std::endl;
    return 0;
}
