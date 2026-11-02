# MLP Optimization on Fashion MNIST: CPU → cuBLAS

A multilayer perceptron written **from scratch in C++**, trained on Fashion-MNIST and optimized step by step: single-core CPU → OpenMP → hand-written CUDA kernels → cuBLAS.

The study behind this repository measured a **158× speedup** over the sequential baseline, taking training time from 237.0 s to 1.5 s.

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![CUDA](https://img.shields.io/badge/CUDA-13.x-76B900)
![OpenMP](https://img.shields.io/badge/OpenMP-4.5-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Results

Best of 10 freshly initialized runs each, batch size 512, as reported in the paper:

| Implementation | Training time | Throughput (samples/s) | Accuracy | Speedup |
| :--- | ---: | ---: | ---: | ---: |
| Sequential (1 CPU core) | 237.0 s | 2,532 | 87.7% | 1.00× |
| OpenMP (16 threads) | 172.7 s | 3,475 | 87.2% | 1.37× |
| Manual CUDA | 1.7 s | 344,850 | 79.7% | 139.41× |
| CUDA + cuBLAS | **1.5 s** | **394,506** | 78.1% | **158.00×** |

Moving the workload to the GPU costs a few accuracy points (8.0 to 9.6 p.p.) because larger batches mean fewer weight updates per epoch — see [Batch size vs. accuracy](#batch-size-vs-accuracy).

## Model

| | |
| :--- | :--- |
| Architecture | 784 → 128 → 64 → 10 (dense, fully connected) |
| Parameters | 109,386 trainable |
| Activations | ReLU (hidden), sigmoid (output) |
| Loss | Mean squared error |
| Weight init | He et al. |
| Bias init | Zeros |
| Epochs | 10 |
| Precision | 32-bit float |

Inputs are the raw 28×28 grayscale pixels normalized to `[0, 1]` (divided by 255.0); targets are one-hot encoded.

## Implementations

| Directory | Technique | Notes |
| :--- | :--- | :--- |
| `sequential/` | Pure C++ on one core | `Matrix` backed by `std::vector`, triple-loop matmul, online (per-sample) updates |
| `openmp/` | Same code + OpenMP | `#pragma omp parallel for collapse(2)` on matmul and element-wise maps |
| `cuda-matrix/` | First GPU prototype | Naive kernel (16×16 thread blocks), `cudaMallocManaged` and a host↔device copy per matmul — kept to show the cost of not having persistent device memory |
| `cuda-manual/` | Hand-written CUDA | Kernels for matmul, bias broadcast, ReLU/sigmoid (+ derivatives), Hadamard, transpose, column sum, scalar ops and add; all weights, biases and workspaces allocated once and kept resident on the device |
| `cuda-cublas/` | Hand-written CUDA + cuBLAS | Identical to `cuda-manual`, but `matmulGpu` delegates to `cublasSgemm` with `CUBLAS_TF32_TENSOR_OP_MATH` |

`cuda-matrix` is the intermediate step that led to `cuda-manual`; the paper reports the four versions in the other directories.

## Repository layout

```
.
├── sequential/          # C++ baseline (Makefile)
├── openmp/              # CPU parallelism (Makefile)
├── cuda-matrix/         # GPU prototype, managed memory (CMake)
├── cuda-manual/         # custom kernels, persistent GPU state (CMake)
├── cuda-cublas/         # custom kernels + cublasSgemm (CMake)
├── data/                # Fashion-MNIST IDX files + MNIST reader (MIT, Baptiste Wicht)
├── docs/
│   ├── paper-en.md      # English translation of the paper
│   └── paper/
│       ├── paper-en.pdf          # English translation, PDF
│       └── ArtigoCUBLAS-pt.pdf   # original paper (pt-BR)
├── scripts/get_data.sh  # downloads the dataset into data/
└── LICENSE
```

Each CUDA directory also contains `src/diagnostic.cu`, a standalone sanity check for the kernels. It is not part of the CMake target.

## Requirements

- `g++` with OpenMP support (tested with GCC 16.2)
- `cmake` ≥ 3.18 and a CUDA Toolkit (tested with NVCC 13.3) for the GPU versions
- An NVIDIA GPU with compute capability 8.9, or edit `CMAKE_CUDA_ARCHITECTURES` in the `CMakeLists.txt` of each CUDA directory
- ~60 MB of free disk for the dataset

`-march=native` is enabled in every build, so binaries are tuned to the machine that compiles them.

## Getting started

### 1. Dataset

```bash
./scripts/get_data.sh
```

It downloads the four uncompressed IDX files into `data/` (the script skips files that are already there). Alternatively, grab them from [zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) and unzip them into `data/`.

### 2. CPU versions

```bash
mkdir -p results

cd sequential
make -j
./mlpSequential > ../results/sequential.csv
```

For OpenMP, replace the directory and the binary name:

```bash
cd openmp
make -j
OMP_NUM_THREADS=16 ./mlpOpenmp > ../results/openmp.csv
```

### 3. GPU versions

```bash
cd cuda-cublas                       # or cuda-manual / cuda-matrix
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/mlpCublas > ../results/cublas.csv
```

Binaries are `mlpGpu` (`cuda-manual`), `mlpCublas` (`cuda-cublas`) and `mlpCuda` (`cuda-matrix`).

> **Run every binary from inside its own directory.** The dataset path is resolved relative to the working directory as `../data`, so `./sequential/mlpSequential` from the repository root will not find the files.

## Output

Each run writes one CSV row per test image to stdout, followed by a summary block:

```
IMAGE_ID,REAL_LABEL,PREDICTED_LABEL,IS_CORRECT,CONFIDENCE
0,9,9,1,91.3
1,2,2,1,88.7
...

#--- FINAL METRICS ---
#Training Time: 1.51234 seconds
#Troughput: 396738 samples/second
#Accuracy: 78.14%
#Class 0 - Precision: 79.2%, Recall:81.4%
...
```

Only the training loop is timed. Because the summary lines start with `#`, the file stays parseable as CSV (skip rows that do not match the numeric pattern).

## Batch size vs. accuracy

Batch size was swept on the cuBLAS version. To rank configurations, the paper defines a relative efficiency metric that trades accuracy against time:

$$E(b) = \frac{A_b - A_{min}}{A_{max} - A_{min}} \times \frac{T_{max} - T_b}{T_{max} - T_{min}}$$

with $A_{max} = 87.5\%$ and $T_{min} = 1.2$ s as the reference bounds.

| Batch size | Time | Accuracy | Throughput | E(b) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 2.5 s | 87.5% | 240,000 | 0.00 |
| 256 | 1.9 s | 81.8% | 321,000 | 0.39 |
| **512** | 1.5 s | 79.1% | 394,000 | **0.60** |
| 1024 | 1.3 s | 73.9% | 450,000 | 0.59 |
| 2048 | 1.2 s | 61.8% | 488,000 | 0.31 |
| 4096 | 1.2 s | 50.1% | 515,000 | 0.00 |

Small batches starve the GPU; large batches buy milliseconds at the cost of dozens of accuracy points. Batch 512 is the best operating point for this setup.

## Hyperparameters in the code

Hyperparameters are compile-time constants at the top of each `src/main.cpp`. The values shipped in the repository are:

| Directory | `LEARNING_RATE` | `BATCH_SIZE` | Update rule |
| :--- | ---: | ---: | :--- |
| `sequential/`, `openmp/` | 0.005 | — | online (one sample per step) |
| `cuda-matrix/`, `cuda-manual/` | 0.1 | 256 | mini-batch |
| `cuda-cublas/` | 0.1 | 4096 | mini-batch |

The benchmarks in the paper used learning rate 0.1 and batch size 512 for all versions. To reproduce them, set those constants in each `main.cpp` and rebuild; the CPU versions use per-sample updates, so matching the paper's batching there requires restructuring the training loop.

## Caveats

- **Accuracy drop on GPU.** Fewer parameter updates per epoch (bigger batches) means slower convergence in 10 epochs. It is a consequence of the batching strategy, not of CUDA itself.
- **TF32 is enabled** in `cuda-cublas` (`CUBLAS_TF32_TENSOR_OP_MATH`), which trades mantissa bits for throughput on Ampere and newer GPUs.
- **Single precision throughout.** No mixed precision or Tensor Core-specific data paths are used beyond the cuBLAS math mode.
- **Timings are machine-specific.** All numbers come from the hardware described in the paper (Ryzen 7 5700X3D, RTX 4060 Ti, 15 GiB DDR4).

## Paper

The full study — methodology, experimental setup and discussion — is available in this repository:

- [**English translation** (Markdown)](docs/paper-en.md) · [PDF](docs/paper/paper-en.pdf)
- [**Original paper** (PDF, pt-BR)](docs/paper/ArtigoCUBLAS-pt.pdf)

### Citation

```bibtex
@misc{campos2026mlp,
  title  = {MLP Optimization on Fashion MNIST: CPU to cuBLAS},
  author = {de Campos, Guilherme and Dall'Igna, Mateus},
  year   = {2026},
  note   = {Course paper, Instituto Federal Catarinense (IFC), Blumenau, SC, Brazil},
  url    = {https://github.com/MateusDallIgna/mlp-optimizations-study}
}
```

## License

Code is released under the [MIT License](LICENSE). The paper and its translation (`docs/`) are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgements

- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) by Zalando Research (MIT).
- `data/include/mnist` is the [MNIST reader](https://github.com/wichtounet/mnist) by Baptiste Wicht (MIT).
