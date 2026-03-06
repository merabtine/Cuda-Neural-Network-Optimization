# CUDA-Based Optimization of Shallow Neural Network Training

## Overview

This project explores the optimization of shallow neural network training using GPU parallelization techniques. The goal is to analyze performance bottlenecks and progressively improve execution efficiency by aligning algorithmic design with GPU architecture.

The study focuses on a matrix-based implementation of a shallow neural network and applies several optimization strategies including memory access optimization, shared memory tiling, full GPU migration, and CUDA stream-based pipelining.

Through systematic profiling and optimization, the project demonstrates how architectural-aware design can significantly accelerate neural network training.

---

## Objectives

The main objectives of this project are:

* Analyze the performance of a baseline neural network implementation.
* Identify computational and memory bottlenecks using GPU profiling tools.
* Improve memory access patterns to reduce bandwidth inefficiencies.
* Increase GPU utilization through shared memory tiling.
* Eliminate CPU bottlenecks by migrating the full training pipeline to the GPU.
* Exploit hardware concurrency using CUDA streams to overlap computation and data transfers.

---

## Methodology

### 1. Baseline Implementation

A shallow neural network with one hidden layer is implemented using matrix-based forward and backward propagation. Initial profiling is performed to understand resource utilization and identify performance bottlenecks.

### 2. Memory Access Optimization

Memory access patterns are improved by transposing weight matrices to enable coalesced memory accesses across GPU threads.

### 3. Shared Memory Tiling

Matrix multiplications are optimized using tiled computation with shared memory, allowing threads within a block to reuse loaded data and reduce global memory traffic.

### 4. Full GPU Migration

The entire training pipeline, including activation functions, gradient computation, and weight updates, is migrated to the GPU to remove CPU bottlenecks and host-device synchronization overhead.

### 5. Streamed GPU Training

CUDA streams are introduced to pipeline multiple training batches, enabling overlap between memory transfers and kernel execution to maximize GPU utilization.

---

## Key Results

The progressive optimization strategy leads to significant performance improvements:

* Improved GPU occupancy and memory efficiency
* Reduced kernel launch overhead
* Elimination of CPU bottlenecks
* Overlapping computation and memory transfers

The final streamed GPU implementation achieves substantial training speedups compared to the baseline implementation, particularly for medium and large datasets.

---

## Technologies Used

* CUDA
* C/C++
* GPU Profiling with NVIDIA Nsight Compute
* Parallel Programming Concepts
* GPU Architecture Optimization

---

## Experimental Environment

Experiments were conducted on a CUDA-enabled GPU environment using an NVIDIA Tesla T4 architecture. The system supports concurrent kernel execution and asynchronous memory transfers, enabling advanced GPU optimization techniques.

---

## Conclusion

This project demonstrates how systematic profiling and architecture-aware optimization can dramatically improve the performance of neural network training on GPUs. By progressively addressing memory inefficiencies, parallel execution constraints, and CPU-GPU synchronization overhead, the training process evolves from a partially parallel implementation to a fully optimized pipelined GPU workflow.

The methodology used in this work provides a reproducible framework for optimizing computational workloads on modern parallel hardware.

---

## Authors

* Aya Malek Merabtine
* Malek Kebbal
* Yasmine Haouas
* Meriem Benyammi

---

## Academic Context

This project was developed as part of a **High Performance Computing (HPC)** course focusing on GPU optimization and parallel computing techniques.
