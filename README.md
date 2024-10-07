# MPI Matrix Multiplication

This project demonstrates a parallel matrix multiplication using the MPI (Message Passing Interface) framework. It distributes rows of matrix A among different processes, multiplies them with matrix B, and gathers the results into matrix C.

## Requirements

To run this project locally, ensure that the following tools are installed on your machine:

- **MPI Library**: This project uses MPI for parallel processing. You can use OpenMPI or MPICH.
- **C Compiler**: To compile the C code, you will need `mpicc` (MPI's C compiler).

### Installing OpenMPI and OpenMP

**MacOS** (using Homebrew):

```bash
brew install open-mpi llvm libomp
```

**Ubuntu**:

```bash
sudo apt update
sudo apt install openmpi-bin openmpi-common libopenmpi-dev libomp
```

**Windows**:

For Windows, you can install OpenMPI through MSYS2 or use WSL (Windows Subsystem for Linux) to emulate a Linux environment.

## Files

- `src/openmp-optimise`: Main source code for applying a convolution filter on a grayscale image with OpenMP
- `src/openmpi-optimise`: Main source code implementing matrix multiplication using MPI and saving the results to a file.
- `src/openmpi-uneven-optimise`: Example scenario of the having rows in AAA not evenly divisble by the number of processes.
- `output/test.txt`: The file with the resultant matrix C.
- `output/uneven-test.txt`: The file where the resultant matrix C with AAA edge case.

## Setup

### Step 1: Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/restartdk/computational-fluid-dynamics-hpi-openmp.git
cd computational-fluid-dynamics-hpi-openmp/
```

### Step 2: Compile the Program

Use `mpicc` to compile the C program:

```bash
mpicc -o matrix_multiply_file matrix_multiply_file.c
```

This will produce an executable named `matrix_multiply_file`.

### Step 3: Run the Program

Run the program using `mpirun` or `mpiexec` with a specified number of processes:

```bash
mpirun -np <NUMBER_OF_PROCESSES> ./matrix_multiply_file
```

### Step 4: View the Output

The resultant matrix `C` will be written to `test.txt`. You can view it using any text editor:

```bash
cat test.txt
```

### Example Output

```text
14 32
32 77
50 122
```

This output shows the result of multiplying matrix A by matrix B and saving it to `test.txt`.

## Customising Matrix Sizes

The matrix sizes are defined by the constants `M`, `K`, and `N` in the source code. You can modify these values as needed in the `openmpi-optimise.c` file:

```c
#define M 5 // Number of rows in A
#define K 3 // Number of columns in A and rows in B
#define N 2 // Number of columns in B
```

## Analysis of performance changes to OpenMP

I tested the convolution operation with different kernel sizes and scheduling strategies to optimise performance. I measured the execution time for each combination of kernel size and scheduling strategy to compare their efficiency.

The results showed that the guided scheduling strategy with a chunk size of 3 was the most efficient for both kernel sizes 3 and 7. For a kernel size of 3, the guided scheduling strategy with a chunk size of 3 achieved a convolution time of 0.088422 seconds, which was faster than the other scheduling strategies tested. For a kernel size of 7, the guided scheduling strategy with a chunk size of 3 achieved a convolution time of 0.563970 seconds for a normal image, 0.452003 seconds for a serial image, and 0.439804 seconds for a blurred image. Compared to the other scheduling strategies tested, these times were the fastest.

I encountered some challenges, such as installing OpenMP on my computer and configuring the development environment to support it. However, I was able to overcome these challenges by following the appropriate installation instructions and properly configuring the compiler and development environment.
