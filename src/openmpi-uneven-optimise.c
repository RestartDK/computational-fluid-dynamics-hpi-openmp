#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define M 5 // Number of rows in A
#define K 3 // Number of columns in A and rows in B
#define N 2 // Number of columns in B

void initialize_matrices(int A[M][K], int B[K][N]) {
  // Initialize matrix A
  int temp_A[M][K] = {
      {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}};

  // Initialize matrix B
  int temp_B[K][N] = {{1, 4}, {2, 5}, {3, 6}};

  // Copy values to A and B
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      A[i][j] = temp_A[i][j];
    }
  }

  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      B[i][j] = temp_B[i][j];
    }
  }
}

void print_matrix(int C[M][N]) {
  // Function to print matrix C
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      printf("%d ", C[i][j]);
    }
    printf("\n");
  }
}

void write_matrix_to_file(int C[M][N], const char *filename) {
  // Open file to write the output
  FILE *file = fopen(filename, "w");
  if (file == NULL) {
    printf("Error opening file!\n");
    return;
  }

  // Write the matrix to the file
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      fprintf(file, "%d ", C[i][j]);
    }
    fprintf(file, "\n");
  }

  // Close the file
  fclose(file);
}

int main(int argc, char **argv) {
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int A[M][K], B[K][N], C[M][N];
  int local_A[M / size + 1][K]; // Buffer to hold a portion of A
  int local_C[M / size + 1][N]; // Buffer to hold a portion of C

  int rows_per_process = M / size;
  int extra_rows = M % size;
  int local_rows; // Rows for the current process

  // Determine how many rows each process gets
  if (rank < extra_rows) {
    local_rows = rows_per_process + 1;
  } else {
    local_rows = rows_per_process;
  }

  // Arrays to specify send counts and displacements for MPI_Scatterv
  int sendcounts[size];
  int displs[size];

  // Compute sendcounts and displs for each process
  int offset = 0;
  for (int i = 0; i < size; i++) {
    sendcounts[i] =
        (i < extra_rows) ? (rows_per_process + 1) * K : rows_per_process * K;
    displs[i] = offset;
    offset += sendcounts[i];
  }

  if (rank == 0) {
    // Initialize matrices A and B in the root process
    initialize_matrices(A, B);
  }

  // Scatter rows of matrix A to all processes
  MPI_Scatterv(&A[0][0], sendcounts, displs, MPI_INT, &local_A[0][0],
               local_rows * K, MPI_INT, 0, MPI_COMM_WORLD);

  // Broadcast matrix B to all processes
  MPI_Bcast(&B[0][0], K * N, MPI_INT, 0, MPI_COMM_WORLD);

  // Perform the multiplication for the local part of A
  for (int i = 0; i < local_rows; i++) {
    for (int j = 0; j < N; j++) {
      local_C[i][j] = 0;
      for (int k = 0; k < K; k++) {
        local_C[i][j] += local_A[i][k] * B[k][j];
      }
    }
  }

  // Gather the local parts of C from all processes
  int recvcounts[size]; // For gathering the computed C matrix
  int displs_C[size];   // Displacements for C matrix

  // Compute recvcounts and displs for gathering C
  offset = 0;
  for (int i = 0; i < size; i++) {
    recvcounts[i] =
        (i < extra_rows) ? (rows_per_process + 1) * N : rows_per_process * N;
    displs_C[i] = offset;
    offset += recvcounts[i];
  }

  // Collect the computed rows of C from all processes to the root process
  MPI_Gatherv(&local_C[0][0], local_rows * N, MPI_INT, &C[0][0], recvcounts,
              displs_C, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    // Print the result
    printf("Resultant Matrix C:\n");
    print_matrix(C);
    write_matrix_to_file(C, "./output/uneven-test.txt");
  }

  MPI_Finalize();
  return 0;
}
