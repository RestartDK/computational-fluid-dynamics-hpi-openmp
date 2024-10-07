#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 1920
#define HEIGHT 1080

// Convolution time for nothing 1: 0.090295
// Kernel size 3 Convolution time for static chunk size 1: 0.145393
// Kernel size 3 Convolution time for dynamic chunk size 1: 0.149825
// Kernel size 3 Convolution time for dynamic chunk size 1: 0.121254
// Kernel size 3 Convolution time for dynamic chunk size 32: 0.102237
// Kernel size 3 Convolution time for guided chunk size 3: 0.088422
// Kernel size 3 Convolution time for guided chunk size 9: 0.117862
// Kernel size 7 Convolution time for guided chunk size 3: 0.551508
// Image blur Kernel size 7 Convolution time for guided chunk size 3: 0.463137
// Image normal Kernel size 7 Convolution time for guided chunk size 3: 0.620178

// Image normal Kernel size 7 Convolution time for guided chunk size 3: 0.563970
// Image serial Kernel size 7 Convolution time for guided chunk size 3: 0.452003
// Image blur Kernel size 7 Convolution time for guided chunk size 3: 0.439804

void apply_convolution(double **image, double **output, double **kernel,
                       int kernel_size) {
  int offset = kernel_size / 2;

#pragma omp parallel for collapse(2) schedule(guided, 3)
  for (int y = offset; y < HEIGHT - offset; y++) {
    for (int x = offset; x < WIDTH - offset; x++) {
      double sum = 0.0;
      for (int ky = 0; ky < kernel_size; ky++) {
        for (int kx = 0; kx < kernel_size; kx++) {
          int ix = x + kx - offset;
          int iy = y + ky - offset;
          sum += image[iy][ix] * kernel[ky][kx];
        }
      }
      output[y][x] = sum;
    }
  }
}

void apply_convolution_serial(double **image, double **output, double **kernel,
                              int kernel_size) {
  for (int y = 1; y < HEIGHT - 1; y++) {
    for (int x = 1; x < WIDTH - 1; x++) {
      double sum = 0.0;
      for (int ky = 0; ky < kernel_size; ky++) {
        for (int kx = 0; kx < kernel_size; kx++) {
          int ix = x + kx - 1;
          int iy = y + ky - 1;
          sum += image[iy][ix] * kernel[ky][kx];
        }
      }
      output[y][x] = sum;
    }
  }
}

int main() {
  // Seed the random number generator
  srand(time(0));
  // Dynamically allocate memory for image and output
  double **image = (double **)malloc(HEIGHT * sizeof(double *));
  double **output = (double **)malloc(HEIGHT * sizeof(double *));
  double **output_serial = (double **)malloc(HEIGHT * sizeof(double *));
  for (int i = 0; i < HEIGHT; i++) {
    image[i] = (double *)malloc(WIDTH * sizeof(double));
    output[i] = (double *)malloc(WIDTH * sizeof(double));
    output_serial[i] = (double *)malloc(WIDTH * sizeof(double));
  }

  // Initialize the image with random values
  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      image[y][x] = rand() % 256;
    }
  }

  // Example kernel (3x3 sharpen filter)
  int kernel_size = 7;
  double **kernel = (double **)malloc(kernel_size * sizeof(double *));
  for (int i = 0; i < kernel_size; i++) {
    kernel[i] = (double *)malloc(kernel_size * sizeof(double));
  }

  double **kernel_blur = (double **)malloc(kernel_size * sizeof(double *));
  for (int i = 0; i < kernel_size; i++) {
    kernel_blur[i] = (double *)malloc(kernel_size * sizeof(double));
  }

  // Normal image
  kernel[0][0] = 0;
  kernel[0][1] = -1;
  kernel[0][2] = 0;
  kernel[1][0] = -1;
  kernel[1][1] = 4;
  kernel[1][2] = -1;
  kernel[2][0] = 0;
  kernel[2][1] = -1;
  kernel[2][2] = 0;

  // Blurred image
  kernel_blur[0][0] = 0 / 9.0;
  kernel_blur[0][1] = -1 / 9.0;
  kernel_blur[0][2] = 0 / 9.0;
  kernel_blur[1][0] = -1 / 9.0;
  kernel_blur[1][1] = 4 / 9.0;
  kernel_blur[1][2] = -1 / 9.0;
  kernel_blur[2][0] = 0 / 9.0;
  kernel_blur[2][1] = -1 / 9.0;
  kernel_blur[2][2] = 0 / 9.0;

  // Apply convolution
  clock_t startTime = clock();
  apply_convolution(image, output, kernel, kernel_size);
  clock_t endTime = clock();
  double elapsedTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;
  printf("Convolution time: %f seconds\n", elapsedTime);

  // Apply convolution on serial
  clock_t startTime_serial = clock();
  apply_convolution(image, output_serial, kernel, kernel_size);
  clock_t endTime_serial = clock();
  double elapsedTime_serial =
      (double)(endTime_serial - startTime_serial) / CLOCKS_PER_SEC;
  printf("Convolution time serial: %f seconds\n", elapsedTime_serial);

  // Apply convolution on blur
  clock_t startTime_blur = clock();
  apply_convolution(image, output, kernel_blur, kernel_size);
  clock_t endTime_blur = clock();
  double elapsedTime_blur =
      (double)(endTime_blur - startTime_blur) / CLOCKS_PER_SEC;
  printf("Convolution time blur: %f seconds\n", elapsedTime_blur);

  // Output the result of a small section
  for (int y = 0; y < 5; y++) {
    for (int x = 0; x < 5; x++) {
      printf("%6.1f ", output[y][x]);
    }
    printf("\n");
  }

  // Free allocated memory
  for (int i = 0; i < HEIGHT; i++) {
    free(image[i]);
    free(output[i]);
    free(output_serial[i]);
  }
  free(image);
  free(output);
  free(output_serial);

  for (int i = 0; i < kernel_size; i++) {
    free(kernel[i]);
    free(kernel_blur[i]);
  }
  free(kernel);
  free(kernel_blur);

  return 0;
}
