#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void quicksort(int *arr, int low, int high) {
    int i, j, pivot, temp;
    if (low < high) {
        pivot = low;
        i = low;
        j = high;
        while (i < j) {
            while (arr[i] <= arr[pivot] && i <= high)
                i++;
            while (arr[j] > arr[pivot] && j >= low)
                j--;
            if (i < j) {
                temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        temp = arr[j];
        arr[j] = arr[pivot];
        arr[pivot] = temp;
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                quicksort(arr, low, j - 1);
            }
            #pragma omp section
            {
                quicksort(arr, j + 1, high);
            }
        }
    }
}

int main() {
    int n = 100000000; // Number of elements
    int *arr = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % n; // Fill array with random integers
    }
    double start_time = omp_get_wtime();
    quicksort(arr, 0, n - 1);
    double end_time = omp_get_wtime();
    printf("Time taken: %f seconds\n", end_time - start_time);
    free(arr);
    return 0;
}
