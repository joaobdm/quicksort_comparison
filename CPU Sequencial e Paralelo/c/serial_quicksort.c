#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

// Function to generate random integers within the range of integer limits
void fill_random(int *array, int n) {
    for (int i = 0; i < n; i++) {
        array[i] = rand();
    }
}

// Comparison function for qsort
int compare(const void *a, const void *b) {
    int arg1 = *(const int *)a;
    int arg2 = *(const int *)b;
    return (arg1 > arg2) - (arg1 < arg2);
}

// QuickSort function
void quicksort(int *array, int low, int high) {
    if (low < high) {
        int pivot = array[high];
        int i = low - 1;

        for (int j = low; j < high; j++) {
            if (array[j] < pivot) {
                i++;
                int temp = array[i];
                array[i] = array[j];
                array[j] = temp;
            }
        }
        int temp = array[i + 1];
        array[i + 1] = array[high];
        array[high] = temp;

        int pi = i + 1;
        quicksort(array, low, pi - 1);
        quicksort(array, pi + 1, high);
    }
}

// Function to compare two arrays
bool compare_arrays(int *a, int *b, int size, double *similarity) {
    bool identical = true;
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (a[i] != b[i]) {
            identical = false;
        } else {
            count++;
        }
    }
    *similarity = (double)count / size * 100.0;
    return identical;
}

int main() {
    int N;
    printf("Enter the number of elements: ");
    scanf("%d", &N);

    int *array = malloc(N * sizeof(int));
    int *control_sorted_array = malloc(N * sizeof(int));

    srand(time(NULL));
    fill_random(array, N);

    memcpy(control_sorted_array, array, N * sizeof(int));
    qsort(control_sorted_array, N, sizeof(int), compare);

    clock_t start_time = clock();
    quicksort(array, 0, N - 1);
    clock_t end_time = clock();

    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    double similarity;
    compare_arrays(array, control_sorted_array, N, &similarity);

    printf("Arrays are %.2f%% similar.\n", similarity);
    printf("Elapsed time: %.6f seconds.\n", elapsed_time);

    free(array);
    free(control_sorted_array);

    return 0;
}
