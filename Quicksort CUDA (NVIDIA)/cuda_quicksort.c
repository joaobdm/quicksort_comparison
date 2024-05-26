// serial_quicksort.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quicksort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

int main() {
    int LENGTH_OF_INT_LIST = 10000;
    int* int_list = (int*)malloc(LENGTH_OF_INT_LIST * sizeof(int));
    if (int_list == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    srand(time(0));
    for (int i = 0; i < LENGTH_OF_INT_LIST; i++) {
        int_list[i] = rand();
    }

    clock_t start_sort_time = clock();
    quicksort(int_list, 0, LENGTH_OF_INT_LIST - 1);
    clock_t end_sort_time = clock();

    printf("Time to sort list of %d integers using serial quicksort: %f seconds.\n", LENGTH_OF_INT_LIST, (double)(end_sort_time - start_sort_time) / CLOCKS_PER_SEC);

    free(int_list);
    return 0;
}
