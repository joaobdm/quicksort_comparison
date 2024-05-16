import sys
from typing import Final
from numba import cuda
import numpy as np
import numba
import time
import copy

@cuda.jit
def partition_kernel(arr, lo, hi, pivot_indices):
    pivot = arr[(lo + hi) // 2]
    i = lo
    j = hi

    while True:
        while arr[i] < pivot:
            i += 1
        while arr[j] > pivot:
            j -= 1
        if i >= j:
            pivot_indices[0] = j
            return
        arr[i], arr[j] = arr[j], arr[i]
        i += 1
        j -= 1

def quicksort(arr):
    d_arr = cuda.to_device(arr)
    stack = [(0, len(arr) - 1)]
    pivot_indices = cuda.device_array(1, dtype=np.int32)
    
    while stack:
        lo, hi = stack.pop()
        if lo < hi:
            partition_kernel[1, 1](d_arr, lo, hi, pivot_indices)
            pivot = pivot_indices.copy_to_host()[0]

            if lo < pivot:
                stack.append((lo, pivot))
            if pivot + 1 < hi:
                stack.append((pivot + 1, hi))

    d_arr.copy_to_host(arr)

LENGTH_OF_INT_LIST: Final = 10_000_000
MAX_POSSIBLE_INT_VALUE = sys.maxsize
MIN_POSSIBLE_INT_VALUE = -sys.maxsize - 1

int_list = np.random.randint(MIN_POSSIBLE_INT_VALUE, MAX_POSSIBLE_INT_VALUE + 1, size=LENGTH_OF_INT_LIST)

int_list_copy = copy.deepcopy(int_list)

print("\nGenerating reference sorted list using Python's \"sorted\" built-in function for validating correctness of serial version and parallel version of quicksort... ", end="")

reference_sorted_list = sorted(int_list)
print("Done!")

start_time = time.time()

quicksort(int_list)
elapsed_time = time.time() - start_time
print("Execution time:", elapsed_time, "seconds")
print("Sorted array:", int_list)

print("Validating result of cuda version of quicksort...")

np.testing.assert_array_equal(np.array(int_list), np.array(reference_sorted_list), err_msg="cuda version of quicksort did not produce a correctly sorted list.")
print("    Congratulations, expected and actual lists are equal!\n")