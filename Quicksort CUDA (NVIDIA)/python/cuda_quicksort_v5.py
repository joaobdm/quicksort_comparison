from numba import cuda
import numpy as np
import sys
from typing import Final
import copy
import time


@cuda.jit
def quicksort_kernel(data, left, right, stack):
    top = -1
    top += 1
    stack[top] = left
    top += 1
    stack[top] = right

    while top >= 0:
        right = stack[top]
        top -= 1
        left = stack[top]
        top -= 1

        pivot = data[(left + right) // 2]
        i = left
        j = right
        while i <= j:
            while data[i] < pivot:
                i += 1
            while data[j] > pivot:
                j -= 1
            if i <= j:
                data[i], data[j] = data[j], data[i]
                i += 1
                j -= 1

        if left < j:
            top += 1
            stack[top] = left
            top += 1
            stack[top] = j
        if i < right:
            top += 1
            stack[top] = i
            top += 1
            stack[top] = right

def quicksort_gpu(data):
    n = data.shape[0]
    d_data = cuda.to_device(data)
    d_stack = cuda.to_device(np.empty(shape=(1024,), dtype=np.int32))  # Adjust size as necessary
    threads_per_block = 256  # Example value, adjust as needed
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block
    quicksort_kernel[blocks_per_grid, threads_per_block](d_data, 0, n - 1, d_stack)
    return d_data.copy_to_host()

# Example usage

LENGTH_OF_INT_LIST: Final = 100_000
MAX_POSSIBLE_INT_VALUE = sys.maxsize
MIN_POSSIBLE_INT_VALUE = -sys.maxsize - 1

int_list = np.random.randint(MIN_POSSIBLE_INT_VALUE, MAX_POSSIBLE_INT_VALUE, size=(LENGTH_OF_INT_LIST,))

int_list = int_list.tolist()

int_list = np.array(int_list)

int_list_copy = copy.deepcopy(int_list)

print("\nGenerating reference sorted list using Python's \"sorted\" built-in function for validating correctness of serial version and parallel version of quicksort... ", end="")

reference_sorted_list = sorted(int_list)
print("Done!")
print(f"\nTime to sort list of {LENGTH_OF_INT_LIST} integers...\n")

start_sort_time = time.time()


# data = np.random.rand(1024).astype(np.float32)  # Adjust the size as necessary
sorted_data = quicksort_gpu(int_list)
print("Sorted data:", sorted_data)
end_sort_time = time.time()


print("Validating result of serial version of quicksort...")
np.testing.assert_array_equal(np.array(int_list), np.array(reference_sorted_list), err_msg="serial version of quicksort did not produce a correctly sorted list.")
print("    Congratulations, expected and actual lists are equal!\n")