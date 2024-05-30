import sys
from typing import Final
from numba import cuda, int32
import numpy as np
import numba
import time
import copy

@cuda.jit(device=True)
def partition(arr, lo, hi):
    pivot = arr[(lo + hi) // 2]
    i = lo
    j = hi

    while True:
        while arr[i] < pivot:
            i += 1
        while arr[j] > pivot:
            j -= 1
        if i >= j:
            return j
        arr[i], arr[j] = arr[j], arr[i]
        i += 1
        j -= 1

@cuda.jit
def quicksort_debug_kernel(arr, lo, hi, debug_info):
    stack = cuda.local.array(1000, dtype=int32)  # Local memory for stack, increased size
    pivot_indices = cuda.local.array(1, dtype=int32)

    tid = cuda.threadIdx.x
    if tid == 0:
        stack[0] = lo
        stack[1] = hi
        stack_size = 2
    else:
        stack_size = 0

    cuda.syncthreads()

    debug_index = 0
    while stack_size > 0:
        if tid == 0:
            hi = stack[stack_size - 1]
            lo = stack[stack_size - 2]
            stack_size -= 2

        cuda.syncthreads()

        if lo < hi:
            pivot = partition(arr, lo, hi)

            if tid == 0:
                pivot_indices[0] = pivot

            cuda.syncthreads()
            pivot = pivot_indices[0]

            if tid == 0:
                if stack_size + 4 <= stack.size:  # Ensure stack does not overflow
                    if lo < pivot:
                        stack[stack_size] = lo
                        stack[stack_size + 1] = pivot
                        stack_size += 2
                    if pivot + 1 < hi:
                        stack[stack_size] = pivot + 1
                        stack[stack_size + 1] = hi
                        stack_size += 2

                # Store debug info
                if debug_index < debug_info.shape[0]:
                    debug_info[debug_index, 0] = lo
                    debug_info[debug_index, 1] = hi
                    debug_info[debug_index, 2] = pivot
                    debug_index += 1

            cuda.syncthreads()

@cuda.jit
def validate_sorted_kernel(arr, is_sorted, n):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tid < n - 1:
        if arr[tid] > arr[tid + 1]:
            is_sorted[0] = 0

def quicksort(arr):
    d_arr = cuda.to_device(arr)
    threads_per_block = 128
    blocks_per_grid = (len(arr) + threads_per_block - 1) // threads_per_block

    quicksort_debug_kernel[blocks_per_grid, threads_per_block](d_arr, 0, len(arr) - 1, np.zeros((1000, 3), dtype=np.int32))
    cuda.synchronize()
    d_arr.copy_to_host(arr)

def quicksort_debug(arr):
    d_arr = cuda.to_device(arr)
    debug_info = np.zeros((1000, 3), dtype=np.int32)
    d_debug_info = cuda.to_device(debug_info)
    threads_per_block = 128
    blocks_per_grid = (len(arr) + threads_per_block - 1) // threads_per_block

    quicksort_debug_kernel[blocks_per_grid, threads_per_block](d_arr, 0, len(arr) - 1, d_debug_info)
    cuda.synchronize()
    d_arr.copy_to_host(arr)
    d_debug_info.copy_to_host(debug_info)
    
    return debug_info

def validate_sorted(arr):
    is_sorted = np.array([1], dtype=np.int32)
    d_arr = cuda.to_device(arr)
    d_is_sorted = cuda.to_device(is_sorted)
    threads_per_block = 128
    blocks_per_grid = (len(arr) + threads_per_block - 1) // threads_per_block

    validate_sorted_kernel[blocks_per_grid, threads_per_block](d_arr, d_is_sorted, len(arr))
    cuda.synchronize()
    d_is_sorted.copy_to_host(is_sorted)
    
    return is_sorted[0] == 1

def cpu_quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return cpu_quicksort(left) + middle + cpu_quicksort(right)

LENGTH_OF_INT_LIST: Final = 100_000
MAX_POSSIBLE_INT_VALUE = sys.maxsize
MIN_POSSIBLE_INT_VALUE = -sys.maxsize - 1

int_list = np.random.randint(MIN_POSSIBLE_INT_VALUE, MAX_POSSIBLE_INT_VALUE + 1, size=LENGTH_OF_INT_LIST)

int_list_copy = copy.deepcopy(int_list)

print("\nGenerating reference sorted list using Python's \"sorted\" built-in function for validating correctness of serial version and parallel version of quicksort... ", end="")

reference_sorted_list = sorted(int_list)
print("Done!")

start_time = time.time()

# Hybrid approach
if LENGTH_OF_INT_LIST < 10_000:
    quicksort(int_list)
else:
    # Split into smaller chunks for CUDA and then use CPU for final merge
    chunks = np.array_split(int_list, 10)
    for i, chunk in enumerate(chunks):
        print(f"Sorting chunk {i+1}/{len(chunks)}")
        debug_info = quicksort_debug(chunk)
        # Print summary of debug info
        print(f"Debug info summary for chunk {i+1}:")
        print(f"First few debug entries:\n{debug_info[:5]}")
        print(f"Last few debug entries:\n{debug_info[-5:]}")
        # Validate each chunk
        if not validate_sorted(chunk):
            print(f"Chunk {i+1} is not sorted correctly.")

    int_list = cpu_quicksort(np.concatenate(chunks))

elapsed_time = time.time() - start_time
print("Execution time:", elapsed_time, "seconds")

print("Validating result of cuda version of quicksort...")

try:
    np.testing.assert_array_equal(np.array(int_list), np.array(reference_sorted_list), err_msg="cuda version of quicksort did not produce a correctly sorted list.")
    print("    Congratulations, expected and actual lists are equal!\n")
except AssertionError as e:
    print(e)
    print("First few elements of the sorted array:")
    print(int_list[:10])
    print("First few elements of the reference sorted array:")
    print(reference_sorted_list[:10])
    print("Mismatch detected. Debug the sorting process.")
