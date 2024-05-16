import sys
from typing import Final
from numba import cuda
import numpy as np
import numba
import time
import copy

@cuda.jit
def quicksort_kernel(arr):
    lo = 0
    hi = len(arr) - 1
    stack = cuda.local.array(shape=128, dtype=numba.int32)  # Adjust size as needed

    # Initialize stack
    top = 0
    stack[top] = lo
    top += 1
    stack[top] = hi
    top += 1

    # Main loop to pop and push items until stack is empty
    while top > 0:
        top -= 1
        hi = stack[top]
        top -= 1
        lo = stack[top]

        pivot = arr[(hi + lo) // 2]
        i = lo
        j = hi
        while i <= j:
            while arr[i] < pivot:
                i += 1
            while arr[j] > pivot:
                j -= 1
            if i <= j:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
                j -= 1

        # Push the indices of elements to be sorted into the stack
        if lo < j:
            top += 1
            stack[top] = lo
            top += 1
            stack[top] = j
        if i < hi:
            top += 1
            stack[top] = i
            top += 1
            stack[top] = hi

def quicksort(arr):
    d_arr = cuda.to_device(arr)
    # Adjust the number of blocks and threads as needed
    threads_per_block = 256
    blocks = (len(arr) + threads_per_block - 1) // threads_per_block
    quicksort_kernel[blocks, threads_per_block](d_arr)
    d_arr.copy_to_host(arr)

LENGTH_OF_INT_LIST: Final = 100_000
MAX_POSSIBLE_INT_VALUE = sys.maxsize
MIN_POSSIBLE_INT_VALUE = -sys.maxsize - 1

int_list = np.random.randint(MIN_POSSIBLE_INT_VALUE, MAX_POSSIBLE_INT_VALUE + 1, size=LENGTH_OF_INT_LIST)  # Random array of size 10 with integer values

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

# import sys
# from typing import Final
# from numba import cuda
# import numpy as np
# import numba
# import time
# import copy

# @cuda.jit
# def quicksort_kernel(arr):
#     lo = 0
#     hi = len(arr) - 1
#     stack = cuda.local.array(shape=64, dtype=numba.int32)  # Adjust size as needed

#     # Initialize stack
#     top = 0
#     stack[top] = lo
#     top += 1
#     stack[top] = hi

#     # Main loop to pop and push items until stack is empty
#     while top > 0:
#         hi = stack[top]
#         top -= 1
#         lo = stack[top]
#         top -= 1

#         pivot = arr[(hi + lo) // 2]
#         i = lo
#         j = hi
#         while i <= j:
#             while arr[i] < pivot:
#                 i += 1
#             while arr[j] > pivot:
#                 j -= 1
#             if i <= j:
#                 arr[i], arr[j] = arr[j], arr[i]
#                 i += 1
#                 j -= 1

#         # Push the indices of elements to be sorted into the stack
#         if lo < j:
#             top += 1
#             stack[top] = lo
#             top += 1
#             stack[top] = j
#         if i < hi:
#             top += 1
#             stack[top] = i
#             top += 1
#             stack[top] = hi

# def quicksort(arr):
#     d_arr = cuda.to_device(arr)
#     # Adjust the number of blocks and threads as needed
#     blocks = 1
#     threads_per_block = 256
#     quicksort_kernel[blocks, threads_per_block](d_arr)
#     d_arr.copy_to_host(arr)


# LENGTH_OF_INT_LIST: Final = 10_000_000
# MAX_POSSIBLE_INT_VALUE = sys.maxsize
# MIN_POSSIBLE_INT_VALUE = -sys.maxsize - 1

# int_list = np.random.randint(MIN_POSSIBLE_INT_VALUE, MAX_POSSIBLE_INT_VALUE+1, size=(LENGTH_OF_INT_LIST,))  # Random array of size 10 with integer values

# int_list = int_list.tolist()

# int_list_copy = copy.deepcopy(int_list)

# print("\nGenerating reference sorted list using Python's \"sorted\" built-in function for validating correctness of serial version and parallel version of quicksort... ", end="")

# reference_sorted_list = sorted(int_list)
# print("Done!")


# start_time = time.time()  

# quicksort(int_list)
# elapsed_time = time.time() - start_time  
# print("Execution time:", elapsed_time, "seconds")
# print("Sorted array:", int_list)

