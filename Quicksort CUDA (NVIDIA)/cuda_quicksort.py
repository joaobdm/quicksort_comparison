from numba import cuda
import numpy as np
import numba
import time

@cuda.jit
def quicksort_kernel(arr):
    lo = 0
    hi = len(arr) - 1
    stack = cuda.local.array(shape=64, dtype=numba.int32)  # Adjust size as needed

    # Initialize stack
    top = 0
    stack[top] = lo
    top += 1
    stack[top] = hi

    # Main loop to pop and push items until stack is empty
    while top > 0:
        hi = stack[top]
        top -= 1
        lo = stack[top]
        top -= 1

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
    blocks = 1
    threads_per_block = 256
    quicksort_kernel[blocks, threads_per_block](d_arr)
    d_arr.copy_to_host(arr)


ARRAY_MAX_INT_VALUE = 1_000_000
ARRAY_SIZE = 1_000_000

arr = np.random.randint(0, ARRAY_MAX_INT_VALUE+1, size=ARRAY_SIZE, dtype=np.int32)  # Random array of size 10 with integer values
print("Original array:", arr)
start_time = time.time()  

quicksort(arr)
elapsed_time = time.time() - start_time  
print("Execution time:", elapsed_time, "seconds")
print("Sorted array:", arr)

