import sys
import time
import numpy as np
import cupy as cp
import copy
from typing import Final

LENGTH_OF_INT_LIST: Final = 100_000_000
MAX_POSSIBLE_INT_VALUE = sys.maxsize
MIN_POSSIBLE_INT_VALUE = -sys.maxsize - 1

# Generate random integer list
int_list = np.random.randint(MIN_POSSIBLE_INT_VALUE, MAX_POSSIBLE_INT_VALUE + 1, size=LENGTH_OF_INT_LIST)

# Create a copy for verification purposes
int_list_copy = copy.deepcopy(int_list)

# Generating reference sorted list using Python's "sorted" function for validation
print("\nGenerating reference sorted list using Python's \"sorted\" built-in function for validating correctness of serial version and parallel version of quicksort... ", end="")
reference_sorted_list = sorted(int_list)
print("Done!")

# Perform sorting using CuPy on GPU
int_list_gpu = cp.asarray(int_list)  # Transfer data to GPU
start_time = time.time()
cp.cuda.Stream.null.synchronize()  # Ensure GPU calculations are done in sync

int_list_gpu = cp.sort(int_list_gpu)  # CuPy's sort function

cp.cuda.Stream.null.synchronize()  # Synchronize again to ensure sorting is complete
elapsed_time = time.time() - start_time
print("Sort completed")
# Transfer data back to CPU
int_list_sorted = cp.asnumpy(int_list_gpu)
print("Execution time:", elapsed_time, "seconds")
print("Sorted array:", int_list_sorted)

# Validate the result of the GPU version of quicksort
print("Validating result of CuPy version of quicksort...")
np.testing.assert_array_equal(int_list_sorted, reference_sorted_list, err_msg="CuPy version of quicksort did not produce a correctly sorted list.")
print("    Congratulations, expected and actual lists are equal!\n")
