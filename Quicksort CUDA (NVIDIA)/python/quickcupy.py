import cupy as cp
import sys
from typing import Final
import time

def quicksort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = arr[arr < pivot]
    middle = arr[arr == pivot]
    right = arr[arr > pivot]

    return cp.concatenate([quicksort(left), middle, quicksort(right)])

# Example usage
LENGTH_OF_INT_LIST: Final = 1_000_000
MAX_POSSIBLE_INT_VALUE = sys.maxsize
MIN_POSSIBLE_INT_VALUE = -sys.maxsize - 1

int_list = cp.random.randint(MIN_POSSIBLE_INT_VALUE, MAX_POSSIBLE_INT_VALUE, size=(LENGTH_OF_INT_LIST,))
int_list_copy = int_list.copy()
reference_sorted_list = cp.sort(int_list_copy)

print(f"\nTime to sort list of {LENGTH_OF_INT_LIST} integers...\n")
start_sort_time = time.time()
sorted_list = quicksort(int_list)  # Capture the returned sorted array
end_sort_time = time.time()
print(f"Quicksort completed in {end_sort_time - start_sort_time:.2f} seconds.")

cp.testing.assert_array_equal(sorted_list, reference_sorted_list, err_msg="Quicksort did not produce a correctly sorted list.")
