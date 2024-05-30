import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

mod = SourceModule("""
__device__ int partition(int *data, int left, int right) {
    int pivot = data[right];
    int i = (left - 1);
    for (int j = left; j <= right - 1; j++) {
        if (data[j] < pivot) {
            i++;
            int temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
    int temp = data[i + 1];
    data[i + 1] = data[right];
    data[right] = temp;
    return (i + 1);
}

__global__ void quick_sort_kernel(int *data, int left, int right) {
    if (left < right) {
        int pi = partition(data, left, right);
        quick_sort_kernel<<<1, 1>>>(data, left, pi - 1);
        quick_sort_kernel<<<1, 1>>>(data, pi + 1, right);
    }
}
""",
options=['--ptxas-options=-v', '-arch=sm_86', '-rdc=true'],  # Correct architecture flag
no_extern_c=True)

quick_sort_kernel = mod.get_function("quick_sort_kernel")

def sort_array(data):
    data = np.array(data, dtype=np.int32)
    n = data.size
    quick_sort_kernel(drv.InOut(data), np.int32(0), np.int32(n - 1), block=(1, 1, 1), grid=(1, 1))
    drv.Context.synchronize()  # Ensure all CUDA operations have completed
    return data

# Example usage:
data = np.random.randint(0, 100, size=10)
sorted_data = sort_array(data)
print("Sorted data:", sorted_data)
