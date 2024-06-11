from parallel_vs_serial_sort import main

NUM_OF_TEST = 5
SAMPLE_ARRAY_SIZES = [100_000,1_000_000,10_000_000]
executionTypes = [True,False]

for executionType in executionTypes:
    for array_size in SAMPLE_ARRAY_SIZES:
        print(f'Array Size: {array_size}')
        main(array_size, NUM_OF_TEST, executionType)