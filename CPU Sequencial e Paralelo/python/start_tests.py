from parallel_vs_serial_sort import main

NUM_OF_TEST = 100_000
SAMPLE_ARRAY_SIZES = [16,32,64]
executionTypes = [True]

for executionType in executionTypes:
    for array_size in SAMPLE_ARRAY_SIZES:
        print(f'Array Size: {array_size}')
        main(array_size, NUM_OF_TEST, executionType)