from parallel_vs_serial_sort import main

NUM_OF_TEST = 5
SAMPLE_ARRAY_SIZES = [100_000,1_000_000,10_000_000]

for array_size in SAMPLE_ARRAY_SIZES:
    print(f'Array Size: {array_size}')
    for x in range(NUM_OF_TEST):
        print(f'Test Number #: {x+1}')
        main(array_size)