def serial_quicksort(a_list):
    if len(a_list) > 1:
        partitioning_element = a_list.pop(len(a_list) // 2)
        no_larger_than_list = [x for x in a_list if x <= partitioning_element]
        larger_than_list = [x for x in a_list if x > partitioning_element]

        serial_quicksort(no_larger_than_list)
        serial_quicksort(larger_than_list)

        a_list[:] = no_larger_than_list + [partitioning_element] + larger_than_list

def partition(a_list, no_larger_than_list, larger_than_list, partitioner):
    no_larger_than_list.extend(x for x in a_list if x <= partitioner)
    larger_than_list.extend(x for x in a_list if x > partitioner)
    a_list.clear()  # This clears the original list as intended

# Usage example kept simple, you can uncomment it to test directly in this script:
# if __name__ == "__main__":
#     test_list = [3, 6, 8, 10, 1, 2, 1]
#     serial_quicksort(test_list)
#     print(test_list)
