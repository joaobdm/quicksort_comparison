def serial_quicksort(a_list, low, high):
    if low < high:
        partition_index = partition(a_list, low, high)
        serial_quicksort(a_list, low, partition_index - 1)
        serial_quicksort(a_list, partition_index + 1, high)

def partition(a_list, low, high):
    pivot = a_list[high]
    i = low - 1
    for j in range(low, high):
        if a_list[j] <= pivot:
            i = i + 1
            a_list[i], a_list[j] = a_list[j], a_list[i]
    a_list[i + 1], a_list[high] = a_list[high], a_list[i + 1]
    return i + 1

# Exemplo de uso
if __name__ == '__main__':
    a_list = [13, 19, 9, 5, 12, 8, 7, 4, 21, 2, 6, 11]
    serial_quicksort(a_list, 0, len(a_list) - 1)
    print(a_list)
