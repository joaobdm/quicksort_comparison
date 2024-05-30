import multiprocessing
from serial_quicksort import serial_quicksort, partition

def parallel_quicksort(a_list, low, high, sending_socket, current_processes_count, MAX_PROCESSES_COUNT):
    assert a_list is not None
    assert sending_socket is not None
    assert current_processes_count is not None
    assert MAX_PROCESSES_COUNT is not None

    if low < high:
        if current_processes_count >= MAX_PROCESSES_COUNT:
            serial_quicksort(a_list, low, high)
            sending_socket.send(a_list[low:high + 1])
            sending_socket.close()
        else:
            partition_index = partition(a_list, low, high)
            recv_socket_low, send_socket_low = multiprocessing.Pipe(duplex=False)
            recv_socket_high, send_socket_high = multiprocessing.Pipe(duplex=False)

            processes = [
                multiprocessing.Process(target=parallel_quicksort, args=(a_list, low, partition_index - 1, send_socket_low, current_processes_count * 2, MAX_PROCESSES_COUNT)),
                multiprocessing.Process(target=parallel_quicksort, args=(a_list, partition_index + 1, high, send_socket_high, current_processes_count * 2, MAX_PROCESSES_COUNT))
            ]

            for process in processes:
                process.start()

            lower_part = recv_socket_low.recv()
            higher_part = recv_socket_high.recv()

            # Copia os subarrays organizados de volta pro array original
            a_list[low:low + len(lower_part)] = lower_part
            a_list[partition_index + 1:partition_index + 1 + len(higher_part)] = higher_part

            sending_socket.send(a_list[low:high + 1])

            for process in processes:
                process.join()
                process.close()

            sending_socket.close()
    else:
        sending_socket.send(a_list[low:high + 1])
        sending_socket.close()

# Exemplo de uso
if __name__ == '__main__':
    a_list = [13, 19, 9, 5, 12, 8, 7, 4, 21, 2, 6, 11]
    parent_conn, child_conn = multiprocessing.Pipe()
    process = multiprocessing.Process(target=parallel_quicksort, args=(a_list, 0, len(a_list) - 1, child_conn, 1, 4))
    process.start()
    sorted_list = parent_conn.recv()
    process.join()
    print(sorted_list)
