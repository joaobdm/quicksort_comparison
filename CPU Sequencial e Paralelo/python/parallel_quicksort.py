import multiprocessing
from serial_quicksort import serial_quicksort, partition

def parallel_quicksort(a_list, sending_socket, current_processes_count, MAX_PROCESSES_COUNT):
    assert a_list is not None
    assert sending_socket is not None
    assert current_processes_count is not None
    assert MAX_PROCESSES_COUNT is not None

    if len(a_list) > 0:
        if current_processes_count >= MAX_PROCESSES_COUNT:
            serial_quicksort(a_list)
            sending_socket.send(a_list)
            sending_socket.close()
        else:
            partitioning_element = a_list.pop(len(a_list) // 2)
            no_larger_than_list, larger_than_list = [], []
            partition(a_list, no_larger_than_list, larger_than_list, partitioning_element)
            recv_socket_nl, send_socket_nl = multiprocessing.Pipe(duplex=False)
            recv_socket_lg, send_socket_lg = multiprocessing.Pipe(duplex=False)

            processes = [
                multiprocessing.Process(target=parallel_quicksort, args=(no_larger_than_list, send_socket_nl, current_processes_count*2, MAX_PROCESSES_COUNT)),
                multiprocessing.Process(target=parallel_quicksort, args=(larger_than_list, send_socket_lg, current_processes_count*2, MAX_PROCESSES_COUNT))
            ]
            
            for process in processes:
                process.start()

            no_larger_than_list = recv_socket_nl.recv()
            larger_than_list = recv_socket_lg.recv()

            a_list.extend(no_larger_than_list + [partitioning_element] + larger_than_list)
            sending_socket.send(a_list)

            for process in processes:
                process.join()
                process.close()

            sending_socket.close()
    else:
        sending_socket.send(a_list)
        sending_socket.close()

# This setup ensures that processes are managed in a loop, reducing duplication and enhancing clarity.
