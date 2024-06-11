import sys
import numpy
from typing import Final
import time
import copy
import multiprocessing
from sys_monitoring import start_monitoring, stop_monitoring 
from serial_quicksort import serial_quicksort
from parallel_quicksort import parallel_quicksort

# LENGTH_OF_INT_LIST: Final = 100_000
MAX_POSSIBLE_INT_VALUE = sys.maxsize
MIN_POSSIBLE_INT_VALUE = -sys.maxsize - 1


def main(array_size, no_of_executions, isSerial):
    
    LENGTH_OF_INT_LIST: Final = array_size

    start_sort_time = time.time()   
    start_monitoring(('Serial Quicksort' if isSerial else 'Parallel Quicksort')+' - for '+str(no_of_executions)+' executions',LENGTH_OF_INT_LIST)

    for executionStep in range(no_of_executions):
        #Criação dos vetores
        int_list = numpy.random.randint(MIN_POSSIBLE_INT_VALUE, MAX_POSSIBLE_INT_VALUE, size=(LENGTH_OF_INT_LIST,))
        int_list = int_list.tolist()
        int_list_copy = copy.deepcopy(int_list)

        #Ordenação do Vetor Controle
        reference_sorted_list = sorted(int_list)    
        print('Execution Step:',executionStep)

        if isSerial:            
           
            serial_quicksort(a_list=int_list,low=0,high=len(int_list)-1)            
            # print(f"...using serial version of quicksort: {end_sort_time - start_sort_time:.6f} seconds.\n")
            resp = numpy.testing.assert_array_equal(numpy.array(int_list), numpy.array(reference_sorted_list), err_msg="serial version of quicksort did not produce a correctly sorted list.")
            if resp is not None:
                print('SORT ERROR!!!')
            # print("    Congratulations, expected and actual lists are equal!\n")

        else:
            receive_sorted_list_socket, send_sorted_list_socket = multiprocessing.Pipe(duplex=False)
            quicksort_parent_process = multiprocessing.Process(target=parallel_quicksort, args=(int_list_copy,0,len(int_list_copy)-1, send_sorted_list_socket, 1, multiprocessing.cpu_count()))
            quicksort_parent_process.start()
            int_list_copy = receive_sorted_list_socket.recv()
            quicksort_parent_process.join()
            # print(f"...using parallel version of quicksort (parallelized over a target of {multiprocessing.cpu_count()} processes): {end_sort_time - start_sort_time:.6f} seconds.\n")
            resp = numpy.testing.assert_array_equal(numpy.array(int_list_copy), numpy.array(reference_sorted_list), err_msg="parallel version of quicksort did not produce a correctly sorted list.")
            if resp is not None:
                print('SORT ERROR!!!')
            # print("    Congratulations, expected and actual lists are equal!\n")

    end_sort_time = time.time()
    stop_monitoring(end_sort_time-start_sort_time)
if __name__ == "__main__":
    main()
