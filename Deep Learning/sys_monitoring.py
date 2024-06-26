import psutil
import GPUtil
import time
import logging
import threading
import numpy as np

# Configuração do log
logging.basicConfig(filename='./logs/10_000_pgn_mpnn_combined.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Variável global para controlar a execução do loop
keep_running = False
monitor_thread = None
cpu_usage_list = []
memory_usage_list = []
gpu_usage_list = []

def log_system_usage(sort_method, array_size, num_of_tests):
    global keep_running
    while keep_running:
        # Uso de CPU
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Uso de memória
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        
        # Uso de GPU
        gpus = GPUtil.getGPUs()
        gpu_usage = gpus[0].load * 100 if gpus else 0  # Considerando apenas a primeira GPU
        
        # Registrando os dados no log
        logging.info(f'Sort Method: {sort_method}, Array Size: {array_size}, Number of Tests: {num_of_tests}, CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%, GPU Usage: {gpu_usage}%')
        
        # Adicionando Logs em listas
        cpu_usage_list.append(cpu_usage)
        memory_usage_list.append(memory_usage)
        gpu_usage_list.append(gpu_usage)

        # Verifica se a execução deve continuar
        if not keep_running:
            break

        # Esperar 1 segundo antes de fazer a próxima medição
        time.sleep(1)

def start_monitoring(sort_method, array_size, num_of_tests):
    logging.info(f'-----Starting Log-----')
    global keep_running, monitor_thread
    keep_running = True
    monitor_thread = threading.Thread(target=log_system_usage, args=(sort_method, array_size, num_of_tests))
    monitor_thread.start()
    print("Monitoring started.")

def stop_monitoring(elapsed_time):
    global keep_running, monitor_thread
    keep_running = False
    if monitor_thread is not None:
        monitor_thread.join()
    
    print("Monitoring stopped.")
    logging.info(f'Média uso de CPU: {np.mean(cpu_usage_list)}%, Média uso de Memória: {np.mean(memory_usage_list)}%, Média uso de GPU: {np.mean(gpu_usage_list)}%')
    logging.info(f'-----Finishing Log, Elapsed Time: {elapsed_time:.6f} seconds-----')

# Exemplo de uso
if __name__ == '__main__':
    # Iniciar o monitoramento
    start_monitoring('Serial Quicksort', 100, 10, 'example')
    
    # Simular tempo de execução (substitua isso pela lógica do seu programa)
    time.sleep(4)
    
    # Parar o monitoramento
    stop_monitoring(4.0)
