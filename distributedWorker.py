from multiprocessing import cpu_count, Process
from distributedManager import TaskManager
import simulation_physic as sp


def __worker_function(job_queue, result_queue):
    while True:
        task = job_queue.get()
        result = sp._mp_move_bodies_circle(*task)
        # In the tuple task should be positions,speed,mass,timestep
        # and probably the index range
        result_queue.put(result)
        job_queue.task_done()


def __start_workers(m):
    job_queue, result_queue = m.get_job_queue(), m.get_result_queue()
    nr_of_processes = cpu_count()
    processes = [Process(target=__worker_function,
                         args=(job_queue, result_queue))
                 for i in range(nr_of_processes)]
    for p in processes:
        p.start()
    return nr_of_processes


if __name__ == '__main__':
    from sys import argv, exit
    if len(argv) < 3:
        print('usage:', argv[0], 'server_IP server_socket')
        exit(0)
    server_ip = argv[1]
    server_socket = int(argv[2])
    TaskManager.register('get_job_queue')
    TaskManager.register('get_result_queue')
    m = TaskManager(address=(server_ip, server_socket), authkey=b'secret')
    m.connect()
    nr_of_processes = __start_workers(m)
    print(nr_of_processes, 'workers started')
