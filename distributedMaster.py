from distributedManager import TaskManager
import time
import numpy as np


def __calculate(positions, speed, mass, timestep): # This will probably receive (positions, speed, mass, timestep)
    job_queue, result_queue = m.get_job_queue(), m.get_result_queue()

    in_list = __create_argument_list(positions, speed, mass, timestep)
    result_list = []

    for arg in in_list:
        job_queue.put(arg)
    job_queue.join()
    while not result_queue.empty():
        result_list.append(result_queue.get())
    return result_list


def __create_argument_list(positions, speed, mass, timestep, l):
    """
    Create a tuple of positions, speed, mass, timestep and indexrange
    """
    # converting mem views to np array
    step = 10 # number of positions every worker has to work with
    count = 1
    pos = np.asarray(positions)
    spe = np.zeros((step, 3), dtype=np.float64)
    mas = np.asarray(mass)

    for i in range(1, mass.shape[0], step):
        index = 0
        indexrange = []
        for j in range(i, step*count+1):
            if j > mass.shape[0]:
                break
            spe[index][0] = speed[j][0]
            spe[index][1] = speed[j][1]
            spe[index][2] = speed[j][2]
            indexrange.append(j)
            index += 1
        count += 1
        l.append((pos, spe, mas, timestep, indexrange))
    return l


if __name__ == '__main__':
    from sys import argv, exit
    if len(argv) != 4:
        print('usage:', argv[0], 'server_IP server_socket amount_of_shots')
        exit(0)
    server_ip = argv[1]
    server_socket = int(argv[2])
    TaskManager.register('get_job_queue')
    TaskManager.register('get_result_queue')
    m = TaskManager(address=(server_ip, server_socket), authkey=b'secret')
    m.connect()

    t1 = time.time()
    result = __calculate(m, int(argv[3]))
    t2 = time.time()
    print(' result: ', result)
    print(' time:   ', t2-t1, ' s\n')
