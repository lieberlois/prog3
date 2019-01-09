def __calculate(): # This will probably receive (positions, speed, mass, timestep)
    chunks = 50 # TODO: Find a good way to determine a reasonable chunk amount (Working Cores * 10 or similar?)
    job_queue, result_queue = m.get_job_queue(), m.get_result_queue()

    in_list = [round(amount/chunks) for _ in range(chunks)]   # in_list and result_list might be faster with numpy
    print(f"Amount of jobs: {len(in_list)}")
    print(f"Job Size: {in_list[0]}")
    result_list = []
    
    for arg in in_list:
        job_queue.put(arg)
    job_queue.join()
    while not result_queue.empty():
        result_list.append(result_queue.get()) 
    return result_list

if __name__ == '__main__':
    from sys import argv, exit
    if len(argv) != 4:
        print('usage:', argv[0], 'server_IP server_socket amount_of_shots')
        exit(0)
    server_ip = argv[1]
    server_socket = int(argv[2])
    TaskManager.register('get_job_queue')
    TaskManager.register('get_result_queue')
    m = TaskManager(address=(server_ip, server_socket), authkey = b'secret')
    m.connect()

    t1 = time.time()
    result = __calculate(m, int(argv[3]))
    t2 = time.time()
    print(' result: ', result)
    print(' time:   ', t2-t1, ' s\n')