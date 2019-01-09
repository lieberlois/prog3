from multiprocessing.managers import BaseManager
from multiprocessing import JoinableQueue, Queue

class TaskManager(BaseManager):
    pass

if __name__ == '__main__':
    from sys import argv, exit
    if len(argv) != 2:
        print('usage:', argv[0], b'socket_nr')
        exit(0)
    master_socket = int(argv[1])
    task_queue = JoinableQueue()
    result_queue = Queue()
    TaskManager.register('get_job_queue', 
                         callable = lambda:task_queue)
    TaskManager.register('get_result_queue', 
                         callable = lambda:result_queue)
    m = TaskManager(address = ('', master_socket), 
                    authkey = b'secret')
    print('starting queue server, socket', master_socket)
    m.get_server().serve_forever()