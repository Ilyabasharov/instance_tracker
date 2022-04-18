import time
import torch


def get_sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class Timer():
    def __init__(self, verbose: bool=False):
        self.sum_time = 0
        self.count = 0
        self.verbose = verbose
        
    def timeit(self, method):
        def timed(*args, **kw):
           ts = get_sync_time()
           result = method(*args, **kw)
           tex = get_sync_time() - ts
           self.count += 1
           self.sum_time += tex
           if self.verbose:
               self.display()
           return result    
        return timed
        
    def display(self):
        print(self.sum_time / self.count, 's, ', self.count / self.sum_time, 'fps')
    


