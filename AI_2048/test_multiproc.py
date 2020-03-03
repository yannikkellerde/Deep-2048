from multiprocessing import Queue,Process
import time

class stupid_class():
    def __init__(self,i):
        self.i = i
    def __str__(self):
        return f"stupid {self.i}"

def putter(q:Queue):
    stupids = []
    for i in range(20):
        stupid = stupid_class(i)
        q.put(stupid)
        time.sleep(0.1)
        stupids.append(stupid)
    q.put("Done")
    for stupid in stupids:
        print(stupid)
def getter(q:Queue):
    while 1:
        try:
            f = q.get(block=False)
        except Exception as e:
            print("Empty")
            continue
        print(f)
        if f == "Done":
            break
        else:
            f.i = -1
q = Queue()
put_proc = Process(
    target=putter, args=[q])
get_proc = Process(
    target=getter, args=[q])
put_proc.start()
get_proc.start()
put_proc.join()