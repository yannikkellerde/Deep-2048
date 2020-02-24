import time
start = time.perf_counter()
for a in range(100000):
    time.time()
print(time.perf_counter() - start)