a = list(range(10))+list(range(10))
"""for i in range(len(a)):
    if a[i]%3==0:
        del a[i]"""

i=0
while i<len(a):
    if a[i]%3==0 or a[i]%4==0:
        del a[i]
        i-=1
    i+=1

print(a)