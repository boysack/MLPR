from time import time

with open("ex02_data") as f:
    lines = f.readlines()
lines = [line.strip().split() for line in lines]
busId = "2187"

start = time()
lines = [line for line in lines if line[0] == busId]
#print(lines)
el01 = time()-start
print(f"elapsed time - {el01}")

start = time()
lines = list(filter(lambda e: e[0] == busId, lines))
#print(lines)
el02 = time()-start
print(f"elapsed time - {el02}")

if el01 > el02:
    print("filter is faster")
else:
    print("comprehension is faster")