import numpy as np
import sys

if len(sys.argv) != 2:
    print("Invalid arguments. Run as python <prog> <n>.")
    exit(2)

def clz(value):
    nZeros = 0
    truncatedValue = value & 0xffffffff
    i = 31
    while i>0:
        if (truncatedValue & (1<<i) == 0):
            nZeros += 1
        else:
            return nZeros
        i-=1
    return nZeros


def h(y):
    return 2


def map3SimplexCoord(x, y, z):
    #if (x>y):
    #    return -1
    #if (y>z):
    #    return -1

    b = 2**(np.floor(np.log2(y+1)))
    q = np.floor(x/b)

    print(f"b:{b} q:{q} - ", end="")

    # first case 
    if (z<n/2):
        if (x<y and y<z):
            return np.array([x,y,z]) + [0, n/2, 0]
        else:
            return np.array([b*(1+q)-x, 2*b*(1+q) - y, 2*b - z + n/2])
    else:
        if (x<y and y<(z-n/2)):
            return np.array([b*(q)+x, y + 2*b*(q), z - n/2])
        else:
            return np.array([b*(1+q)-x, 2*b*(1+q) - y, 2*b - z + n/2])
    return -1


n = int(sys.argv[1])

if ((n & (n - 1)) != 0):
    print("n is not a power of 2")
    exit(2)

grid = [int(n/2), int(n/2), int(np.ceil(3*(n-1)/4))]

print(f"Grid size = ({grid[0]}, {grid[1]}, {grid[2]})")

for z in range(grid[2]):
    for y in range(grid[1]):
        for x in range(grid[0]):
            print(f"[{x}, {y}, {z}] -> " + str(map3SimplexCoord(x, y, z)))
