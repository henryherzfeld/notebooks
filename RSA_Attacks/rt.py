from time import time
# --------------------------------------------------
# Functions for computing running time
#
# Function that measures running time of function f with input n
#
def rt(f, n):
    starttime = time()
    return ([f(n), time() - starttime])


# Computes average run time of f(n) over m trials.
def rt_average(f, n, m):
    rt_total = 0
    for i in range(m):
        rt_total += rt(f, n)[1]
    return rt_total / m


# Measure running time of function f with inputs a and b
def rt2(f, a, b):
    starttime = time()
    return ([f(a, b), time() - starttime])


# Measure running time of function f with inputs a, b, and c
def rt3(f, a, b, c):
    starttime = time()
    return ([f(a, b, c), time() - starttime])


# Measure running time of function f with inputs a, b, c, and d
def rt4(f, a, b, c, d):
    starttime = time()
    return ([f(a, b, c, d), time() - starttime])

# Computes average run time of f(n) over m trials.
def rt_average_2(f, a, b, m):
    rt_total = 0
    for i in range(m):
        rt_total += rt2(f, a, b)[1]
    return rt_total / m

def rt_average_4(f, a, b, c, d, m):
    rt_total = 0
    for i in range(m):
        rt_total += rt4(f, a, b, c, d)[1]
    return rt_total / m
