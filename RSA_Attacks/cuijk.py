from powmodn import bit_pow_mod_n
from RSA import RSA
from scipy.stats import norm
from scipy import mean, std
from random import randrange
import time

def get_ti(M, n, d):
    t = time.time()

    if d:
        M *= M**2 % n

    return M, time.time()-t

bits = 16
scheme = RSA(powmodn=bit_pow_mod_n, sign=True)
p, q, n, l, e, d, public_key, private_key = scheme.generate_keys(bit_length=bits)
n_m = 16
max_m = 2**1024
conv = 100
messages = [randrange(max_m) for x in range(n_m)]
M = messages[0]
num = 2**(bits-1)
qqq = []

while num:
    d = 0

    if num & 1:
        d = 1

    M, Ti = get_ti(M, n, d)
    qqq.append(Ti*conv)

    num >>= 1
