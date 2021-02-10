from powmodn import bit_pow_mod_n
from rt import rt2, rt3, rt4
from RSA import RSA
from util import string2int
from math import log
from scipy.stats import pearsonr
import time
import matplotlib.pyplot as plt
from random import randrange

def get_ti(M, n, d):
    t = time.time()

    if d:
        M *= M**2 % n

    return M, time.time()-t

# create ti using physical specs
max_bitlength = 16
bits = 16
e = 65537
scheme = RSA(powmodn=bit_pow_mod_n, sign=True)
tis = []
p, q, n, l, e, d, public_key, private_key = scheme.generate_keys(bit_length=bits)


n_m = 16
max_m = 2**1024
conv = 100

messages = [randrange(max_m) for x in range(n_m)]


tis = []
for m in messages:
    _, t = rt4(scheme.rsa_sign, m, public_key, p, q)
    tis.append(t*conv)


nums = (2**(bits-1), (2**bits)-1)
bits = 16
M = messages[0]


Tis = []

for num in nums:

    qqq = []

    while num:
        d = 0

        if num & 1:
            d = 1

        M, Ti = get_ti(M, n, d)
        qqq.append(Ti*conv)
        print(Ti*conv)

        num >>= 1

    Tis.append(qqq)



print(pearsonr(Tis[1], tis))

ax1 = plt.subplot(2,1,1)
# ax1.set(xlim=(0, .2))
plt.hist(Tis[0])
plt.hist(Tis[1])
plt.title('theoretical')
plt.subplot(2,1,2, sharex=ax1)
plt.hist(tis)
plt.title('actual')
plt.show()
