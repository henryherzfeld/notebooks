from util import clear_bit_n, set_bit_n, get_last_n_bits, clear_last_n_bits, remove_outliers
from rt import rt3x_average
from gmpy2 import powmod
import numpy as np
from RSA import RSA
from random import randrange
from scipy import mean, std
import matplotlib.pyplot as plt

BL = 128 # The key length in bits. This is the size of n = pq.
NN = 1024 # The number of messages to be signed/decrypted
averages = 100 # number of measurements to be averaged per message

# Generate keys. Keep doing it until we get one where bit b1 of d is 1.
d = 0

scheme = RSA(gmp=True)

while not (d & 2):
    (p, q, n, l, e, d, pk, sk) = scheme.generate_keys(bit_length = BL//2)

M = [randrange(2**BL) for i in range(NN)]

# The guess g. Initial guess is 1, corresponding to b_0 = 1 and all other bits 0.
g = 1


# Get the running times t_i for m_i^x for the messages m_i in M
def measure_times(M, x, n, number_of_runs_per_message = 1):
    return np.array([rt3x_average(powmod, m, x, n, number_of_runs_per_message) for m in M])


TT = measure_times(M, d, n, averages) # These are the T_i's
tt0 = measure_times(M, g, n, averages) # These are the t_i's when g = 0b1

g = set_bit_n(g, 1) # set g to 0b11
tt1 = measure_times(M, g, n, averages) # these are the t_i's when g is 0b11

D0 = remove_outliers(TT - tt0)
D1 = remove_outliers(TT - tt1)


# Compute the standard deviations for the time differences
(sd0, sd1) = map(std, (D0, D1))


print("\nStandard deviation of time differences: ", sd0, sd1)

print("\nIn binary, d = ", bin(d))


num_bins = 32

nn, bins, patches = plt.hist(D0, num_bins, facecolor='blue', alpha=0.5, label='D0')
plt.show()

nn, bins, patches = plt.hist(D1, num_bins, facecolor='blue', alpha=0.5, label='D1')
plt.show()

msg = M[0]
d64 = clear_last_n_bits(d, 64)
b64 = get_last_n_bits(d, 64)
