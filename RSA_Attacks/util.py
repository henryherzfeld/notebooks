from math import gcd
from Crypto.Util import number
from random import randrange

def lcm(a, b):
    return a * b // gcd(a, b)


def string2int(s):
    # return int.from_bytes(s.encode("utf-8"), byteorder = "big")
    return number.bytes_to_long(s.encode("utf-8"))


def int2string(n):
    # return (n.to_bytes(((n.bit_length() + 7) //8), byteorder = "big")).decode("utf-8")
    return number.long_to_bytes(n)


# For the random fault attack described in section 5.2, the attacker needs
# to corrupt the contents of exactly one of w or x.
# This function flips exactly one bit at a position chosen at random:
def flip_random_bit(x):
    return x ^ (1 << randrange(x.bit_length()))

# Multiplicative inverse of a in Z_n* given by Wikipedia article
# def inverse(a, n):
#    (t, new_t, r, new_r) = (0, 1, n, a)
#    while new_r != 0:
#        quotient = r // new_r
#        (t, new_t) = (new_t, t - quotient * new_t)
#        (r, new_r) = (new_r, r - quotient * new_r)
#    if r > 1:
#        return -1 # a is not invertible in Z/nZ
#    else:
#	     return t % n
