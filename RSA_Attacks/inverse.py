from math import gcd

# My rewrite
def rec_inverse_helper(r, new_r, n, t=0, new_t=1):
    if new_r <= 0:
        return t % n
    else:
        return rec_inverse_helper(new_r, r % new_r, n, new_t, t - (r // new_r) * new_t)

# Multiplicative inverse of a in Z_n*
def rec_inverse(a, n):
    if gcd(a, n) == 1:
        return rec_inverse_helper(n, a, n)
    else:
        return "a is not invertible in Z/nZ"
