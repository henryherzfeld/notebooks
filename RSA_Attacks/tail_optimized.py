from tailcaller import TailCall, TailCaller
from math import gcd

# uses tailcaller class and function to perform tail-recursion
@TailCaller # tailcaller class decorator
def tail_rec_pow_mod_n_helper(b, m, n, acc):
    if not m:
        return acc

    if not (m % 2):
        m = m // 2
        acc = acc**2
    else:
        m -= 1
        acc *= b

    return TailCall(tail_rec_pow_mod_n_helper, b, m, n, acc % n)


def tail_rec_pow_mod_n(b, m, n):
  acc = b
  return tail_rec_pow_mod_n_helper(b, m, n, acc)


# My rewrite
@TailCaller
def tail_rec_inverse_helper(r, new_r, n, t=0, new_t=1):
    if new_r <= 0:
        return t % n
    else:
        return TailCall(tail_rec_inverse_helper, new_r, r % new_r, n, new_t, t - (r // new_r) * new_t)

# Multiplicative inverse of a in Z_n*
def tail_rec_inverse(a, n):
    if gcd(a, n) != 1:
        return "a is not invertible in Z/nZ"
    return tail_rec_inverse_helper(n, a, n)
