from montgomery import Montgomery

# Naïve recursive implementation of the square-and-multiply fast exponentiation.
# returns bˆm (mod n)
def rec_pow_mod_n(b, m, n):
    if not m:
        return 1
    elif not (m % 2):
        return rec_pow_mod_n(b, m // 2, n)**2 % n
    else:
        return b * rec_pow_mod_n(b, m - 1, n) % n


# bit-twiddling way to compute b^m (mod n). No recursion needed.
# If you can optimize this further, do so!
def bit_pow_mod_n(b, m, n):
    acc = 1  # the accumulated product
    z = b % n
    while m:
        if m & 1:  # check the least significant bit
            acc = (acc * z) % n
        z = (z * z) % n
        m >>= 1  # the bit shift
    return acc


# Uses montgomery reduction to perform bitwise powmodn without needing modulus divide
# NOT OUR CODE: https://rosettacode.org/wiki/Montgomery_reduction
def mon_pow_mod_n(b, m, n):
        mont = Montgomery(n)
        t1 = b * mont.rrm
        t2 = m * mont.rrm

        r1 = mont.reduce(t1)
        r2 = mont.reduce(t2)
        r = 1 << mont.n

        prod = mont.reduce(mont.rrm)
        base = mont.reduce(b * mont.rrm)
        exp = m
        while exp.bit_length() > 0:
            if (exp & 1) == 1:
                prod = mont.reduce(prod * base)
            exp = exp >> 1
            base = mont.reduce(base * base)
        return(mont.reduce(prod))
