# Functions

from math import *
from time import *
from random import *
from Crypto.Util import \
    number  # See https://www.dlitz.net/software/pycrypto/api/current/toc-Crypto.Util.number-module.html
import sys
from flint import nmod, fmpz
from Crypto.Hash import SHA
from Crypto.Hash import MD5

sys.setrecursionlimit(16384)


def add(a, b, n):
    return (a + b) % n


def mult(a, b, n):
    return (a * b) % n


# Naïve recursive implementation of the square-and-multiply fast exponentiation.
# returns bˆm (mod n)
def rec_pow_mod_n(b, m, n):
    if m == 0:
        return 1
    elif m % 2 == 0:
        t = rec_pow_mod_n(b, m // 2, n)
        return mult(t, t, n)
    else:
        return mult(b, rec_pow_mod_n(b, m - 1, n), n)


# bit-twiddling way to compute b^m (mod n). No recursion needed.
# If you can optimize this further, do so!
def bit_pow_mod_n(b, m, n):
    acc = nmod(1, n)
    z = nmod(b, n)
    while m:
        if m & 1:  # check the least significant bit
            acc *= z
        z *= z
        m >>= 1  # the bit shift
    return fmpz(acc)

# Python supports first-class functions:
pow_mod_n = bit_pow_mod_n


def lcm(a, b):
    return a * b // gcd(a, b)


def rsa_encrypt(m, public_key):
    n, e = public_key
    return pow_mod_n(m, e, n)


# It's "isomorphic" to the encrypt function! Maybe just reuse the encrypt function.
def rsa_decrypt(c, private_key):
    n, d = private_key
    return pow_mod_n(c, d, n)


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
#        return t % n


# My rewrite
def rinv_helper(r, new_r, t=0, new_t=1):
    if new_r > 0:
        quotient = r // new_r
        return rinv_helper(new_r, r % new_r, new_t, t - (r // new_r) * new_t)
    else:
        return t


# Multiplicative inverse of a in Z_n*
def inverse(a, n):
    if gcd(a, n) == 1:
        return fmpz(rinv_helper(n, a) % n)
    else:
        return "a is not invertible in Z/nZ"


def generate_keys(bit_length=1024, e=65537):
    p = fmpz(number.getPrime(bit_length))
    q = fmpz(number.getPrime(bit_length))
    n = p * q
    # Carmichael's totient function, which is the same as Euler's in this case.
    l = fmpz(lcm(p - 1, q - 1))
    public_key = (n, e)
    d = inverse(e, l)
    private_key = (n, d)
    return (p, q, n, l, e, d, public_key, private_key)


def string2int(s):
    # return int.from_bytes(s.encode("utf-8"), byteorder = "big")
    return fmpz(number.bytes_to_long(s.encode("utf-8")))


def int2string(n):
    # return (n.to_bytes(((n.bit_length() + 7) //8), byteorder = "big")).decode("utf-8")
    return number.long_to_bytes(n)


def encrypt(plaintext_string, public_key):
    return int2string(rsa_encrypt(string2int(plaintext_string), public_key))


def decrypt(ciphertext_string, private_key):
    return int2string(rsa_decrypt(string2int(ciphertext_string), private_key))


# --------------------------------------------------
# Faster decryption using the Chinese Remainder Theorem (CRT)
#
# Map an element a of Z_n* to an element of Z_p* x Z_q*
def Zn_to_ZpxZq(a, p, q):
    return ((a % p, a % q))


# Map an element (a, b) of Z_p* x Z_q* to Z_n*
def ZpxZq_to_Zn(a, p, b, q, n):
    s = inverse(p, q)
    t = inverse(q, p)

    # s is the inverse of p in Z_q*. It means sp = 1 (mod q).
    # Obviously sp = 0 (mod p). So sp in Z_n* maps to (0, 1) in Z_p* x Z_q*.
    # Similarly, The isomorphism maps tq in Z_n* to (1, 0) in Z_p* x Z_q*.

    # Therefore, spb = b (mod q) and (obviously) spb = 0 (mod p).
    # So spb in Z_n* maps to (0, b) in Z_p* x Z_q*. Similarly, the isomorphism
    # maps tqa in Z_n* to (a, 0) in Z_p* x Z_q*.

    # Combining the two, we get that spb + tqa = a (mod p) = b (mod q).
    # That is, the isomorphism maps spb + tqa in Z_n* to (a, b) in Z_p* x Z_q*.

    return ((s * p * b + t * q * a) % n)


# For the random fault attack described in section 5.2, the attacker needs
# to corrupt the contents of exactly one of w or x.
# This function flips exactly one bit at a position chosen at random:
def flip_random_bit(x):
    return x ^ (1 << randrange(x.bit_length()))


# ^ is the bitwise XOR operator. a << b shifts a by b bits.


# Square-and-multiply using the CRT. Computes b^m (mod n).
def CRT_pow_mod_n(b, m, n, p, q, faulty=False):
    (u, v) = Zn_to_ZpxZq(b, p, q)

    w = pow_mod_n(u, m % (p - 1), p)
    x = pow_mod_n(v, m % (q - 1), q)

    if faulty:  # used by the random fault attack
        if randint(0, 1):
            w = flip_random_bit(w)
        else:
            x = flip_random_bit(x)

    return ZpxZq_to_Zn(w, p, x, q, n)


def CRT_rsa_decrypt(c, private_key, p, q, faulty=False):
    n, d = private_key
    return CRT_pow_mod_n(c, d, n, p, q, faulty)


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


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Test

print("\n\n-------------------------------------------------------------")
print("RSA decryption or signing using the Chinese Remainder Theorem (CRT)")


def run_test(message="The quick brown fox jumps over the lazy dog.",
             bit_length=2048, e=65537):
    # Generate a pair of primes
    # Use PyCrypt's library for generating primes for now. If we need to make
    # a home-brew version of these, we will, later...
    print("Generating ", bit_length, "-bit primes...")

    (p, q, n, l, e, d, public_key, private_key) = generate_keys(bit_length, e)

    print("\nOriginal plaintext message: ", message)

    m = string2int(message)
    print("\nEncrypting message...")
    c = rsa_encrypt(m, public_key)
    ciphertext = int2string(c)
    print("  Ciphertext: ", ciphertext)

    print("\nDecrypting message...")
    [m2, running_time] = rt2(rsa_decrypt, c, private_key)
    message2 = int2string(m2)
    print("  Message decrypted by rsa_decrypt: ", message2)
    print("  Running time for rsa_decrypt: ", running_time)

    print("\nDecrypting message using Chinese Remainder Theorem...")
    [m3, running_time] = rt4(CRT_rsa_decrypt, c, private_key, p, q)
    message3 = int2string(m3)
    print("  Message decrypted by CRT_rsa_decrypt: ", message3)
    print("  Running time for CRT_rsa_decrypt: ", running_time)

    return (p, q, n, l, e, d, public_key, private_key, m, c, message, ciphertext)


# Run the test with default parameters and save the results:
(p, q, n, l, e, d, public_key, private_key, m, c, message, ciphertext) = run_test()

print("\n\n-------------------------------------------------------------")
print("Creating csv table of running times for rsa_decrypt and CRT_rsa_decrypt...\n")


def create_decrypt_running_time_table(message, e, start, stop, step, trials):
    m = string2int(message)

    print("bit_length,Naïve_fast_exponentiation,CRT_fast_exponentiation")

    for bit_length in range(start, stop + 1, step):
        sum_rt = sum_rt_crt = 0

        for i in range(trials):
            (p, q, n, l, e, d, public_key, private_key) = generate_keys(bit_length, e)
            c = rsa_encrypt(m, public_key)

            # Accumulate naïve recursive running times across trials
            sum_rt += rt2(rsa_decrypt, m, private_key)[1]

            # ditto for CRT
            sum_rt_crt += rt4(CRT_rsa_decrypt, c, private_key, p, q)[1]

        print(bit_length, sum_rt / trials, sum_rt_crt / trials, sep=',')


# Run it:
create_decrypt_running_time_table("Testing 123", 65537, 128, 1280, 128, 5)

# -------------------------------------------------------------------------
# Mount the random fault attack decribed in section 5.2
#
# Signing and verifying are just the same functions as decrypting and encrypting:
CRT_rsa_sign = CRT_rsa_decrypt
rsa_verify = rsa_encrypt

bit_length = 1024;
e = 65537

print("\n\n-------------------------------------------------------------")
print("\nThe random fault attack:")
print("Generating", bit_length, "-bit keys...")
(p, q, n, l, e, d, public_key, private_key) = generate_keys(bit_length, e)

message = "Please attach your signature."
print("\nAlice asks Bob to sign this message:\n\n\t", message)
m = string2int(message)
print("\nIn numeric form, this message is m =", m)

print("\nNormally, Bob signs the message with his private key. The signature is ")
s = CRT_rsa_sign(m, private_key, p, q)
print("\ns =", s)

print("\nAlice can verify this signature using the public key:")
print("\nrsa_verify(s, public_key) = ", rsa_verify(s, public_key))

print("\nIt is the same as the original message m.")

print("\nThe random fault attack works on the CRT implementation of rsa_sign")
print("(same as rsa_decrypt) function.")
print("With CRT, RSA signing/decrypting first sends the message m in Z_n*")
print("to the corresponding element (u, v) in Z_p* x Z_q*.")
print("Exponentiation is done in Z_p* x Z_q* where it is less computationally expensive:")
print("\n\t x = u ^ (d mod (p - 1))")
print("\n\t w = v ^ (d mod (q - 1))")
print("\nAfterwards, send (x, w) in Z_p* x Z_q* to the corresponding element in Z_n*.")
print("This can be done using the extended euclidean algorithm, where we find r and t,")
print("the inverses of p in Z_q*, and q in Z_p*, respectively, so that")
print("\n\trpx + tqw = w (mod p) = x (mod q)")
print("\nis the corresponding element in Z_n*. This is the signature s = m ^ d (mod n).")
print("Note that spx = 0 (mod p) and tqw = 0 (mod q). The random fault attack")
print("can be mounted if exactly one of w or x can be corrupted.")

print("\nSuppose exactly one of x or w has bit errors, say x becomes x'.")
print("Then Alice doesn't get the original message m when she verifies the signature.")
print("Instead of s^e (mod n) = m, Alice gets ")
print("\n\tm' = (rpx' + tqw)^e (mod n).")
print("\t   = (rpx')^e + (tpq)^e (mod n) # All cross products are 0 mod n = pq.")
print("\nThe difference between this and m = s^d = (rpx) ^ e + (tpq) ^ e (mod n)")
print("is (rpx') ^ e - (rpx)^e (mod n). This difference is equivalent to 0 (mod p),")
print("and just as important, this difference is nonzero. That is,")
print("\n\tm - verify(s', public_key) = 0 (mod p) and is nonzero.")
print("\nThus, computing gcd(n, m - verify(s' public_key)) reveals one of the")
print("factors of n.")
print("\nWhen we introduce even a single bit error in x or w, we don't get the")
print("original message m when we verify the corrupted signature.")

print("\nThe CRT_rsa_sign function has a 'feature' that lets us introduce a")
print("bit error at a random position in either x or w:")
print("\n\ts1 = CRT_rsa_sign(m, private_key, p, q, faulty = True)")

s1 = CRT_rsa_sign(m, private_key, p, q, faulty=True)  # random bit flip

print("\nThe corrupted signature: s' =", s1)

m1 = rsa_verify(s1, public_key)

print("\nAlice doesn't get the original message back when she verifies s':")
print("\nm' = verify(s', public_key) =", m1)

print("\nThe difference between this and the original message is equivalent to 0")
print("modulo one of the prime factors of n. Alice can recover the factorization by computing the gcd.")

recovered_factor = gcd(n, m - m1)

print("\ngcd(n, m - m') =", recovered_factor)
print("\nIt should be equal to one of the prime factors of n:")
print("\np =", p)
print("\nq =", q)

# --------------------------------------------------------------------------
# Proposed SHA1 commitment
#
print("\n\n-------------------------------------------------------------")
print("Proposed SHA1 commitment")

ID = b"HKV"  # Our initials, à la R, S, and A
m = b"Resistance is futile."
print("\nID =", ID)
print("m =", m)

# Create the 128-bit MD5 digest using PyCrypto's MD5 class
r = MD5.new()
r.update(m)
print("\nMD5(m) =", r.digest())
print("In printable hex, MD5(m) =", r.hexdigest())

# Concatenate
M = ID + m + r.digest()
print("ID || m || r =", M)

# Commitment, C = SHA1(M), which is a 160-bit digest
C = SHA.new()
C.update(M)
print("\nC = SHA1(ID || m || r) =", C.digest())
print("In printable hex, C =", C.hexdigest())
print("In binary, C =", bin(int(C.hexdigest(), 16)))
