from Crypto.Util import \
    number  # See https://www.dlitz.net/software/pycrypto/api/current/toc-Crypto.Util.number-module.html
from util import lcm, flip_random_bit, int2string, string2int
from random import randint
# inverse algorithms
from inverse import rec_inverse
from tail_optimized import tail_rec_inverse

class RSA:
    def __init__(self, powmodn=0, inverse=rec_inverse, sign=False, gmp=False):

        self.powmodn = powmodn
        self.inverse = inverse
        self.gmp = gmp

        if self.gmp:
            from gmpy2 import powmod
            self.powmodn = powmod

        if sign:
            self.rsa_verify = self.rsa_encrypt
            self.rsa_sign = self.CRT_rsa_decrypt

        print("RSA scheme using {} and {} algorithms".format(self.powmodn.__name__, inverse.__name__))

    def rsa_encrypt(self, m, public_key):
        n, e = public_key
        return self.powmodn(m, e, n)

    # It's "isomorphic" to the encrypt function! Maybe just reuse the encrypt function.
    def rsa_decrypt(self, c, private_key):
        n, d = private_key
        return self.powmodn(c, d, n)

    def generate_keys(self, bit_length=1024, e=65537):

        p = number.getPrime(bit_length)
        q = number.getPrime(bit_length)
        n = p * q
        # Carmichael's totient function, which is the same as Euler's in this case.
        l = lcm(p - 1, q - 1)
        d = self.inverse(e, l)

        if self.gmp:
            from gmpy2 import mpz
            public_key = (mpz(n), mpz(e))
            private_key = (mpz(n), mpz(d))
            return (mpz(p), mpz(q), mpz(n), mpz(l), mpz(e), mpz(d), public_key, private_key)

        else:
            public_key = (n, e)
            private_key = (n, d)
            return (p, q, n, l, e, d, public_key, private_key)


    def encrypt(self, plaintext_string, public_key):
        return int2string(rsa_encrypt(string2int(plaintext_string), public_key))

    def decrypt(self, ciphertext_string, private_key):
        return int2string(rsa_decrypt(string2int(ciphertext_string), private_key))

    # Square-and-multiply using the CRT. Computes b^m (mod n).
    def CRT_pow_mod_n(self, b, m, n, p, q, faulty=False):
        (u, v) = self.Zn_to_ZpxZq(b, p, q)

        w = self.powmodn(u, m % (p - 1), p)
        x = self.powmodn(v, m % (q - 1), q)

        if faulty:  # used by the random fault attack
            if randint(0, 1):
                w = flip_random_bit(w)
            else:
                x = flip_random_bit(x)

        return self.ZpxZq_to_Zn(w, p, x, q, n)

    def CRT_rsa_decrypt(self, c, private_key, p, q, faulty=False):
        n, d = private_key
        return self.CRT_pow_mod_n(c, d, n, p, q, faulty)

    # Map an element a of Z_n* to an element of Z_p* x Z_q*
    def Zn_to_ZpxZq(self, a, p, q):
        return ((a % p, a % q))

    # Map an element (a, b) of Z_p* x Z_q* to Z_n*
    def ZpxZq_to_Zn(self, a, p, b, q, n):
        s = self.inverse(p, q)
        t = self.inverse(q, p)

        # s is the inverse of p in Z_q*. It means sp = 1 (mod q).
        # Obviously sp = 0 (mod p). So sp in Z_n* maps to (0, 1) in Z_p* x Z_q*.
        # Similarly, The isomorphism maps tq in Z_n* to (1, 0) in Z_p* x Z_q*.

        # Therefore, spb = b (mod q) and (obviously) spb = 0 (mod p).
        # So spb in Z_n* maps to (0, b) in Z_p* x Z_q*. Similarly, the isomorphism
        # maps tqa in Z_n* to (a, 0) in Z_p* x Z_q*.

        # Combining the two, we get that spb + tqa = a (mod p) = b (mod q).
        # That is, the isomorphism maps spb + tqa in Z_n* to (a, b) in Z_p* x Z_q*.

        return ((s * p * b + t * q * a) % n)
