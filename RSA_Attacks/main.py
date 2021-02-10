import sys
from tests import test_RSA, test_RSA_sign, create_decrypt_running_time_table, prompt_for_powmodn
from tail_optimized import tail_rec_inverse
from inverse import rec_inverse

val = int(input("\n1 Test RSA Encryption\
           \n2 Test RSA Signature\
           \n3 Generate RSA Decryption CSV\n\ninput: "))

if val:
    if val == 1:
        print("\n\n-------------------------------------------------------------")
        print("RSA decryption using the Chinese Remainder Theorem (CRT)")

        # prompt user input for powmodn algorithm type and assign to alg obj
        pow_alg = prompt_for_powmodn()

        ans = int(input("use tail rec inverse? 0 no, 1 yes"))
        if ans:
            inv_alg=tail_rec_inverse
        else:
            sys.setrecursionlimit(16384)
            inv_alg = rec_inverse

        # Run the test with default parameters and save the results:
        (p, q, n, l, e, d, public_key, private_key, m, c, message, ciphertext) =\
        test_RSA(inv_alg, powmodn=pow_alg)

    if val == 2:
        print("\n\n-------------------------------------------------------------")
        print("Creating csv table of running times for rsa_decrypt and CRT_rsa_decrypt...\n")

        # prompt user input for powmodn algorithm type and assign to alg obj
        alg = prompt_for_powmodn()

        # rough prompt for using tail optimized inverse
        ans = int(input("use tail rec inverse? 0 no, 1 yes"))
        if ans:
            inv_alg=tail_rec_inverse
        else:
            inv_alg = rec_inverse

        create_decrypt_running_time_table("Testing 123", 65537, 128,1280, 128, 5,
                                          powmodn=pow_alg, inverse=inv_alg)

    if val == 3:
        # Test fault attack
        test_RSA_sign()
