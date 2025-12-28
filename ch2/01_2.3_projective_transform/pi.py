
#%%
from decimal import Decimal, getcontext

# set precision (total digits, not just decimals)
getcontext().prec = 100

def leibniz_pi(n_terms):
    pi_over_4 = Decimal(0)
    sign = Decimal(1)

    for k in range(n_terms):
        pi_over_4 += sign / Decimal(2 * k + 1)
        sign = -sign

    return pi_over_4 * Decimal(4)


pi = leibniz_pi(2000000)
print(pi)
