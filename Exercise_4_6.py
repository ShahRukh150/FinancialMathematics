import numpy as np
from MV_Functions import CMVL, MVS_EF

print("\nBOOK: Portfolio Theory and Risk Management")
print("Exercise 4.6 \n")

# =============================================================================
# Data
# =============================================================================
n=3
# Covariance Matrix
C=np.array([[0.01, 0, 0],
   [0, 0.02, 0.02],
   [0, 0.02, 0.04]])
# Returns Matrix
mu=np.array([[0.1],
             [0.2],
             [0.3]])

# =============================================================================
# Find the vectors a, b described in Corollary (4.7). Using a and b
# compute the vector on the minimum variance line corresponding to m = 20%.
# =============================================================================
# Data
m = 0.20

a,b=CMVL(n, mu, C)
w=m*a+b

print("a= ", a,"\n")
print("b= ",b ,"\n")
print("w= (in %)", 100*w, "\n")

p=1 # allowing plots

_,_,_,_ = MVS_EF(n, a, b, C, m, p)