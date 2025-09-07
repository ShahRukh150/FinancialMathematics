import numpy as np
from MV_Functions import CMVL, MVS_EF

print("\n Topic 4: Practical Applications")
print("Problem 2 \n")

# =============================================================================
# Data
# =============================================================================
# Aramco/STC without Almaria
n1=2
# Returns Matrix
mu1 = np.array([[0.07],
       [0.09]])
# Covariance Matrix
C1 = np.array([[0.0196, 0.0056],
       [0.0056, 0.0256]])

# Aramco/STC with Almaria
n2=3
# Returns Matrix
mu2 = np.array([[0.08],
       [0.12],
       [0.10]])
# Covariance Matrix
C2 = np.array([[0.0225, 0.009 , 0.0054],
       [0.009 , 0.04  , 0.009 ],
       [0.0054, 0.009 , 0.0324]])

m=0.09 # Assumed for calculations and comparision

# =============================================================================
# Compare the efficient frontiers with/without Almarai
# =============================================================================

a1,b1 = CMVL(n1, mu1, C1)
a2,b2 = CMVL(n2, mu2, C2)

p=1 # Allowing MVS_EF to plot

r1,sd1,r_ef1,sd_ef1=MVS_EF(n1, a1, b1, C1, m, p)
r2,sd2,r_ef2,sd_ef2=MVS_EF(n2, a2, b2, C2, m, p)

# =============================================================================
# Explain the change in portfolio allocation
# =============================================================================
w1 = m*a1 + b1
w2 = m*a2 + b2

print("Weights without Almarai: (in %)", 100*w1)
print("Weights with Almarai: (in %)", 100*w2)