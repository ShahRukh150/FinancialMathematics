import numpy as np
from MV_Functions import mu_3, Cov_3, M1_M2_MVL

print("\nBOOK: Portfolio Theory and Risk Management")
print("Exercise 4.5 \n")

# =============================================================================
# Data
# =============================================================================
n=3
# Asset 1
var_A=0.01
mu_A=0.10
# Asset 2
var_B=0.02
mu_B=0.20
# Asset 3
var_C=0.04
mu_C=0.30

#correlation
corr_AB=0
corr_AC=0
corr_BC=0

# =============================================================================
# Using (4.13) compute the portfolio which solves the problem (4.12) for m = 25%.
# =============================================================================
# Data
m = 0.25

# Returns Matrix
mu = mu_3(mu_A, mu_B, mu_C)

# Covariance Matrix
sd_A=np.sqrt(var_A)
sd_B=np.sqrt(var_B)
sd_C=np.sqrt(var_C)
C = Cov_3(sd_A, sd_B, sd_C, corr_AB, corr_AC, corr_BC)

D_M,D_M1,D_M2 = M1_M2_MVL(n, m, mu, C)

One_V=np.ones((n,1))        # Column Matrix of ones
C_inverse = np.linalg.inv(C)   # Inverse of C

# =============================================================================
# Computing Minimum Variance Portofolio for given expected return "m"
# =============================================================================
w = (1/D_M) * C_inverse @ (D_M1 * mu  + D_M2 * One_V)
print("Minimum Variance Portfolio, weights (in %)= ")
print(100*w)