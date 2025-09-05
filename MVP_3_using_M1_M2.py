import numpy as np

print("MINIMUM VARIANCE PORTFOLIO FOR GIVEN EXPECTED RETURN m WITH n=3 STOCKS")
print()

# =============================================================================
# Values Using from Example (Apple, Microsoft, Google)
# =============================================================================
n=3                           # Number of Stocks/Assets
# Returns Vector
Returns = (0.09,0.11,0.13)    # Returns 9%,11%,13%
# Covariance Matrix
C = [[0.030, 0.018, 0.014],
     [0.018, 0.040, 0.020],
     [0.014, 0.020, 0.050]]
m = 0.11                      # Expected Return m=11%

# =============================================================================
# Defining Matrices and Variables
# =============================================================================
One_V=np.ones((n,1))          # Row Matrix of ones
mu=np.zeros((n,1))              # Row matrix of returns

# =============================================================================
# Calculating mu Matrix
# =============================================================================
for i in range(n):
    mu[i,0]=Returns[i]

# =============================================================================
# Computing entries of M, M_1 and M_2
# =============================================================================
C_inverse = np.linalg.inv(C) # Inverse of C
mu_T=mu.transpose()
One_V_T=One_V.transpose()

# =============================================================================
# Computing Determinant of M, M_1, and M_2
# =============================================================================
Term1 = mu_T @ C_inverse @ mu
Term2 = mu_T @ C_inverse @ One_V
Term3 = Term2
Term4 = One_V_T @ C_inverse @ One_V

Det_M = Term1 * Term4 - Term2 * Term3
Det_M1 = m * Term4 - Term2 * 1
Det_M2 = Term1 * 1 - m * Term3

# =============================================================================
# Computing Minimum Variance Portofolio for given expected return "m"
# =============================================================================
w = (1/Det_M) * C_inverse @ (Det_M1 * mu  + Det_M2 * One_V)
print(w)