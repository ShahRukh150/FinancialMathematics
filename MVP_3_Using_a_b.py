import numpy as np

print("MINIMUM VARIANCE PORTFOLIO FOR GIVEN EXPECTED RETURN m WITH n=3 STOCKS")
print("Using a and b")

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
One_V=np.ones((n,1))        # Row Matrix of ones
mu=np.zeros((n,1))            # Row matrix of returns

# =============================================================================
# Calculating mu Matrix
# =============================================================================
for i in range(n):
    mu[i,0]=Returns[i]

# =============================================================================
# Computing entries of M
# =============================================================================
C_inverse = np.linalg.inv(C)   # Inverse of C
mu_T=mu.transpose()
One_V_T=One_V.transpose()

# =============================================================================
# Computing Determinant of M, and Values of a and b
# =============================================================================
Term1 = mu_T @ C_inverse @ mu
Term2 = mu_T @ C_inverse @ One_V
Term3 = Term2
Term4 = One_V_T @ C_inverse @ One_V

Det_M = Term1 * Term4 - Term2 * Term3
a = (1/Det_M) * C_inverse @ (Term4 * mu  - Term2 * One_V)
b = (1/Det_M) * C_inverse @ (Term1 * One_V  - Term3 * mu)

# =============================================================================
# Computing Minimum Variance Portofolio for given expected return "m"
# =============================================================================
w = m*a+b
print(w)