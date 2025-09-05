import numpy as np
import matplotlib.pyplot as plt

print("Minimum Variance Set vs Effective Frontier")
print()

# =============================================================================
# Values Using from Example (Aramco Vs STC)
# =============================================================================
n=2                           # Number of Stocks/Assets
# Returns Vector
Returns = (0.08,0.12)         # Returns 8%,12%
# Covariance Matrix
C = [[0.0225, 0.009],
     [0.009, 0.04]]
m = 0.10                      # Expected Return m=11%


# =============================================================================
# # =============================================================================
# # Values Using from Example (Apple, Microsoft, Google)
# # =============================================================================
# n=3                           # Number of Stocks/Assets
# # Returns Vector
# Returns = (0.09,0.11,0.13)    # Returns 9%,11%,13%
# # Covariance Matrix
# C = [[0.030, 0.018, 0.014],
#      [0.018, 0.040, 0.020],
#      [0.014, 0.020, 0.050]]
# m = 0.11
# =============================================================================


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
print("Weights (in percentages) = ")
print(w)

# =============================================================================
# Computing Minimum Variance Set and Efficient Frontier
# =============================================================================

r=np.linspace(0.01, 0.15)   # Expected Returns between 1% to 15%
sd=np.zeros(len(r))         # Standard Deviations of Minimum Variance Portfolios against Expected Returns
r_ef = []                    # Efficient Frontier Expected Returns Variable
sd_ef = []                   # Efficient Frontier Standard Deviation Variable

for i in range(len(r)):
    w_new = r[i]*a+b           # Minimum Variance Portfolio for ith Expected Return in r
    w_new_T = w_new.transpose()
    sd[i] = np.sqrt(w_new_T @ C @ w_new) # Standard Deviation of the Portfolio
    if r[i] > m:
        r_ef.append(r[i])       # Add this Expected Return to Efficient Frontier
        sd_ef.append(sd[i])     # Add this Standard Deviation to Efficient Frontier

# =============================================================================
# Plotting Standard Deviation against Expected Returns
# =============================================================================

# Create the plot
plt.plot(sd, r, color='red', linestyle='-', label= 'Minimum Variance Set')
plt.plot(sd_ef, r_ef, color='green', linestyle='-', label= 'Efficient Frontier')
plt.axhline(y=m, color='black', linestyle='-')
plt.plot()
plt.xlabel("Portfolio Risk (Standard Deviation)")
plt.ylabel("Expected Returns")
plt.title("Minimum Variance Set Vs Efficient Frontier for 2 stocks")
plt.legend()
plt.show()