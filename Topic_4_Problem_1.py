import numpy as np
from MV_Functions import mu_3, Cov_3, GMVP, CMVL

print("\n Topic 4: Practical Applications")
print("Problem 1 \n")

# =============================================================================
# Data
# =============================================================================
n=3
# Saudi Aramco
mu_A=0.08
sd_A=0.14
# Al Rajhi Bank
mu_B=0.09
sd_B=0.16
# SABIC
mu_C=0.12
sd_C=0.20
#correlations
corr_AB=0.25
corr_AC=-0.30
corr_BC=0.20
#Target Return
m=0.09

# =============================================================================
# Find the optimal portfolio weights and its risk.
# =============================================================================

# Covariance Matrix
C = Cov_3(sd_A, sd_B, sd_C, corr_AB, corr_AC, corr_BC)

# Returns Matrix
mu = mu_3(mu_A, mu_B, mu_C)

# Co-efficients of MVL
a,b = CMVL(n, mu, C)

# MVP of m=9%
w_m = m * a + b

# Standard Deviation of this MVP of m= 9%
risk = np.sqrt(w_m.transpose() @ C @ w_m)

print("Optimal Portfolio Weights for 9% Target Returns is: ", 100*w_m)
print("Risk associated with portfolio for 9% Target Returns is: ", 100*risk,"% \n")


print("INTERESTING OBSERVATION \n")
# Global MVP
w_min = GMVP(n, C)
m_min = mu.transpose() @ w_min
risk_min = np.sqrt(w_min.transpose() @ C @ w_min)

print("Global Minimum Variance Portfolio Weights is: (in percentages)", 100*w_min)
print("Returns on GMVP is: ", m_min*100,"%")
print("Risk associated with GMVP is: ", risk_min*100,"%")

print("\n THERE EXISTS ANOTHER PORTFOLIO THAT GIVES MORE RETURNS FOR LESS RISK.")