import numpy as np
from MV_Functions import CMVL, Cov_3, mu_3, GMVP

print("\n Topic 2: Three-Asset Minimum Variance Portfolios")
print("Problem 2 \n")

# =============================================================================
# Data
# =============================================================================
n=3
# Riyad Bank
mu_A=0.06
sd_A=0.10
# Saudi Electricity
mu_B=0.05
sd_B=0.08
# Jarir Bookstore
mu_C=0.15
sd_C=0.25
#correlations
corr_AB=0.10
corr_AC=-0.20
corr_BC=0.05

# =============================================================================
# (a) Calculate the global minimum variance portfolio
# =============================================================================
print("Part (a) \n")

# Covariance Matrix
C = Cov_3(sd_A, sd_B, sd_C, corr_AB, corr_AC, corr_BC)

# Global Minimum Variance Portfolio
w = GMVP(n,C)

print("Global Minimum Variance Portfolio, weights (in %)= ")
print(100*w,"\n")

# =============================================================================
# (b) Find two distinct efficient portfolios and verify the Two-Fund Theorem
# =============================================================================
print("Part (b) \n")

# Returns Matrix
mu = mu_3(mu_A, mu_B, mu_C)

# Constants of MVL
a,b = CMVL(n, mu, C)

# Let w_1=m_1*a+b and w_2=m_2*a+b be two efficient portfolios for where m_1=0.30 and m_2=0.45
m_1=0.30
m_2=0.45

w_1 = m_1 * a + b
w_2 = m_2 * a + b

return_percentage = np.linspace(0.01,1.0,100) #between 1% to 100%
flag=0

print("Errors (in %) of Weights Calculated with w=m*a+b and w=alpha*w_1 + (1-alpha)*w_2: \n")
for m in return_percentage:
    alpha = (m-m_2)/(m_1-m_2)
    w_ab = m*a+b
    w_alpha = alpha*w_1+(1-alpha)*w_2
    Residual_weights = 100*w_ab - 100*w_alpha
    print("Return m= ",100*m,"%")
    print("Error: ", Residual_weights, "\n")
    for i in Residual_weights:
        if abs(i)>=1:
            flag+=1
print("\n The number of times when error in weights is more than or equal to 1 percent is", flag,".")