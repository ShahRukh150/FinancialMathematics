from MV_Functions import CMVL,Cov_3,mu_3

print("\n Topic 2: Three-Asset Minimum Variance Portfolios")
print("Problem 1 \n")

# =============================================================================
# Data
# =============================================================================
n=3
# Saudi Aramco
mu_A=0.08
sd_A=0.15
# STC
mu_B=0.12
sd_B=0.20
# Almarai
mu_C=0.10
sd_C=0.18
#correlations
corr_AB=0.30
corr_AC=0.20
corr_BC=0.25

# =============================================================================
# (a) Compute the covariance matrix  
# =============================================================================
print("Part (a) \n")
# Computing Covariance Matrix Using the function from MV_Functions
C=Cov_3(sd_A,sd_B,sd_C,corr_AB,corr_AC,corr_BC)
print("Covariance Matrix C= ")
print(C, "\n")

# =============================================================================
# (b) Find the MVL portfolio weights for target return m = 9%
# =============================================================================
print("Part (b) \n")
#Data
m=0.09

# Computing Returns Column Matrix Using the function from MV_Functions
mu = mu_3(mu_A, mu_B, mu_C)              # Returns Matrix

# Computing Constants of Minimum Variance Line Using the function from MV_Functions
a,b=CMVL(n, mu, C)

# Computing MVP for m=9%
w = m*a + b
print("MVP weights (in %)= ")
print(100*w, "\n")

# =============================================================================
# (c) Compute the portfolio variance
# =============================================================================
print("Part (c) \n")

MVP_var = w.transpose() @ C @ w
print("Portfolio Variance= ")
print(MVP_var)