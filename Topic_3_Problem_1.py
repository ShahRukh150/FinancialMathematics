import numpy as np
from MV_Functions import GMVP, CMVL

print("\n Topic 3: Efficient Frontier Properties")
print("Problem 1 \n")

# =============================================================================
# Data
# =============================================================================
# Saudi Aramco
mu_A=0.07
sd_A=0.14
# Al Rajhi Bank
mu_B=0.09
sd_B=0.16
#correlation
corr_AB=0.25

# =============================================================================
# Prove that for any portfolio w on the MVL: Cov(K_w,K_w_min) = (σ_min)^2
# =============================================================================
# Number of Stocks/Assets
n=2
# Returns Matrix
mu=np.array([[0.07],
       [0.09]])
# Covariance Matrix
C=np.array([[0.0196, 0.0056],
       [0.0056, 0.0256]])
# Global Minimum Variance Portfolio
w_min=GMVP(n, C)


# =============================================================================
# Right Hand Side ((σ_min)^2)
# =============================================================================
var_min = w_min.transpose() @ C @ w_min

# =============================================================================
# Left Hand Side (Cov(K_w,K_w_min))
# =============================================================================

# Coefficients of Minimum Variance Line 
a,b=CMVL(n, mu, C)

return_percentage = np.linspace(0.01,1.0,100) # Between 1% to 100%
flag=0

print("Errors when subtracting values of both sides for different m: \n")
for m in return_percentage:
    w=m*a+b
    Left_Term = w.transpose() @ C @ w_min
    Residual_expression = Left_Term - var_min
    print("Return m= ",100*m,"% , Error: ", Residual_expression)
    if Residual_expression >=0.01:
        flag+=1
print("\n The number of times when error is more than or equal to 0.01 is", flag,".")