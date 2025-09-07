import numpy as np
from MV_Functions import CMVL
import matplotlib.pyplot as plt

print("\nBOOK: Portfolio Theory and Risk Management")
print("Exercise 4.7 \n")

# Consider the data from Exercise 4.6.
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
# Consider two portfolios corresponding to m = 10% and m = 20%: Find the
# variances of, as well as the covariance between, their returns.
# =============================================================================
#Data
# Target Returns
m_1=0.10
m_2=0.20

# Coefficients of MVL
a,b=CMVL(n, mu, C)

# Weights of MVP corresponding to m_1 and m_2
w_1 = m_1 * a + b
w_2 = m_2 * a + b

# Variances of Returns of portfolios
var_w_1 = w_1.transpose() @ C @ w_1
var_w_2 = w_2.transpose() @ C @ w_2

# Covariances of Returns of portfolios
Covariance_kw1_kw2 = w_1.transpose() @ C @ w_2

print("Variance of Returns of Portfolio 1: ", var_w_1)
print("Variance of Returns of Portfolio 2: ", var_w_2)
print("Covariance Between Returns of Portfolio 1 and Portfolio 2: ", Covariance_kw1_kw2,"\n")

print("Plot Using Two Fund Theorem \n")

# =============================================================================
# Use these to plot the minimum variance line in the (sd; mu) plane.
# =============================================================================
returns_range = np.linspace(0.01,1.0,100) # Between 1% to 100%
sd_range=np.zeros(len(returns_range))
i=0

for m in returns_range:
    alpha = (m-m_2)/(m_1-m_2)
    w_temp = alpha*w_1+(1-alpha)*w_2
    sd_temp = np.sqrt(w_temp.transpose() @ C @ w_temp)
    sd_range[i]=sd_temp
    i+=1

# Create the plot
plt.plot(sd_range, returns_range)
plt.xlabel("Portfolio Risk (Standard Deviation)")
plt.ylabel("Expected Returns")
plt.title("Minimum Variance Line")
plt.legend()
plt.show()