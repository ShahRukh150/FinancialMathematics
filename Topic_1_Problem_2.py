import numpy as np
from MV_Functions import CMVL
import matplotlib.pyplot as plt

print("\n Topic 1: Minimum Variance Line (Two Assets)")
print("Problem 2 \n")

# =============================================================================
# Data
# =============================================================================
# STC
mu_A=0.10
sd_A=0.18
# SABIC
mu_B=0.12
sd_B=0.22
#correlation
corr_AB=0.3

# =============================================================================
# (a) Derive the equation of the MVL in (σ, μ) space
# =============================================================================
print("Part (a) \n")
# Defining Variables for Entries of Det M and Calculating Values of a,b
n=2

C=np.zeros((n,n))           # Covariance Matrix
C[0,0]=sd_A**2
C[1,1]=sd_B**2
C[0,1]=corr_AB*sd_A*sd_B
C[1,0]=C[0,1]

mu=np.zeros((n,1))          # Returns Matrix
mu[0,0]=mu_A
mu[1,0]=mu_B

# Computing Constants of Minimum Variance Line Using the function from MV_Functions
a,b=CMVL(n, mu, C)

print("Equation of Minimum Variance Line= m(a)+b, where a= \n", a)
print("and b= \n", b,"\n")

# =============================================================================
# (b) Plot the MVL and mark the minimum variance portfolio
# =============================================================================
print("Part (b) \n")
m=np.linspace(0.01, 0.15)   # Expected Returns between 1% to 15%
sd=np.zeros(len(m))         # Standard Deviations of Minimum Variance Portfolios against Expected Returns

for i in range(len(m)):
    w_new = m[i]*a+b           # Minimum Variance Portfolio for ith Expected Return in m
    w_new_T = w_new.transpose()
    sd[i] = np.sqrt(w_new_T @ C @ w_new) # Standard Deviation of the Portfolio

# Finding Minimum Value Portfolio
m_sd=min(sd)            # minimum risk
ind= np.argmin(sd)     # index of minimum risk
min_m=m[ind]            # Expected Return of Minimum risk

# Plotting MVL and marking MVP

# Create the plot
plt.plot(sd, m, color='black', linestyle='-', label= 'Minimum Variance Line')
plt.plot(m_sd, min_m, marker='o', label="Minimum Variance Portfolio")
plt.xlabel("Portfolio Risk (Standard Deviation)")
plt.ylabel("Expected Returns")
plt.title("Minimum Variance L")
plt.legend()
plt.show()