import numpy as np
from MV_Functions import CMVL

print("Topic 1: Minimum Variance Line (Two Assets)")
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
# (a) Compute the covariance matrix  
# =============================================================================
print("Part (a) \n")
n=2
C=np.zeros((n,n))
C[0,0]=sd_A**2
C[1,1]=sd_B**2
C[0,1]=corr_AB*sd_A*sd_B
C[1,0]=C[0,1]
print("Covariance Matrix C= ")
print(C)
print()

# =============================================================================
# (b) Find the portfolio weights on the Minimum Variance Line (MVL) for target
# returns m = 7.5%, 8%, and 8.5%
# =============================================================================
print("Part (b) \n")
# Given Data
m=[0.075, 0.08, 0.085]

# Defining return matrix mu
mu=np.zeros((n,1))              # Returns Matrix
mu[0,0]=mu_A
mu[1,0]=mu_B

# Computing Constants of Minimum Variance Line Using the function from MV_Functions
a,b=CMVL(n,mu,C)

w=[]
for i in m:
    w_temp=i*a + b
    w.append(w_temp)
    print("Expected Return= ", i)
    print("Minimum Variance Portfolio= (in %)")
    print(100*w_temp, "\n")

# =============================================================================
# (c) Calculate the standard deviation for each portfolio
# =============================================================================
print("Part (c) \n")
sd=[]
for i in range(len(m)):
    sd_temp = np.sqrt(w[i].transpose() @ C @ w[i])
    sd.append(sd_temp)
    print("Expected Return (in %)= ", 100*m[i],"%")
    print("Standard Deviation/RIsk (in %)= ", 100*sd_temp, "\n")