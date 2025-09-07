import numpy as np

print("\n Topic 3: Efficient Frontier Properties")
print("Problem 2 \n")

# Demonstrate the Two-Fund Theorem with:

# =============================================================================
# Data
# =============================================================================
n=2
# Portfolio 1 (Aramco,STC)
w_1=np.array([[0.70],
              [0.30]])
# Portfolio 2 (Aramco,STC)
w_2=np.array([[0.50],
              [0.50]])
#Target Return
m=0.085

# =============================================================================
# Show how to construct a new portfolio with target return m = 8.5%
# =============================================================================

# Data used from Topic 1 Problem 1
# Returns Matrix
mu=np.array([[0.07],
       [0.09]])
# Covariance Matrix
C=np.array([[0.0196, 0.0056],
       [0.0056, 0.0256]])

# Returns of both Portfolios
m_1 = mu.transpose() @ w_1
m_2 = mu.transpose() @ w_2

# Finding alpha
alpha = (m-m_2)/(m_1-m_2)

# New Portfolio on MVL
w_new = alpha*w_1+(1-alpha)*w_2

m_new = mu.transpose() @ w_new

print("New Portfolio with Target Return m = 8.5% is (in %): ", 100*w_new)

print("Return of Portfolio 1: ", m_1*100,"%")
print("Return of Portfolio 2: ", m_2*100,"%")
print("Return of New Portfolio: ", m_new*100,"%")