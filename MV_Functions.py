import numpy as np
import matplotlib.pyplot as plt

def CMVL(n,mu,C):
    '''
    Function: Coeffiencts of Minimum Variance Line
    Parameters
    ----------
    n : Number of Stocks/Assets.
    mu : Expected Returns Column Matrix of Stocks/Assets.
    C : Covariance Matrix.

    Returns
    -------
    a : Vector Showing How Weights Change as m Increases.
    b : Base Portfolio with Minimum Variance and No Return Constraint.
    '''
    # =============================================================================
    # Defining Matrices and Variables
    # =============================================================================
    One_V=np.ones((n,1))        # Column Matrix of ones
    
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
    return a,b

def M1_M2_MVL(n,m,mu,C):
    '''
    Function: Determinant of M1, M2 and M in Minimum Variance Line
    Parameters
    ----------
    n : Number of Stocks/Assets.
    m : Expected Return on Portfolio.
    mu : Expected Returns Column Matrix of Stocks/Assets.
    C : Covariance Matrix.

    Returns
    -------
    Det_M  : (mu^T C^-1 mu) * (1^T C^-1 1) - (mu^T C^-1 1) * (mu^T C^-1 1)
    Det_M1 : m * (1^T C^-1 1) - (mu^T C^-1 1) * 1.
    Det_M2 : (mu^T C^-1 mu) * 1 - m * (mu^T C^-1 1).
    '''
    # =============================================================================
    # Defining Matrices and Variables
    # =============================================================================
    One_V=np.ones((n,1))        # Column Matrix of ones
    
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
    Det_M1 = m * Term4 - Term2 * 1
    Det_M2 = Term1 * 1 - m * Term3
    return Det_M,Det_M1,Det_M2

def MVS_EF(n,a,b,C,m,p):
    '''
    Function: Minimum Variance Set Vs Efficient Frontier
    Parameters
    ----------
    n : Number of Stocks/Assets.
    a : Vector Showing How Weights Change as m Increases.
    b : Base Portfolio with Minimum Variance and No Return Constraint.
    C : Covariance Matrix.
    m : Expected Return on Portfolio.
    p : If plot is required then p=1 otherwise no plot.

    Returns
    -------
    Plot of Minimum Variance Line and Effective Frontier (if p=1).
    r       : Vector Showing How Weights Change as m Increases.
    sd      : Base Portfolio with Minimum Variance and No Return Constraint.
    r_ef    : Covariance Matrix.
    sd_ef   : Expected Return on Portfolio.
    '''
    # =============================================================================
    # Computing Minimum Variance Set and Efficient Frontier
    # =============================================================================
    
    r1=np.linspace(0.01, 0.35)   # Expected Returns between 1% to 35%
    r=r1.tolist()
    r.append(m)
    r.sort()
    sd=np.zeros(len(r))         # Standard Deviations of Minimum Variance Portfolios against Expected Returns
    r_ef = []                    # Efficient Frontier Expected Returns Variable
    sd_ef = []                   # Efficient Frontier Standard Deviation Variable
    
    for i in range(len(r)):
        w_new = r[i]*a+b           # Minimum Variance Portfolio for ith Expected Return in r
        w_new_T = w_new.transpose()
        sd[i] = np.sqrt(w_new_T @ C @ w_new) # Standard Deviation of the Portfolio
        if r[i] >= m:
            r_ef.append(r[i])       # Add this Expected Return to Efficient Frontier
            sd_ef.append(sd[i])     # Add this Standard Deviation to Efficient Frontier
        
    if p==1:
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
        plt.title("Minimum Variance Set Vs Efficient Frontier")
        plt.legend()
        plt.show()
    return r,sd,r_ef,sd_ef

def Cov_3(sd_A,sd_B,sd_C,corr_AB,corr_AC,corr_BC):
    '''
    Function: Covariance Matrix for 3 Stocks/Assets
    Parameters
    ----------
    sd_A : Standard Deviation of Stock A.
    sd_B : Standard Deviation of Stock B.
    sd_C : Standard Deviation of Stock C.
    corr_AB : Correlation of Stock A to Stock B.
    corr_AC : Correlation of Stock A to Stock C.
    corr_BC : Correlation of Stock B to Stock C.

    Returns
    -------
    C   : Covariance Matrix.
    '''
    n=3
    C=np.zeros((n,n))
    C[0,0]=sd_A**2
    C[1,1]=sd_B**2
    C[2,2]=sd_C**2
    C[0,1]=corr_AB*sd_A*sd_B
    C[1,0]=C[0,1]
    C[0,2]=corr_AC*sd_A*sd_C
    C[2,0]=C[0,2]
    C[1,2]=corr_BC*sd_B*sd_C
    C[2,1]=C[1,2]
    return C

def mu_3(mu_A,mu_B,mu_C):
    '''
    Function: Returns Matrix for 3 Stocks/Assets
    Parameters
    ----------
    mu_A : Expected Returns on Stock A.
    mu_B : Expected Returns on Stock B.
    mu_C : Expected Returns on Stock C.

    Returns
    -------
    mu   : Expected Returns Matrix.
    '''
    n=3
    # Defining return matrix mu
    mu=np.zeros((n,1))              # Returns Matrix
    mu[0,0]=mu_A
    mu[1,0]=mu_B
    mu[2,0]=mu_C
    return mu

def GMVP(n,C):
    '''
    Function: GLobal Minimum Variance Portfolio
    Parameters
    ----------
    n : Number of Stocks/Assets.
    C : Covariance Matrix.
    
    Returns
    -------
    w_min : Global Minimum Variance Portfolio Weights.
    '''
    C_inverse = np.linalg.inv(C)   # Inverse of C
    One_V=np.ones((n,1))        # Column Matrix of ones
    One_V_T=One_V.transpose()
    w_min = (1/ (One_V_T @ C_inverse @ One_V)) * C_inverse @ One_V
    return w_min
