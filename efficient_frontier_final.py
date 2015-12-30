# based on blog post here: http://blog.quantopian.com/markowitz-portfolio-optimization-2/

import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd

import mpld3
from mpld3 import plugins

np.random.seed(123)

# Turn off progress printing 
solvers.options['show_progress'] = False

## NUMBER OF ASSETS
n_assets = 4

## NUMBER OF OBSERVATIONS
n_obs = 1000  #original 1000

#results in a n_assets x n_obs vector, with a return for each asset in each observed period
return_vec = np.random.randn(n_assets, n_obs) 


## Additional code demonstrating the formation of a Markowitz Bullet from random portfolios:

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)


def random_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma

## Uncomment to see visualization
#n_portfolios = 500
#means, stds = np.column_stack([
#    random_portfolio(return_vec) 
#    for _ in xrange(n_portfolios)
#])

#plt.plot(stds, means, 'o', markersize=5)
#plt.xlabel('std')
#plt.ylabel('mean')
#plt.title('Mean and standard deviation of returns of randomly generated portfolios')
#plt.show()



def convert_portfolios(portfolios):
    ''' Takes in a cvxopt matrix of portfolios, returns list '''
    port_list = []
    for portfolio in portfolios:
        temp = np.array(portfolio).T
        port_list.append(temp[0].tolist())
        
    return port_list


def optimal_portfolio(returns):
    ''' returns an optimal portfolio given a matrix of returns '''
    n = len(returns)
    #print n  # n=4, number of assets
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))  #S is the covariance matrix. diagonal is the variance of each stock

    
    pbar = opt.matrix(np.mean(returns, axis=1))
    print "pbar:", pbar

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]

    port_list = convert_portfolios(portfolios)
 
   
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]  #Different than input returns
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios] #np.sqrt returns the stdev, not variance
    
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    #print m1 # result: [ 159.38531535   -3.32476303    0.4910851 ]
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x'] #Is this the tangency portfolio? X1 = slope from origin?  
    print "wt, optimal portfolio:", wt
    return np.asarray(wt), returns, risks, port_list



def covmean_portfolio(covariances, mean_returns):
    ''' returns an optimal portfolio given a covariance matrix and matrix of mean returns '''
    n = len(mean_returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    S = opt.matrix(covariances)  # how to convert array to matrix?  

    pbar = opt.matrix(mean_returns)  # how to convert array to matrix?

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    port_list = convert_portfolios(portfolios)
    
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    frontier_returns = [blas.dot(pbar, x) for x in portfolios]  
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios] 
    
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(frontier_returns, risks, 2)
    #print m1 # result: [ 159.38531535   -3.32476303    0.4910851 ]
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']  

    return np.asarray(wt), frontier_returns, risks, port_list


## Example Input from Estimates

covariances = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] # inner lists represent columns, diagonal is variance
 

mean_returns = [1.5,3.0,5.0,2.5] # Returns in DALYs

weights, returns, risks, portfolios = covmean_portfolio(covariances, mean_returns)


#plt.plot(stds, means, 'o') #if you uncomment, need to run 500 porfolio random simulation above

## Matplotlib Visualization:

plt.ylabel('mean')
plt.xlabel('std')
plt.plot(risks, returns, 'y-o') #risks and returns are just arrays of points along the frontier
plt.show()


## Optional interactive plot using mpld3:
## Source: http://mpld3.github.io/examples/html_tooltips.html

# fig, ax = plt.subplots()
# ax.grid(True, alpha=0.3)

# labels = []
# for i in range(len(risks)):
#     label = " Risk: " + str(risks[i]) + " Return: " + str(returns[i]) + " Portfolio Weights: " + str(portfolios[i])
#     labels.append(str(label))

# points = ax.plot(risks, returns, 'o', color='b',
#                  mec='k', ms=15, mew=1, alpha=.6)

# ax.set_xlabel('standard deviation')
# ax.set_ylabel('return')
# ax.set_title('Efficient Frontier', size=20)

# tooltip = plugins.PointHTMLTooltip(points[0], labels,
#                                    voffset=10, hoffset=10)
# plugins.connect(fig, tooltip)

# mpld3.show()



















