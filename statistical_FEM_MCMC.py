# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:54:02 2019

@author: Ben Boys


Sample Y and U data for statistical finite element method, need about 100 samoples
"""
import numpy as np
from scipy.special import gamma
import scipy.stats as sp
import matplotlib.pyplot as plt
import time
from header import model
from mpl_toolkits.mplot3d import axes3d


# Define sensor positions
X_VALUES = np.linspace(0.0, 220.0, 60)
#seperator = ', '
#print(seperator.join(map(str, X_VALUES)))

# Using the model class, get the 'True' deflection values, U
model =  model(X_VALUES)

U_VALUES =  np.multiply(5, model.getModelMean())
#print(seperator.join(map(str, U_VALUES)))  

# get the K and Sigma matrices
K = model.getK(0.01)
SIGMA = model.getSigma(0.8)

# Sample delta

ERROR_MEAN = model.getErrorMean()

delta = sp.multivariate_normal.rvs(ERROR_MEAN, K, 1)
#print(seperator.join(map(str, delta)))


eta = sp.multivariate_normal.rvs(ERROR_MEAN, SIGMA, 1)
#print(seperator.join(map(str, eta)))

plt.figure()
plt.scatter(X_VALUES, delta)
plt.title('$\delta$')
plt.show


plt.figure()
plt.scatter(X_VALUES, eta)
plt.title('$\epsilon$')
plt.show

total_error = delta + eta

plt.figure()
plt.scatter(X_VALUES, total_error)
plt.title('$\delta$ + $\epsilon$')
plt.show

Y_VALUES = U_VALUES + total_error
#print(seperator.join(map(str, Y_VALUES)))

# Figures to show data
plt.figure()
plt.scatter(X_VALUES, U_VALUES)
plt.title('$u$')
plt.show
plt.figure()
plt.scatter(X_VALUES, Y_VALUES)
plt.title('$y$')
plt.show


def getKernelDensityEstimate(data):
    """ Input: data
        Retun: Plot of the kernel density estimate
    """
    
    kernel = sp.gaussian_kde(data) 
    x = np.linspace(0.0, 1.3, 100)
    y= np.linspace(0.0, 0.14, 100)
    
    X, Y = np.meshgrid(x, y)

    # Get Z values
    pos= np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(pos).T, X.shape)
    minZ = np.min(Z)
    maxZ = np.max(Z)
    
    
    # Plot contour map of Z
    plt.figure()
    breaks = np.linspace(minZ, maxZ, 1000)
    tick = np.linspace(minZ, maxZ, 11)
        
    plt.contourf(x, y, Z, breaks, cmap='viridis')
    plt.colorbar(ticks= tick, orientation='vertical')
    
    plt.xlabel('$\sigma$')
    plt.ylabel('$\lambda$')
    plt.grid()
    plt.title('Kernel Density Estimation')
    
    
    # Plot a 3D plot of Z
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis',linewidth=0)
    ax.set_xlabel('X avis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()
    
    return None

#Metropolis Hastings

def getSamples(Y_VALUES, U_VALUES, samples):
    """ Inputs: Get samples from the distribution given a likelihood and prior density
        Outputs: Data array of samples
        Description: Uses a simple multivariate metropolis hastings algorithm.
        """
    
    # Define start point of the MCMC sampler w[1] is lambda, w[0] is sigma
    w_prev = [1.0, 0.1]
    
    # Define proposal density of the MCMC sampler
    w_cov = [[0.001, 0.0],[0.0, 0.001]]
    
    # Define initial guesses of K and Sigma matrices
    K_GUESS = model.getK(w_prev[1])
    SIGMA_GUESS = model.getSigma(w_prev[0])
    # Get the likelihood
    cov_ = np.add(K_GUESS, SIGMA_GUESS)
    likelihood_prev = sp.multivariate_normal.pdf(Y_VALUES, U_VALUES, cov_)
      
    # Define alpha_sigma and alpha_lambda
    alpha_sigma = 1.0
    alpha_lambda = 1.0
    
    # Evaluate the pdf of the distribution we want to sample from
    prior_prev = sp.gamma.pdf(w_prev[0], alpha_sigma)*sp.gamma.pdf(w_prev[1], alpha_lambda, scale=0.01)
    
    data = [[],[]]
    total_samples = 0
    
    
    # Metropolis Hastings 'proper', just sampling from 1 chain now, because
    # we assume the chain has converged
    while total_samples < samples:
        total_samples+=1
        w = sp.multivariate_normal.rvs(w_prev, w_cov, 1)
        
        # Multiply two single variate prior distributions
        prior = sp.gamma.pdf(w[0], alpha_sigma)*sp.gamma.pdf(w[1], alpha_lambda, scale=0.01)
        if prior == 0:
            None
        else:
            # Get the likelihood this bit takes a while
            K_guess = model.getK(w[1])
            SIGMA_guess = model.getSigma(w[0])
            cov_ = np.array(np.add(K_guess, SIGMA_guess))
            likelihood = sp.multivariate_normal.pdf(Y_VALUES, U_VALUES, cov_)
            #print(likelihood)
            
            #compute acceptance ratio
            r = (prior*likelihood)/ (prior_prev*likelihood_prev)
        
            #generate u from a uniform distribution
            u =  np.random.uniform()
            if u <= r:
                #accept the sample
                data[0].append(w[0])
                data[1].append(w[1])
                w_prev = w
                prior_prev = prior
                likelihood_prev = likelihood
            else:
                None
    
    
    print(data)
    
    # Perform the burn on the first 100 values
    burn = 200
    
    data[0] = data[0][burn:]
    data[1] = data[1][burn:]
    
    NO_BINS = 100
    xstart = -1.5
    xfinish = 1.5
    ystart = -6.0
    yfinish = -3.0
    xedges = np.linspace(xstart, xfinish, NO_BINS)
    yedges = np.linspace(ystart, yfinish, NO_BINS)
    H, xedges, yedges = np.histogram2d(data[0], data[1], bins=(xedges, yedges))
    print(np.max(H))
    print(np.argmax(H))
    ind = np.unravel_index(np.argmax(H), H.shape)
    print(H.shape)
    print(ind)
    print('The percentage of accepted samples was {}%'.format(len(data[0])*100/(total_samples)))
        
    plt.figure()
    plt.hist2d(data[0], data[1], bins=100)
    plt.xlabel('$\sigma$')
    plt.ylabel('$\lambda$')
    plt.show()
    
# =============================================================================
#     # Write data to a file
#     with open(r"C:\Users\Ben Boys\Documents\Gyroid Beam Project\python\week 5\Metropolis hastings\mcmc.csv", 'w', newline='') as myfile:
#      wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#      data_zipped = zip(*data)
#      wr.writerow(data_zipped)
# =============================================================================

    # Get the Kernel Density Estimate and assosciated plots
    getKernelDensityEstimate(data)
    
    return None


t1_start = time.process_time()
getSamples(Y_VALUES, U_VALUES, 40000)
t1_stop = time.process_time()
print('elapsed time', t1_stop - t1_start)



