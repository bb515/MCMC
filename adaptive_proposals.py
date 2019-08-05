# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:10:26 2019

@author: Ben Boys

Metropolis Hastings Adaptive Proposals

For efficient sampling


Adapt the proposal distibution after every 100 samples


Proposal distribution covariance matrix is 0.2 times the current covariance matrix of the samples taken
"""



import scipy.stats as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
import time
import csv


# Mean vector
mean = [2.0, 2.0]

#Covariance matrix
cov = [[1.0, 0.95],[0.95,1.0]]

# Multivariate PDF
sample = sp.multivariate_normal.rvs(mean, cov, 1)

# Draw the pdf on a contour plot

# Create a linspace of x and y values

x = np.linspace(-5.0, 5.0, 100)
y= np.linspace(-5.0, 5.0, 100)

X, Y = np.meshgrid(x, y)

# this is key creates pairs of x and y values in the position they are in the meshgrid
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y

# Getting the Z values
rv = sp.multivariate_normal(mean, cov)


Z = rv.pdf(pos)
minZ = np.min(Z)
maxZ = np.max(Z)

breaks = np.linspace(minZ, maxZ, 1000)
tick = np.linspace(minZ, maxZ, 11)
    
plt.contourf(x, y, rv.pdf(pos), breaks, cmap='viridis')
plt.colorbar(ticks= tick, orientation='vertical')
    
plt.xlabel('alpha')
plt.ylabel('beta')
plt.grid()
plt.title('Marginal likelihood')
plt.show

# Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis',linewidth=0)
ax.set_xlabel('X avis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

def getKernelDensityEstimate(data, mean, cov):
    """ Input: data
        Retun: Plot of the kernel density estimate
    """
    
    kernel = sp.gaussian_kde(data) 
    x = np.linspace(-5.0, 5.0, 100)
    y= np.linspace(-5.0, 5.0, 100)
    
    X, Y = np.meshgrid(x, y)

    
    
    # Get Z values
    pos= np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(pos).T, X.shape)
    minZ = np.min(Z)
    maxZ = np.max(Z)
    
    # Get W values
    # this is key creates pairs of x and y values in the position they are in the meshgrid
    posW = np.empty(X.shape + (2,))
    posW[:, :, 0] = X; posW[:, :, 1] = Y

    # Getting the U and T values
    rv = sp.multivariate_normal(mean, cov)
    W = rv.pdf(posW)
    
    U = np.abs(np.subtract(W, Z))
    minU = np.min(U)
    maxU = np.max(U)
    
    T = (np.subtract(W, Z))
    
    
    # Plot Z
    plt.figure()
    breaks = np.linspace(minZ, maxZ, 1000)
    tick = np.linspace(minZ, maxZ, 11)
        
    plt.contourf(x, y, Z, breaks, cmap='viridis')
    plt.colorbar(ticks= tick, orientation='vertical')
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid()
    plt.title('Kernel Density Estimation')
    
    #Plot U
    plt.figure()
    breaks = np.linspace(minU, maxU, 1000)
    tick = np.linspace(minU, maxU, 11)
        
    plt.contourf(x, y, U, breaks, cmap='viridis')
    plt.colorbar(ticks= tick, orientation='vertical')
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid()
    plt.title('Difference between KDE and PDF')
    
    # Make a 3D plot of Z
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis',linewidth=0)
    ax.set_xlabel('X avis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()
    
    # Make a 3D plot of T
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, T, cmap='viridis',linewidth=0)
    ax.set_xlabel('X avis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_zlim([-0.08,0.08])
    plt.show()
    return None

def getBurnIn(mean, cov, chains):
    """ Inputs: Mean vector and covariance of a Gaussian to sample from using metropolis hastings
                Number of chains used to calculate chain variances (convergence).
        Outputs: Histogram of the distribution
        Description: Uses a simple multivariate metropolis hastings algorithm to sample from a Gaussian distribution without
        sampling directly from it. This can be used to sample from more complicated distributions which are impossible to sample from.
    """
    
    #initiate data array, data mean array and data cov array
    data = []
    data_mean = []
    data_cov = []
    MEAN = []
    
    #initiate some scatter plot arrays
    plt_R1 = []
    plt_R2 = []
    plt_burn = []
    
    # build data array- must be list because we rely heavily on append function
    
    for i in range(chains):
        data.append([[],[]])
        data_mean.append([])
        data_cov.append([])
        MEAN.append(mean)
   
    # Initiate R values at (arbritarily) larger than 1.01 for each parameter
    R1 = 100.0
    R2 = 100.0
    
    # Define small constant of covariance matrix
    W_VAR = 0.2
    
    # Initiate the proposal density covariance matrix
    w_cov= np.multiply(W_VAR, cov)
    
    # Define the starting points of the chains
    START_POINTS = [[0.0, -1.0], [1.0, 3.0], [4.0, -1.0]]
    
    # Initialise exponent for no. samples for the burn
    j = 3
    burn = 10
    while (R1 > 1.01) or (R2 > 1.01):
        for i in range(chains):
            
            # Location vector in 2D parameter space
            w_prev = sp.multivariate_normal.rvs(START_POINTS[i], w_cov, 1)
    
            # Evaluate the pdf of the distribution we want to sample from
            post_prev = sp.multivariate_normal.pdf(w_prev, mean, cov)
            
            while len(data[i][0]) < burn:
                w = sp.multivariate_normal.rvs(w_prev, w_cov, 1)
                post = sp.multivariate_normal.pdf(w, mean, cov)
                
                q_ij = sp.multivariate_normal.pdf(w_prev, w, w_cov)
                q_ji = sp.multivariate_normal.pdf(w, w_prev, w_cov)
                
                #compute acceptance ratio
                r = (post/ post_prev)* (q_ji/q_ij)
        
                #generate u from a uniform distribution
                u =  np.random.uniform()
                if u <= r:
                    #accept the sample
                    data[i][0].append(w[0])
                    data[i][1].append(w[1])
                    w_prev = w
                    post_prev = post
                else:
                    None
                
            ##CALCULATE R
            
            #mean of chain for current number (value) of samples
            data_mean[i] = [np.mean(data[i][0]), np.mean(data[i][1])]
        
            #cov of chain
            data_cov[i] = np.cov(data[i][0], data[i][1])
            
        #mean squared error
        mean_error = np.subtract(data_mean, MEAN)
        mean_squared_error = np.power(mean_error, 2)
            
        #covariance sum
        cov_sum = np.sum(data_cov, axis=0)
            
        B1 = np.multiply(burn/(chains-1), np.sum(mean_squared_error[:,0]))
        W1 = np.multiply(1/chains, cov_sum[0][0])
        V1 = (burn -1)/(burn) * W1 + (1/burn)* B1
            
        B2 = np.multiply(burn/(chains-1), np.sum(mean_squared_error[:,1]))
        W2 = np.multiply(1/chains, cov_sum[1][1])
        V2 = (burn -1)/(burn) * W2 + (1/burn)* B2
            
        R1 = np.sqrt(V1/W1)
        plt_R1.append(R1)
        R2 = np.sqrt(V2/W2)
        plt_R2.append(R2)

        plt_burn.append(burn)
    
        # Increase burn period, since we haven't converged
        j+=1
        burn = round(np.exp(j))
    
        # Save the current proposal distribution
        
        # Adapt the proposal distribution
        w_cov = np.multiply(W_VAR, data_cov[0])
        
        START_POINTS = [[data[0][0][-1],data[0][1][-1]], [data[1][0][-1],data[1][1][-1]], [data[2][0][-1],data[2][1][-1]]]
    plt.figure()
    plt.scatter(plt_burn, plt_R1)
    plt.scatter(plt_burn, plt_R2)
    plt.ylabel('$\hat{R}$')
    plt.xlabel('samples')
    plt.xscale('log')
    plt.show
    return(START_POINTS, w_cov)


def getSamples(mean, cov, chains, samples):
    
    START_POINTS, w_cov = getBurnIn(mean, cov, chains)
    
    print(w_cov)
    print(START_POINTS)
    data = [[],[]]
    data_mean = []
    total_samples = 0
    
    
    # Location vector in 2D parameter space
    w_prev = sp.multivariate_normal.rvs(START_POINTS[0], w_cov, 1)
    
    # Evaluate the pdf of the distribution we want to sample from
    post_prev = sp.multivariate_normal.pdf(w_prev, mean, cov)
    
    # Metropolis Hastings 'proper', just sampling from 1 chain now, because
    # we know that it has converged
    while len(data[0]) < samples:
        total_samples +=1
        w = sp.multivariate_normal.rvs(w_prev, w_cov, 1)
        post = sp.multivariate_normal.pdf(w, mean, cov)
        q_ij = sp.multivariate_normal.pdf(w_prev, w, w_cov)
        q_ji = sp.multivariate_normal.pdf(w, w_prev, w_cov)
                
        #compute acceptance ratio
        r = (post/ post_prev)* (q_ji/q_ij)
        
        #generate u from a uniform distribution
        u =  np.random.uniform()
        if u <= r:
            #accept the sample
            data[0].append(w[0])
            data[1].append(w[1])
            w_prev = w
            post_prev = post
        else:
            None
    
    xedges = np.linspace(-3.0, 7.0, 100)
    yedges = np.linspace(-3.0, 7.0, 100)
    H, xedges, yedges = np.histogram2d(data[0], data[1], bins=(xedges, yedges))
    plt.figure()
    plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.show()
    print('The percentage of accepted samples was {}%'.format(samples*100/(total_samples)))
    
    data_mean.append(np.mean(data[0]))
    data_mean.append(np.mean(data[1]))
    data_cov = np.cov(data[0], data[1])
    
# =============================================================================
#     # Write data to a file
#     with open(r"C:\Users\Ben Boys\Documents\Gyroid Beam Project\python\week 5\Metropolis hastings\mcmc.csv", 'w', newline='') as myfile:
#      wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#      data_zipped = zip(*data)
#      wr.writerow(data_zipped)
# =============================================================================
    
    print('The mean of the MCMC samples is', data_mean)
    print('The covariance of the MCMC samples is', data_cov)
    
    # Get the Kernel Density Estimate and assosciated plots
    getKernelDensityEstimate(data, mean, cov)
    
    return data

# Define amount of chains to sample from. Mean and covariance are already defined above.
chains = 3

t1_start = time.process_time()
getSamples(mean, cov, chains, 10000)
t1_stop = time.process_time()
print('elapsed time', t1_stop - t1_start)