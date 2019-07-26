# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:11:27 2019

@author: Ben Boys


A multivariate Metropolis hastings algorithm
"""

import pandas as pd
import scipy.stats as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Make a contour plot

#Contour fill plot

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


# Metropolis hastings


def mhmv(mean, cov):
    """ Inputs: Play god by taking a mean vector and a covariance matrix of a pre-defined Gaussian,
        Outputs: Histogram of the distribution
        Description: Uses a simple multivariate metropolis hastings algorithm to sample from a Gaussian distribution without
        sampling directly from it. This can be used to sample from more complicated distributions which are impossible to sample from.
        """

    size =  50000
    
    
    #Accepted samples is a matrix of x and y pairs
    accepted_x = []
    accepted_y = []
    
    
    w_sigma = 1.0
    w_cov = w_sigma*np.identity(2)
    
    
    # initialise location
    w0 = [0.0, 0.0]
    # Location vector in 2D parameter space
    w_prev = sp.multivariate_normal.rvs(w0, w_cov, 1)
    
    # Evaluate the pdf of the distribution we want to sample from
    post_prev = sp.multivariate_normal.pdf(w_prev, mean, cov)
    
    # Initiate the s vector
    s = 0
    # Initiate the number of rejected samples
    rejected = 0
    while s <= size:
        s += 1
        w = sp.multivariate_normal.rvs(w_prev, w_cov, 1)
        post = sp.multivariate_normal.pdf(w, mean, cov)
        
        #compute acceptance ratio
        r = (post/ post_prev)
        
        if r >= 1:
            #accept the sample
            accepted_x.append(w[0])
            accepted_y.append(w[1])
            w_prev = w
            post_prev = post
        else:
            #generate u from a uniform distribution
            u =  np.random.uniform()
            if u <= r:
                #accept the sample
                accepted_x.append(w[0])
                accepted_y.append(w[1])
                w_prev = w
                post_prev = post
            else:
                rejected += 1
        
    
    
    xedges = np.linspace(-3.0, 7.0, 100)
    yedges = np.linspace(-3.0, 7.0, 100)
    H, xedges, yedges = np.histogram2d(accepted_x, accepted_y, bins=(xedges, yedges))
    
    #fig = plt.figure(figsize=(7, 3))
    #ax = fig.add_subplot(131, title='imshow: square bins')
    plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.show()
    
    print('The percentage of rejected samples was {}%'.format(rejected*100/size))
    return None



mhmv(mean, cov)
