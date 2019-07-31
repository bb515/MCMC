# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:11:27 2019

@author: Ben Boys


A multivariate Metropolis hastings algorithm

Have set myself a task: How accuract can you get it to converge to
Gaussian distribution parameters given that you can sample only 10,000 times.

Starta

"""

import scipy.stats as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

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


# Metropolis hastings


def mhmv(mean, cov, chains):
    """ Inputs: Mean vector and covariance of a Gaussian to sample from using metropolis hastings
                Number of chains used to calculate chain variances (convergence).
        Outputs: Histogram of the distribution
        Description: Uses a simple multivariate metropolis hastings algorithm to sample from a Gaussian distribution without
        sampling directly from it. This can be used to sample from more complicated distributions which are impossible to sample from.
        """
    samples =  50000
    chains = 3
    
    #Accepted samples is a matrix of x and y pairs
    accepted_x = [[],[],[]]
    accepted_y = [[],[],[]]
    
    # R values for the parameters
    R1 = [[],[]]
    w_var = 0.7
    w_cov= np.multiply(w_var, cov)
    
    # Define the starting points of the chains
    start_points = [[0.0, -1.0], [1.0, 3.0], [4.0, -1.0]]

    
    for i in range(chains):
        
        print('sampling Markov Chain %d...'%i)
        # Location vector in 2D parameter space
        w_prev = sp.multivariate_normal.rvs(start_points[i], w_cov, 1)
    
        # Evaluate the pdf of the distribution we want to sample from
        post_prev = sp.multivariate_normal.pdf(w_prev, mean, cov)
    
        # Initiate the s vector for number of samples taken
        s = 0
    
        while s <= samples:
            s += 1
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
                accepted_x[i].append(w[0])
                accepted_y[i].append(w[1])
                w_prev = w
                post_prev = post
            else:
                None
        
    # The sample burn length to test R has converged yet
    samples_burn =  [2,3,4,5,6,7,8,9, 10, 15, 20, 30, 40, 60, 80, 100, 200, 300, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000]
    
    # For all parameters, compute within and between chain variances, W and B
    # x1, parameter on the x-axis
    for i in range(len(samples_burn)):
        #Mean and covariance values of the chains
        mu_c = []
        cov_c = []
        accepted_x_burn = [accepted_x[0][:samples_burn[i]], accepted_x[1][:samples_burn[i]], accepted_x[2][:samples_burn[i]]]
        accepted_y_burn = [accepted_y[0][:samples_burn[i]], accepted_y[1][:samples_burn[i]], accepted_y[2][:samples_burn[i]]]
        for j in range(chains):  
            #mean of chain for current number (value) of samples
            mu_c.append([np.mean(accepted_x_burn[j]), np.mean(accepted_y_burn[j])])
        
            #cov of chain
            cov_c.append(np.cov(accepted_x_burn[j], accepted_y_burn[j]))
        #mean squared error
        me = np.subtract(mu_c, [mean, mean, mean])
        mse = np.power(me, 2)
        cov_c = np.sum(cov_c, axis=0)
        
        # Calculate each R value for each parameter
        for j in range(2):
        
            B = np.multiply(samples/(chains-1), np.sum(mse[:,j]))
            W = np.multiply(1/chains, cov_c[j][j])
    
            V = (samples -1)/(samples) * W + (1/samples)* B
            R = np.sqrt(V/W)
        
            R1[j].append(R)

        if R1[0][i] <1.01 and R1[1][i] < 1.01:
            burn = samples_burn[i]
            print('Convergence on R<1.01 reached, burning first %d samples'%burn)
            break
        else:
            assert i < (len(samples_burn)-1), "Not enough samples to converge!"
            pass

    scatter_x_values = (samples_burn[:len(R1[0])])
    plt.figure()
    plt.plot(scatter_x_values, R1[0],c='g', label= 'x1')
    plt.plot(scatter_x_values, R1[1],c='k', label= 'x2')
    
    horiz_line_data = np.array([1.01 for i in range(len(scatter_x_values))])
    plt.plot(scatter_x_values, horiz_line_data, 'r--') 
    plt.xlabel('samples')
    plt.ylabel('$\hat{R}$')
    plt.xscale('log')
    plt.grid()
    plt.title('Burn in')
    
    
    # Combine data of all chains
    accepted_x_np = np.append(np.append(accepted_x[0][burn:], accepted_x[1][burn:]), accepted_x[2][burn:])
    accepted_y_np = np.append(np.append(accepted_y[0][burn:], accepted_y[1][burn:]), accepted_y[2][burn:])
    
    xedges = np.linspace(-3.0, 7.0, 100)
    yedges = np.linspace(-3.0, 7.0, 100)
    H, xedges, yedges = np.histogram2d(accepted_x_np, accepted_y_np, bins=(xedges, yedges))
    plt.figure()
    plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.show()
    print('The percentage of accepted samples was {}%'.format(len(accepted_x_np)*100/(samples*chains)))
    
    return None

t1_start = time.process_time()
mhmv(mean, cov, 3)
t1_stop = time.process_time()



print('time', t1_stop - t1_start)


