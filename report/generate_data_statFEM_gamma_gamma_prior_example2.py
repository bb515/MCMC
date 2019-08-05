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
from gyroid import model
from mpl_toolkits.mplot3d import axes3d


# Define sensor positions
X_VALUES = [0.0, 5.641025641025641, 11.282051282051283, 16.923076923076923, 22.564102564102566, 28.205128205128208, 33.84615384615385, 39.48717948717949, 45.12820512820513, 50.769230769230774, 56.410256410256416, 62.05128205128206, 67.6923076923077, 73.33333333333334, 78.97435897435898, 84.61538461538463, 90.25641025641026, 95.8974358974359, 101.53846153846155, 107.17948717948718, 112.82051282051283, 118.46153846153847, 124.10256410256412, 129.74358974358975, 135.3846153846154, 141.02564102564102, 146.66666666666669, 152.30769230769232, 157.94871794871796, 163.5897435897436, 169.23076923076925, 174.8717948717949, 180.51282051282053, 186.15384615384616, 191.7948717948718, 197.43589743589746, 203.0769230769231, 208.71794871794873, 214.35897435897436, 220.0]


# Using the model class, get the 'True' deflection values, U
model =  model(X_VALUES)

U_VALUES =  [0.0, -0.6917008035697372, -1.379760278057291, -2.060537094380478, -2.7303899234571154, -3.38567743620502, -4.022758303542007, -4.637991196385896, -5.2277347856545, -5.788347742265639, -6.316188737137127, -6.807616441186781, -7.25898952533242, -7.666666660491861, -8.027006517582917, -8.336367767523406, -8.591109081231146, -8.787589129623953, -8.922166583619646, -8.991200114136038, -8.991200114136038, -8.922166583619646, -8.787589129623953, -8.591109081231146, -8.336367767523406, -8.027006517582917, -7.666666660491859, -7.25898952533242, -6.807616441186781, -6.316188737137126, -5.788347742265635, -5.227734785654498, -4.637991196385895, -4.022758303542006, -3.3856774362050195, -2.7303899234571127, -2.0605370943804764, -1.3797602780572895, -0.6917008035697364, 0.0] 

# get the K and Sigma matrices
#K = model.getK(0.04)
#print(K)

#SIGMA = model.getSigma(0.6)
#print(SIGMA)

# Sample delta
# This is a multivariate distribution, but how?

ERROR_MEAN = model.getErrorMean()
#print(DELTA_MEAN)

delta = [0.4325753742751227, 0.8597690653620631, 0.10986340450489712, -0.20780206652317548, 0.675951077361762, 1.061675839732957, -0.19111922774940399, 0.23165419229777073, 2.12808084278264, 0.05927382885870158, -0.9341515390669421, 0.6241475769931719, 1.3865716182322658, 0.5720426726711803, 1.2652670663462147, 0.3896290742833012, 0.6212797188244559, 0.8959812797745113, 0.37342365651220344, 1.1016045570147661, 0.6472621195239258, 0.27038744706185014, -0.3329744612047532, 0.5571055337176949, -0.11749410147473882, -0.7297011266665977, -0.02336818528131035, 1.0325433361806897, -0.0766858759749875, 0.4630551676872521, -0.13921414586478253, 0.7247964087883284, 1.239785779830234, -0.9602512010738592, -0.5785584603284616, 1.4458347095264155, 0.7173180166961813, 0.18603376769369467, 0.3095597540970292, -1.2747436895509352]



eta = [1.1375820329515813, -1.3042833222615762, 0.45544680598217796, -0.17856922921815913, 0.07687474968287308, -0.07097542940544153, 0.4099084270006645, -0.11603566813377994, -0.5129402608517263, 0.23399785207789428, -0.6784919903026088, -1.246479249800404, -0.15304780219848835, -0.11751551835730248, 0.2119241908800703, -0.03130125316826417, 0.5422974853342402, 1.3656088603002834, -1.1602788558128982, 1.1386467171859593, 1.0190861496388035, -1.2220102837147848, -0.05439937212820866, -0.4970302829927861, -0.3674016638702154, -0.4482809378648819, 0.40692340918176834, -0.2833207422774037, -0.7867679578007589, -0.19821985851123575, -0.17858210258849305, 0.0952823945567891, -0.7829900022164352, -1.5139380416343622, -0.3717892222779043, -0.3846742391861939, -0.1681103332871662, 0.0813107036261154, -0.33507046690708864, 0.0680510655406832]


plt.figure()
plt.scatter(X_VALUES, delta)
plt.title('$\delta$')
plt.show


plt.figure()
plt.scatter(X_VALUES, eta)
plt.title('$\epsilon$')
plt.show

total_error = np.add(delta, eta)

plt.figure()
plt.scatter(X_VALUES, total_error)
plt.title('$\delta$ + $\epsilon$')
plt.show

Y_VALUES = U_VALUES + total_error
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
    x = np.linspace(0.0, 1.5, 100)
    y= np.linspace(0.0, 1.5, 100)
    
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
    prior_prev = sp.gamma.pdf(w_prev[0], alpha_sigma)*sp.gamma.pdf(w_prev[1], alpha_lambda, scale= 0.01)
    
    data = [[],[]]
    total_samples = 0
    
    
    # Metropolis Hastings 'proper', just sampling from 1 chain now, because
    # we assume the chain has converged
    while total_samples < samples:
        total_samples+=1
        w = sp.multivariate_normal.rvs(w_prev, w_cov, 1)
        
        # Multiply two single variate prior distributions
        prior = sp.gamma.pdf(w[0], alpha_sigma)*sp.gamma.pdf(w[1], alpha_lambda, scale= 0.01)
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
    
    
    
# =============================================================================
#     data_mean.append(np.mean(data[0]))
#     data_mean.append(np.mean(data[1]))
#     data_cov = np.cov(data[0], data[1])
#     print('The mean of the MCMC samples is', data_mean)
#     print('The covariance of the MCMC samples is', data_cov)
#     plt.figure()
#     plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
#     plt.xlabel('sigma')
#     plt.ylabel('mu')
#     plt.show()
# =============================================================================
    
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
getSamples(Y_VALUES, U_VALUES, 200000)
t1_stop = time.process_time()
print('elapsed time', t1_stop - t1_start)



