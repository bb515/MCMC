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

#U_VALUES =  np.multiply(5, model.getModelMean())
U_VALUES = [0.0, -0.6917008035697372, -1.379760278057291, -2.060537094380478, -2.7303899234571154, -3.38567743620502, -4.022758303542007, -4.637991196385896, -5.2277347856545, -5.788347742265639, -6.316188737137127, -6.807616441186781, -7.25898952533242, -7.666666660491861, -8.027006517582917, -8.336367767523406, -8.591109081231146, -8.787589129623953, -8.922166583619646, -8.991200114136038, -8.991200114136038, -8.922166583619646, -8.787589129623953, -8.591109081231146, -8.336367767523406, -8.027006517582917, -7.666666660491859, -7.25898952533242, -6.807616441186781, -6.316188737137126, -5.788347742265635, -5.227734785654498, -4.637991196385895, -4.022758303542006, -3.3856774362050195, -2.7303899234571127, -2.0605370943804764, -1.3797602780572895, -0.6917008035697364, 0.0]

# get the K and Sigma matrices
#K = model.getK(0.008229747049)
#SIGMA = model.getSigma(0.6)

#ERROR_MEAN = model.getErrorMean()

#delta = sp.multivariate_normal.rvs(ERROR_MEAN, K, 1)
delta = [-1.3991268260435217, -0.4356506445178837, 0.6342301338518543, 0.5261469353697341, -0.4273580509268169, -1.4383377692403694, -1.8991145555935425, -1.361662990764935, -0.3368817019748327, -0.08410515855117062, -0.43113547325652724, -0.2974860154564678, 0.29109619311075974, 0.3554045669319598, 0.32098053662522197, 1.0089803031403655, 1.21208188233744, 0.2918052384593704, -0.5928308665936727, -0.7610701908158691, -0.7178794870220119, -0.6957852530142863, -0.9665574092012075, -0.7647913929282227, 0.23230550716552423, 0.6411038205195978, 0.2233326186780903, -0.35337429596545383, -1.107727365666808, -1.3368848767786012, -0.7627195530577132, -0.15818654819061745, -0.3551377884605607, -0.8718753033995937, -0.7573598046514857, -0.5773703592227549, -0.7727605433542638, -1.3865627060855112, -1.5868123056558952, -0.8779227276877142]


#eta = sp.multivariate_normal.rvs(ERROR_MEAN, SIGMA, 1)
eta = [-0.038339474326146226, 0.7999563597665051, 1.0910267140826848, -0.34422744859423987, -0.613700547011062, 0.906207150859059, 0.36084108171753987, 0.299545955449888, -1.0127298854129059, 1.1400199484771225, -0.2023146209512375, -1.2478812334460763, -0.9683386978252362, 0.6920360285886752, 0.3304248467382131, -0.5159056014582426, 0.18976233695622066, 0.5101550008293197, 0.22190494577712336, -0.002289195184798024, 0.24625735694338607, 0.4372044721013408, -0.8522016485554791, -0.23135750801085786, -1.0162669836614269, 0.41688732764639025, -0.08510798952474972, -0.0976313565356886, 0.06002567500997098, 0.2889130127652829, -0.2645306696416941, 1.0738900416983865, 0.36499166267318317, 0.28721909701071485, 0.7049194388178616, 0.41839506425401224, 0.2995435573595405, 0.5800504834038354, -0.14346121205197368, 0.33963874951724565]

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

#Y_VALUES = U_VALUES + total_error
Y_VALUES = [-1.437466300369668, -0.32739508832111575, 0.3454965698772481, -1.8786176076049839, -3.771448521394994, -3.91780805458633, -5.56103177741801, -5.700108231700943, -6.577346373042238, -4.732432952339687, -6.949638831344892, -8.352983690089324, -7.936232030046896, -6.619226064971226, -7.375601134219482, -7.843293065841283, -7.189264861937485, -7.985628890335263, -9.293092504436196, -9.754559500136706, -9.462822244214664, -9.180747364532591, -10.60634818738064, -9.587257982170227, -9.120329244019308, -6.969015369416929, -7.5284420313385185, -7.709995177833562, -7.855318131843617, -7.364160601150444, -6.8155979649650416, -4.312031292146729, -4.628137322173272, -4.607414509930885, -3.4381178020386436, -2.8893652184258554, -2.5337540803752, -2.1862725007389656, -2.421974321277605, -0.5382839781704685]
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
    
    # Perform the burn on the first 100 values
    burn = 200
    
    data[0] = data[0][burn:]
    data[1] = data[1][burn:]
    
    print(data)
    
    NO_BINS = 100
    xstart = -1.5
    xfinish = 1.5
    ystart = -6.0
    yfinish = -3.0
    xedges = np.linspace(xstart, xfinish, NO_BINS)
    yedges = np.linspace(ystart, yfinish, NO_BINS)
    H, xedges, yedges = np.histogram2d(data[0], data[1], bins=(xedges, yedges))
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
getSamples(Y_VALUES, U_VALUES, 500000)
t1_stop = time.process_time()
print('elapsed time', t1_stop - t1_start)



