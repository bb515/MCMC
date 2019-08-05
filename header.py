# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 19:27:45 2019

@author: Ben Boys


Models class for creating the models, getting deflections
"""
import numpy as np

def x():
    #X values of the data
    x_values = [28.9269,
                44.1511,
                59.3753,
                74.5995,
                89.8237,
                105.0479,
                120.2721,
                135.4963,
                150.7205,
                165.9447,
                181.1689,
                196.3931]
    return(x_values)

class model:
    
    def __init__(self, x_values):
        self.x_values = x_values
        pass
    
    # define parameters
    # Boundary conditions and shared geometry of the models
    #BEAM_VOLUME = 32000.0 #Volume of the test gyroid beam in mm^3
    BEAM_LENGTH = 220.0 #Length between the two supports in mm^3
    YOUNGS_MODULUS = 2.30*10**3 #Youngs modulus of PLA in N/mm^2
    
    # Model 1 - I Beam - dimensions in mm
    MODEL1_DEPTH = 40.0
    MODEL1_BREADTH = 52.73
    MODEL1_THICKNESS = 1.0
    
    # Model 2 - Solid rectangular section - dimensions in mm
    MODEL2_BREADTH= 12.06
    MODEL2_DEPTH= 12.06
    
    #Model 3 - Cylinder - dimensions in mm
    MODEL3_RADIUS = 6.804
    
    #Model 4 - Fabricated model to try and match our deflections
    
    
    # Second moments of area
    
    MODEL1_I = (MODEL1_DEPTH**3*MODEL1_THICKNESS)/12 + (MODEL1_BREADTH*MODEL1_THICKNESS*MODEL1_DEPTH**2)/2
    
    MODEL2_I = (MODEL2_BREADTH*MODEL2_DEPTH**3)/12
    
    MODEL3_I = (np.pi/4)*(MODEL3_RADIUS**4)
    
    MODEL4_I = 13395.7327
    
    MODEL5_I = 19561.11
    
    # Create dictionaries for the constants
    
    models = {1: {'SECOND_MOMENT' : MODEL1_I, 'YOUNGS_MODULUS' : YOUNGS_MODULUS, 'BEAM_LENGTH' : BEAM_LENGTH},
              2: {'SECOND_MOMENT' : MODEL2_I, 'YOUNGS_MODULUS' : YOUNGS_MODULUS, 'BEAM_LENGTH' : BEAM_LENGTH},
              3: {'SECOND_MOMENT' : MODEL3_I, 'YOUNGS_MODULUS' : YOUNGS_MODULUS, 'BEAM_LENGTH' : BEAM_LENGTH},
              4: {'SECOND_MOMENT' : MODEL4_I, 'YOUNGS_MODULUS' : YOUNGS_MODULUS, 'BEAM_LENGTH' : BEAM_LENGTH},
              5: {'SECOND_MOMENT' : MODEL5_I, 'YOUNGS_MODULUS' : YOUNGS_MODULUS, 'BEAM_LENGTH' : BEAM_LENGTH},
              }
    
    def getDeflection(self, x, models, model_number, load):
        """Inputs: 
    
        Outputs:
            given the model and distance from left hand support,
        will return a vertical deflection, upwards positive, using Euler-Bernolli beam theory."""
    
        I = models[model_number]['SECOND_MOMENT']
        E = models[model_number]['YOUNGS_MODULUS']
        L = models[model_number]['BEAM_LENGTH']
        W = load
        
        if 0.0 <= x < L/2:
            X = x
        elif L/2 <= x <= L:
            X = L - x
        else:
            raise Exception('x should be between 0 and the length of the beam')
    
        bending_stiffness = (I*E)
        cubic = (W*X*L**2)/(-16.0) + (W*X**3)/12.0
        deflection = (1/bending_stiffness)*cubic
        return(deflection)
    
    
    def getModelMean(self, models=models, model_number=4, weight_in_newtons=250):
        
        # fill model mean vector
        model_data = []
        for x in self.x_values:
            model_data.append(self.getDeflection(x, models, model_number, weight_in_newtons))
        return(model_data)
        
    def getK(self, lambd):
        """ Inputs: A length scale, lambd and a vector of x values, x
        Return: K matrix
        Description: The K matrix describes the covariance of the error
        """
        K = np.empty([len(self.x_values), len(self.x_values)])
    
        for i in range(len(self.x_values)):
            for j in range(len(self.x_values)):
                K[i][j] = np.exp(-lambd*pow((self.x_values[i]-self.x_values[j]), 2))
        return K
    
    def getK2(self, zeta):
        """ Inputs: A length scale, lambd and a vector of x values, x
        Return: K matrix
        Description: The K matrix describes the covariance of the error
        """
        K = np.empty([len(self.x_values), len(self.x_values)])
    
        for i in range(len(self.x_values)):
            for j in range(len(self.x_values)):
                K[i][j] = np.exp(-np.exp(zeta)*pow((self.x_values[i]-self.x_values[j]), 2))
        return K
    
    def getSigma(self, sigma):
        """ Inputs: A sigma, and a vector x with sensor position values.
        Returns: An isotropic covariance matrix with dimensions len(x), len(x)
        """ 
        SIGMA = pow(sigma, 2)*np.identity(len(self.x_values))
        return(SIGMA)
    def getErrorMean(self):
        """ Inputs: A sigma, and a vector x with sensor position values.
        Returns: An isotropic covariance matrix with dimensions len(x), len(x)
        """ 
        ERROR = np.zeros(len(self.x_values))
        return(ERROR)
    


def u(weight_in_newtons):
    """Inputs: 
    
        Outputs:
            """
    FEM = {200: {'data': [-0.622404969,
                          -0.91947751,
                          -1.179565705,
                          -1.396546797,
                          -1.586007955,
                          -1.683644721,
                          -1.684520116,
                          -1.486430748,
                          -1.368146308,
                          -1.104878184,
                          -0.893170393,
                          -0.598540144,]},
        250: {'data' : [-0.778006226,
                        -1.14934691,
                        -1.474457159,
                        -1.745683529,
                        -1.982509981,
                        -2.104555939,
                        -2.105650183,
                        -1.85803847,
                        -1.710182916,
                        -1.381097756,
                        -1.116463012,
                        -0.748175194]},
        300: {'data' : [-0.933607454,
                      -1.379216265,
                      -1.769348557,
                      -2.094820196,
                      -2.379011933,
                      -2.52546708,
                      -2.526780173,
                      -2.229646122,
                      -2.052219461,
                      -1.657317276,
                      -1.339755589,
                      -0.897810216]}
        }
    return(FEM[weight_in_newtons]['data'])



def y(weight_in_newtons):
    """Inputs: 
    
        Outputs:
            """
    data = {200: {'data' : [-0.623,
                            -0.6505,
                            -1.0025,
                            -1.163,
                            -1.2565,
                            -1.423,
                            -1.344,
                            -1.198,
                            -1.3435,
                            -1.1115,
                            -0.7195,
                            -0.555]},
              250: {'data' : [-0.761,
                              -0.985,
                              -1.3025,
                              -1.641,
                              -1.777,
                              -1.823,
                              -1.8195,
                              -1.5925,
                              -1.419,
                              -1.266,
                              -0.908,
                              -0.715]},
              300: {'data' : [-0.922,
                              -1.2715,
                              -1.5915,
                              -1.818,
                              -2.0725,
                              -2.3365,
                              -2.138,
                              -1.943,
                              -1.762,
                              -1.474,
                              -1.162,
                              -0.796]}}
        
        
    return(data[weight_in_newtons]['data'])