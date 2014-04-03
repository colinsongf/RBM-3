'''
   Copyright (c) 2014, Joonhee Han.

   This is an implementation of the paper 'Training Restricted Boltzmann Machines: An Introduction' by Asja Fischer.
 
   This file is part of Restricted Boltzmann Machines.
   RBM is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.
'''

import numpy as np


class Rbm:
  def __init__(self, num_v, num_h, learning_rate):
    self.w = np.random.randn(num_h, num_v) * 0.1 #sd = 0.1
    self.b = np.random.randn(num_v) * 0.1 # bias term for visible unit
    self.c = np.random.randn(num_h) * 0.1 # bias term for hidden unit
    self.num_v = num_v
    self.num_h = num_h
    self.learning_rate = learning_rate
    
  def sigmoid(self, x):
    return 1.0 / (1.0 + np.exp(-x))

  def prob_h_given_v(self, v):
    return sigmoid( np.dot( self.w, v ) + self.c )

  def prob_v_given_h(self, h):
    return sigmoid( np.dot( self.w.T, h ) + self.b )

  def sample_v_using_gibbs_sampling(self, num_samples, sample_after=1000):
    samples_v   = np.zeros( self.num_v * num_samples ).reshape( num_samples, self.num_v )
    samples_p_v = np.zeros( num_samples )
    v = np.random.choice(2, self.num_v)

    for i in range(sample_after):
      p_h = self.prob_h_given_v( v )
      h = np.random.rand( self.num_h ) > p_h
      p_v = self.prob_v_given_h( h )
      v = np.random.rand( self.num_v ) > p_v
    
    for i in range(num_samples):
      p_h = self.prob_h_given_v( v )
      h = np.random.rand( self.num_h ) > p_h
      p_v = self.prob_v_given_h( h )
      samples_v[i,:] = np.random.rand( self.num_v ) > p_v
      samples_p_v[i] = p_v
      
    return samples_v, samples_p_v

  def expectation_of_data(self, training_samples):
    w = np.zeros( num_h * num_v ).reshape( num_h, num_v )
    p_h_avg = np.zeros( num_h )

    for i, v in enumerate( training_samples ):
      p_h = self.prob_h_given_v( v )
      p_h_avg += p_h
      w += np.dot( p_h, v.T )

    return w / len( training_samples ), p_h_avg / len( training_samples )
      
  def expectation_of_model(self, gibbs_samples):
    w = np.zeros( num_h * num_v ).reshape( num_h, num_v )
    p_h_avg = np.zeros( num_h )

    for i, v in enumerate( gibbs_samples ):
      p_h = self.prob_h_given_v( v )
      p_h_avg += p_h
      w += np.dot( p_h, v.T )
      
    return w / len( gibbs_samples ), p_h_avg / len( gibbs_samples )

  # training_samples = N x V
  def train(self, iterations, training_samples):
    num_samples = 100

    for i in range(iterations):
      # N x V
      gibbs_samples = self.sample_v_using_gibbs_sampling( num_samples )
      
      w_data_avg, p_h_data_avg = self.expectation_of_data( training_samples )
      w_model_avg, p_h_model_avg = self.expectation_of_model( gibbs_samples )

      self.w += w_data_avg - w_model_avg 
      self.b += np.mean( training_samples, axis=0 ) - np.mean( gibbs_samples, axis=0 )
      self.c += p_h_data_avg - p_h_model_avg
