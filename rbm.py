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


class RBM:
  def __init__(self, num_v, num_h, learning_rate):
    self.w = np.random.randn(num_h, num_v) * 0.1 #sd = 0.1
    self.b = np.random.randn(num_v).reshape(num_v, 1) * 0.1 # bias term for visible unit
    self.c = np.random.randn(num_h).reshape(num_h, 1) * 0.1 # bias term for hidden unit
    self.num_v = num_v
    self.num_h = num_h
    self.learning_rate = learning_rate
    
  def sigmoid(self, x):
    return 1.0 / (1.0 + np.exp(-x))

  # v = num_v x 1
  def prob_h_given_v(self, v):
    return self.sigmoid( np.dot( self.w, v ) + self.c )

  # h = num_h x 1
  def prob_v_given_h(self, h):
    return self.sigmoid( np.dot( self.w.T, h ) + self.b )

  # samples_v = num_v x num_samples
  def sample_v_using_gibbs_sampling(self, num_samples, sample_after=500):
    samples_v  = np.zeros( self.num_v * num_samples ).reshape( self.num_v, num_samples )
    v = np.random.choice(2, self.num_v).reshape( self.num_v, 1 )

    for i in range(sample_after):
      p_h = self.prob_h_given_v( v )
      h = np.asarray( np.random.rand( self.num_h ) ).reshape( self.num_h, 1) < p_h
      p_v = self.prob_v_given_h( h )
      v = np.asarray( np.random.rand( self.num_v ) ).reshape( self.num_v, 1) < p_v

    for i in range(num_samples):
      p_h = self.prob_h_given_v( v )
      h = np.asarray( np.random.rand( self.num_h ) ).reshape( self.num_h, 1) < p_h
      p_v = self.prob_v_given_h( h )
      v = np.asarray( np.random.rand( self.num_v ) ).reshape( self.num_v, 1) < p_v
      samples_v[:, i] = v[:,0]
      
    return samples_v

  # training_samples = V x N
  def expectation_of_data(self, training_samples):
    w = np.zeros( self.num_h * self.num_v ).reshape( self.num_h, self.num_v )
    p_h_avg = np.zeros( self.num_h ).reshape( self.num_h, 1 )

    for i, v in enumerate( training_samples.T ):
      v = v.reshape( self.num_v, 1)
      p_h = self.prob_h_given_v( v )
      p_h_avg += p_h
      w += np.dot( p_h, v.T )

    return w / training_samples.shape[1], p_h_avg / training_samples.shape[1]
      
  # gibbs_samples = V x N
  def expectation_of_model(self, gibbs_samples):
    w = np.zeros( self.num_h * self.num_v ).reshape( self.num_h, self.num_v )
    p_h_avg = np.zeros( self.num_h ).reshape( self.num_h, 1)

    for i, v in enumerate( gibbs_samples.T ):
      v = v.reshape( self.num_v, 1 )
      p_h = self.prob_h_given_v( v )
      p_h_avg += p_h
      w += np.dot( p_h, v.T )

    return w / gibbs_samples.shape[1], p_h_avg / gibbs_samples.shape[1]

  # training_samples = V x N
  def train(self, iterations, training_samples):
    num_samples = 50

    for i in range(iterations):
      print 'iteration============', i
      gibbs_samples = self.sample_v_using_gibbs_sampling( num_samples )
      w_data_avg, p_h_data_avg = self.expectation_of_data( training_samples )
      w_model_avg, p_h_model_avg = self.expectation_of_model( gibbs_samples )

      self.w += self.learning_rate * (w_data_avg - w_model_avg)
      self.b += self.learning_rate * ((np.mean( training_samples, axis=1 ) - np.mean( gibbs_samples, axis=1 )).reshape(self.num_v, 1))
      self.c += self.learning_rate * (p_h_data_avg - p_h_model_avg)


  def test(self, v):
    v = v.reshape( self.num_v, 1 )
    p_h = self.prob_h_given_v( v )
    h = np.asarray( np.random.rand( self.num_h ) ).reshape( self.num_h, 1) > p_h
    return p_h, h
    

if __name__ == '__main__':
  rbm = RBM(num_v = 6, num_h = 2, learning_rate = 0.1)
  training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0],[0,0,1,1,0,0],[0,0,1,1,1,0]]).T
  rbm.train( 1000, training_data )

  test_data = np.array([[0,0,0,1,1,0]]) 
  p_h, h= rbm.test( test_data )
  print p_h, h 

  test_data = np.array([[1,1,1,0,0,0]]) 
  p_h, h= rbm.test( test_data )
  print p_h, h 
