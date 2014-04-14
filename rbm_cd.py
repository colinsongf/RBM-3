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
import warnings

#warnings.filterwarnings('error')


class RBM:
  def __init__(self, num_v, num_h, learning_rate, gaussian=False):
    scale = 1.0
    self.w = np.random.randn(num_h, num_v) * scale  #sd
    self.b = np.random.randn(num_v).reshape(num_v, 1) * scale # bias term for visible unit
    self.c = np.random.randn(num_h).reshape(num_h, 1) * scale # bias term for hidden unit
    self.num_v = num_v
    self.num_h = num_h
    self.learning_rate = learning_rate
    self.gaussian = gaussian

  def sigmoid(self, x):
    try:
        return 1.0 / (1.0 + np.exp(-x))
    except Warning as e:
        print x

  # v = V x 1
  # eq.(21)
  def prob_h_given_v(self, v):
    return self.sigmoid( np.dot( self.w, v ) + self.c )

  # h = H x 1
  # eq.(22)
  def prob_v_given_h(self, h):
    return self.sigmoid( np.dot( self.w.T, h ) + self.b )


  def sample_v_using_cd_k(self, v_0, k=1):
    v = v_0

    for _k in range(k):
      p_h = self.prob_h_given_v( v )
      h = np.asarray( np.random.rand( self.num_h ) ).reshape( self.num_h, 1) < p_h
      p_v = self.prob_v_given_h( h )
      v = np.asarray( np.random.rand( self.num_v ) ).reshape( self.num_v, 1) < p_v

    return v 


  # training_samples = V x N
  def train(self, iterations, training_samples):

    for i in range(iterations):
      print 'iteration============', i

      for i, v in enumerate( training_samples ):
        v_0 = v.reshape( self.num_v, 1 )
        v_k = self.sample_v_using_cd_k( v_0 )
        p_h_0 = self.prob_h_given_v( v_0 )
        p_h_k = self.prob_h_given_v( v_k )

        self.w += self.learning_rate * ( np.dot( p_h_0, v_0.T ) - np.dot( p_h_k, v_k.T ) )
        self.b += self.learning_rate * ( v_0 - v_k )
        self.c += self.learning_rate * ( p_h_0 - p_h_k )


  def get_hidden(self, v):
    v = v.reshape( self.num_v, 1 )
    p_h = self.prob_h_given_v( v )
    h = np.asarray( np.random.rand( self.num_h ) ).reshape( self.num_h, 1) < p_h
    return h


  def get_visible(self, h):
    h = h.reshape( self.num_h, 1 )
    p_v = self.prob_v_given_h( h )

    if self.gaussian:
        v = p_v
    else:
        v = np.asarray( np.random.rand( self.num_v ) ).reshape( self.num_v, 1) < p_v
    return v 


if __name__ == '__main__':
  rbm = RBM(num_v = 6, num_h = 2, learning_rate = 0.1, gaussian=True)
  training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0],[0,0,1,1,0,0],[0,0,1,1,1,0], [0,0,1,1,0,1]])
  rbm.train( 1000, training_data )
 
  test_data = np.array([[0,0,0,1,1,0]]) 
  h= rbm.get_hidden( test_data )
  print h

  test_data = np.array([[1,1,1,0,0,0]]) 
  h= rbm.get_hidden( test_data )
  print h 
