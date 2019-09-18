'''
Joshua Bruton
Last updated: 23 May 2019
Sigmoid Function
https://github.com/snooky23/K-Sparse-AutoEncoder
'''
import numpy as np

def sigmoid_function(signal, derivative=False):
	if derivative:
		return np.multiply(signal, 1.0 - signal)
	else:
		return 1.0 / (1.0 + np.exp(-signal))
