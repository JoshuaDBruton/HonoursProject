'''
Joshua Bruton
13 May 2019
Adapted from: https://github.com/snooky23/K-Sparse-AutoEncoder
'''
import numpy as np
from utils.activations import sigmoid_function as sf

class Linear:
	def __init__(self, name, num_in, num_out):
		self.name=name
		self.res=[]
		self.weights=2*np.random.random((num_in, num_out))-1
		self.biases=np.zeros(num_out)
	def get_res(self,x):
		res=sf(x.dot(self.weights)+self.biases)
		self.res=res
		return res
class Sparse:
	def __init__(self, name, num_in, num_out, k=20):
		Linear.__init__(self, name, num_in, num_out)
		self.k=k
	def get_res(self, x):
		result=sf(x.dot(self.weights)+self.biases)
		k=self.k
		if k<result.shape[1]:
			for raw in result:
				indices=np.argpartition(raw, -k)[-k:]
				mask=np.ones(raw.shape, dtype=bool)
				mask[indices]=False
				raw[mask]=0
		self.res=result
		return result
