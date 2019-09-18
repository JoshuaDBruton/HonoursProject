'''
Joshua Bruton
Last updated: 23 May 2019
Generalised Implementation (along with other files) of Overcomplete k-Sparse Autoencoder
Adapted from:
	https://github.com/snooky23/K-Sparse-AutoEncoder, and
	"The Appropriateness of k-Sparse Autoencoders..." by Pushkar Bhatkoti
'''
from utils.layers import *
from utils.activations import sigmoid_function as sf
import random


# Loss used, others are acceptable (representation error may be preferred)
def subtract_error(outputs, targets):
	res = outputs - targets
	return res

# Class for k-Sparse autoencoder, depends on layer class
class KSparseAutoencoder:
	'This class creates and trains k-Sparse Autoencoder and has some other functionality relevant to the project'
	def __init__(self, inputSize, num_hidden, k=20):
		self.inputSize=inputSize
		self.num_hidden=num_hidden
		self.k=k
		self.data=None
		self.eta=None
		self.epochs=None
		self.batch_size=None
		self.layers=[Sparse(name='hidden', num_in=inputSize, num_out=num_hidden, k=k), Linear(name='output', num_in=num_hidden, num_out=inputSize)]

	# Draws random signal from data
	def drawRand(self):
		rand = random.randint(0,self.data.shape[0]-1)
		return self.data[rand]

	# Initials W_1 of AE to random signals from data
	def initW(self):
		for i,_ in enumerate(self.layers[1].weights):
			self.layers[1].weights[i,:] = self.drawRand()

	# Prints the architecture of the network
	def print_arch(self):
		print('======================================')
		print('=========== Architecture =============')
		print('======================================')
		for layer in self.layers:
			print('%s: %s' % (layer.name, layer.weights.shape))
		print('======================================')

	# Prints hyper-parameters of the network
	def print_parameters(self):
		print('======================================')
		print('============ Parameters ==============')
		print('======================================')
		print('Learning rate: %s\nEpochs: %s\nBatch Size: %s' % (self.eta, self.epochs, self.batch_size))
		print('======================================')

	# Fit the model, called after class initialisation
	def fit(self, x, y, eta=0.01, epochs=10000, batch_size=256, evince=False):
		self.eta=eta # learning rate
		self.epochs=epochs
		self.batch_size=batch_size
		self.data=x
		self.initW()
		if evince==True:
			self.print_arch()
			self.print_parameters()
			print('\nStart Training')

		for k in range(epochs):
			indices=np.random.randint(x.shape[0], size=batch_size)
			batch_x=x[indices]
			batch_y=y[indices]
			res=self.feed_forward(batch_x)
			error=subtract_error(res[-1],batch_y)

			if (k+1)%500==0:
				loss=np.mean(np.abs(error))
				message='epochs: {0}, loss: {1:4f}'.format((k+1), loss)
				if evince==True:
					print(message)

			deltas=self.bp(res, error)
			self.update(res, deltas, eta)

		print('Training complete.')

	# Updates the weights
	def update(self, res, deltas, eta):
		for i in range(len(self.layers)):
			layer=self.layers[i]
			layer_result=res[i]
			delta=deltas[i]
			layer.weights-=eta*layer_result.T.dot(delta)

	# Backprop
	def bp(self, results, error):
		last=self.layers[-1]
		deltas=[error*sf(results[-1], derivative=True)]

		for i in range(len(results)-2, 0, -1):
			layer=self.layers[i]			
			delta=deltas[-1].dot(layer.weights.T)*sf(results[i], derivative=True)
			deltas.append(delta)

		deltas.reverse()
		return deltas

	# Requires vector of input, returns the result after full forward pass through all 3 layers
	def feed_forward(self, x):
		results=[x]
		for i in range(len(self.layers)):
			res=self.layers[i].get_res(results[i])
			results.append(res)
		return results

	# This is special, just returns output of hiddel_layer (approximates, maybe, sparse coefficients)
	def predict(self, x):
		return self.feed_forward(x)[-1]

	# Also special, extracts W_1, an approximate dictionary, from the AE
	def extract_dict(self):
		return self.layers[1].weights
