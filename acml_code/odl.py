'''
Joshua Bruton
Last updated: 23 May 2019
Implementation of Online Dictionary Learning
See: J Mairal "Online Dictionary Learning for Sparse Coding
'''
import numpy as np
import random
import matplotlib.pyplot as plt
import omp

# Dictionary update step from ODL, taken from J Mairal "Online Dictionary Learning for Sparse Coding"
def updateDict(D, A, B):	
	D = D.copy()
	DA  = D.dot(A)
	count = 0
	for j in range(D.shape[1]):
		u_j = (B[:, j] - np.matmul(D, A[:, j])) / A[j, j] + D[:, j]
		D[:, j] = u_j/max([1, np.linalg.norm(u_j)]) 
	return D

class Dict:
	'This class creates, trains and outputs dictionaries with the online dictionary learning algorithm'
	def __init__(self, num_coms):
		self.num_coms = num_coms
		self.signals = None
		self.atoms = None
		self.coefs = None
		self.regTerm = None
		self.max_iter = None
	
	# Outputs a the learnt dictionary
	def showDict(self):
		self.atoms=self.atoms.transpose()
		split=int(np.sqrt(self.atoms.shape[1]))
		new_tiles=np.zeros(((self.num_coms,split,split)))
		for i in range(self.num_coms):
			new_tiles[i]=np.reshape(self.atoms[i], (split,split))
		f, ax=plt.subplots(self.num_coms//split,split)
		count=0
		for i in range(self.num_coms//split):
			for j in range(split):
				ax[i][j].imshow(new_tiles[count],cmap="gray")
				ax[i][j].axis("off")
				count+=1
		plt.show()
		self.atoms=self.atoms.transpose()

	# Returns the dictionary as array of atoms, the most recently saved one
	def getAtoms(self):
		return self.atoms

	# SETS and returns coeficients
	def getCoefs(self):
		self.coefs = np.zeros((self.signals.shape[0],self.num_coms))
		for i in range(self.signals.shape[0]):
			self.coefs[i] = omp.omp(self.atoms, self.signals[i], self.regTerm)
		return self.coefs

	# Shows the current error in the sparse approximation
	def showError(self):
		self.getCoefs()
		res = self.signals - (self.coefs.dot(self.atoms.T))
		errors = np.linalg.norm(res, axis=1)**2
		overall_error = np.sum(errors)
		print('Representation Error: ' + str(overall_error), end='\r')

	# Draws a random signal from the data
	def drawRand(self):
		rand = random.randint(0,self.signals.shape[0]-1)
		return self.signals[rand]

	# Gets a single coefficient (from OMP implementation)
	def get_co(self, signal):
		return omp.omp(self.atoms, signal, self.regTerm)

	# Initialises dictionary to random signals from data
	def initialDict(self):
		self.atoms = np.zeros((self.signals.shape[1], self.num_coms))
		for i in range(self.num_coms):
			self.atoms[:,i] = self.drawRand()

	# Prints the progress to terminal (nicely)
	def update_progress(self, curr_it):
		print('[{0}] {1}%'.format((curr_it), np.rint((curr_it/self.max_iter)*100)), end='\r')

	# Trains
	def fit(self, signals, reg_term=100, max_iter=100, showRepError=False,showDictionary=False, showRecExample=False, prog=False):
		# Initialisation
		self.orig = signals
		self.regTerm=reg_term
		self.signals = signals
		self.initialDict()
		self.max_iter = max_iter

		if showRepError == True:
			self.showError()

		signal = self.drawRand()
		A = np.zeros((self.num_coms, self.num_coms))
		B = np.zeros((self.signals.shape[1], self.num_coms))
		for t in range(1,max_iter):
			signal = self.drawRand()
			alpha = self.get_co(signal)
			A += (alpha.dot(alpha.T))
			B += (signal[:, None]*alpha[None,:])
			self.atoms = updateDict(self.atoms, A, B)
			if prog==True:
				self.update_progress(t)

		if showRepError == True:
			print('\nDone Training.')
			self.showError()

		if showRecExample == True:
			alpha = omp.omp(self.atoms, signal, reg_term)
			plt.plot(signal, '-g', label='Original')
			plt.plot((self.atoms.dot(alpha)), ':b', label='Reconstruction')
			plt.legend()
			plt.show()

		if showDictionary==True:
			self.showDict()
