import numpy as np
import matplotlib.pyplot as plt
from math import isclose
import random
#from sklearn.decomposition import MiniBatchDictionaryLearning as dl #for in-buil version of MBODL
from online_dictionary_learning import odl			# Own implementation of dictionary learning, presently uses OMP exclusively (targeted sparsity)
from comet_ml import Experiment #For Comet.ml
from cluster_meths import knn
from utils import data_processing as dp

# some_param = "some value"
# experiment.log_parameter("param name", some_param)

def accuracy(preds, targets):
	return (sum(1 for a, b in zip(preds, targets) if a==b)/preds.size)

data_set='Indian Pines'				# Will be used later to specify training set
atom_nums=[220, 240, 260, 280]									# The number of atoms in the dictionary
ks=[7]														# The number of nearest neighbors for KNN
odl_iters=[10000]									# The maximum number of iterations for ODL
sparse=[20, 40, 60, 80]										# The sparsity targeted by OMP
verbose=True									# Output or not
shuffle=True									# Shuffle data or not
comet=True										# Save experiment to Comet.ml
test_number=len(atom_nums)*len(ks)*len(odl_iters)*len(sparse)

for odl_iter in odl_iters:
	for atom_num in atom_nums:
		for sparsity in sparse:
			if verbose:
				print('Online Dictionary Learning:')

			# Initialise the data
			data=dp.extract(shuffle=shuffle)

			# Normalise
			signals=np.array([l.values for l in data], dtype=np.float64)
			labels=np.array([l.label for l in data], dtype=np.float64)
			signals-=np.mean(signals)
			signals/=np.std(signals)

			# Initialise the dictionary
			dic=odl.Dict(num_coms=atom_num)

			# Trains the dictionary
			dic.fit(signals, reg_term=sparsity, max_iter=odl_iter, showDictionary=False, showRecExample=False, prog=verbose)

			if verbose:
				print('\nExtracting coefficients:\n', end = '\r')

			# Extract coefficients
			coefs=dic.getCoefs()

			if verbose:
				rep_error = dic.showError(recal=False)

			# Split data
			train_signals, test_signals=dp.split(signals, portion=0.9)
			train_coefs, test_coefs=dp.split(coefs, portion=0.9)
			train_labels, test_labels=dp.split(labels, portion=0.9)

			for k in ks:
				if comet:
					# Report any information you need by:
					hyper_params = {"Data set": data_set, "k": k, "Shuffle": shuffle, "Max. Iterations": odl_iter}
					experiment.log_parameters(hyper_params)
					experiment.log_metric("Sparsity", sparsity, step=1)
					experiment.log_metric("Number of components", atom_num, step=1)

				if verbose:
					print('\nRunning KNN:')

				# Initialising for KNN
				numtest=500#test_labels.shape[0]
				model=knn.knn(k=k)
				pred_sigs=np.zeros(numtest)
				pred_coefs=np.zeros(numtest)

				# Running KNN for testing data
				for i in range(numtest):
					pred_sigs[i]=model.fit(train_signals, test_signals[i], train_labels)
					pred_coefs[i]=model.fit(train_coefs, test_coefs[i], train_labels)
					if verbose:	
						if i+1<numtest:
							print('[' + str(i+1) + '] ' + str(np.round((((i+1)/numtest)*100),2)) + '%', end='\r')
						else:
							print('[' + str(i+1) + '] ' + str(np.round((((i+1)/numtest)*100),2)) + '%', end='\n')

				# Fetching and printing accuracy
				acc_sigs=accuracy(pred_sigs, test_labels[:numtest])
				acc_coefs=accuracy(pred_coefs, test_labels[:numtest])

				if verbose:
					print("\nSignal Accuracy: " + str(acc_sigs) + "\nCoefficient Accuracy: " + str(acc_coefs))

				if comet:
					experiment.log_metric("Representation Error", rep_error, step=1)
					experiment.log_metric("Signal Accuracy", acc_sigs, step=None)
					experiment.log_metric("Coefficient Accuracy", acc_coefs, step=None)
