import numpy as np
from online_dictionary_learning import odl			# Own implementation of dictionary learning, presently uses OMP exclusively (targeted sparsity)
from comet_ml import Experiment 								#	For Comet.ml
from utils import data_processing as dp					# For extracting HSI data and stuff

data_sets=['Pavia']#['Salinas','Indian_Pines', 'Pavia']						# Data set to use
odl_iters_all=[[10000]]#[[5000],[5000],[5000]]									# The maximum number of iterations for ODL
atom_nums_all=[[120]]#[[220, 306, 408, 1000],[220, 300, 400, 1000],[120, 160, 206, 600]]										# Number of atoms in the data
sparsities_all=[[2]]#[[2, 5, 10, 20], [2, 5, 10, 20], [2, 5, 10, 20]]												# Sparsity of the sparse coding step in DL
shuffle=True
comet=True												# Load experiment to comet
save=True													# Save the dictionary in .npy file
step=0

for i, data_set in enumerate(data_sets):
	odl_iters=odl_iters_all[i]
	atom_nums=atom_nums_all[i]
	sparsities=sparsities_all[i]
	for odl_iter in odl_iters:
		for atom_num in atom_nums:
			for sparse in sparsities:
				label='results/%s/t%d_k%d_L%d'%(data_set, odl_iter, atom_num, sparse)
				if comet:
					# Create an experiment
					experiment = Experiment(api_key="8dKtcvfN32cdY2uuE8m5vZvmV", project_name="Honours_Project", workspace="joshuabruton")
					experiment.set_name(label)
					# Report any information you need by:
					experiment.log_parameter('Data set', data_set, step=None)
					experiment.log_parameter('Shuffled data', shuffle, step=None)
					experiment.log_parameter('Number of components', atom_num, step=None)
					experiment.log_parameter('Maximum Iterations', odl_iter, step=None)
					experiment.log_parameter('Sparsity', sparse, step=None)
				else:		
					experiment=None

				# Initialise the data
				data=dp.extract(shuffle=shuffle, data_set=data_set)

				# Normalise
				signals=np.array([l.values for l in data], dtype=np.float64)
				labels=np.array([l.label for l in data], dtype=np.float64)
				signals-=np.mean(signals)
				signals/=np.std(signals)

				# Initialise the dictionary
				dic=odl.Dict(num_coms=atom_num)

				# Trains the dictionary
				dic.fit(signals, reg_term=sparse, max_iter=odl_iter, showDictionary=False, showRecExample=False, prog=False, comet=experiment, logRep=comet, save=save, label=label)

				# Calculate representation error
				rep_error = dic.showError(recal=True)
				if comet:
					experiment.log_metric('Representation error', rep_error, step=odl_iter)
