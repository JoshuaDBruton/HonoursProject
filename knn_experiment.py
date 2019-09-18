import numpy as np
from cluster_meths import knn
from utils import data_processing as dp
from online_dictionary_learning import omp
from comet_ml import Experiment #For Comet.ml

def accuracy(preds, targets):
	return (sum(1 for a, b in zip(preds, targets) if a==b)/preds.size)

comet=False
verbose=True
shuffle=True
D=np.load('results/440_5_10000_dictionary.npy')
sparsity=5
ks=[5]
numtest=500
smetric='euclidean'				# Cosine and hamming don't work effectively, euclidean is okay
cmetric='euclidean'

if comet:
	# Create an experiment
	experiment = Experiment(api_key="8dKtcvfN32cdY2uuE8m5vZvmV", project_name="honours-project", workspace="joshuabruton")
	experiment.set_name('KNN test')
	hyper_params={'Data set':'Indian Pines', 'Dictionary':'440_5_10000_dictionary', 'Shuffled data':shuffle, 'Sparsity':sparsity, 'Number of tests':numtest, 'Signal metric':smetric, 'Coefficient metric':cmetric}
	experiment.log_parameters(hyper_params)

# Initialise the data
data=dp.extract(shuffle=shuffle)

# Normalise
signals=np.array([l.values for l in data], dtype=np.float64)
labels=np.array([l.label for l in data], dtype=np.float64)
signals-=np.mean(signals)
signals/=np.std(signals)

# Initialise coefficients
coefs=np.zeros((signals.shape[0],D.shape[1]))

# Extract coefficients from dictionary
for i,x in enumerate(signals):
	coefs[i]=omp.omp(D, x, L=sparsity, eps=None)
	if verbose:	
		if i+1<signals.shape[0]:
			print('[' + str(i+1) + '] ' + str(np.round((((i+1)/signals.shape[0])*100),2)) + '%', end='\r')
		else:
			print('[' + str(i+1) + '] ' + str(np.round((((i+1)/signals.shape[0])*100),2)) + '%', end='\n')

# Split into training and testing data
train_signals, test_signals=dp.split(signals, portion=0.9)
train_coefs, test_coefs=dp.split(coefs, portion=0.9)
train_labels, test_labels=dp.split(labels, portion=0.9)

for k in ks:
	if comet:
		experiment.log_parameter('k', k, step=None)
	
	# Initialise model and prediction arrays
	model=knn.knn(k=k)
	pred_sigs=np.zeros(numtest)
	pred_coefs=np.zeros(numtest)

	# Running KNN for testing data
	for i in range(numtest):
		pred_sigs[i]=model.fit(train_signals, test_signals[i], train_labels, metric=smetric)
		pred_coefs[i]=model.fit(train_coefs, test_coefs[i], train_labels, metric=cmetric)
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
		experiment.log_metric('Signal accuracy', acc_sigs, step=k)
		experiment.log_metric('Coefficient accuracy', acc_coefs, step=k)												
