from utils import data_processing as dp
import numpy as np
import spams
from cluster_meths import knn
from scipy import sparse

def accuracy(preds, targets):
	return (sum(1 for a, b in zip(preds, targets) if a==b)/preds.size)

tileDim=3
dictionary='220_1_10000_dictionary'
D=np.load('results/%s.npy' %dictionary)

D = np.asfortranarray(D / np.tile(np.sqrt((D*D).sum(axis=0)),(D.shape[0],1)),dtype=np.double)
#D=np.asfortranarray(D, dtype=np.double)

tiles, labels = dp.extract_tiles(shuffle=False, dim=tileDim)

labels=np.reshape(labels, (labels.shape[0]*labels.shape[1]))
tiles=np.reshape(tiles,(tiles.shape[0]*tiles.shape[1], tiles.shape[2]))
tiles=np.asfortranarray(tiles.T, dtype=np.double)

ind_groups=np.array(np.arange(0,tiles.shape[1],tileDim*tileDim), dtype=np.int32)

A = spams.somp(tiles, D, ind_groups, L=10, eps=0.1, numThreads=-1)

A = sparse.coo_matrix(A)
A = A.todense()
A = A.T

train_coefs, test_coefs = dp.split(A,portion=0.9)
train_labels, test_labels = dp.split(labels, portion=0.9)

numtest = 50

model = knn.knn(k=5)
preds = np.zeros(numtest)

for i in range(numtest):
	preds[i]=model.fit(train_coefs, test_coefs[i], train_labels, metric='hamming')
	if i+1<numtest:
		print('[' + str(i+1) + '] ' + str(np.round((((i+1)/numtest)*100),2)) + '%', end='\r')
	else:
		print('[' + str(i+1) + '] ' + str(np.round((((i+1)/numtest)*100),2)) + '%', end='\n')

acc=accuracy(preds, test_labels[:numtest])

print('The coefficient accuracy was ' + str(acc) + '.')
