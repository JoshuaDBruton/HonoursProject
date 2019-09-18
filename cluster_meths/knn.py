import numpy as np

def hamming_dist(x, y):
	count=0
	for i, j in zip(x.T, y):
		if i > 0 and j > 0:
			count+=1
	return count

class knn:
	def __init__(self, k):
		self.k=k
		self.prediction=-1
		self.distance=0
	def fit(self, refs, qp, train_labels, metric='euclidean'):
		if metric=='euclidean':
			qp=np.array(qp)
			refs=np.array(refs)
			distances=[np.sqrt(np.sum((np.array(a)-np.array(qp))*(np.array(a)-np.array(qp)))) for a in refs]
		elif metric=='hamming':
			distances=np.zeros(refs.shape[0])
			#print(qp.shape)
			#print(refs)
		elif metric=='cosine':
			distances=[(np.sum(np.array(qp)*np.array(a)))/(np.sqrt(np.sum(np.array(qp)*np.array(qp)))*np.sqrt(np.sum(np.array(a)*np.array(a)))) for a in refs]
		sortIndex=np.argsort(distances)
		labels=[None]*self.k
		for i in range(self.k):
			labels[i]=int(train_labels[sortIndex[i]])
		return max(labels, key=labels.count)
