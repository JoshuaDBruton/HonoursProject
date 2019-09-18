import scipy.io as spio
import numpy as np

data_mat=spio.loadmat("data/IP/Indian_pines_corrected.mat")
gt_mat=spio.loadmat("data/IP/Indian_pines_gt.mat")
IP_image=data_mat["indian_pines_corrected"]
IP_gt=gt_mat['indian_pines_gt']

pavia_mat = spio.loadmat('data/PaviaU/PaviaU.mat')
pavia_gt_mat = spio.loadmat('data/PaviaU/PaviaU_gt.mat')
pavia_image = pavia_mat['paviaU']
pavia_gt = pavia_gt_mat['paviaU_gt']

salinas_mat = spio.loadmat('data/Salinas/salinas.mat')
salinas_gt_mat = spio.loadmat('data/Salinas/salinas_gt.mat')
salinas_image = salinas_mat['salinasA_corrected']
salinas_gt = salinas_gt_mat['salinasA_gt']

image = None
gt = None

class Point:
	def __init__(self, values, label):
		self.values=np.array(values)
		self.label=label

def extract(shuffle=False, data_set='Indian_Pines'):
	global image
	global gt
	if data_set=='Indian_Pines':
		image = IP_image
		gt = IP_gt
	elif data_set=='Pavia':
		image = pavia_image
		gt = pavia_gt
	elif data_set=='Salinas':
		image = salinas_image
		gt = salinas_gt
	signals=[]
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):	
			if gt[i,j]!=0:
				signals.append(Point(image[i,j,:],gt[i,j]))
	if shuffle:
		np.random.shuffle(signals)
	return signals

def extract_tiles(shuffle=False, dim=5):
	h=image.shape[0]//dim
	w=image.shape[1]//dim
	depth=image.shape[2]
	tiles=[]
	labels=[]
	count=0
	for i in range(h):
		for j in range(w):
			if gt[i,j]!=0:
				tiles.append(image[i*dim:i*dim+dim,j*dim:j*dim+dim, :])
				labels.append(gt[i*dim:i*dim+dim,j*dim:j*dim+dim])
				count+=1
	tiles=np.array(tiles, dtype=np.double)
	labels=np.array(labels)
	tiles=np.reshape(tiles, (tiles.shape[0], dim*dim, depth))
	labels=np.reshape(labels, (labels.shape[0], dim*dim))	
	if shuffle:
		merge=list(zip(tiles, labels))
		np.random.shuffle(merge)
		tiles, labels = zip(*merge)
	return tiles, labels

def split(data,portion=0.9, size=False):
	train_data=data[0:int(portion*len(data))]
	test_data=data[int(portion*len(data)):]
	return train_data, test_data
