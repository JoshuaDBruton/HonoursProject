'''
Joshua Bruton
Last updated: 23 May 2019
Some image processing things I needed, mostly image->signals and signals->image
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

def signalsToImage(signals, dim=8, showImage=False, saveImage=False, saveName='default'):
	n = signals.shape[0]
	tiles = np.zeros((n,dim,dim))
	for i in range(n):
		tiles[i] = np.reshape(signals[i],(dim,dim))
	image = np.zeros((n//dim,n//dim))
	count=0
	for i in range(signals.shape[1]):
		for j in range(signals.shape[1]):
			image[0+i*dim:dim+i*dim,0+j*dim:dim+j*dim]=tiles[count]
			count+=1
	if showImage==True:
		plt.imshow(image,cmap='gray')
		plt.show()
	if saveImage==True:
		misc.imsave('Results/' + saveName + '.png', image)

def imageToSignals(image, dim=8, showTiles=False):
	tiles=imageToTiles(image, dim, showTiles)
	signals=np.zeros((tiles.shape[0], tiles.shape[1]*tiles.shape[2]))
	for i, tile in enumerate(tiles):
		signals[i]=np.ravel(tile)
	return signals

def imageToTiles(image, dim=32, showTiles=False):
	h=image.shape[0]//dim
	w=image.shape[1]//dim
	tiles=np.zeros((h*w,dim,dim))
	count=0
	for i in range(h):
		for j in range(w):
			tiles[count]=image[i*dim:i*dim+dim,j*dim:j*dim+dim]
			count+=1
	if showTiles==True:
		f, (ax)=plt.subplots(h,w)
		plt.suptitle('Tiled Image')
		count=0
		for i in range(h):
			for j in range(w):
				ax[i][j].imshow(tiles[count],cmap="gray")
				ax[i][j].axis("off")
				count+=1
		plt.show()
	return tiles
