from autoenc import *
from utils.imageProcessing import *
import matplotlib.pyplot as plt
import numpy as np

def showDict(atoms):
	print(atoms.shape)
	#atoms=atoms.transpose()
	split=int(np.sqrt(atoms.shape[1]))
	new_tiles=np.zeros(((atoms.shape[0],split,split)))
	for i in range(atoms.shape[0]):
		new_tiles[i]=np.reshape(atoms[i], (split,split))
	f, ax=plt.subplots(atoms.shape[0]//split,split)
	count=0
	for i in range(atoms.shape[0]//split):
		for j in range(split):
			ax[i][j].imshow(new_tiles[count],cmap="gray")
			ax[i][j].axis("off")
			count+=1
	plt.show()

dim=8
num_hidden=72
k=20

ally=np.array(plt.imread('data/ally.png'))
signals=imageToSignals(ally, dim=dim, showTiles=False)
ksae=KSparseAutoencoder(signals.shape[1], num_hidden, k=k)
ksae.fit(signals, signals, eta=0.001, epochs=5000, batch_size=1024, evince=True)
new_signals=np.zeros(signals.shape)
new_signals=ksae.predict(signals)
signalsToImage(new_signals, dim=dim, showImage=True, saveImage=False, saveName='ally_ae_recon_k20_h72_e5000_b1024_e1e-3')
#atoms=ksae.extract_dict()
#showDict(atoms)
