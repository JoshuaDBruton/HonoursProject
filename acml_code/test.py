import odl
import numpy as np
import matplotlib.pyplot as plt
from utils.imageProcessing import *

dim=8
img=np.array(plt.imread('data/ally.png'))
signals=imageToSignals(img, dim=dim, showTiles=False)
signals-=np.mean(signals)
signals/=np.std(signals)
dict=odl.Dict(72)
dict.fit(signals, reg_term=5, max_iter=1000, showRepError=False, showDictionary=False, showRecExample=False, prog=True)
atoms = dict.getAtoms()
coefs = dict.getCoefs()
signalsToImage(coefs.dot(atoms.T), dim=dim, showImage=True, saveImage=False, saveName='ally_odl_reg20dir_i1000_k72_omp')
