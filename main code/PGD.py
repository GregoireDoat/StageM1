import pickle as rick

import numpy as np
import skimage as ski

import torch as tc 
from torch import nn
from torchvision import datasets, transforms
from torcheval.metrics.functional import peak_signal_noise_ratio

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation

from tqdm import tqdm,trange

from main_code import SuperResolution, make_noisy

'''
    Tout les calculs relatifs à la PGD sont effectués dans ce code
'''

# taille de l'esapce latent
dim_latent = 100 # 200, 400, 800

class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder = nn.Sequential( #two layers encoder
            nn.Linear(784,1500),
            nn.Sigmoid(), #ReLU, Tanh, etc.
            nn.Linear(1500,dim_latent),   
            nn.Sigmoid(),#input is in (0,1), so select this one
        )
        self.decoder=nn.Sequential( #two layers decoder
            nn.Linear(dim_latent, 1500),
            nn.Sigmoid(),
            nn.Linear(1500, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return decoded

if __name__=='__main__':


###     Auto-encodeur et dataset


    # charchement du test set
    TrainSet = datasets.MNIST('../dataset', train=True, transform=transforms.ToTensor(), download=False) #=True)

    # chargement de l'auto-encodeur
    SupRes = SuperResolution(path2autoencoder='../resultats/autoencoder/Autoencoder 100', path2save='../resultats')


###     Différentes initialisations [DONE]
    

        # Set up

    # chargement de nouvelles images
    nb = 8
    indexes = np.random.randint(0, len(TrainSet), nb)   # [51213, 19566, 39819, 755, 12677, 35878, 35991, 11036] : indexes du rapport
    print(f"\n Indexes associés aux images : {indexes}\n")

    imgs = [TrainSet[i][0].squeeze() for i in indexes]


        # On fait tourner

    # initalisation par backprojection
    SupRes.set_passebas(filtre='sans')
    SupRes.multiplot_multitarget(methode='PGD', targets=imgs, inits=['tA(y_0)']*nb, pas=[1.25]*nb, Niter=20, saveas='backproj-s')
    
    SupRes.set_passebas(filtre='gaussien', parametre=0.6)
    SupRes.multiplot_multitarget(methode='PGD', targets=imgs, inits=['tA(y_0)']*nb, pas=[4.25]*nb, Niter=20, saveas='backproj-g')
    
    # initialisation aléatoire uniforme
    SupRes.set_passebas(filtre='sans')
    SupRes.multiplot_multitarget(methode='PGD', targets=imgs, inits=['random']*nb, pas=[1.25]*nb, Niter=25, saveas='rand_unif-s')
    
    SupRes.set_passebas(filtre='gaussien', parametre=0.6)
    SupRes.multiplot_multitarget(methode='PGD', targets=imgs, inits=['random']*nb, pas=[4.25]*nb, Niter=25, saveas='rand_unif-g')
    
    # initialisation aléatoire gaussien
    inits = [tc.normal(mean=0.0, std=2., size=(28, 28)) for _ in range(nb)]

    SupRes.set_passebas(filtre='sans')
    SupRes.multiplot_multitarget(methode='PGD', targets=imgs, inits=inits, pas=[1.25]*nb, Niter=25, saveas='rand_gauss-s')
    
    SupRes.set_passebas(filtre='gaussien', parametre=0.6)
    SupRes.multiplot_multitarget(methode='PGD', targets=imgs, inits=inits, pas=[4.25]*nb, Niter=25, saveas='rand_gauss-g')