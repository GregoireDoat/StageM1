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
    
    '''
        # Set up

    # chargement de nouvelles images
    nb = 8
    indexes = [51213, 19566, 39819, 755, 12677, 35878, 35991, 11036]#np.random.randint(0, len(TrainSet), nb)   #  : indexes du rapport
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
    '''


###     Différentes taille d'espace latent


        # Préparatifs

    # chargement d'une nouvelle image
    index = np.random.randint(0, len(TrainSet))
    print(f"\n Indexe associé à l'image : {index}\n")

    img = TrainSet[index][0].squeeze()
    plt.imshow(img, cmap='magma')
    plt.show()


    #  trois tailles d'esapce latent 

    dims = [100, 200, 400, 800]

    # adaptation du pas en fonction de l'AE / du passe-bas
    pas_s = [1.25, 2.75, 2.75, 2.3]
    pas_g = [4.25, 4.75, 6.5, 6.15]        # c'est pas forcément les meilleurs choix mais ils sont pas trop mal
    

        # On fait tourner

    for d, p_s, p_g in zip(dims, pas_s, pas_g):
    
        # chargement de la taille de l'espace latent
        dim_latent = d
        SupRes.set_autoencoder(f'../resultats/autoencoder/Autoencoder {d}')
        SupRes.set_sizes(u_lenth=d)

        # différentes initialisations
        inits = ['tA(y_0)', make_noisy(img, bruit='uniforme', param=0.5), make_noisy(img, bruit='gaussien', param=0.5), 'random']

        # descente sans passe-bas
        SupRes.set_passebas(filtre='sans')
        SupRes.multiplot_descente(methode='PGD', target=img, inits=inits, pas=[p_s]*4, Niter=20, saveas=f'lat-s_{d}')
        
        # descente avec passe-bas
        SupRes.set_passebas(filtre='gaussien', parametre=0.6)
        SupRes.multiplot_descente(methode='PGD', target=img, inits=inits, pas=[p_g]*4, Niter=20, saveas=f'lat-g_{d}')
        plt.show()



    
