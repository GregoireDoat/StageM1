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


###     Initialisation par backprojection
    

        # Set up

    # changement d'une nouvelle image
    nbs = np.random.randint(0, len(TrainSet), 8)
    print(f"\n Indexes associés aux images : {nbs}\n")

    imgs = [TrainSet[nb][0].squeeze() for nb in nbs]

    #  changement d'auto-encodeur
    dim_latent = 100   
    SupRes.AE = tc.load(f'../resultats/autoencoder/Autoencoder {dim_latent}')
    SupRes.set_sizes(u_lenth=dim_latent)


        # On fait tourner

    SupRes.set_passebas(filtre='sans')
    SupRes.multiplot_multitarget(methode='PGD', targets=imgs, inits=['tA(y_0)']*8, pas=[1.75]*8, Niter=20, saveas='backproj')
    
    SupRes.set_passebas(filtre='gaussien', parametre=0.6)
    SupRes.multiplot_multitarget(methode='PGD', targets=imgs, inits=['tA(y_0)']*8, pas=[4.25]*8, Niter=20, saveas='backproj')

    plt.show()


###     Initialisations aléatoires
    '''
    

        # Set up

    # changement d'une nouvelle image
    nbs = np.random.randint(0, len(TrainSet), 8)
    print(f"\n Indexes associés aux images : {nbs}\n")

    imgs = [TrainSet[nb][0].squeeze() for nb in nbs]

    #  changement d'auto-encodeur
    dim_latent = 100   
    SupRes.AE = tc.load(f'../resultats/autoencoder/Autoencoder {dim_latent}')
    SupRes.set_sizes(u_lenth=dim_latent)


        # On fait tourner

    SupRes.set_passebas(filtre='sans')
    SupRes.multiplot_multitarget(methode='PGD', targets=imgs, inits=['random']*4+['rand+AE']*4, pas=[1.75]*8, Niter=20, saveas='backproj')
    
    SupRes.set_passebas(filtre='gaussien', parametre=0.6)
    SupRes.multiplot_multitarget(methode='PGD', targets=imgs, inits=['random']*4+['rand+AE']*4, pas=[4.25]*8, Niter=20, saveas='backproj')

    plt.show()
    ''' 

   
###     Plein de descente partie de vecteur aléatoire
    '''

        # Set up

    # changement d'une nouvelle image
    nbs = np.random.randint(0, len(TrainSet), 4)
    print(f"\n Indexes associés aux images : {nbs}\n")

    imgs = [TrainSet[nb][0].squeeze() for nb in nbs]


        # Descentes  

    SupRes.set_passebas(filtre='sans')
    SupRes.multiplot_multitarget(methode='PGD', targets=imgs, inits=['random']*4, pas=[0.05]*4, Niter=300, saveas=f'multarg-n-s')


    SupRes.set_passebas(filtre='gaussien', parametre=0.6)
    SupRes.multiplot_multitarget(methode='PGD', targets=imgs, inits=['random']*4, pas=[0.25]*4, Niter=300, saveas=f'multarg-n-g')
    '''


###     Différents niveau de compression
    '''

        # Préparatifs

    # chargement d'une nouvelle image
    nb = np.random.randint(0, len(TrainSet))
    print(f"\n Index associé à l'image : {nb}\n")

    img = TrainSet[4618][0].squeeze()
    plt.imshow(img, cmap='magma')
    plt.show()

    # plusieurs valeurs pour p et q
    pas_s = [0.01, 0.05, 0.1, 0.1, 0.1]
    pas_g = [0.01, 0.5, 0.1, 0.1, 0.2]
    sizes = ['mini', 'small', 'mid1', 'mid2', 'big']
    Ps = [5, 7, 14, 7, 14]
    Qs = [5, 7, 7, 14, 14]


        # On fait tourner
    
    for i, (size, p,q, p_s, p_g) in enumerate(zip(sizes, Ps, Qs, pas_s, pas_g)):

        # changement de niveau de compression
        SupRes.set_sizes(y_shape=(p,q))

        # descente sans passe-bas
        SupRes.set_passebas(filtre='sans')

        SupRes.PGD(target=img, init='tA(y_0)', pas=p_s, Niter=300) 
        SupRes.plot_descente(methode='PGD', img3=SupRes.info['y_0'], saveas=f'size-s_{size}')
        
        # descente avec passe-bas
        SupRes.set_passebas(filtre='gaussien', parametre=0.6)

        SupRes.PGD(target=img, init='tA(y_0)', pas=p_g, Niter=300) 
        SupRes.plot_descente(methode='PGD', img3=SupRes.info['y_0'], saveas=f'size-g_{size}')

    
        # Remise à la valeur par défaut des tailles d'image 

    SupRes.set_sizes()
    '''


###     Différentes taille d'espace latent

    '''
        # Préparatifs

    # chargement d'une nouvelle image
    nb = np.random.randint(0, len(TrainSet))
    print(f"\n Indexe associé à l'image : {nb}\n")

    img = TrainSet[33186][0].squeeze()
    #plt.imshow(img, cmap='magma')
    #plt.show()


    #  trois tailles d'esapce latent 

    dims = [100, 200, 400, 800]

    # adaptation du pas en fonction de l'AE / du passe-bas
    pas_s = [1.25, 2.75, 2.75, 2.3]
    pas_g = [4.25, 4.75, 6.5, 6.15]        # c'est pas forcément les meilleurs choix mais il sont pas trop mal
    

        # On fait tourner

    for d, p_s, p_g in zip(dims, pas_s, pas_g):
    
        # changement de la taille de l'espace latent
        dim_latent = d
        SupRes.set_autoencoder(f'../resultats/autoencoder/Autoencoder {d}')
        SupRes.set_sizes(u_lenth=d)

        # différentes initialisations
        inits = ['tA(y_0)', make_noisy(img, bruit='uniforme', param=0.5), make_noisy(img, bruit='gaussien', param=0.5), 'random']

        # descente sans passe-bas
        SupRes.set_passebas(filtre='sans')

        SupRes.multiplot_descente(methode='PGD', target=img, inits=inits, pas=[p_s]*4, Niter=300, saveas=f'lat-s_{d}')
        
        # descente avec passe-bas
        #SupRes.set_passebas(filtre='gaussien', parametre=0.6)

        #SupRes.multiplot_descente(methode='PGD', target=img, inits=inits, pas=[p_g]*4, Niter=300, saveas=f'lat-g_{d}')
    '''


    
