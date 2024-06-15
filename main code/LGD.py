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
    Tout les calculs relatifs à la LGD sont effectués dans ce code
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


###     Différentes initialisations
    '''

        # Set up

    # chargement d'une nouvelle image
    index = np.random.randint(0, len(TrainSet))
    print(f"\n Index associé à l'image : {index}\n")

    img = TrainSet[index][0].squeeze()
    plt.imshow(img, cmap='magma')
    plt.show()

    #  initialisation égale à f_E(tA(y_0)), à x_0 bruité, aléatoire passé dans l'autoencoder, aléatoire...
    encoded = SupRes.AE.encoder(img.view(1,-1))

    inits = ['f_E(tA(y_0))', make_noisy(encoded, bruit='uniforme', param=0.5), make_noisy(encoded, bruit='gaussien', param=0.5), 'random']


        # On fait tourner

    SupRes.set_passebas(filtre='sans')
    SupRes.multiplot_descente(methode='LGD', target=img, inits=inits, pas=[0.05]*6, Niter=300, saveas='inits-s')
    
    SupRes.set_passebas(filtre='gaussien', parametre=0.6)
    SupRes.multiplot_descente(methode='LGD', target=img, inits=inits, pas=[0.25]*6 , Niter=300, saveas='inits-g')
    '''

###     Initialisations aléatoires 


        # Set up

    # chargement de nouvelles images
    nb = 8
    indexes = [19493, 31811,  3917, 50784, 21097, 12192,  9334, 13113] #np.random.randint(0, len(TrainSet), nb)
    print(f"\n Indexes associés aux images : {indexes}\n")

    imgs = [TrainSet[i][0].squeeze() for i in indexes]


        # Bruit uniforme  

    # les initialisations
    inits = ['random']*nb

    SupRes.set_passebas(filtre='sans')
    #SupRes.multiplot_multitarget(methode='LGD', targets=imgs, inits=inits, pas=[0.05]*nb, Niter=300, saveas=f'multarg_unif-s')   # f'multarg_gauss-s')


    SupRes.set_passebas(filtre='gaussien', parametre=0.6)
    SupRes.multiplot_multitarget(methode='LGD', targets=imgs, inits=inits, pas=[0.25]*nb, Niter=300, saveas=f'multarg_unif-g')   # f'multarg_gauss-g')


        # Bruit gaussien

    # les initialisations
    inits = [tc.normal(mean=0.0, std=0.2, size=(1, dim_latent)) for _ in range(nb)]

    SupRes.set_passebas(filtre='sans')
    #SupRes.multiplot_multitarget(methode='LGD', targets=imgs, inits=inits, pas=[0.05]*nb, Niter=300, saveas=f'multarg_gauss-s')


    SupRes.set_passebas(filtre='gaussien', parametre=0.6)
    #SupRes.multiplot_multitarget(methode='LGD', targets=imgs, inits=inits, pas=[0.25]*nb, Niter=300, saveas=f'multarg_gauss-g')
    plt.show()


'''
###     Différents niveau de compression


        # Préparatifs

    # chargement d'une nouvelle image
    index = np.random.randint(0, len(TrainSet))
    print(f"\n Index associé à l'image : {index}\n")

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

        # chargement de niveau de compression
        SupRes.set_sizes(y_shape=(p,q))

        # descente sans passe-bas
        SupRes.set_passebas(filtre='sans')

        SupRes.LGD(target=img, init='f_E(tA(y_0))', pas=p_s, Niter=300) 
        SupRes.plot_descente(methode='LGD', img3=SupRes.info['y_0'], saveas=f'size-s_{size}')
        
        # descente avec passe-bas
        SupRes.set_passebas(filtre='gaussien', parametre=0.6)

        SupRes.LGD(target=img, init='f_E(tA(y_0))', pas=p_g, Niter=300) 
        SupRes.plot_descente(methode='LGD', img3=SupRes.info['y_0'], saveas=f'size-g_{size}')

    
        # Remise à la valeur par défaut des tailles d'image 

    SupRes.set_sizes()


###     Différentes taille d'espace latent


        # Préparatifs

    # chargement d'une nouvelle image
    index = np.random.randint(0, len(TrainSet))
    print(f"\n Indexe associé à l'image : {index}\n")

    img = TrainSet[33186][0].squeeze()
    #plt.imshow(img, cmap='magma')
    #plt.show()


    #  trois tailles d'esapce latent 

    dims = [100, 200, 400, 800]

    # adaptation du pas en fonction de l'AE / du passe-bas
    pas_s = [0.05, 0.15, 0.3, 0.75]
    pas_g = [0.3, 8., 10., 10.]        # c'est pas forcément les meilleurs choix mais il sont pas trop mal
    

        # On fait tourner

    for d, p_s, p_g in zip(dims, pas_s, pas_g):
        # chargement de la taille de l'espace latent
        dim_latent = d
        SupRes.set_autoencoder(f'../resultats/autoencoder/Autoencoder {d}')
        SupRes.set_sizes(u_lenth=d)

        # différentes initialisations
        encoded = SupRes.AE.encoder(img.view(1,-1))
        inits = ['f_E(tA(y_0))', make_noisy(encoded, bruit='uniforme', param=0.5), make_noisy(encoded, bruit='gaussien', param=0.5), 'random']

        # descente sans passe-bas
        SupRes.set_passebas(filtre='sans')

        SupRes.multiplot_descente(methode='LGD', target=img, inits=inits, pas=[p_s]*4, Niter=300, saveas=f'lat-s_{d}')
        
        # descente avec passe-bas
        SupRes.set_passebas(filtre='gaussien', parametre=0.6)

        SupRes.multiplot_descente(methode='LGD', target=img, inits=inits, pas=[p_g]*4, Niter=300, saveas=f'lat-g_{d}')
'''