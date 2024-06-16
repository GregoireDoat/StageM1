import pickle as rick

import numpy as np
import skimage as ski

from functools import lru_cache

import torch as tc 
from torch import nn
from torchvision import datasets, transforms
from torcheval.metrics.functional import peak_signal_noise_ratio

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation

from tqdm import tqdm, trange

'''
    Le coeur du code de découpe en 3 classes:

    - La première classe est le réseau autoencodeur, la taille de son esapce latent est donnée pas dim_latent juste au dessus.


    - La deuxième classe 'Affichage' est dédié à l'affichage (si, si), elle est parente à la classe suivante qui effectue les calculs.
        Pour cette raison, elle n'est jamais appelé qu'à travers 'SuperResolution'.
        A chaque affiche du rapport est associé une fonction.


    - La troisème classe, 'SuperResolution', effectue les descentes de gradient (PGD et LGD) et se décompose en 9 fonctions :

        - __init__ :        Initialisation les varaibles qui seront nécessaire aux calculs. En particlier, elle appelle les 3 fonctions suivantes

        - set_autoencoder : Charge l'auto-encodeur au chemin donné

        - set_sizes :       Met à jours les tailles d'images sous-résolues (p,q), super-résolues (n,m) et la taille de l'esapce latent d

        - set_pass-bas :    Met à jour le filtre passe-bas de l'opérateur de mesure A (sans, gaussien, porte)

        - A :               L'opérateur de mesure, réduit une image de taille (n,m) en une de taille (p,q) avec le passe-bas actuellement set

        - tA :              Application transposé de A

        - PGD :             Effectue la déscente de gradient projeté. Si son paramètre back_tracking=True, 
                                elle passe par l'application PGD_backtrack pour ajuster le pas pendant la descente

        - PGD_backtrack :   Ajuste le pas de descente par backtracking. Ne fonctionne pas pour des raisons obscures,
                                elle boucle à l'infinie sans trouver de pas satisfaisant

        - LGD :             Effectue la déscente de gradient depuis l'espace latent de l'autoencodeur chargé. 

        - LGD_backtrack :   zIdem que PGD_backtrack, ne marche pas non plus

    Il y a également une fonction make_noisy qui ajoute un bruit additif au paramètre ajustable.
        La présentation des filtres passe-bas, les calculs de gamma et deux exemples de PGD et LGD sont donnés après le if __name__=='__main__'.
        Le reste des résultats sont obtenues respectivement avec les codes LGD.py et PGD.py

    Par souci de lisibilité, les tenseurs de taille (x_height, x_width)=(n,m) ou (1, x_height * x_width) seront toujours notés x (ou x_0),
        les tenseurs de taille (y_height, y_width)=(p,q) ou (1, y_height * y_width) seront toujours noté y (ou y_0)
        et les vecteurs de l'esapace latent de taille (1, u_height)=(1,d) toujours noté u.
        Le reste des notations sont reprises sur rapport.

    Le code est interminanble mais forte heureusement, ce qui se trouve après le if __name__=='__main__' est (normalement)
        suffisament compréhensible en soi! (idem pour PGD.py et LGD.py) Bonne lecture :)
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
        

class Affichage:

    # auto-encodeur et filtre

    def plot_perfAE(self, imgs, saveas) -> None:
        '''
            :param list[tc.Tensor] types:   liste d'image (x_height, x_width)
            :param str saveas:              nom pour sauvegarde si précisé

            Affiche les images 'imgs' avec en dessous leur version auto-encoder avec le PSNR entre les deux
            Si 'saveas' est précisé, sauvegarde les figures et chaque image individuellement
        '''

            # Plot

        fig, ax = plt.subplots(2, len(imgs))

        for i, img in tqdm(enumerate(imgs)):
            
            # passage dans l'auto-encodeur
            output = Net(img.view(1,-1)).detach().view(self.x_height, self.x_width)

            # plot des images
            ax[0,i].set_title(f'input {i+1}')
            ax[0,i].imshow(img, cmap=self.color_map)

            ax[1,i].set_title(f'output {i+1}')
            ax[1,i].imshow(output, cmap=self.color_map)

        # éventuelle sauvegarde de la figure avant le show
        if saveas is not None:
            plt.savefig(f'{self.path2save}/{saveas}-fig.png')
        plt.show()


            # Sauvegarde

        if saveas is not None:

            # création du dossier de sauvegarde
            save_name = f'{self.path2save}/{saveas}'

            for i, img in tqdm(enumerate(imgs)):
                
                # passage dans l'auto-encodeur
                output = Net(img.view(1,-1)).detach().view(self.x_height, self.x_width)

                # calcul du PSNR
                PSNR = peak_signal_noise_ratio(img, output, data_range=1.).detach().numpy()
                PSNR = np.round(PSNR, decimals=2)

                # sauvegarde des images
                plt.imsave(f'{save_name}_{i+1}_input.png', img, cmap=self.color_map)
                plt.imsave(f'{save_name}_{i+1}_{PSNR=}_output.png', output, cmap=self.color_map)

    def compare_filtre(self, img, types, params, shift=False, saveas=None) -> None:
        '''
            :param tc.Tensor img:       image (x_height, x_width)
            :param list[str] types:     liste des types de filtres (voir SuperResolution.set_passebas)
            :param list[float] params:  paramètres associésau filtre
            :param bool shift:          dicte si le filtre doit être shifté à l'affichage ou non
            :param str saveas:              nom pour sauvegarde si précisé 

            Affiche les filtres 'types' de paramètre respectif 'params' dans l'esapce de Fourier (éventuellement shifté) avec en dessous l'image img,
                la mesure A(img) et la backprojection tAA(img)
            Si 'saveas' est précisé, sauvegarde les figures et chaque image individuellement
        '''

        fig, ax = plt.subplots(4, min(len(types), len(params)), figsize=(16, 8))
        
        for i, (filtre, param) in enumerate(zip(types, params)):

            self.set_passebas(filtre=filtre, parametre=param)

            compressed = SupRes.A(img)
            transposed = SupRes.tA(compressed)

            #img[0,0], compressed[0,0], transposed[0,0] = 0, 0, 0
            #img[1,0], compressed[1,0],  transposed[1,0] = 1, 1, 1


            ax[0,i].set_title(f"{SupRes.info['filtre_param']}")

            if filtre == 'sans':
                ax[0,i].imshow(np.ones_like(img))
            elif shift:
                ax[0,i].imshow(tc.fft.fftshift(self.h))
            else:
                ax[0,i].imshow(self.h)

            ax[1,i].imshow(img, cmap=self.color_map)

            ax[2,i].imshow(compressed, cmap=self.color_map)

            ax[3,i].imshow(transposed, cmap=self.color_map)

        ax[0,0].set_ylabel('filtre')
        ax[1,0].set_ylabel('x')
        ax[2,0].set_ylabel('Ax')
        ax[3,0].set_ylabel('tA(Ax)')
        plt.show()


            # Sauvegarde

        if saveas is not None:
            plt.imsave(save_name + '-x.png', img, cmap=self.color_map)

            # sauvegarde des filtres avec leur application à img
            for i, (filtre, param) in enumerate(zip(types, params)):

                # changement de filtre
                self.set_passebas(filtre=filtre, parametre=param)
                
                # calcul de Ax et tA(Ax) en conséquant
                compressed = self.A(img)
                transposed = self.tA(compressed)

                
                if filtre == 'sans':
                    # le filtre (dans Fourier)
                    plt.imsave(f'{save_name}-f-filtre={filtre[0]}.png', np.ones_like(img), cmap='viridis')
                    # A et tA(Ax)
                    plt.imsave(f'{save_name}-Ax-filtre={filtre[0]}.png', compressed, cmap=self.color_map)
                    plt.imsave(f'{save_name}-tA(Ax)-filtre={filtre[0]}.png', transposed, cmap=self.color_map)

                else:
                    # le filtre (dans Fourier)
                    plt.imsave(f'{save_name}-f-filtre={filtre[0]}_param={param}.png', np.fft.fftshift(self.h), cmap='viridis')
                    # A et tA(Ax)
                    plt.imsave(f'{save_name}-Ax-filtre={filtre[0]}_param={param}.png', compressed, cmap=self.color_map)
                    plt.imsave(f'{save_name}-tA(Ax)-filtre={filtre[0]}_param={param}.png', transposed, cmap=self.color_map)

    # descente

    def plot_descente(self, methode, img3, saveas=None) -> None:
        '''
            :param str methode:         méthode utilisée pour la descente (PGD ou LGD)
            :param tc.Tensor/array img3:   image à affiche en plus de résultats, de la target et des graphs
            :param str saveas:          nom auquel sauvegarder les résultats 

            Affiche les résultats de la 'methode' : target, initialisation, résultats de la descente, l'image img3 et le graphs du loss et du PSNR.
            Si saveas est donné, sauvegarde la figure et les données de la descente à ce nom
        '''
        x_0 = self.info['x_0']
        Imgs = self.histo['img']
        Loss = self.histo['value']
        PSNRs = self.histo['PSNR']


            # Plot

        fig, ax = plt.subplots(2,3, figsize=(16, 8))
        plt.suptitle(f'Résultats {methode} (PSNR={PSNRs[-1]})')

        # plot des images
        ax[0,0].set_title('target')
        ax[0,0].imshow(x_0, cmap=self.color_map)

        ax[0,1].set_title('guess')
        ax[0,1].imshow(Imgs[-1], cmap=self.color_map)
        # plot d'une image au choix
        ax[1,0].set_title(f'image 3')
        ax[1,0].imshow(img3, cmap=self.color_map)

        ax[1,1].set_title('init')
        ax[1,1].imshow(Imgs[0], cmap=self.color_map)

        # plot des valeurs
        ax[0,2].set_title('Evolution de F')
        ax[0,2].plot(Loss, color='k')

        ax[1,2].set_title('Compaisons')
        ax[1,2].axhline(self.info['PSNR_etalon'], label='PSNR étalon', color=(0.5, 0.5, 0.5), ls='--')
        ax[1,2].plot(PSNRs, label='PSNR', color='orange')

        ax[0,2].set_yscale('log')
        ax[1,2].legend()

        if saveas is not None:
            with open(f'{self.path2save}/{methode}/{saveas}.pkl', 'wb') as f:
                rick.dump([self.info, self.histo], f)
            plt.savefig(f'{self.path2save}/{methode}/{saveas}_fig.png')
        
        #plt.show()

    def multiplot_descente(self, methode, target, inits, pas, Niter=100, ten_first=False, saveas=None) -> None:
        '''
            :param str methode:         méthode utilisée pour la descente (PGD ou LGD)
            :param tc.Tensor target:       cible pour les plusieurs descentes
            :param list[tc.Tensor|str] inits:   liste des initialisations pour chaque descente
            :param list[tc.Tensor|str] pas:     liste des pas pour chaque descente
            :param int Niter:               nombre d'itérations à faire pour la descente
            :param bool ten_first:          dicte s'il faut afficher les 10 premières itérations ou non
            :param str saveas:          nom auquel sauvegarder les résultats 

            Affiche un résumé des résultats des descentes de cible commune 'target', initialiser respectivement par les 'inits',
                avec les pas de descente 'pas' et sur 'Niter' itération. Le nombre de descente effectué est donné par la longueur 
                de la plus petite des liste ('inits' et 'pas').
            Affiche éventuellement les 10 premières images de la descentes et éventuellement sauvegarde le tout (les 10 images, la figure et les données)
                au nom 'saveas' si fourni.
        '''
        l = min(len(inits), len(pas))
        plot_tenfirst = []

        fig = plt.figure(figsize=(16, 8))
        gs = mpl.gridspec.GridSpec(4,l+2)

        # target
        targ = fig.add_subplot(gs[:2, :2])
        
        targ.set_title('target')
        targ.imshow(target, cmap=self.color_map)

        # loss
        losses = fig.add_subplot(gs[2:, :(l+2)//2])
        losses.set_title('Evolution des loss')
        losses.set_yscale('log')

        # PSNR
        PSNRs = fig.add_subplot(gs[2:, (l+2)//2:])
        PSNRs.set_title('Evolution des PSNR')        

        for j, (init, p) in enumerate(zip(inits, pas)):

            if methode == 'PGD':
                self.PGD(target, init, p, Niter=Niter)
            elif methode == 'LGD':
                self.LGD(target, init, p, Niter=Niter)

            if ten_first == True:
                plot_tenfirst.append(self.histo['img'][:10])

            ax = fig.add_subplot(gs[0, j+2])

            ax.set_title(f'init image {j+1}')
            ax.imshow(self.histo['img'][0], cmap=self.color_map)


            ax = fig.add_subplot(gs[1, j+2])

            ax.set_title(f'guess {j+1}')
            ax.imshow(self.histo['img'][-1], cmap=self.color_map)


            losses.plot(self.histo['value'],label=f'{j+1}')
            PSNRs.plot(self.histo['PSNR'] , label=f'{j+1}')
            
            if saveas is not None:
                with open(f'{self.path2save}/{methode}/{saveas}_{j+1}.pkl', 'wb') as f:
                    rick.dump([self.info, self.histo],f)
       

        PSNRs.axhline(self.info['PSNR_etalon'], label='PSNR étalon', color=(0.5, 0.5, 0.5), ls='--')
        losses.legend()
        PSNRs.legend()

        if saveas is not None:
            plt.savefig(f'{self.path2save}/{methode}/{saveas}_fig.png')
        #plt.show()


        if ten_first == True:
            fig, ax = plt.subplots(2*l, 5)

            for i in range(l):
                for j in range(5):
                    ax[2* i ,j].imshow(plot_tenfirst[i][ j ], cmap=self.color_map)
                    ax[2*i+1,j].imshow(plot_tenfirst[i][j+5], cmap=self.color_map)
        

            if saveas is not None:
                plt.savefig(f'{self.path2save}/{methode}/{saveas}_10first.png')
            plt.show()

    def multiplot_multitarget(self, methode, targets, inits, pas, Niter=100, ten_first=False, saveas=None) -> None:
        
        l = min(len(inits), len(pas))
        plot_tenfirst = []

        fig = plt.figure(figsize=(16, 8))
        gs = mpl.gridspec.GridSpec(5,l)

        # loss
        losses = fig.add_subplot(gs[-2:, :l//2])
        losses.set_title('Evolution des loss')
        losses.set_yscale('log')

        # PSNR
        PSNRs = fig.add_subplot(gs[-2:, l//2:])
        PSNRs.set_title('Evolution des PSNR')        

        for j, (target, init, p) in enumerate(zip(targets, inits, pas)):

            if methode == 'PGD':
                self.PGD(target, init, pas=p, Niter=Niter)
            elif methode == 'LGD':
                self.LGD(target, init, pas=p, Niter=Niter)

            if ten_first == True:
                plot_tenfirst.append(self.histo['img'][:10])

            # initialisation
            ax = fig.add_subplot(gs[0, j])

            ax.set_title(f'init image {j+1}')
            ax.imshow(self.histo['img'][0], cmap=self.color_map)

            # guess
            ax = fig.add_subplot(gs[1, j])

            ax.set_title(f'guess {j+1}')
            ax.imshow(self.histo['img'][-1], cmap=self.color_map)

            # target
            ax = fig.add_subplot(gs[2, j])

            ax.set_title(f'target {j+1}')
            ax.imshow(self.info['x_0'], cmap=self.color_map)


            losses.plot(self.histo['value'],label=f'{j+1}')
            PSNRs.plot(self.histo['PSNR'] , label=f'{j+1}')
            
            if saveas is not None:
                with open(f'{self.path2save}/{methode}/{saveas}_{j+1}.pkl', 'wb') as f:
                    rick.dump([self.info, self.histo],f)
       
        losses.legend()
        PSNRs.legend()

        if saveas is not None:
            plt.savefig(f'{self.path2save}/{methode}/{saveas}_fig.png')
        #plt.show()


        if ten_first == True:
            fig, ax = plt.subplots(2*l, 5)

            for i in range(l):
                for j in range(5):
                    ax[2* i ,j].imshow(plot_tenfirst[i][ j ], cmap=self.color_map)
                    ax[2*i+1,j].imshow(plot_tenfirst[i][j+5], cmap=self.color_map)
        

            if saveas is not None:
                plt.savefig(f'{self.path2save}/{methode}/{saveas}_10first.png')
            plt.show()

    # animation

    def animation(self, N) -> None:
        '''
        Produit une animation des N premières itérations de la descente (PGD ou LGD) à partir des données 
            qui se trouve dans la classe SuperResolution. Il faut donc qu'une descente ait déjà été effectué.
        '''
        # set up figure
        self.fig = plt.figure(figsize=(16, 8))
        self.gs = mpl.gridspec.GridSpec(2,3)

        self.ax = []
        # image
        self.ax.append(self.fig.add_subplot(self.gs[:2, :2]))
        # loss
        self.ax.append(self.fig.add_subplot(self.gs[0, 2]))
        # PSNR
        self.ax.append(self.fig.add_subplot(self.gs[1, 2]))
        
        #echelle des graphes
        self.ax[1].set_xlim(0, N)
        self.ax[1].set_ylim(0, max(self.histo['value']))

        self.ax[2].set_xlim(0, N)
        self.ax[2].set_ylim(0, max(self.histo['PSNR']))

        # lancement de l'animation
        anim = animation.FuncAnimation(fig=self.fig, func=self.anim_update, frames=N, init_func=self.anim_init)
        plt.show()

    def anim_init(self) -> None:
        self.line = [None]*3

        self.ax[0].imshow(self.histo['img'][0], cmap=self.color_map)

        self.line[0], = self.ax[1].plot([0], self.histo['value'][0], color='black')
        self.line[2], = self.ax[2].plot([0], self.histo['PSNR'][0], color='orange')

    def anim_update(self, n) -> None:
        absc = np.arange(n)

        self.ax[0].imshow(self.histo['img'][n], cmap=self.color_map)

        self.line[0].set_data(absc, self.histo['value'][:n])
        self.line[2].set_data(absc, self.histo['PSNR'][:n])


class SuperResolution(Affichage):

    def __init__(self, path2autoencoder, path2save) -> None:
        '''
            :param str path2autoencoder:    dossier où sont stockés les paramètres d'auto-encoder
            :param str pth2save:            dossier où sauvegarder les résultats

            Cette classe est parente la classe affichage. Donc unitile d'appeler Affichage.
        '''

        # sockage des infos
        self.info = {}

        # chargement de l'AE
        self.set_autoencoder(path2autoencoder)
        # set des taille des images
        self.set_sizes()
        # construction du filtre passe-bas
        self.set_passebas()

        # affichage et sauvegarde
        self.color_map = 'magma'    # rend mieux les petites variations IMO
        self.path2save = path2save  # chemin où sauvegarder les résultats

    # mise à jour d'hyper-paramètres

    def set_autoencoder(self, path2autoencoder) -> None:
        '''
            :param str path2autoencoder:    dossier où sont stockés les paramètres d'auto-encoder

            Charge l'auto-encodeur au chemin donnée.
            Attention : il faut rappeler ensuite set_sizes si les dimension d'entrée et de l'espace latent changent
        '''
        self.AE = tc.load(path2autoencoder) 
        self.info['Autoencodeur'] = path2autoencoder
        print(f'\n{self.AE}')

    def set_sizes(self, x_shape=[28,28], y_shape=[14,14], u_lenth=100) -> None:
        '''
            :param list(int) x_shape:    tailles des images super-résolue, (n,m)
            :param list(int) y_shape:    tailles des images sous-échantillon, (p,q)
            :param int u_lenth:         dimension de l'esapce latent, d

            Change les valeurs des tailles de tenseur.
            Attention : il faut rappeler ensuite set_passebas si x_shape et y_shape sont modifiés.
        '''
        # tailles des images avant et après mesure 
        self.x_height, self.x_width = x_shape   # = (n, m)
        self.y_height, self.y_width = y_shape   # = (p, q)

        # calcul des rations  n/p  et  m/q  pour  S 
        self.gap_x = self.x_height//self.y_height
        self.gap_y = self.x_width//self.y_width

        # taille de l'espace latent
        self.u_height = u_lenth                 # = d

    def set_passebas(self, filtre='sans', parametre=1) -> None:
        '''
            :param str filtre:          'sans', 'gaussien', 'porte'. type de filtre passe-bas.
            :param float parametre:     paramètre associé au filtre

            Créé le filtre passe-bas directement dans l'esapce des fréquances, sauf dans le cas 'sans', où il n'y a pas de filtre.
        '''


            # Plusieurs choix de filtre (3)

        # Pas de filtre

        if filtre == 'sans' :

            # construction du filtre
            self.h = None

            # stockage des infos
            self.info['filtre_type'] = filtre
            self.info['filtre_param'] = 'None'
            self.info['filtre'] = self.h

            print("\n Compression sans passe-bas")

        # Filtre gaussien

        if filtre == 'gaussien':

            # paramètres : écrat-type
            sdv = parametre 

                # Construction de la gaussienne

            # valeurs où calculer
            freqs_x = tc.linspace(-1, 1, self.x_height)
            freqs_y = tc.linspace(-1, 1, self.x_width)
            freqs = tc.stack(tc.meshgrid(freqs_x, freqs_y), axis=2)

            # version transformé par Fourier
            h = (sdv*tc.pi * freqs)**2
            h = (-h.sum(axis=2)).exp()  
            
            # normalisation
            norm1 = tc.fft.ifft2(tc.fft.ifftshift(h)).real.sum()
            self.h = tc.fft.ifftshift(h / norm1)

                # Stockage des infos

            self.info['filtre_type'] = filtre
            self.info['filtre_param'] = parametre
            self.info['filtre'] = self.h.numpy()

            print(f"\n Filtre passe-bas gaussien d'écart-type {sdv}")
            
        # Filtre porte normalisé

        if filtre == 'porte':
            # paramètres : bornes de la porte [-a,a]x[-a,a]
            a = parametre  

            # liste des fréquances shifftées
            freqs_x = tc.linspace(-1,1, self.x_height)
            freqs_y = tc.linspace(-1,1, self.x_width)
            #print(f'{freqs=}')

            # produit de la porte avec les fréquances
            porte_x = tc.where(freqs_x >= a, 0, 1)
            porte_x = tc.where(freqs_x <=-a, 0, porte_x)

            porte_y = tc.where(freqs_y >= a, 0, 1)
            porte_y = tc.where(freqs_y <=-a, 0, porte_y)

            # mise en carré
            porte_x, porte_y = tc.meshgrid(porte_x, porte_y)
            porte = porte_x * porte_y

            # produit dans l'esapce des fréquances avant le return
            #norm1 = tc.linalg.norm(tc.fft.ifft(porte).view(-1), ord=1)
            self.h = tc.fft.ifftshift(porte)#/norm1

                # Stockage des infos

            self.info['filtre_type'] = filtre
            self.info['filtre_param'] = parametre
            self.info['filtre'] = self.h.numpy()

            print(f"\n Filtre porte passe-bas sur [-{a},{a}]x[-{a},{a}]")

    # estiamtion de la RIP constante

    def estimate_RIP(self, DataSet, saveas=None) -> None:
        '''
            :param iterable DataSet:    set d'image sur lequel estimer la RIP constant
            :param bool saveas:         s'il faut sauvegarder ou non les résultats (format pkl)

            Estime la RIP constant sur le DataSet via la formule (6) du rapport et affiche l'histogramme
        '''
        val = -tc.ones(len(DataSet))
        
        for i, (img,_) in tqdm(enumerate(DataSet)):
            img = img.squeeze()
            norm_x = tc.einsum('ij, ij -> ', img, img)
            norm_Ax = tc.einsum('ij, ij -> ', self.A(img), self.A(img))

            val[i] += norm_Ax/norm_x

        val = np.absolute(val.numpy())
        gammax = max(val)
        print(f'\n Valeur minimal pour gamma :  {gammax}')
        val, bornes = np.histogram(val, bins=500)

        fig = plt.figure(figsize=(16, 8))

        plt.suptitle("nombre d'image vérifiant le RIP pour chaque gamma")
        plt.xlabel('gamma')
        plt.ylabel("nombre d'image")
        plt.stairs(val, bornes, fill=True)

        if saveas is not None:
            f = open(f'{self.path2save}/estim_gamma-{saveas}.pkl', 'wb')
            rick.dump([val, bornes, gammax], f)
            plt.savefig(f'{self.path2save}/estim_gamma-{saveas}.jpg')

        plt.show()
    
    # matrice A et sa transposé
        
    def A(self, x) -> tc.Tensor:
        '''
            :param tc.Tensor x:    image à sous-échantillonnée
            Applique l'opérateur de mesure A à x. Voir annexe B du rapport pour plus de détail sur don fonctionnement.
        '''
        if self.h is None:
            return x[::self.gap_x, ::self.gap_y]

        else:
            Fx = tc.fft.fft2(x)
            C_hx = tc.fft.ifft2(self.h*Fx).real
            return C_hx[::self.gap_x, ::self.gap_y]

    def tA(self, y) -> tc.Tensor:
        '''
            :param tc.Tensor y:    image sous-échantillonnée à agrandir

            Applique la transposé de A à y. Voir annexe B du rapport pour plus de détail sur don fonctionnement.
        '''
            # Produit de y avec la transposé tS

        # préremplissage de x=tS(y) par des 0
        x = tc.zeros((self.x_height,self.x_height))

        # le-dit remplissage (voir rapport pour les détails)
        for i in range(self.y_height):
            for j in range(self.y_width):
                x[i*self.gap_x, j*self.gap_y] = y[i, j]
        

            # Produit de x avec tC_h
        if self.h is None:
            return x

        else:
            iFx = tc.fft.ifft2(x)

            return  tc.fft.fft2(self.h * iFx).real

    # les descentes de gradien

    def PGD(self, target, init, pas, Niter, back_tracking=False) -> np.array:
        '''
            :param tc.Tensor target:       image cible de la descente
            :param str|tc.Tensor init:     'tA(y_0)', 'random', 'rand+AE', tc.Tensor : initialisation de la descente
            :param float pas:           pas de descente
            :param int Niter:           nombre d'itérations
            :param bool back_tracking:  si oui ou non il faut faire du back_tracking

            :rtn np.Array self.histo['img'][-1]: resultats de la descente

            Effectue la descente de gradient projetée avec self.AE comme projection. L'historique de la descente est ajouté au dictionnaire self.histo
                et les informations au dictionnaire self.info.
            Plusieurs initialisation sont possibles : par backprojection avec 'tA(y_0)', par une image aléatoire, ou aléatoire puis passé dans l'auto-encoder
                pour effectuer une premier projection. 
            A chaque itérations le PSNR entre l'image actuelle et la target est comparer à celui la target avec sa version auto-encoder.
            Il y a deux boucles pour la descente : avec et sans backtracking.            
        '''

            # Calculs préliminaires

        # cible de la déscente
        x_0 = target.squeeze()           # image que l'on veut retrouver        
        y_0 = self.A(x_0)   # sa version sous-échantillonnée (à reconstruire)  

        # PSNR check
        psnr_etalon = peak_signal_noise_ratio(self.AE(x_0.view(1,-1)).view(self.x_height, self.x_width), x_0, 1.)
        print(f'\n PSNR étallon :  {psnr_etalon}')
        psnr_check = False

        # stockage des l'évolution des x et F(x)
        Xs = tc.empty((Niter, *x_0.shape))
        Fs = tc.empty(Niter)
        PSNRs = tc.empty(Niter)

        # stockage des infos sur la descente
        if back_tracking == True:
            self.info.update({'pas':'bk', 'x_0':x_0.detach().numpy(), 'y_0':y_0.numpy(), 'PSNR_etalon':psnr_etalon.numpy()})
        else:
            self.info.update({'pas':pas, 'x_0':x_0.detach().numpy(), 'y_0':y_0.numpy(), 'PSNR_etalon':psnr_etalon.numpy()})


            # Initialisation

        # backtracking de y_0
        if init == 'tA(y_0)':
            Xs[0] = self.tA(y_0)

        # prise d'un vecteur au hasard
        elif init == 'random':
            Xs[0] = tc.rand_like(x_0)
            #Xs[0] = tc.normal(mean=0.0, std=1.0, size=(self.x_height, self.x_width))  

        # prise d'un vecteur au hasard mais passé dans l'autoencoder pour qu'il soit de la forme f_D(u)
        elif init == 'rand+AE':
            Xs[0] = self.AE(tc.rand_like(x_0.view(1,-1))).view(self.x_height, self.x_width)
            #Xs[0] = self.AE(tc.normal(mean=0.0, std=1.0, size=(1, self.x_height*self.x_width))).view(self.x_height, self.x_width)

        # sinon, on prend le vecteur donné en entrée (reshape au cas où)
        else:
            Xs[0] = init.squeeze()


            # Premiers calculs pour lancer la boucle

        diff = self.A(Xs[0])-y_0
        Fs[0] = tc.linalg.norm(diff)


            # La descente de gradient (enfin on y vient)

        # avec back tracking pour le pas 
        if back_tracking == True:
        
            for n in trange(Niter-1):

                grad = self.tA(diff)

                pas, x, Fs[n+1], diff = self.PGD_backtrack(pas, Xs[n], Fs[n], grad, y_0)

                Xs[n+1] = self.AE(x.view(1,-1)).view(self.x_height, self.x_width)

                # check pour savoir si on peut faire meiux
                PSNRs[n] = peak_signal_noise_ratio(Xs[n], x_0, 1.)

                if psnr_check != True and PSNRs[n] > psnr_etalon:
                    psnr_check = True
                    print(f"\t PSNR check passé à l'itérations {n}\n")

        # sans back tracking, e.i. à pas fixe
        else:
            for n in trange(Niter-1):

                # calculs gradient
                grad = self.tA(diff)

                # descente projetée
                x = Xs[n] - pas*grad
                Xs[n+1] = self.AE(x.view(1,-1)).view(self.x_height, self.x_width)

                # stockage des F(x_{n+1})
                diff = self.A(Xs[n+1]) - y_0
                Fs[n+1] = tc.linalg.norm(diff)


                # check pour savoir si on peut faire meiux
                PSNRs[n] = peak_signal_noise_ratio(Xs[n], x_0, 1.).detach()
                
                if psnr_check != True and PSNRs[n] > psnr_etalon:
                    psnr_check = True
                    print(f"\t PSNR check passé à l'itérations {n}\n")


        # mise en accès des historiques (en ne gardant que les valeurs vrtargetent utilisées, à savoir les i premières)
        PSNRs[-1] = peak_signal_noise_ratio(Xs[-1], x_0, 1.).detach()

        self.histo = { 'img'  : Xs.detach().numpy(),
                      'value' : Fs.detach().numpy(),
                       'PSNR' : np.round(PSNRs.numpy(), decimals=2)}

        return self.histo['img'][-1]

    @lru_cache(maxsize=None)
    def PGD_backtrack(self, pas, x, Fx, grad, y_0) -> list:

        norm_grad = tc.einsum('ij, ij -> ', grad, grad)    # /!\ norme au carré 

        x_next = x - pas*grad
        diff_next = self.A(x_next) - y_0
        F_next = tc.einsum('ij, ij -> ', diff_next, diff_next)

        if F_next > Fx - pas/2 * norm_grad:
            print('réduction du pas')
            return self.PGD_backtrack(pas/2, x, Fx, grad, y_0)

        else:
            return pas, x_next, F_next, diff_next

    def LGD(self, target, init, pas, Niter, back_tracking=False) -> np.array:
        '''
            :param tc.Tensor target:       image cible de la descente
            :param str|tc.Tensor init:     'f_E(tA(y_0))', 'random', 'rand+f_E', tc.Tensor : initialisation de la descente
            :param float pas:           pas de descente
            :param int Niter:           nombre d'itérations
            :param bool back_tracking:  si oui ou non il faut faire du back_tracking

            :rtn np.Array self.histo['img'][-1]: resultats de la descente

            Effectue la descente de gradient depuis l'espace latent. L'historique de la descente est ajouté au dictionnaire self.histo
                et les informations au dictionnaire self.info.
            Plusieurs initialisation sont possibles : par backprojection avec 'f_E(tA(y_0))', par une image aléatoire, ou aléatoire puis passé dans l'auto-encoder
                pour effectuer une premier projection. 
            A chaque itérations le PSNR entre l'image actuelle et la target est comparer à celui la target avec sa version auto-encoder.
            Il y a deux boucles pour la descente : avec et sans backtracking.            
        '''

            # Calculs préliminaires

        # le fonction enco/deco
        f_E = self.AE.encoder
        f_D = self.AE.decoder

        # cible de la déscente
        x_0 = target.squeeze()           # image que l'on veut retrouver        
        y_0 = self.A(x_0)       # sa version sous-échantillonnée (à reconstruire)  

        # psnr check
        psnr_etalon = peak_signal_noise_ratio(self.AE(x_0.view(1,-1)).view(self.x_height, self.x_width), x_0, 1.)
        print(f'\n PSNR étallon :  {psnr_etalon}')
        psnr_check = False

        # stockage des l'évolution des x et F(x)
        Xs = tc.empty((Niter, *x_0.shape))
        Fs = tc.empty(Niter)
        PSNRs = tc.empty(Niter)

        # stockage des infos sur la descente
        if back_tracking == True:
            self.info.update({'pas':'bk', 'x_0':x_0.detach().numpy(), 'y_0':y_0.numpy(), 'PSNR_etalon':psnr_etalon.numpy()})
        else:
            self.info.update({'pas':pas, 'x_0':x_0.detach().numpy(), 'y_0':y_0.numpy(), 'PSNR_etalon':psnr_etalon.numpy()})


            # Initialisation

        # backprojection 
        if init == 'f_E(tA(y_0))':
            u = f_E(self.tA(y_0).view(1,-1))

        # prise d'un vecteur au hasard
        elif init == 'random':
            u = tc.rand((1, self.u_height))
            #u = tc.normal(mean=0.0, std=5.0, size=(1, self.u_height))

        # prise d'un vecteur au hasard mais passé dans l'encoder
        elif init == 'rand+f_E':
            u = f_E(tc.rand_like(x_0.view(1,-1)))

        # sinon, on prend le vecteur donné en entrée
        else:
            u = tc.empty_like(init)
            u.copy_(init)
        


            # Premiers calculs pour lancer la boucle

        Xs[0] = f_D(u.detach()).view(self.x_height, self.x_width)
        diff = self.A(Xs[0])-y_0
        Fs[0] = tc.linalg.norm(diff)

        # PSNR check


            # La descente de gradient (enfin on y vient)

        # avec back tracking pour le pas 
        if back_tracking == True:
            for n in trange(Niter-1):

                # calculs gradient
                Jacf_D = tc.autograd.functional.jacobian(f_D, u).squeeze()
                grad = self.tA(diff)
                grad = tc.mm(Jacf_D.transpose_(1,0), grad.view(-1,1)).view(1,-1)
                
                pas, u, Xs[n+1], Fs[n+1], diff = self.LGD_backtrack(pas, u, Fs[n], grad, y_0)

                # calculs de PSNR
                PSNRs[n] = peak_signal_noise_ratio(Xs[n], x_0, 1.)

                if psnr_check != True and PSNRs[n] > psnr_etalon:
                    psnr_check = True
                    print(f"\t PSNR check passé à l'itérations {n}")
        
        else:
            for n in trange(Niter-1):

                # calculs gradient
                Jacf_D = tc.autograd.functional.jacobian(f_D, u).squeeze()

                grad = self.tA(diff)
                grad = tc.mm(Jacf_D.transpose_(1,0), grad.view(-1,1)).view(1,-1)

                # descente
                u -= pas*grad

                # pré-calcul (pour éviter les redondances)
                Xs[n+1] = f_D(u).view(self.x_height, self.x_width)

                # stockage de F(x_{n+1})
                diff = self.A(Xs[n+1]) - y_0
                Fs[n+1] = tc.linalg.norm(diff)

                # calculs de PSNR
                PSNRs[n] = peak_signal_noise_ratio(Xs[n], x_0, 1.).detach()

                if psnr_check != True and PSNRs[n] >= psnr_etalon:
                    psnr_check = True
                    print(f'\t PSNR check passed après {n} itérations')


        # mise en accès des historiques
        PSNRs[-1] = peak_signal_noise_ratio(Xs[-1], x_0, 1.).detach()

        self.histo = { 'img' : Xs.detach().numpy(),
                      'value': Fs.detach().numpy(),
                      'PSNR' : np.round(PSNRs.numpy(), decimals=2)}

        return self.histo['img'][-1]

    @lru_cache(maxsize=None)
    def LGD_backtrack(self, pas, u, Fu, grad, y_0) -> list:

        norm_grad = tc.einsum('ij, ij -> ', grad, grad)    # /!\ norme au carré 

        u_next = u - pas*grad
        x_next = self.AE.decoder(u).view(self.x_height, self.x_width)

        diff_next = self.A(x_next) - y_0
        F_next = tc.einsum('ij, ij -> ', diff_next, diff_next)

        if F_next > Fu - pas/2 * norm_grad:
            print('réduction du pas')
            return self.LGD_backtrack(pas/2, u, Fu, grad, y_0)

        else:
            return pas, u_next, x_next, F_next, diff_next

def make_noisy(img, bruit='gaussien', param=0.01) -> tc.Tensor:
    '''
        :param tc.Tensor img:  tenseur à bruité
        :param str bruit:   type de bruit : 'gaussien' ou 'uniforme'
        :param float param: paramètre associé au bruit 

        Ajoute un bruit additif au tenseur img (qui n'est pas forcément une image d'ailleurs) en fonction du type de bruit :
            - Si gaussien, param donne l'écart type du bruit 
            Si uniforme, param donne l'intervalle [-param, param] où seront tirés les valeurs du bruit   
    '''

    if bruit == 'gaussien':
        return img + tc.normal(1., param, img.shape)

    if bruit == 'uniforme':
        return img + (tc.rand_like(img)*2 - 1)*param


if __name__=='__main__':


###     Loading datas

        # Auto-encodeur et dataset

    # charchement du set de test
    TrainSet = datasets.MNIST('../dataset', train=True, transform=transforms.ToTensor(), download=False) #=True)

    # initialisation de la classe pour les calculs
    SupRes = SuperResolution(path2autoencoder='../resultats/autoencoder/Autoencoder 100', path2save='../resultats')


###     Plot de quelque auto-encodage
    '''
    indexes = np.random.randint(0, len(TrainSet), 8)
    print(f'\n {indexes=}\n')

    imgs = [iTrainSet[i][0].squeeze() for i in indexes]
    for d in [800, 400, 200, 100]:

        # changement de la taille de l'espace latent
        dim_latent = d
        SupRes.set_autoencoder(f'../resultats/autoencoder/Autoencoder {d}')
        SupRes.set_sizes(u_lenth=d)

        Savings.plot_perfAE(imgs, saveas=f'autoencoder/guess-{d}')
    '''


###     Estimation de la RIP constant gamma
    '''
    
    # sans passe-bas
    SupRes.set_passebas(filtre='sans')
    SupRes.estimate_RIP(TrainSet, saveas='s')

    # avec passe-bas gaussien
    SupRes.set_passebas(filtre='gaussien', parametre=0.6)
    SupRes.estimate_RIP(TrainSet, saveas='g')
    '''


###     Comparaison des passe-bas
    '''
    
        # Changement d'image

    index = np.random.randint(0, len(TrainSet))
    print(f"\n Index associé à l'image : {index}\n")

    img = TrainSet[index][0].squeeze()


        # Gaussien

    # selection du filtre
    filtres = ['sans'] + ['gaussien']*7
    params = [0] + [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]   # sdv dans l'esapce des fréquances

    # plot (+ sauvegarde)
    SupRes.compare_filtre(img, filtres, params, shift=True, saveas=)


        # Porte

    # selection du filtre
    #filtres = ['sans'] + ['porte']*7
    #params = [0] + [0.9, 0.75,  0.6, 0.4, 0.25, 0.1]  # taille de la porte

    # plot (+ sauvegarde)
    SupRes.compare_filtre(img, filtres, params, shift=True, saveas=)

    
    '''


###     Premiers résultats de descentes
    '''
        
        # Set up

    # changement d'une nouvelle image
    index = np.random.randint(0, len(TrainSet))
    print(f"\n Indexe associé à l'image : {index}\n")

    img = TrainSet[index][0].squeeze()
    plt.imshow(img, cmap='magma')
    plt.show()

    # filtre
    #SupRes.set_passebas(filtre='sans')
    SupRes.set_passebas(filtre='gaussien', parametre=0.6)

        # PGD

    # la descente
    SupRes.PGD(target=img, init='tA(y_0)', pas=4.25, Niter=50)#, back_tracking=True)

    # plot du résultat
    SupRes.plot_descente(methode='PGD', img3=SupRes.info['y_0'], saveas='test')#, saveas='premiere-g') #-s')

    # petite animation pour voir ce qu'il se passe
    SupRes.animation(N=50)


        # LGD

    # la descente
    SupRes.LGD(target=img, init='', pas=0.3, Niter=100)#, back_tracking=True)

    # plot du résultat
    SupRes.plot_descente(methode='LGD', img3=SupRes.info['y_0'])#, saveas='premiere-g') #-s')

    # petite animation pour voir ce qu'il se passe
    SupRes.animation(N=100)
    '''

###     Sand box

    index = np.random.randint(0, len(TrainSet))
    print(f"\n Index associé à l'image : {index}\n")

    img = TrainSet[index][0].squeeze()


    pas_o = [0.1, 1, 2.5, 5, 10, 20 , 50]
    pas_s = [4, 4.25, 4.5, 4.75]
    pas_g = [2.5, 3, 3.5, 4, 4.5, 5]

    SupRes.set_sizes(y_shape=(14,7))


    SupRes.set_passebas(filtre='sans')
    SupRes.multiplot_descente(methode='PGD', target=img, inits=['tA(y_0)']*10, pas=pas_s, Niter=20)#, saveas='')
    
    SupRes.set_passebas(filtre='gaussien', parametre=0.6)
    SupRes.multiplot_descente(methode='PGD', target=img, inits=['tA(y_0)']*10, pas=pas_g, Niter=20)#, saveas='')
    plt.show()





        # Descente en masse pour valider

    # selection
    nb = 8
    indexes = np.random.randint(0, len(TrainSet), nb)
    print(f"\n Indexes associés aux images : {indexes}\n")

    imgs = [TrainSet[i][0].squeeze() for i in indexes]

    # backproj
    SupRes.set_passebas(filtre='sans')
    SupRes.multiplot_multitarget(methode='PGD', targets=imgs, inits=['tA(y_0)']*nb, pas=[3.75]*nb, Niter=20)#, saveas='')
    
    SupRes.set_passebas(filtre='gaussien', parametre=0.6)
    SupRes.multiplot_multitarget(methode='PGD', targets=imgs, inits=['tA(y_0)']*nb, pas=[3.5]*nb, Niter=20)#, saveas='')

    plt.show()

    # random
    SupRes.set_passebas(filtre='sans')
    SupRes.multiplot_multitarget(methode='PGD', targets=imgs, inits=['random']*nb, pas=[3.75]*nb, Niter=20)#, saveas='')
    
    SupRes.set_passebas(filtre='gaussien', parametre=0.6)
    SupRes.multiplot_multitarget(methode='PGD', targets=imgs, inits=['random']*nb, pas=[3.5]*nb, Niter=20)#, saveas='')

    plt.show()


'''  TODO

    PGD :

        
    STRUCTURE

    - checks les fautes 


    POUR L'ARTICLE (rien que ça !)

    - check l'état de l'art si tes résultats existes,
        si ca a ses limites, est-ce qu'elles sont explorer

    - est-ce que y'a besoin de plus sofistiqué qu'un MLP ?

    - essaye de Fashion-MNIST pour la robustesse
'''