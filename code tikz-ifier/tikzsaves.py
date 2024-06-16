import os
import pickle as rick

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import tikzplotlib as tpl
#pip install matplotlib==3.7.0 pour tikzer
# a faire dans un venv évidemment

class Save2Tikz():

    def __init__(self, path2data, path2save, color_map='gray') -> None:
        self.path2data = path2data
        self.path2save = path2save 

        self.color_map = color_map

    def get_result(self, methode, name) -> list[dir]:
        ''' Charge le résultat 'name' obtenue par la méthode 'methode' '''
        with open(f'{self.path2data}/{methode}/{name}.pkl', 'rb') as f :
            return rick.load(f)

    def make_path(self, directory, savename) -> str:
        ''' Retourne le chemin pour le fichier 'savename' dans le dossier 'directroy
                en créant le dossier si besoin '''
        # création du dossier de sauvegarde
        path2save = f'{self.path2save}/{directory}'

        if not os.path.exists(path2save):
            os.makedirs(path2save)

        # nom pour sauvegarde
        return f'{path2save}/{savename}'

    # fonctions de sauvegardes

    def save_RIPestimation(self, name, saveas) -> None:
        ''' Sauvegarde les données produites par la fonction 'estimate_RIP' du main_code au format Tikz '''

        save_name = self.make_path('RIPestimation', saveas)

        val, borne, gammax = rick.load(open(f'{self.path2data}/{name}.pkl', 'rb'))

        plt.plot(borne[:-1], val)
        tpl.clean_figure()
        tpl.save(f'{save_name}.tex', axis_width=r'\figwidth', axis_height=r'\figheight')
        plt.close()

    def save_AElearning(self, name, saveas) -> None:
        ''' Sauvegarde les graphes de performances de l'auto-encoder 'name' au format Tikz '''
        # nom pour sauvegarde
        save_name = self.make_path('auto-encoder', saveas)

        # chargement des données
        loss_trn, PSNR_trn, loss_tst, PSNR_tst = np.load(f'{self.path2data}/autoencoder/Loss-PSNRs - {name}.npy')   

        # plot des loss
        plt.plot(loss_trn)
        plt.plot(loss_tst)
        plt.gca().set_yscale('log')
        tpl.clean_figure()
        tpl.save(f'{save_name}_Loss.tex', axis_width=r'\figwidth', axis_height=r'\figheight')
        plt.close()

        # plot des PSNR
        plt.plot(PSNR_trn)
        plt.plot(PSNR_tst)
        tpl.clean_figure()
        tpl.save(f'{save_name}_PSNR.tex', axis_width=r'\figwidth', axis_height=r'\figheight')
        plt.close()

    def save_result(self, methode, name, saveas, n_first=0) -> None:
        ''' Sauvegarde en format Tikz les infos de la figure produite par la fonction 'plot_descente' '''
        # nom pour sauvegarde
        save_name = self.make_path(methode, saveas)

        # chargement des résultats
        info, histo = self.get_result(methode=methode, name=name)

        # info ajoutées aux noms
        data = f"pas={info['pas']}_filtre={info['filtre_type'][0]}-{info['filtre_param']}"


            # Sauvegardes des images target, A(target), initialisation et résultat

        plt.imsave(f'{save_name}-target-{data}.png',  info['x_0'], cmap=self.color_map)
        plt.imsave(f'{save_name}-comptarg-{data}.png',  info['y_0'], cmap=self.color_map)
        plt.imsave(f"{save_name}-init-{data}.png",   histo['img'][0],  cmap=self.color_map)
        plt.imsave(f"{save_name}-guess-{data}.png",  histo['img'][-1], cmap=self.color_map)

        if n_first != 0:
            for i in range(1,n_first):
                plt.imsave(f"{save_name}-iter{i}-{data}.png", histo['img'][i],  cmap=self.color_map)



            # Sauvegardes des courbes 

        # F & norme de u (s'il existe)
        plt.plot(histo['value'], color='blue')

        tpl.clean_figure()
        tpl.save(f'{save_name}-F-{data}.tex', axis_width=r'\figwidth', axis_height=r'\figheight')
        plt.close()

        # PSNR
        plt.plot(histo['PSNR'], color='orange')
        plt.axhline(info['PSNR_etalon'], color=(0.5, 0.5, 0.5))

        tpl.clean_figure()
        tpl.save(f'{save_name}-PSNR-{data}.tex', axis_width=r'\figwidth', axis_height=r'\figheight')
        plt.close()

    def save_multplot(self, methode, globname, names, saveas, multitarget=False) -> None:
        ''' Sauvegarde en format Tikz les infos de la figure produite par la fonction 'multiplot_descente' '''
        # nom pour sauvegarde
        save_name = self.make_path(methode, saveas)
        data = None

            # Sauvegardes
        values = []
        PSNRs = [0.]
        Us = []

        for name in names:

            # chargement des résultats
            info, histo = self.get_result(methode=methode, name=f'{globname}_{name}')

            # info ajoutées aux noms
            data = f"pas={info['pas']}_filtre={info['filtre_type'][0]}-{info['filtre_param']}"

            values.append(histo['value'])
            PSNRs[0] = info['PSNR_etalon']
            PSNRs.append(histo['PSNR'])


            # sauvegardes des images target, A(target), initialisation et résultat
            plt.imsave(f"{save_name}_{name}-init-{data}.png",   histo['img'][0],  cmap=self.color_map)
            plt.imsave(f"{save_name}_{name}-guess-{data}.png",  histo['img'][-1], cmap=self.color_map)


            if multitarget == True:
                plt.imsave(f"{save_name}_{name}-target-{data}.png", info['x_0'], cmap=self.color_map)
            
        data = info['filtre_type'][0]
        plt.imsave(f"{save_name}_{name}-compatarget-{data}.png", info['y_0'], cmap=self.color_map)
        if multitarget == False:
            plt.imsave(f"{save_name}-target-{data}.png", info['x_0'], cmap=self.color_map)

        # sauvegardes des courbes F
        for i, val in enumerate(values):
            plt.plot(val, label=f'{i+1}')
        
        tpl.clean_figure()
        tpl.save(f'{save_name}-Fs-{data}.tex', axis_width=r'\figwidth', axis_height=r'\figheight')
        plt.close()

        # sauvegardes des courbes PSNR
        for i, PSNR in enumerate(PSNRs[1:]):
            plt.plot(PSNR, label=f'{i+1}')
        plt.axhline(PSNRs[0], color=(0.5, 0.5, 0.5))
        #plt.legend()

        tpl.clean_figure()
        tpl.save(f'{save_name}-PSNRs-{data}.tex', axis_width=r'\figwidth', axis_height=r'\figheight')
        plt.close()

    # animation

    def animation(self, methode, name, Nframe, saveas) -> None:
        ''' Produit une animation comme 'animation' du 'main_code' mais cette fois la sauvegarde est possible !'''
            # Travail préliminaire

        # nom pour sauvegarde
        save_name = self.make_path(animation, f'{methode} {saveas}')

        # chargement des résultats
        self.methode = methode
        info, self.histo = self.get_result(methode=methode, name=name)


        # info à ajouter aux noms
        data = f"pas={info['pas']}_filtre={info['filtre_type'][0]}-{info['filtre_param']}"


            # Set up figure pour animation

        # création de la figure
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
        self.ax[1].set_xlim(0, Nframe)
        self.ax[1].set_ylim(0, max(self.histo['value']))

        self.ax[2].set_xlim(0, Nframe)
        self.ax[2].set_ylim(0, max(self.histo['PSNR']))

            # Lancement et sauvegarde

        # lancement de l'animation
        anim = animation.FuncAnimation(fig=self.fig, func=self.anim_update, frames=Nframe, init_func=self.anim_init)

        # sauvegarde 
        #writer = animation.PillowWriter(fps=30)
        anim.save(f'{saveas}.gif', fps=15, dpi=100, writer='pillow', progress_callback=self.anim_verbose)
        plt.close()

    def anim_init(self) -> None:
        self.line = [None]*2

        self.ax[0].imshow(self.histo['img'][0], cmap=self.color_map)

        self.line[0], = self.ax[1].plot([0], self.histo['value'][0], color='black')
        self.line[1], = self.ax[2].plot([0], self.histo['PSNR'][0], color='orange')

    def anim_update(self, n) -> None:
        absc = np.arange(n)

        self.ax[0].imshow(self.histo['img'][n], cmap=self.color_map)

        self.line[0].set_data(absc, self.histo['value'][:n])
        self.line[1].set_data(absc, self.histo['PSNR'][:n])

    def anim_verbose(self, n, Nframe) -> None:
        if n==0:
            print("\nSauvegarde d'animation en cours...")
        print(f'{n} frame sur {Nframe}')



if __name__=='__main__':


# Chemin vers les données / pour sauvegarde

    Savings = Save2Tikz(path2data='../resultats', path2save='../latex/resultats', color_map='magma')

    
# Sauvegarde des historique de perf des auto-encodeurs
    #for dim in [100, 200, 400, 800]:
    #    Savings.save_AElearning(f'Autoencoder {dim}', saveas=f'AE-{dim}')



# Sauvegarde des estimation de la RIP constante

    #Savings.save_RIPestimation(name='estim_gamma-s', saveas='gamma-s')
    #Savings.save_RIPestimation(name='estim_gamma-g', saveas='gamma-g')
    


# Sauvegarde d'animation

    #Savings.animation(methode='PGD', name=f'test', Nframe=50, saveas='../resultats/PGD etrange')
    #Savings.animation(methode='LGD', name=f'premiere-g', Nrame=50, saveas='../resultats/LGD animation')


#   Sand box

    #Savings.save_result('PGD', 'test', 'etrange-g')