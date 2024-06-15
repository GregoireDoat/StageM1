import numpy as np
import skimage as ski

import torch as tc 
import torchvision as tcv
from torch import nn
from torcheval.metrics.functional import peak_signal_noise_ratio

import matplotlib.pyplot as plt
from tqdm import tqdm,trange

'''
    Entrainement d'auto-encodeur
'''


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder = nn.Sequential( #two layers encoder
            nn.Linear(784,1500),    # 784 = 15*15
            nn.Sigmoid(), #ReLU, Tanh, etc.
            nn.Linear(1500,800),
            nn.Sigmoid(),# input is in (0,1), so select this one
        )
        self.decoder=nn.Sequential( #two layers decoder
            nn.Linear(800, 1500),
            nn.Sigmoid(),
            nn.Linear(1500, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return decoded



if __name__=='__main__':


###     Set des données


        # Charchement des données

    path2dataset = '../dataset'
    TrainSet = tcv.datasets.MNIST(path2dataset, train=True, transform=tcv.transforms.ToTensor(), download=False) #=True)
    TestSet = tcv.datasets.MNIST(path2dataset, train=False, transform=tcv.transforms.ToTensor(), download=False) #=True)


        # Quelque infos pour savoir où on en est

    print(f'\n Taille du train set : {len(TrainSet)}\t Taille du test set : {len(TestSet)}')
    print(f'\n Taille des images : {TrainSet[0][0].squeeze().shape}\n')




###     La Descente


        # Set up de la déscente

    # hyper-param du réseau
    LearningRate = 0.001
    BatchSize = 100
    Nepoch = 60

    # loaders
    TrainLoader = tc.utils.data.DataLoader(TrainSet, batch_size=BatchSize, shuffle=True)
    TestLoader = tc.utils.data.DataLoader(TestSet, batch_size=BatchSize, shuffle=False)

    # le réseau
    Net = AutoEncoder()
    print(Net)

    # norme 2 en fonction loss
    Loss = tc.nn.MSELoss()
    optimizer = tc.optim.Adam(Net.parameters(), lr=LearningRate)     

    # stockage des Loss/PSNR
    lstLoss_trn = tc.zeros(Nepoch)     # /!\ Loss moyen par epoch
    lstPSNR_trn = tc.zeros(Nepoch) # /!\ PSNR par batch moyen par epoch

    lstLoss_tst = tc.zeros(Nepoch)     # /!\ Loss moyen par epoch
    lstPSNR_tst = tc.zeros(Nepoch) # /!\ PSNR par batch moyen par epoch


        # La descente de gradient        

    # Boucle sur les époches
    for n in range(Nepoch):

        # Boucle sur les batchs
        for x,_ in tqdm(TrainLoader):
            # mise à la bonne shape de x :
            #   (100, 784) au lieu de (100, 1, 28, 28)
            x = x.view(BatchSize,-1)
            # pass forward
            y = Net(x)

            # calcul du Loss
            loss = Loss(y, x)

            # pass backaward
            optimizer.zero_grad()       # remise à 0 du gradient
            loss.backward()             # calcul du gradient 
            optimizer.step()            # maj des poids

            # stockage des loss et PSNR
            lstLoss_trn[n] += loss.detach()
            lstPSNR_trn[n] += peak_signal_noise_ratio(x, y, data_range=1.).detach()


            # Cacul du PSNR sur le test set

        with tc.no_grad():
            for x, _ in tqdm(TestLoader):

                x = x.view(BatchSize, -1)
                y = Net(x)

                lstLoss_tst[n] += Loss(x, y).detach()
                lstPSNR_tst[n] += peak_signal_noise_ratio(x, y, data_range=1.).detach()
        
        # normalisation des résultats
        lstLoss_trn[n] /= len(TrainLoader)
        lstPSNR_trn[n] /= len(TrainLoader)
        
        lstLoss_tst[n] /= len(TestLoader)
        lstPSNR_tst[n] /= len(TestLoader)
    
        print(f"\t Epoch : {n+1}\t |\t MSE on train : {lstLoss_trn[n]} |\t PSNR on train : {lstPSNR_trn[n]}\t |\t PSNR on test : {lstPSNR_tst[n]}\n")
    

    # Mise au format numpy des résultats

    lstLoss_trn = lstLoss_trn.detach().numpy()
    lstPSNR_trn = lstPSNR_trn.detach().numpy()

    lstLoss_tst = lstLoss_tst.detach().numpy()
    lstPSNR_tst = lstPSNR_tst.detach().numpy()
 



###     Les résultats


        # Sauvegarde du réseau

    # nom et chemin de sauvegarde
    path2save = '../resultats/autoencoder'
    autoenco_name = 'Autoencoder 800'

    # les-dites sauvegardes
    tc.save(Net, f'{path2save}/{autoenco_name}')
    np.save(f'{path2save}/Loss-PSNRs - {autoenco_name}.npy', np.array([lstLoss_trn, lstPSNR_trn, lstLoss_tst, lstPSNR_tst]))


        # Plot du loss

    # graph du loss et des PSNRs
    fig, ax = plt.subplots(1,2)

    ax[0].set_title('Evolutiont du loss')
    ax[0].plot(lstLoss_trn, label='on train set')
    ax[0].plot(lstLoss_tst, label='on test set')
    ax[0].set_yscale('log')

    ax[1].set_title('Evolutiont du PSNR')
    ax[1].plot(lstPSNR_trn, label='on train set')
    ax[1].plot(lstPSNR_tst, label='on test set')

    ax[1].legend()
    plt.show()


       # Plot de quelques prédictions

    # liste des images
    lst1 = np.random.randint(0,len(TestSet), 8)
    lst2 = np.random.randint(0,len(TestSet), 8)

    # le-dit plot
    fig, ax = plt.subplots(4, len(lst1))

    for i, (j1,j2) in enumerate(zip(lst1, lst2)):

        # première et deuxième lignes
        img = TestSet[j1][0].squeeze()
        AEimg = Net(img.view(1,-1)).detach().view(28, 28)

        ax[0,i].set_title('Image à prédure')
        ax[0,i].imshow(img, cmap='magma')

        ax[1,i].set_title("Prédiction")
        ax[1,i].imshow(AEimg, cmap='magma')

        # troisème et quatrième ligne
        img = TestSet[j2][0].squeeze()
        AEimg = Net(img.view(1,-1)).detach().view(28, 28)

        ax[2,i].set_title('Image à prédure')
        ax[2,i].imshow(img, cmap='magma')

        ax[3,i].set_title("Prédiction")
        ax[3,i].imshow(AEimg, cmap='magma')

    plt.show()