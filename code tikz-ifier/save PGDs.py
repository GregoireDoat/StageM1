from tikzsaves import *



    # Chemin vers les données / pour sauvegarde
Savings = Save2Tikz(path2data='../resultats', path2save='../latex/resultats', color_map='magma')

    # Différentes initialisation
#Savings.save_multplot(methode='PGD', globname='differentes initialisations/inits-s', names=range(1,5), saveas='inits')
#Savings.save_multplot(methode='PGD', globname='differentes initialisations/inits-g', names=range(1,5), saveas='inits')


    # En fonction de p et q
tailles = ['mini', 'small', 'mid1', 'mid2', 'big']
#Savings.save_multplot(methode='PGD', globname='differentes tailles/size-s', names=tailles, saveas='size', )
#Savings.save_multplot(methode='PGD', globname='differentes tailles/size-g', names=tailles, saveas='size', )


    # En fonction de d
lenths = [100, 200, 400, 800]

for d in lenths:
    Savings.save_multplot(methode='PGD', globname=f'differents latents/lat-s_{d}', names=range(1,5), saveas=f'lat-{d}')
    Savings.save_multplot(methode='PGD', globname=f'differents latents/lat-g_{d}', names=range(1,5), saveas=f'lat-{d}')
