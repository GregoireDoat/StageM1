from tikzsaves import *



    # Chemin vers les données / pour sauvegarde
Savings = Save2Tikz(path2data='../resultats', path2save='../latex/resultats', color_map='magma')


    # Initialisation par backprojection
Savings.save_multplot(methode='PGD', globname='backproj-s', names=range(1,9), saveas='backproj')
Savings.save_multplot(methode='PGD', globname='backproj-g', names=range(1,9), saveas='backproj')


    # Initialisation random uniforme
Savings.save_multplot(methode='PGD', globname='rand_unif-s', names=range(1,9), saveas='rand_unif')
Savings.save_multplot(methode='PGD', globname='rand_unif-g', names=range(1,9), saveas='rand_unif')


    # Initialisation random gaussienne
Savings.save_multplot(methode='PGD', globname='rand_gauss-s', names=range(1,9), saveas='rand_gauss')
Savings.save_multplot(methode='PGD', globname='rand_gauss-g', names=range(1,9), saveas='rand_gauss')



    # En fonction de d
lenths = [100, 200, 400, 800]

for d in lenths:
    Savings.save_multplot(methode='PGD', globname=f'lat-s_{d}', names=range(1,5), saveas=f'lat-{d}')
    Savings.save_multplot(methode='PGD', globname=f'lat-g_{d}', names=range(1,5), saveas=f'lat-{d}')
