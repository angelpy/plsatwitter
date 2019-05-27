#!/usr/bin/env python
# -*- coding: latin-1 -*-
"""
MÃ³dulo para comparar distintas inicializaciones SVD.
Algoritmos utilizados: 'NNDSVD' | 'AV-SVD', 'NC-SVD'
"""
import time
from os.path import dirname, abspath
from os.path import join
from math import sqrt
import numpy as np
from numpy.linalg import norm
import scipy.sparse as sp
from scipy.stats import entropy
import matplotlib.pyplot as plt
import pandas as pd
from plsatwitter.models.plsa_nmf import PlsaNmf
from plsatwitter.utils.funciones import svd_init, svd_tfm, kl_div
from plsatwitter.config.folders import folders

def run(num_opc=range(1,7)):
    opciones = {
            1: ('x_dense', 'frobenius', 'x_dense_frobenius.png'),
            2: ('x_dense', 'kl-1', 'x_dense_kl1.png'),
            3: ('x_sparse', 'frobenius', 'x_sparse_frobenius.png'),
            4: ('x_sparse', 'kl-1', 'x_sparse_kl1.png'),
            5: ('x_twitter', 'frobenius', 'x_twitter_frobenius.png'),
            6: ('x_twitter', 'kl-1', 'x_twitter_kl1.png'),
            7: None,
    }
    algoritmos = ['AV-SVD', 'NC-SVD', 'NNDSVD']
    tipos = ['X densa', 'X sparse', 'X Twitter']
    k = 100
    if num_opc == 7:
        densidades = [0.01, 0.05, 0.1, 0.2]
        labels = ["{}%".format(str(number*100)) for number in densidades]
        results = np.zeros((4, 3))
        for i, dens in enumerate(densidades):
            print("Probando: " + str(i))
            X = sp.rand(2500, 700, density=dens)
            X /= X.sum()
            W, H = svd_tfm(X, n_components=k, criteria='abs')
            results[i,0] = kl_div(X, W @ H)
            W, H = svd_tfm(X, n_components=k, criteria='count')
            results[i,1] = kl_div(X, W @ H)
            W, H = svd_init(X, n_components=k)
            results[i,2] = kl_div(X, W @ H)

        df = pd.DataFrame(results.T, index=algoritmos, columns=labels)
        titulo = "Matrices dispersas (k = {})".format(k)
        ax = df.plot(kind='bar', title=titulo)
        fig = ax.get_figure()
        ruta = folders['svd'] + "matrices_dispersas.png"
        fig.savefig(ruta, dpi=1000)
        
    else:
        for elegida in num_opc:
            print("Opción {}.\n".format(elegida))
            ruta = folders['svd']
            tipo_X, metrica, tituloFig = opciones[elegida]
            ruta = ruta + tituloFig
            # Plot options
            if metrica == 'frobenius':
                title = r'$\Vert X - X_p \Vert_F$'
            elif metrica == 'kl-1':
                title = r'$D_{KL}(X \Vert X_p)$'
            else:
                title = r'$D_{KL}(X_p \Vert X)$'
                
            # Añado al título el tipo de matriz
            if tipo_X == 'x_twitter':
                plsa_1 = PlsaNmf()
                datasets_route = join(dirname(dirname(abspath(__file__))), 'datasets')
                periodicos_route = join(datasets_route, 'periodicos.csv')
                plsa_1.read_from_csv(periodicos_route)
                X, _, _ = plsa_1.generate_matrix()
                print("sparsity: {}".format(X.count_nonzero() / (X.shape[0] * X.shape[1])))
                title = "Matriz real (twitter) - {}".format(title)
                
            elif tipo_X == 'x_dense':
                X = np.random.random((2500, 500))
                if metrica != 'frobenius':
                    X /= X.sum()
    #                tituloFig = ruta + 'svd_X_dense.png'
                title = "Matriz sintética densa - {}".format(title)
            elif tipo_X == 'x_sparse':
                X = sp.rand(2500, 500, density=0.05)
                if metrica != 'frobenius':
                    X /= X.sum()
                title = "Matriz sintética dispersa - {}".format(title)
            m, n = X.shape
            results = dict.fromkeys(algoritmos)
            k = 10 
            for key in results.keys():
                if key == 'AV-SVD':
                    W, H = svd_tfm(X, k, criteria='abs')
                elif key == 'NC-SVD':
                    W, H = svd_tfm(X, k ,criteria='count')
                elif key == 'NNDSVD':
                    W, H = svd_init(X, k)
                else:
                    W, H = np.random.rand(m, k), np.random.rand(k, n)
                # Se coge la mÃ©trica elegida
                if metrica == 'frobenius':
                    results[key] = norm(X - W @ H)
                elif metrica == 'kl-1':
                    Xp = W @ H
                    results[key] = kl_div(X, Xp)
                elif metrica == 'kl-2':
                    Xp = W @ H
                    results[key] = kl_div(Xp, X)
                else:
                    raise ValueError('Opción no especificada')
                    break
        
            # Plot configuration
            SMALL_SIZE = 14
            MEDIUM_SIZE = 16
            BIGGER_SIZE = 18
            
            
            
            plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
            plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
            
            print(results)
            fig = plt.figure()
            plt.bar(np.arange(len(results.keys())), list(results.values()),
                    width=0.35)
            plt.xticks(np.arange(len(results.keys())), results.keys())
            plt.title(title)
            plt.show()
            fig.savefig(ruta)
    

if __name__ == '__main__':
    run()


