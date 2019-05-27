# -*- coding: latin-1 -*-
"""
Python script used to compare different NMF methods for
the Kullback-Leibler optimization problem.
"""

from plsatwitter.models.plsa_nmf import PlsaNmf
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
from plsatwitter.utils.funciones import *

if __name__ == "__main__":
    
    metodos = ['EM', 'Lin_PG', 'sklearn', 'HALS', 'ANLS_BLOCKPIVOT']
    labels = ['EM', 'PG', 'MU', 'HALS', 'Blockpivot']
    iteraciones = [5, 10, 15, 20, 25, 30]
    componentes = [5, 10, 15, 20]
    N_ITER = 30
    
    
    
    # Clase Plsa
    CUENTAS = ['telecoupm', 'informaticaupm']
    plsa_1 = PlsaNmf()
    plsa_1.genera_corpus(CUENTAS, porCuenta=500, configPath='../config.ini')
    X, _, _ = plsa_1.corpus2matrix(1000)
    M, N = X.shape
     # Initial values for EM algorithm
    fila_EM = []
    tiempos_EM = []
    ruta = '/home/angel/Nextcloud/PycharmProjects/TFM/results/images/'
        
    print("Probando algoritmo E-M")
    for n_comp in componentes:
        print("...para {} componentes.\n".format(n_comp))
        W0 = np.random.random((M, n_comp))
        H0 = np.random.random((n_comp, N))
        kl_0 = kl_div(X, W0 @ H0)
        for i in range(5):
            Wp, Hp = svd_tfm(X, n_comp, 'count')
            if kl_div(X, Wp @ Hp) < kl_0:
                W0, H0 = Wp, Hp
        P0, Q, C, S = nmf2plsa(W0, H0)
        t_init = time.time()
        kl_array, iter_array, X_em, P1, P2, P3 = plsa_1.EM_algorithm_v2(init=(Q, C, S), n_topics = n_comp, calc_div=False, n_iter=N_ITER)
        tiempos_EM.append(time.time() - t_init)
        fila_EM.append(kl_div(X, X_em))

    fila_EM = np.array(fila_EM)
    tiempos_EM = np.array(tiempos_EM)
    resultados = fila_EM
    tiempos = tiempos_EM
    
    for metodo in metodos[1:]:
        print("Testing method {}.\n".format(metodo))
        fila_metodo = []
        tiempos_metodo = []
        for n_comp in componentes:
            print("...for {} components.\n".format(n_comp))
            W0, H0 = svd_tfm(X, n_comp)
            t_init = time.time()
            W, H = plsa_1.calc_nmf(n_iter=N_ITER, n_components=n_comp, method=metodo, wi=W0, hi=H0)
            tiempos_metodo.append(time.time() - t_init)
            Ptemp, _, _, _ = nmf2plsa(W, H)
            fila_metodo.append(kl_div(X,Ptemp))
        fila_metodo = np.array(fila_metodo)
        tiempos_metodo = np.array(tiempos_metodo)
        resultados = np.vstack((resultados, fila_metodo))
        tiempos = np.vstack((tiempos, tiempos_metodo))
            
    df_resultados = df = pd.DataFrame(resultados.T, index=componentes, columns=labels)
    df_tiempos = df = pd.DataFrame(tiempos.T, index=componentes, columns=labels)
    df_resultados.round(3).to_csv()
    df_tiempos.round(3).to_csv()

    fig = plt.figure()
    nombreFigura = 'comparacion_resultados_kl_div.png'
    for method in df_resultados.columns:
        plt.plot(df_resultados.index, df_resultados[method])
    plt.xlabel('Componentes')
    plt.title(r'$D_{KL}(X \Vert X_{app})$')
    plt.legend(df_resultados.columns, fontsize=10)
    plt.xticks(ticks=df_resultados.index)
    fig.savefig(ruta + nombreFigura)
            

    fig = plt.figure()       
    nombreFigura = 'comparacion_tiempos_kl_div.png'
    for method in df_tiempos.columns:
        plt.plot(df_tiempos.index, df_tiempos[method])
    plt.xlabel('Componentes')
    plt.title(r'$D_{KL}(X \Vert X_{app})$')
    plt.legend(df_tiempos.columns, fontsize=10)
    plt.xticks(ticks=df_tiempos.index)
    fig.savefig(ruta + nombreFigura)            
                