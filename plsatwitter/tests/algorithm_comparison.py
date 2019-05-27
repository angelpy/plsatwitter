#!/usr/bin/env python3
# -*- coding: latin-1 -*-

"""
En este script se compara la evoluciÃ³n de distintos algoritmos
en funciÃ³n de k para 20 iteraciones.
"""
from numpy.linalg import norm
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from os.path import dirname, abspath
from os.path import join
from plsatwitter.models.plsa_nmf import PlsaNmf
from plsatwitter.utils.funciones import kl_div, svd_tfm, svd_init
from plsatwitter.utils.matrix_factorization import *
from plsatwitter.utils.visuals import plot_dict, plot_dict_grande
from plsatwitter.config.folders import folders
import pandas as pd
import time

def test_convergence(k=10, n_iter=50):
    algoritmos = {
#            "sklearn": [],
            "EM": []
    }
    tiempos = {
            "sklearn": [],
            "EM": []
            }
    plsa_instance = PlsaNmf()
    datasets_route = join(dirname(dirname(abspath(__file__))), 'datasets')
    periodicos_route = join(datasets_route, 'periodicos_small.csv')
#    config_route = join(datasets_route, 'config.ini')
#    cuentas = ['telecoupm', 'La_UPM']
#    plsa_instance.generate_corpus(cuentas, config_path=config_route, tweets_per_user=1000, corpus_path='univers.csv')
    
    plsa_instance.read_from_csv(periodicos_route)
    X_emp, frecs, palabras = plsa_instance.generate_matrix()
   
    W0, H0 = svd_tfm(X_emp, k, 'count')
    X_mod, *_= plsa_instance.compute_probabilities(method='EM',
                                 n_topics=k,
                                 n_iters=n_iter, init=(W0, H0)
                                 )

    
def div_vs_k(n_iter=50):
    algoritmos = {
            "sklearn": [],
            "EM": []
    }
    tiempos = {
            "sklearn": [],
            "EM": []
            }
    plsa_instance = PlsaNmf()
    datasets_route = join(dirname(dirname(abspath(__file__))), 'datasets')
    periodicos_route = join(datasets_route, 'periodicos_small.csv')
    
    plsa_instance.read_from_csv(periodicos_route)
    # Generate X and frecs
    X_emp, frecs, palabras = plsa_instance.generate_matrix()
    k_vals = range(1,80)
    
    for k in k_vals:
        print("k = {}".format(k))
        W0, H0 = svd_tfm(X_emp, n_components=k, criteria='count')
        for met in algoritmos.keys():
            t0 = time.time()
            Xm, *_ = plsa_instance.compute_probabilities(method=met, n_topics=k,
                             n_iters=n_iter, init=(W0, H0))
            algoritmos[met].append(kl_div(X_emp, Xm))
            tiempos[met].append(time.time() - t0)
            
    algoritmos['MU'] = algoritmos.pop('sklearn')
    tiempos['MU'] = tiempos.pop('sklearn')
    df = pd.DataFrame.from_dict(algoritmos)
    df.index = list(k_vals)
    ax = df.plot(marker='o')
    fig = ax.get_figure()
    ax.set_xlabel("Nº componentes")
    ax.set_ylabel(r'$D_{KL}(X \Vert X_p)$')
    ruta = folders['divergencia'] + 'div_vs_k.png'
    fig.savefig(ruta, dpi=1000, bbox_inches="tight")
    
    
    dft = pd.DataFrame.from_dict(tiempos)
    dft.index = list(k_vals)
    ax = dft.plot(marker='o')
    fig = ax.get_figure()
    ax.set_xlabel("Nº componentes")
    ax.set_ylabel("Tiempo de ejecución (s)")
    ruta = folders['divergencia'] + 'time_vs_k.png'
    fig.savefig(ruta, dpi=1000, bbox_inches="tight")
    
    
def div_vs_iters(n_components=10):
    algoritmos = {
            "EM": [],
            "sklearn": [],
        }  
    # ObtenciÃ³n de tweets
    plsa_instance = PlsaNmf()
    datasets_route = join(dirname(dirname(abspath(__file__))), 'datasets')
    periodicos_route = join(datasets_route, 'periodicos_small.csv')
    
    plsa_instance.read_from_csv(periodicos_route)
    # Generate X and frecs
    X_emp, frecs, palabras = plsa_instance.generate_matrix()
    # Obtener las matrices iniciales
    W0, H0 = svd_tfm(X_emp, n_components=n_components, criteria='count')
    for met in algoritmos.keys():
        algoritmos[met].append(kl_div(X_emp, W0 @ H0))
    # Para cada iteraciÃ³n
    for iter_num in range(1, 30):
        for met in algoritmos.keys():
            Xm, *_ = plsa_instance.compute_probabilities(method=met, n_topics=n_components,
                         n_iters=iter_num, init=(W0, H0))
            algoritmos[met].append(kl_div(X_emp, Xm))      
    algoritmos['MU'] = algoritmos.pop('sklearn')
    df = pd.DataFrame.from_dict(algoritmos)
    ax = df.plot(marker='o')
    ax.set_xlabel("Número de iteraciones")
    fig = ax.get_figure()
    ruta = folders['divergencia'] + 'kl_div_vs_iters.png'
    fig.savefig(ruta, dpi=1000, bbox_inches="tight")
def run():
    # ParÃ¡metros
    
    algorithms = ['sklearn', 'EM']
    resultados = {}
    for alg in algorithms:
        resultados[alg] = []
    
    
    # ObtenciÃ³n de tweets
    plsa_instance = PlsaNmf()
    datasets_route = join(dirname(dirname(abspath(__file__))), 'datasets')
    periodicos_route = join(datasets_route, 'periodicos.csv')
    
    plsa_instance.read_from_csv(periodicos_route)
    # Generate X and frecs
    X_emp, frecs, palabras = plsa_instance.generate_matrix()
    valores_k = [5, 10]
    
    
    for k in valores_k:
        
        W, H = svd_tfm(X_emp, n_components=k, criteria='count')
        for algorithm in algorithms:
            print(algorithm)
            # ObtenciÃ³n del modelo
            X_modelo, *_ = plsa_instance.compute_probabilities(method=algorithm, n_topics=k, n_iters=20, init=(W, H))
            resultados[algorithm].append(kl_div(X_emp, X_modelo))
            
    for key in resultados:
        plt.plot(valores_k, resultados[key], marker='o')
        plt.title('Divergencia')
        plt.xlabel('NÂº de componentes')
        plt.ylabel('Divergencia')
    plt.legend(resultados.keys())
    
def frobenius(M=600, N=100, from_WH = True, add_noise = False, dens=0.5, k=10):
    """
    Para estudiar el comportamiento de algunos algoritmos
    reduciendo la norma de Frobenius.
    Tipos de matrices:
        - X densa
    """
    if from_WH:
        W_init = sp.random(M, k, dens)
        H_init = sp.random(k, N, dens)
        X = (W_init @ H_init).toarray()
        X_n = X + np.random.random(X.shape)
    else:
        X = np.random.rand(M, N)
    iter_max = 50
    algoritmos = {
        'Lin_PG': [],
        'simple_MU': [],
        'HALS': [],
        'ANLS_BLOCKPIVOT': [],
    }
    algoritmos_ruido = {
        'Lin_PG': [],
        'simple_MU': [],
        'HALS': [],
        'ANLS_BLOCKPIVOT': [],
    }
    k_values = range(1, 50)
    k_results = {
        'Lin_PG': [],
        'simple_MU': [],
        'HALS': [],
        'ANLS_BLOCKPIVOT': [],
    }
    k_results_ruido = {
        'Lin_PG': [],
        'simple_MU': [],
        'HALS': [],
        'ANLS_BLOCKPIVOT': [],
    }
    # Test 1: 
    W0, H0 = svd_init(X, k)
    W0n = np.random.random((M, k))
    H0n = np.random.random((k, N))
    # Rellenar primer valor
    for algoritmo in algoritmos.keys():
        algoritmos[algoritmo].append(0.5 * np.linalg.norm(X - W0 @ H0) ** 2)
        algoritmos_ruido[algoritmo].append(0.5 * np.linalg.norm(X_n - W0n @ H0n) ** 2)
    # Ver cuál converge antes
    for it_number in range(1,iter_max+1):
        print("Nº de iteraciones: {}.\n".format(it_number))
        for i, algoritmo in enumerate(algoritmos.keys()):
            W, H = calc_nmf(X, n_iter = it_number, n_components =k, 
                     method=algoritmo, wi = W0, 
                     hi = H0, metrica='frobenius')
            Wn, Hn = calc_nmf(X_n, n_iter = it_number, n_components=k, 
                     method=algoritmo, wi = W0, 
                     hi = H0, metrica='frobenius')
            algoritmos[algoritmo].append(0.5 * np.linalg.norm(X - W @ H) ** 2)
            algoritmos_ruido[algoritmo].append(0.5 * np.linalg.norm(X_n - Wn @ Hn) ** 2)
    if from_WH:
        filename = folders['frobenius'] + 'n_iter_vs_norm_k_{}_dens_{}.png'.format(k, dens)
    else:
        filename = folders['frobenius'] + 'norm_vs_iter_WH_unknown.png'.format(k)
    algoritmos['PG'] = algoritmos.pop('Lin_PG')
    algoritmos['MU'] = algoritmos.pop('simple_MU')
    algoritmos['ANLS BPP'] = algoritmos.pop('ANLS_BLOCKPIVOT')
    algoritmos_ruido['PG'] = algoritmos_ruido.pop('Lin_PG')
    algoritmos_ruido['MU'] = algoritmos_ruido.pop('simple_MU')
    algoritmos_ruido['ANLS BPP'] = algoritmos_ruido.pop('ANLS_BLOCKPIVOT')

    
    file_sinruido = folders['frobenius']+ 'n_iter_vs_norm_sinruido.png'
    file_conruido = folders['frobenius']+ 'n_iter_vs_norm_conruido.png'
    plot_dict(algoritmos, range(0, iter_max+1), 'Nº iteraciones', r'$\Vert X-WH\Vert_F^2$',
              'X sin ruido', file_sinruido)
    
    plot_dict_grande(algoritmos_ruido, range(0, iter_max+1), 'Nº iteraciones', r'$\Vert X-WH\Vert_F^2$',
              'X con ruido', file_conruido)
    # Estudiar cómo se comportan los algoritmos en función
    # del número de componentes
    for k_new in k_values:
        print("k = {}".format(k_new))
        W0, H0 = svd_init(X, k_new)
        for algoritmo in k_results.keys():
            W, H = calc_nmf(X, n_iter = 30, n_components =k_new, 
                         method=algoritmo, wi = W0, 
                         hi = H0, metrica='frobenius')
            Wn, Hn = calc_nmf(X_n, n_iter = 30, n_components =k_new, 
                         method=algoritmo, wi = W0, 
                         hi = H0, metrica='frobenius')
            k_results[algoritmo].append(0.5 * np.linalg.norm(X - W @ H) ** 2)
            k_results_ruido[algoritmo].append(0.5 * np.linalg.norm(X_n - Wn @ Hn) ** 2)
    filename_k = folders['frobenius'] + 'n_comp_vs_norm_k_opt={}_sinruido.png'.format(k)
    filename_k_conruido = folders['frobenius'] + 'n_comp_vs_norm_k_opt={}_conruido.png'.format(k)
    
    plot_dict(k_results, k_values, "Nº componentes", r'$\Vert X-WH\Vert_F^2$',
              'X sin ruido', filename_k)
    plot_dict(k_results_ruido, k_values, "Nº componentes", r'$\Vert X-WH\Vert_F^2$',
              'X con ruido', filename_k_conruido)
    
    k_results['PG'] = k_results.pop('Lin_PG')
    k_results['MU'] = k_results.pop('simple_MU')
    k_results['ANLS BPP'] = k_results.pop('ANLS_BLOCKPIVOT')
    k_results_ruido['PG'] = k_results_ruido.pop('Lin_PG')
    k_results_ruido['MU'] = k_results_ruido.pop('simple_MU')
    k_results_ruido['ANLS BPP'] = k_results_ruido.pop('ANLS_BLOCKPIVOT')
    final_results = np.zeros((len(algoritmos.keys()), 1))
    final_results_ruido = np.zeros((len(algoritmos.keys()), 1))
    for k, v in enumerate(algoritmos.values()):
        final_results[k] = v[-1]
    for k, v in enumerate(algoritmos_ruido.values()):
        final_results_ruido[k] = v[-1]
    df_sinruido = pd.DataFrame(final_results, index = algoritmos.keys(), columns=['f(W, H)'])
    df_conruido = pd.DataFrame(final_results_ruido, index = algoritmos.keys(), columns=['f(W, H)'])
    ax_sin = df_sinruido.plot(kind='bar')
    ax_con = df_conruido.plot(kind='bar')
    fig_sin = ax_sin.get_figure()
    fig_con = ax_con.get_figure()
    ruta_sin = folders['frobenius'] + 'bar_graph_sinruido.png'
    ruta_con = folders['frobenius'] + 'bar_graph_conruido.png'
    fig_sin.savefig(ruta_sin, dpi=1000, bbox_inches = "tight")
    fig_con.savefig(ruta_con, dpi=1000, bbox_inches = "tight")
    