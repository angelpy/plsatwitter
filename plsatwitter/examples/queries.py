"""
Script used to 
"""
import numpy as np
import scipy.sparse as sp
from plsatwitter.models import plsa_nmf as plsa_nmf
import plsatwitter.utils as utils
from os.path import dirname, abspath
from os.path import join

def run():
    # Create class
    plsa_instance = plsa_nmf.PlsaNmf()
    # Load data from CSV
    datasets_route = join(dirname(dirname(abspath(__file__))), 'datasets')
    periodicos_route = join(datasets_route, 'periodicos.csv')
    plsa_instance.read_from_csv(periodicos_route)
    # Generate X and frecs
    X, frecs, words = plsa_instance.generate_matrix()
    # Compute PLSA probabilities
    Xp, P1, P2, P3 = plsa_instance.compute_probabilities(
            method="sklearn", n_iters=80, n_topics=200
            )
    
    # Search query
    q = input("Introduce your search query: ")
    # Print related tuits
    plsa_instance.search(q, P1, P2, P3)
    