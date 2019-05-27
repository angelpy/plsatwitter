"""
Script de ejemplo con cuentas de periódicos
"""
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
    X, frecs, palabras = plsa_instance.generate_matrix()
    # Compute PLSA probabilities
    Xp, P1, P2, P3 = plsa_instance.compute_probabilities(method="sklearn")
    # Prints top words in a cloud of words
    utils.visuals.top_words(P1, palabras, cloud_mode=True)
     
if __name__ == "__main__":
    run()