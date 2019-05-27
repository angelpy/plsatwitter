#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains a class called PlsaNmf that
allows to perform a probabilistic latent semantic
analysis with NMF techniques, as well as a search
engine using queries.
"""
import string
import configparser
import csv
import matplotlib.pyplot as plt
import twitter
import scipy.sparse as sp
import pandas as pd
import plsatwitter.utils.twitter as tw
import plsatwitter.utils.funciones as mat
from nltk.corpus import stopwords
import numpy as np
from sklearn.decomposition import NMF as NMF_sklearn
from sklearn.feature_extraction.text import CountVectorizer
import nimfa
from nonnegfac.nmf import NMF_ANLS_AS_GROUP, NMF_ANLS_AS_NUMPY,\
     NMF_ANLS_BLOCKPIVOT, NMF_MU, NMF_HALS

def plot_residual(n_iter, residual):
    plt.plot(n_iter, residual['W'], '-o', n_iter, residual['H'], '-o')
    plt.xlabel('Number of iterations')
    plt.title('KKT residuals')
    plt.legend(['kkt(W)', 'kkt(H)'])
    plt.show()

class PlsaNmf():

    # Punctuation signs
    punctuation = list(string.punctuation)
    # More stopwords
    others = ['si', 'RT', 'rt', 'via', 'https', 'http', 'jaja', 'jajaja', 'aún']
    # Spanish and English stopwords
    stop = stopwords.words('spanish') + stopwords.words('english') + punctuation + others

    def __init__(self):
        self.X = None
        self.frequencies = None
        self.words = None
        self.corpus = None

    def read_from_csv(self, path, sep="\t"):
        """
        Creates a corpus from a .csv file.
        :param path: the path to the csv
        :param sep: separator, default \t
        :return: None
        """
        self.corpus = pd.read_csv(path, sep=sep)

    def generate_corpus(self, accounts, config_path, tweets_per_user=200,
                        corpus_path='accounts.csv'):
        """
        Generates a tweet corpus from a specified list of accounts and saves
        it to a .csv file. A .ini configuration file must be provided.
        :param accounts: list of Twitter accounts
        :param config_path: .ini file with Twitter API keys
        :param tweets_per_user: number of tweets per account
        :param corpus_path: name of csv saved file
        :return:
        """
        config = configparser.ConfigParser()
        config.read(config_path)
        if 'Twitter' in config:
            twitter_api = twitter.Api(consumer_key=config['Twitter']['consumer_key'],
                                      consumer_secret=config['Twitter']['consumer_secret'],
                                      access_token_key=config['Twitter']['access_token_key'],
                                      access_token_secret=config['Twitter']['access_token_secret'],
                                      tweet_mode='extended')

        statuses = []

        for (_, account) in enumerate(accounts):
            print('Downloading tweets from @{}.\n'.format(account))
            tweets = tw.get_tweets(twitter_api, account)
            tweets = tweets[:tweets_per_user]
            statuses += tweets

        # Write to CSV
        with open(corpus_path, 'w') as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(['created_at', 'full_text', 'screen_name', 'edited_text'])
            for s in statuses:
                cleaned_tweet = tw.clean_tweet(s.full_text)
                writer.writerow([s.created_at, s.full_text, s.user.screen_name, cleaned_tweet])
        self.corpus = pd.read_csv(corpus_path, sep="\t")

    def get_corpus(self):
        """
        Returns the matrix corpus
        :return: corpus
        """
        return self.corpus

    def search(self, query, p1, p2, p3):
        """
        Performs a search on the tweet corpus based on
        a query string using PLSA and returns at most 5 tweets if
        the conditioned probability P(d|q) is above a certain
        threshold.
        :param query: the query string to be used
        :param p1: P(w|z)
        :param p2: P(d|z)
        :param p3: P(z)
        :return: None
        """
        k = p2.shape[0]
        if not query:
            raise ValueError("A query string must be provided.")
        q = [word.lower() for word in query.split() if word not in self.stop]
        n_qw = np.zeros((self.X.shape[0], 1))
        # Get word count
        for idx, word in enumerate(self.words):
            if word in q:
                n_qw[idx] += 1
        n_qw[n_qw == 0] = 1e-12
        p2_n = np.random.random((k, 1))
        # Perform the EM algorithm on the new P(q|z) while keeping P(w|z) and P(z) untouched
        p2_q = self.EM_algorithm_query(p1, p2_n, p3, n_qw, n_iter=70)
        scores = (p2_q.T @ p2).ravel()
        relevant_tweets = (-scores).argsort()
        scores[::-1].sort()
        top_related = relevant_tweets[:5]
        top_probabilities = scores[:5]
        for idx, tweet_number in enumerate(top_related):
            # Print tweet t if P(t|q) is above a certain threshold
            if top_probabilities[idx] > 1e-2:
                print("@{} - (prob = {:.2f})".format(self.corpus.iloc[tweet_number].screen_name,
                                                     top_probabilities[idx]))
                print(self.corpus.iloc[tweet_number].full_text)
                print("________________________________________")

    def generate_matrix(self):
        """
        Once specified a corpus, this method uses
        a Vectorizer to count word occurrences (frecs),
        normalizes the matrix to have
        a probability distribution (X) and re
        :return:
            -X: probability distribution
            -frequencies: word count
            -words: list of words of size X.shape[1]
        """
        text_list = list(self.corpus.astype('U')['edited_text'])
        count_vec = CountVectorizer()
        X = count_vec.fit_transform(text_list).T
        X = X / X.sum()
        frequencies = count_vec.fit_transform(text_list).T
        self.X = X
        self.frequencies = frequencies
        self.words = count_vec.get_feature_names()
        return X, frequencies, self.words

    def compute_probabilities(
            self, method='EM', n_topics=10,
            n_iters=20, init=None,
            X_alt=None, frequencies_alt=None):
        """
        This method computes PLSA probabilities
        from a tweet corpus.
        :param method (str): algorithm used to compute probabilities
        :param n_topics: number of topics
        :param n_iters: number of iterations
        :param init (np.ndarray, np.ndarray): initial values for the algorithm
        :param X_alt:
        :param frequencies_alt:
        :return:
            X_mod: probabilistic model for the empiric distribution
            -P1: P(w|z), posterior probability of words given topics
            -P2: P(d|z), posterior probability of tweets given topics
            -P3: P(z), topic probabilities
        """
        if method == 'EM':
            if init != None:
                _, *em0 = mat.nmf2plsa(init[0], init[1])
                return self.EM_algorithm(
                    n_iter=n_iters, n_topics=n_topics,
                    init=em0, X_alt=X_alt,
                    frequencies_alt=frequencies_alt)
            else:
                return self.EM_algorithm(
                    n_iter=n_iters, n_topics=n_topics,
                    init=None)
        else:
            if init is not None:
                W, H = self.calc_nmf(
                    n_iter=n_iters, n_components=n_topics,
                    method=method, wi=init[0],
                    hi=init[1])
            else:
                W, H = self.calc_nmf(
                    n_iter=n_iters, n_components=n_topics,
                    method=method)
            return mat.nmf2plsa(W, H)

    def calc_nmf(
            self, n_iter=20, n_components=10,
            method='sklearn', wi=None,
            hi=None, metrica='kullback-leibler'):
        """
        This class method computes a nonnegative matrix factorization
        for a given matrix using two different metrics: the Frobenius
        norm and the Kullback-Leibler divergence.
        :param n_iter: number of iterations
        :param n_components: number of components
        :param method: method
        :param wi: initial value of the basis matrix
        :param hi: initial value of the coefficient matrix
        :param metrica: ('frobenius' | 'kullback-leibler') metric
        :return:
            W: basis matrix
            H: coefficient matrix
        """
        if wi is not None and hi is not None:
            init_kim = (wi, hi.T)
        else:
            init_kim = None
        if method == 'sklearn':
            if wi is not None and hi is not None:
                nmf = NMF_sklearn(
                    n_components=n_components, max_iter=n_iter,
                    random_state=1, beta_loss=metrica, solver="mu",
                    alpha=0, l1_ratio=0, init='custom')
                W = nmf.fit_transform(self.X, W=wi, H=hi)
                H = nmf.components_
            else:
                nmf = NMF_sklearn(
                    n_components=n_components, max_iter=n_iter,
                    random_state=1, beta_loss=metrica, solver="mu",
                    alpha=0, l1_ratio=0)
                W = nmf.fit_transform(self.X)
                H = nmf.components_
        if method == 'nimfa_MU':
            nmf = nimfa.Nmf(
                self.X, W=wi, H=hi,
                rank=n_components, max_iter=n_iter,
                update='divergence', objective='div')
            nmf_fit = nmf()
            print(nmf)
            W = nmf_fit.basis()
            W = np.asarray(W)
            H = nmf_fit.coef()
            H = np.asarray(H)
            print(W.shape, H.shape)
        if method == 'Lin_PG':
            nmf = nimfa.Lsnmf(
                self.X, W=wi, H=hi,
                H1=hi, rank=n_components,
                max_iter=n_iter, sub_iter=10,
                inner_sub_iter=10, beta=0.1)
            nmf_fit = nmf()
            W = nmf_fit.basis()
            W = np.asarray(W)
            H = nmf_fit.coef()
            H = np.asarray(H)
        if method == 'simple_MU':
            W, H, _ = NMF_MU().run(
                self.X, n_components, init=init_kim,
                max_iter=n_iter, verbose=-1)
            H = H.T
        if method == 'ANLS_BLOCKPIVOT':
            W, H, _ = NMF_ANLS_BLOCKPIVOT().run(
                self.X, n_components, init=init_kim,
                max_iter=n_iter, verbose=-1)
            H = H.T
        if method == 'ANLS_AS_NUMPY':
            W, H, _ = NMF_ANLS_AS_NUMPY().run(
                self.X, n_components, max_iter=n_iter,
                verbose=-1, init=init_kim)
            H = H.T
        if method == 'ANLS_AS_GROUP':
            W, H, _ = NMF_ANLS_AS_GROUP().run(
                self.X, n_components, max_iter=n_iter,
                verbose=-1, init=init_kim)
            H = H.T
        if method == 'HALS':
            W, H, _ = NMF_HALS().run(
                self.X, n_components, max_iter=n_iter,
                verbose=-1, init=init_kim)
            H = H.T
        return W, H
    def EM_algorithm(
            self, n_iter=20, n_topics=10,
            init=None, X_alt=None,
            frequencies_alt=None):
        """
        Expectation-Maximization algorithm.
        :param n_iter: number of iteration
        :param n_topics: number of topics
        :param init: if specified, initial value of the matrices
        :param X_alt: alternative value of the empiric model
        :param frequencies_alt: alternative matrix for the word count
        :return:
        """
        X = self.X
        frequencies = self.frequencies
        if X_alt is not None:
            X = X_alt
        if frequencies_alt is not None:
            frequencies = frequencies_alt
        M, N, K = X.shape[0], X.shape[1], n_topics
        kkt_residual = {
            'W': [],
            'H': []
            }

        if init != None:
            p1 = init[0].copy()
            p1 = p1.T
            p2 = init[1].copy()
            p3 = np.diagonal(init[2]).copy()
            p3 = p3.astype('float32')
            p1 = np.asfortranarray(p1, dtype='float32')
            p2 = np.asfortranarray(p2, dtype='float32')
        else:
            p1 = np.random.rand(K, M)
            p1 /= p1.sum(1)[:, None]
            p2 = np.random.rand(K, N)
            p2 /= p2.sum(1)[:, None]
            p3 = np.random.rand(K)
            p3 /= p3.sum()
        p1[p1 < np.finfo('float32').eps] = np.finfo('float32').eps
        p2[p2 < np.finfo('float32').eps] = np.finfo('float32').eps
        for _ in range(n_iter):
            try:
                # E-step
                prod = p3[:, None, None] * \
                        (p1[:, :, None] @ p2[:, None, :])
                Pz_td = prod / prod.sum(0)
#                Pz_td = p3[:, None, None] * \
#                        (p1[:, :, None] @ p2[:, None, :]) / \
#                        (p1.T @ np.diag(p3) @ p2)[None,:,:]
                # M-step
                if frequencies_alt is not None:
                    if sp.issparse(frequencies_alt):
                        Pz_td *= self.frequencies.A
                    else:
                        Pz_td *= frequencies
                else:
                    Pz_td *= self.frequencies.A
                den = Pz_td.sum((1, 2))[:, None]
                p1 = Pz_td.sum(2) / den  # (K, M)
                p2 = Pz_td.sum(1) / den
                p3 = (den / den.sum()).ravel()
                # Threshold
                p1[p1 < np.finfo('float32').eps] = np.finfo('float32').eps
                p2[p2 < np.finfo('float32').eps] = np.finfo('float32').eps
                # Compute KKT residuals
                kkt_residual['W'].append(mat.kkt_W(X, p1.T @ np.diag(p3), p2))
                kkt_residual['H'].append(mat.kkt_H(X, p1.T @ np.diag(p3), p2))
            except MemoryError:
                raise MemoryError("Error de memoria.")
#        plot_residual(range(1, n_iter+1), kkt_residual)
        return p1.T @ np.diag(p3) @ p2, p1.T, p2, p3
    def EM_algorithm_query(
            self, p1, p2, p3,
            n_qw, n_iter=50):
        """
        EM algorithm for a query based retrieval.
        :param p1: P(w|z)
        :param p2: initial value of P(q|z)
        :param p3: P(z)
        :param n_qw: word count for the query
        :param n_iter: number of iterations
        :return:
            p2: P(q|z)
        """
        tol = 1e-12
        p1[p1 < tol] = tol
        p2[p2 < tol] = tol
        if p3.ndim == 2:
            p3 = np.diagonal(p3).copy()
        for _ in range(n_iter):
            Pz_td = p3[:, None, None] *(p1.T[:, :, None] @ p2[:, None, :]) / \
                    (p1 @ np.diag(p3) @ p2)[None, :, :]
            # M-step
            Pz_td *= n_qw
            den = Pz_td.sum((0, 1))[:, None]
            p2 = Pz_td.sum(1) / den
            p2[p2 < tol] = tol
        return p2  # P(q|z)
