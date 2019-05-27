Python package for Probabilistic Latent Semantic Analysis
====================

This package is used to extract latent topics from tweets.


Requirements
------------

This package uses the following packages for computing NMF factorizations:

* [nimfa](https://github.com/marinkaz/nimfa)
* [nonnegfac](https://github.com/kimjingu/nonnegfac-python)

It has been tested with:

* python: 3.7.0
* numpy: 1.16.2
* sklearn: 0.20.3
* Ubuntu version: 18.04.2 LTS


Installation
------------

Use **setup.py** to install this package:

```python
    python setup.py install
```

Configuration
------------
On Linux, create in /home/[your_username]/ a folder called **plsatwitter-results** with three folders inside it:

* csv
* svd
* topics


Usage Instructions
------------
An example can be tested using [Jupyer Qtconsole](https://qtconsole.readthedocs.io/en/stable/)
```python3
   import plsatwitter
   plsatwitter.examples.periodicos.run()
```

You can also extract topics from specified accounts. You need to use a configuration file and [get your Twitter application tokens](https://python-twitter.readthedocs.io/en/latest/getting_started.html). Once done, create a configuration file with .ini extension and the following content:
```
    [Twitter]
    consumer_key=your_consumer_key
    consumer_secret=your_consumer_secret
    access_token_key=your_access_token_key
    access_token_secret=your_access_token_secret
    tweet_mode=extended
```

The following script works this way:

1. Create an instance of the PlsaNmf class.
2. Create a list of accounts.
3. Generates the corpus using the previous configuration file and stores the csv in **/home/user/plsatwitter-results/csv/spanish_teams.csv**
4. Generates objective matrix and list of words in the vocabulary.
5. Computes the probability matrices for PLSA.
6. Extracts 10 top words for each topic.

By default, the function _compute_probabilities_ uses as parameters `n_iters=20` and `n_topics=10`. The default option `method='EM'`is slower and offers the same performance than the option `method='sklearn'`.






```python3
    from plsatwitter.models.plsa_nmf import PlsaNmf
    from plsatwitter.config.folders import folders
    
    pn = PlsaNmf()
    accounts = ['realmadrid', 'FCBarcelona_ES', 'Atleti', 'valenciacf']
    pn.generate_corpus(cuentas=accounts, config_path='config.ini', corpus_path=folders['csv'] + 'spanish_teams.csv')
    X, frecs, words = pn.generate_matrix()
    Xp, P1, P2, P3 = pn.compute_probabilities(method="sklearn")
    top_words(P1, words, cloud_mode=True)
```

It is also possible to perform a query-based retrieval, as:
```python3
   import plsatwitter as pt
  pt.examples.queries.run()

  Introduce your search query: juego de tronos
  @el_pais - (prob = 0.23)
  "La violación no es una herramienta para que un personaje sea más fuerte". La actriz Jessica Chastain carga contra ‘Juego de Tronos’ por su uso de la violencia sexual [ojo, spoilers] https://t.co/Iq7umkzyPG
  ________________________________________
  @publico_es - (prob = 0.15)
  Café del Starbucks en 'Juego de Tronos', coches que se fabricaron 10 años después de la época en la que discurre la película...

  Vía @Strambotic https://t.co/jRHnRx779x
  ________________________________________
  @abc_es - (prob = 0.15)
  El significado real de las últimas palabras de un protagonista de «Juego de Tronos» antes de morir #GOT #JuegoDeTronos8x04  https://t.co/YB2L5fCyK4
  ________________________________________
  @abc_es - (prob = 0.15)
  Resuelto el misterio del vaso de Starbucks en el último capítulo de «Juego de Tronos» #JuegoDeTronos #GOT https://t.co/S7ir2DoZ46
  ________________________________________
  @elespanolcom - (prob = 0.14)
  El espectacular gazapo del último episodio de 'Juego de Tronos' que llena de memes la Red https://t.co/CrUE9hdejw
  ________________________________________
```

Remarks and clarifications
----------
This project is almost complete. There are some modules which are included in the package but not explained in this file because they are only of academic interest. However, if one is interested in the results of comparing several algorithms for nonnegative matrix factorization, including initialization, the subpackage ```plsatwitter.tests``` may be of interest. 

Module and class docstrings are expected to be added in the future, as well as a GUI instead of the current CLI working mode.


References
----------
1. Hofmann, Thomas (1999). “Probabilistic latent semantic indexing”. En: Proceedings of the 22nd annual international ACM SIGIR conference on Research and
development in information retrieval - SIGIR ’99. ACM Press. doi: 10.1145/
312624.312649.



Feedback
--------
Please send bug reports, comments or questions to [Ángel Pérez](alperezmi@hotmail.com). Spelling and grammar corrections are also welcomed.
