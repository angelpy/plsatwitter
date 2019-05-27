import io
import numpy as np
import matplotlib.pyplot as plt
import wordcloud
from plsatwitter.config.folders import folders

def plot_residual(n_iter, residual):
    plt.plot(n_iter, residual['W'],'-o',  n_iter, residual['H'], '-o')
    plt.xlabel('Nº iteraciones')
    plt.legend(['kkt(W)', 'kkt(H)'])
    plt.show()
    
def plot_dict_grande(dictionary, xvalues, xlabel, ylabel, title, filename):
    fig = plt.figure()
    markers = ['-o', '-v', '-s', '-p', '-P', '-*',]
    index = 0
    for k,v in dictionary.items():
        if type(v) != list:
            raise ValueError("El valor debe ser un array.")
        if not xvalues:
            raise ValueError("Hay que pasar como parámetro el array x.")
        if not xlabel:
            raise ValueError("Hay que pasar como parámetro la etiqueta de x.")
        if not ylabel:
            raise ValueError("Hay que pasar como parámetro la etiqueta de y.")
        if len(xvalues) != len(v):
            raise ValueError("Error, el array x y el array y tienen distinto tamaño.")
        plt.plot(xvalues, v, markers[index], markersize=3)
        index += 1
    plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(2000, 5000)
    plt.title(title)
    plt.legend(list(dictionary.keys()))
    fig.savefig(filename, dpi=1000, bbox_inches = "tight")

def plot_dict(dictionary, xvalues, xlabel, ylabel, title, filename):
    fig = plt.figure()
    markers = ['-o', '-v', '-s', '-p', '-P', '-*',]
    index = 0
    for k,v in dictionary.items():
        if type(v) != list:
            raise ValueError("El valor debe ser un array.")
        if not xvalues:
            raise ValueError("Hay que pasar como parámetro el array x.")
        if not xlabel:
            raise ValueError("Hay que pasar como parámetro la etiqueta de x.")
        if not ylabel:
            raise ValueError("Hay que pasar como parámetro la etiqueta de y.")
        if len(xvalues) != len(v):
            raise ValueError("Error, el array x y el array y tienen distinto tamaño.")
        plt.plot(xvalues, v, markers[index], markersize=3)
        index += 1
    plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(list(dictionary.keys()))
    fig.savefig(filename, dpi=1000, bbox_inches = "tight")

def top_words(topic_term_dist, term_list, n_top_words=10, saveRoute = None, cloud_mode = False):
    r"""
    Prints the n_top_words words of a topic-word distribution.
    Args:
        topic_term_dist: array-like
            Topic-word distribution of shape (K, M)
            with K < M.
        term_list: list
            List of words used in PLSA
        n_top_words: int
            Number of top words.
        saveRoute: str
            Storage route of generated files.
        cloud_mode: bool
            If true, ignores the n_top_words attribute
            and prints a cloud of words for each topic.
            If false (default) a text summary is generated.
    Returns:
        None
    """
    if topic_term_dist is None:
        raise ValueError("No topic-term distribution was defined.")
    if term_list is None:
        raise ValueError("No list of terms was defined.")
    if len(term_list) == 0:
        raise ValueError("The provided list of terms is empty.")
    if n_top_words <= 0:
        raise ValueError("The number of top words should be positive.")
    if type(saveRoute) is not str and saveRoute:
        raise ValueError("The saveRoute parameter should be a string.")
    if type(cloud_mode) is not bool:
        raise ValueError("The cloud_mode parameter should be a bool.")
    if topic_term_dist.shape[0] > topic_term_dist.shape[1]:
        topic_term_dist = topic_term_dist.T
    for topic_number, topic in enumerate(topic_term_dist):
        resumen = ""
        if cloud_mode:
            fig = plt.figure()
            name = 'topic_{}'.format(topic_number)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(
                wordcloud.WordCloud(background_color='white').fit_words(
                    dict(zip(
                        term_list,
                        topic
                    ))
                ).to_image()
            )
            
            if saveRoute:
                ruta = folders['topics']
                fig.savefig(ruta + name + '.png', dpi=800)
        else:
            # Text representation
            message = "Topic #%d: " % topic_number
            message += " ".join([term_list[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            message += "\n"
            resumen += message
            if saveRoute:
                f1 = io.open('summary_topics.txt', 'w', encoding='utf-8')
                f1.write(resumen)
                f1.close()
            else:
                print(resumen)

def print_top_tweets(dist, corpus):
    argmax = np.argmax(dist.T, axis=1)
    for topic_idx, topic in enumerate(dist.T):
        print("----------Topic {}------------".format(topic_idx))
        print(corpus.iloc[topic_idx].full_text)
        
def top_tweets_topic(dist, corpus, n_topic=0, n_top_tweets=5):
    idx_tweets = (-dist[n_topic, :]).argsort()[:n_top_tweets]
    print("Tuits más relevantes para el tema {}".format(n_topic))
    for idx in idx_tweets:
        mensaje = """
Tweet de @{}
Contenido:
{}
        """.format(corpus.iloc[idx].screen_name,
        corpus.iloc[idx].full_text)
        print(mensaje)
def write_top_tweets(self, dist):
    message = ""
    argmax = np.argmax(dist.T, axis=1)
    for topic_idx, topic in enumerate(dist.T):
        message += "----------Topic {}------------".format(topic_idx)
        message += self.statuses[argmax[topic_idx]].full_text