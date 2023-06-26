import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def matriz_similitud(df):
    """
    Generacion de la matriz de similitud del coseno para el Data Frame ingresado.
    Se ingresa un DataFrame con la información necesaria para generar la matriz de similud del coseno y se otorga dicha matriz.
    Se especifica en parametros la estructura del DataFrame necesaria
    
    Parametros
    ----------
    df : pd.DataFrame()

        DataFrame con la informacion empleada para la generacion de la matriz de similitud del coseno.
        df[['titles', 'genres', 'overview']] --> titles --> str overview --> str genres --> ['genero1', 'genero2',... ]

    Retorno
    -------
    numpy.ndarray

        matriz  de similtud del coseno que expresa la simiaridad entre los vectores que representan cada pelicula.

    Ejemplo
    --------
    >>> matriz_similitud(movies) \t
    >>> array([[1.        , 0.29039095, 0.22495375, ..., 0.35094275, 0.17067707, 0.12071039],
                [0.29039095, 1.        , 0.        , ..., 0.        , 0.        , 0.        ],....]

    """

    #Se genera la columa con el texto de entrada
    df['texto_combinado'] = df['genres'].apply(lambda x: ' '.join(x)) + ' ' + df['title'] + ' ' + df['overview']
    #Elimino los signos de puntuacion
    df['texto_combinado'] = df['texto_combinado'].apply( lambda x: re.sub(r'[^\w\s]', '', x) if pd.notnull(x) else '' )

    # Crear una matriz TF-IDF a partir de los datos empleando las stop words en idioma ingles
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['texto_combinado'])

    #Se genera la matriz de similitud del coseno a partir de la matriz anterior
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim

def obtener_recomendaciones(indice_pelicula, matriz_sim, df, top_n=5):
    """
    Peliculas recomendadas en función de la de entrada, se ingresa el indice asociado al DataFrame de entrada, mismo con el que se genero la matriz_similitud

    Parametros
    ----------
    indice_pelicula: int

        Indice de la pelicula a buscar recomendaciones

    matriz_sim: numpy.ndarray 

        Matriz de similitud del coseno asociada al df, calculada previamente

    df : pd.DataFrame()

        DataFrame empleado para la generacion de la matriz  de similitud del coseno.

    top_n : int

        Numero de recomendaciones a obtener, por defecto 5.
    
    Retorno
    -------
    list

        Lista de titulos de peliculas más similares a la asociada al indice ingresad en orden descendente de similitud.

    Ejemplo
    --------
    >>> obtener_recomendaciones('titulo') \t
    >>> ['titulo1','titulo2','titulo3','titulo4','titulo5']

    """

    #Dada la correspondencia de los indices del df con la matriz me quedo con los puntajes de similitud en torno a esa pelicula
    sim_scores = list(enumerate(matriz_sim[indice_pelicula]))
    #Ordeno de mayor a menor los scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    #Selecciono lo más similares en funcion de top_n, evadiendo el primero que corresponde a la misma pelicula
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]

    #Me quedo con los titulos y la similitud
    top_movies = df['title'].iloc[top_indices].values

    return top_movies



if __name__ == '__main__':
    df = pd.read_csv('data/cleanMovies.csv')
    df['genres'] = df.genres.apply( lambda x: eval(x))

    sim_matrix = matriz_similitud(df[0:5000])

    print(obtener_recomendaciones( 863,sim_matrix, df[0:5000],5 ))

