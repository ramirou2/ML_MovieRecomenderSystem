import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.neighbors import NearestNeighbors

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
    df['texto_combinado'] = df['genres'].apply(lambda x: ' '.join(x)) + ' ' + df['title']   + ' ' + df['overview']
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
    # scores = sim_scores[1:top_n+1]
    return top_movies #, top_indices, scores

def modelos_knn(df, k = 5):
    """
    Generacion de modelos para la obtencion de recomendaciones empleando knn y las features de genres, production companies, release_year y popularity.
    Se especifica en parametros la definicion de las estructura del DataFrame

    Parametros
    ----------
    df : pd.DataFrame()

        DataFrame empleado para la generacion de los modelos previos.
        df[['title', 'genres', 'production_companies', 'release_year', 'popularity']] --> titles --> str genres --> ['genero1', 'genero2',... ] release_year --> int  popularity --> float 
    k : int

        Cantidad de recomendaciones a devolver posteriormente
    
    Retorno
    -------
    modelo:  sklearn.neighbors._unsupervised.NearestNeighbors

        modelo de knn entrenado previamente con los datos de df sin desordenar

    mlb_genres: sklearn.preprocessing._label.MultiLabelBinarizer

        modelo para el one hot encoding de genres

    scaler: sklearn.preprocessing._data.StandardScaler

        Modelo de escalado de 'release_year', 'popularity'

    Ejemplo
    --------
    >>> modelos_knn(df, k=5) \t
    >>>  (model, scaler, mlb_genres, mlb_prod) Ver en retorno

    """
    # Variables de entrada
    X_genres = df['genres']
    X_year_popularity = df[['release_year', 'popularity']]

    # Codificar los géneros utilizando one-hot encoding
    mlb_genres = MultiLabelBinarizer()
    X_genres_encoded = pd.DataFrame(mlb_genres.fit_transform(X_genres), columns=mlb_genres.classes_)

    # Escalar los datos de popularidad y año de lanzamiento
    scaler = StandardScaler()
    X_year_popularity_scaled = scaler.fit_transform(X_year_popularity)

    # Combinar todas las variables de entrada
    X = np.concatenate([X_genres_encoded.values, X_year_popularity_scaled], axis=1)

    # Entrenar el modelo de k-NN
    # Número de vecinos a considerar
    k = k + 1
    model = NearestNeighbors(n_neighbors=k)
    model.fit(X)
    return model, scaler, mlb_genres

def recomendaciones_knn(df, indice, knn, scaler, mlb_genres):
    """
    Peliculas recomendadas en función de la de entrada, se ingresa el indice asociado al DataFrame de entrada, mismo con el que se generan los modelos subsiguientes. 
    Se retorna una lista de titulos recomendados. El

    Parametros
    ----------
    indice: int

        Indice de la pelicula a buscar recomendaciones

    knn:  sklearn.neighbors._unsupervised.NearestNeighbors

        modelo de knn entrenado previamente con los datos de df sin desordenar

    mlb_genres: sklearn.preprocessing._label.MultiLabelBinarizer

        modelo para el one hot encoding de genres

    scaler: sklearn.preprocessing._data.StandardScaler

        Modelo de escalado de 'release_year', 'popularity'
        
    df : pd.DataFrame()

        DataFrame empleado para la generacion de los modelos previos.
        df[['title', 'genres', 'production_companies', 'release_year', 'popularity']] --> titles --> str genres --> ['genero1', 'genero2',... ] release_year --> int  popularity --> float 
    
    Retorno
    -------
    list

        Lista de titulos de peliculas más similares a la asociada al indice ingresad en orden descendente de similitud.

    Ejemplo
    --------
    >>> recomendaciones_knn('titulo') \t
    >>> ['titulo1','titulo2','titulo3','titulo4','titulo5']

    """
    
    # Ejemplo de consulta de una película y recomendación
    peli = df[df.index == indice]
    query_movie_genres = peli['genres']
    query_movie_year_popularity = peli[['release_year', 'popularity']]
    query_movie_year_popularity_scaled = scaler.transform(query_movie_year_popularity)

    # Codificar los géneros de la película consultada
    query_movie_genres_encoded = mlb_genres.transform(query_movie_genres)

    # Combinar las variables de consulta
    query_movie = np.concatenate( [query_movie_genres_encoded,query_movie_year_popularity_scaled], axis = 1)

    # Obtener los índices de las películas recomendadas
    distances, indices = knn.kneighbors(query_movie)
    # Obtener las películas recomendadas
    recommended_titles = df['title'].iloc[indices[0][1:6]].values
    return recommended_titles

if __name__ == '__main__':
    df = pd.read_csv('data/cleanMovies.csv')
    df['genres'] = df.genres.apply( lambda x: eval(x))
    #df['production_companies'] = df.production_companies.apply( lambda x: eval(x))
    indice = df[df['title'] == 'Jumanji' ].index[0]
    df['popularity'] = df['popularity'].fillna(0)
#USO PARA SIMILITUD
    # sim_matrix = matriz_similitud(df[0:20000])
    # print(df.iloc[indice][['title', 'genres']])
    # print(obtener_recomendaciones( indice,sim_matrix, df[0:20000],5 ))
    # print(df.iloc[[13337, 15987, 6160, 9495, 8795]][['title', 'genres']])
# USO PARA KNN
    model, scaler, mlb_genres = modelos_knn(df[0:2000], k = 5)
    titulos = recomendaciones_knn(df[0:2000], indice, model, scaler, mlb_genres)
    print(titulos)
    #print(type(model),type(scaler), type(mlb_genres), type(mlb_prod) )
