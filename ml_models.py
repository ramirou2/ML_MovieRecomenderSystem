import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.neighbors import NearestNeighbors

def knn_movies(df, id, k = 5):
    df['genres'] = df.genres.apply( lambda x: eval(x))
    df['production_companies'] = df.production_companies.apply( lambda x: eval(x))

    # Variables de entrada
    X_genres = df['genres']
    X_prod_companies = df['production_companies']
    X_year_popularity = df[['release_year', 'popularity']]

    # Codificar los géneros utilizando one-hot encoding
    mlb_genres = MultiLabelBinarizer()
    X_genres_encoded = pd.DataFrame(mlb_genres.fit_transform(X_genres), columns=mlb_genres.classes_)

    mlb_prod = MultiLabelBinarizer()
    X_prod_companies_encoded = pd.DataFrame(mlb_prod.fit_transform(X_prod_companies), columns=mlb_prod.classes_)

    # Escalar los datos de popularidad y año de lanzamiento
    scaler = StandardScaler()
    X_year_popularity_scaled = scaler.fit_transform(X_year_popularity)

    # Combinar todas las variables de entrada
    X = np.concatenate([X_genres_encoded.values,X_prod_companies_encoded, X_year_popularity_scaled], axis=1)

    # Entrenar el modelo de k-NN
    # Número de vecinos a considerar
    k = k + 1
    model = NearestNeighbors(n_neighbors=k)
    model.fit(X)

    # Ejemplo de consulta de una película y recomendación
    peli = df[df['id'] == id]

    query_movie_genres = peli['genres']
    query_movie_prod = peli['production_companies']
    query_movie_year_popularity = (peli[['release_year', 'popularity']])
    query_movie_year_popularity_scaled = scaler.transform(query_movie_year_popularity)

    # Codificar los géneros de la película consultada
    query_movie_genres_encoded = mlb_genres.transform(query_movie_genres)

    query_movie_prod_encoded = mlb_prod.transform(query_movie_prod)
    # Combinar las variables de consulta
    query_movie = np.concatenate( [query_movie_genres_encoded, query_movie_prod_encoded,query_movie_year_popularity_scaled], axis = 1)

    # Obtener los índices de las películas recomendadas
    distances, indices = model.kneighbors(query_movie)
    # Obtener las películas recomendadas
    recommended_movies = df.iloc[indices[0], :]
    recommended_titles = recommended_movies['title']
    return recommended_titles

if __name__ == '__main__':
    # Cargar el conjunto de datos de películas
    df = pd.read_csv('data/cleanMovies.csv')
    movies = knn_movies(df, 863,6)
    print('helloword')
    print(movies)
