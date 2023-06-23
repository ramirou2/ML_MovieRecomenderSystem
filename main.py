from fastapi import FastAPI, Path, Query, Body
from pydantic import BaseModel, Field 
from typing import Optional 
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime

from ml_models import knn_movies

# CARGANDO LOS ARCHIVOS NECESARIOS PARA LOS ENDPOINTS
df = pd.read_csv('data/cleanMovies.csv', parse_dates = ['release_date'])
df2 = pd.read_csv('data/cleanCredits.csv')
df3 = pd.merge(df, df2, on='id', how='left')
# Casteo de datos a tipo lista para posterior manipulacion
df3['cast'] = df3['cast'].apply(lambda x: eval(x) if pd.notnull(x) else list([]))
df3['director'] = df3['director'].apply(lambda x: eval(x) if pd.notnull(x) else list([]))


#DATA GENERAL DE LA API
app = FastAPI()
app.title = "Titulo de la API"
app.version = "Version de la APi"

#Root API
@app.get("/")
def index():
    return {"message": ""}

#Endpoint 1
class Mes(str, Enum):
    enero = "enero"
    febero = "febrero"
    marzo = "marzo"
    abril = "abril"
    mayo = "mayo"
    junio = "junio"
    julio = "julio"
    agosto = "agosto"
    septiembre = "septiembre"
    octubre = "octubre"
    noviembre = "noviembre"
    diciembre = "diciembre"
meses = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12
}
@app.get("/peliculas/get_month/{mes}", tags=['Peliculas'])
def cantidad_filmaciones_mes( mes: Mes ):
    """
    Cantidad de filmaciones estrenadas el mes indicado.

    Esta función recibe un mes en idioma español y retorna el número de peliculas estrenadas dicho mes sin importar cual es el año.

    Parametros
    ----------
    mes : int

        mes en idioma español enero,febrero,marzo,...

    Retorno
    -------
    int:

        Cantidad de peilculas estrenadas en el mes numero ''mes''.
    
    Ejemplo
    --------
    >>> cantidad_filmaciones_mes(febrero)

        X cantidad de películas fueron estrenadas en el mes de febrero
    """
    mesNum = meses[mes]
    salida = df[df['release_date'].dt.month == mesNum].release_date.count()
    return {"mes":mes, "peliculas": int(salida) }

#Endpoint 2
@app.get("/peliculas/get_month/{dia}", tags=['Peliculas'])
def cantidad_filmaciones_dia( dia: int = Path(ge=1, le=31) ):
    """
    Cantidad de filmaciones estrenadas el dia indicado, incluyendo todos los meses.

    Esta función recibe un dia en idioma español y retorna el número de peliculas estrenadas dicho día sin importar cual es el mes o año.

    Parametros
    ----------
    dia : int
        dia en idioma español 1-31

    Retorno
    -------
    int
        Cantidad de peilculas estrenadas en el mes de ''mes''.

    Ejemplo
    --------
    >>> cantidad_filmaciones_dia(30)
        X cantidad de películas fueron estrenadas en los días 30
    """
    salida = df[df['release_date'].dt.day == dia].release_date.count()
    return {"dia":dia, "peliculas": int(salida) }

#Endpoint 3
@app.get("/pelicula/get_popularidad/{titulo}", tags=['Pelicula'])
def score_titulo( titulo: str ):
    """
    Score asociada a la pelicula con el titulo indicado.

    Esta función recibe un título de pelicula en idioma ingles y retorna título, el año de estreno y el score asociado redondeado a 2 decimales.

    Parametros
    ----------
    titulo : str
        titulo en idioma ingles

    Retorno
    -------
    JSON:
        {'title':titulo, 'release_year': release_year, 'popularity':popularity}

    Ejemplos
    --------
    >>> score_titulo('Father of the Bride Part II') {caso de exito} \t
    >>> {'title':'Father of the Bride Part II', 'release_year': 1995, 'popularity':8.39}

    >>> score_titulo('Father of theBride') {contexto de error o no existencia}  \t
    >>> {'title':'', 'release_year': '', 'popularity':''}

    """
    titulo = titulo.lower()
    coincidencias = df[df['title'] == titulo] 

    if coincidencias.shape[0] > 0:
        salida_df = coincidencias[['title', 'release_year', 'popularity']].iloc[0]
        salida_json =  {'title':titulo, 'release_year': int( salida_df['release_year']), 'popularity': round(salida_df['popularity'], 2) } 
    else:
        salida_json = {'title':'', 'release_year': '', 'popularity':''}

    return salida_json

#Endpoint 4
@app.get("/pelicula/get_votos/{titulo}", tags=['Pelicula'])
def votos_titulo( titulo: str ):
    """
    Cantiad de votos y valor promedio de las votaciones asociada a la pelicula con el titulo indicado, en caso de que tenga al menos 2000 valoraciones.
    Caso contrario, no se devuelve ningun valor.

    Esta función recibe un título de pelicula en idioma ingles y retorna título, el año de estreno, la cantidad de votaciones y el valor promedio de las votaciones.

    Parametros
    ----------
    titulo : str
        titulo en idioma ingles

    Retorno
    -------
    registro:
        {'title':titulo, 'release_year': release_year, 'vote_count': vote_count, 'vote_average':vote_average }
    Ejemplo
    --------
    >>> votor_titulo('Father of the Bride Part II') {caso de exito} \t
    >>> {'title':'Father of the Bride Part II', 'release_year': 1995, 'vote_count': 2000, 'vote_average':5.7 } CORREGIR CON UNO QUE SEA REAL

    >>> votor_titulo('Father of the Bride Part II') {contexto de no existencia o de votos insuficientes}  \t
    >>> {'title':'', 'release_year': '', 'vote_count': '', 'vote_average':'' }
    """
    titulo = titulo.lower()

    coincidencias = df[df['title'] == titulo]

    salida_json = {'title':'', 'release_year': '', 'vote_count': '', 'vote_average':'' }
    if coincidencias.shape[0] > 0:
        salida_df = coincidencias[['title', 'release_year', 'vote_count', 'vote_average']].iloc[0] #Me quedo con la primer aparicion
        if salida_df['vote_count'] >= 2000:
            salida_json = {'title':titulo, 'release_year': int( salida_df['release_year']), 'vote_count': int(salida_df['vote_count']), 'vote_average':round(salida_df['vote_average'], 2) }        

    return salida_json

@app.get("/actor/get_actor/{actor}", tags=['Actores'])
def get_actor( actor: str ):
    """
    Éxito del acotor indicado medido a través del retorno. Cantidad de películas que en las que ha participado y el promedio de retorno.

    Esta función recibe el nombre completo de un actor y devuelve la cantidad de peliculas en las que ha participado, el retorno total y el promedio de retorno.

    Parametros
    ----------
    nombreActor : str
        Nombre completo del actor.

    Retorno
    -------
    JSON
        JSON con determinado formato.

    Ejemplo
    --------
    >>> get_actor('Tom Hanks') {caso de exito} \t 
    >>> {'actor':'Tom Hanks', 'cantidad': , 'retorno': , 'retorno_promedio':  }

    >>> get_actor('Pepe El grillo') {caso error o sin existencia} \t
    >>> {'actor':'', 'cantidad': '', 'retorno':'', 'retorno_promedio': '' }
    """ 
    actor = actor.lower()
    cantidad = 0
    retorno = 0.0
    retorno_promedio = 0.0
    indices = []
    for index, movie in df3.iterrows():
        if actor in movie['cast']:
            cantidad += 1
            retorno += movie['return']
            indices.append(index)
    if cantidad == 0:
        salida_json = {'actor':'', 'cantidad': '', 'retorno':'', 'retorno_promedio': '' }
    else:
        retorno_promedio = retorno / cantidad
        salida_json = {'actor':actor, 'cantidad': cantidad, 'retorno':round(retorno, 2), 'retorno_promedio': round(retorno_promedio, 2)}
    return salida_json 

@app.get("/director/get_director/{director}", tags=['Directores'])
def get_director(director: str):
    """
    Éxito del director indicado medido a través del retorno. Peliculas dirigidas con fecha, costo y ganancia individual.

    Esta función recibe el nombre completo de un director y devuelve las peliculas dirigidas con fecha, costo y ganancia individual.

    Parametros
    ----------
    nombreDirector : str
        Nombre completo del director.

    Retorno
    -------
    JSON
        { 'director':'director', 'return': round(retorno, 2),  'movies':  [{pelicula1}, {pelicula2} ....]}.

    Ejemplo
    --------
    >>> get_director('John Lasseter') {caso de exito} \t
    >>> { 'director':'John Lasseter', 'return': round(retorno, 2),  'movies': [{pelicula1}, {pelicula2} ....]}

    >>> get_director('Pepe el grillo') {caso de inexistencia} \t
    >>> { 'director':'', 'return': '',  'movies': ''}
    """
    director = director.lower()
    retorno = 0.0
    indices = []
    for index, movie in df3.iterrows():
        if director in movie['director']:
            indices.append(index)
            retorno += movie['return']

    if len(indices) > 0:
        peliculas = df3.iloc[indices]
        peliculas = peliculas[['title', 'release_date', 'budget', 'revenue']]
        # Si bien el formato ya esta en datetime.date, el to_json necesita el uso de strftime para otorgar la salida correcta, agregar en el procesamiento previo
        peliculas["release_date"] = peliculas["release_date"].dt.strftime('%Y-%m-%d')
        salida_json = peliculas.to_json(orient='records')
        salida = { 'director':director, 'return': round(retorno, 2),  'movies': salida_json}
    else:
        salida = { 'director':'', 'return': '',  'movies': ''}
    return salida

@app.get("/recomendacion/get_recomendacion/{pelicula}", tags=['Sistema de Recomendacion'])
def get_recomendacion(pelicula: str):
    """ COMPLETAR DOCSTRING
    
    """
    entrada = df[0:100]
    salida = knn_movies(entrada, 862,6)
    return {'peliculas': salida }