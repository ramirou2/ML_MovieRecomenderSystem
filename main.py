from fastapi import FastAPI, Path, Query, Body
from pydantic import BaseModel, Field 
from typing import Optional 
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime

from ml_models import matriz_similitud, obtener_recomendaciones

# CARGANDO LOS ARCHIVOS NECESARIOS PARA LOS ENDPOINTS
df = pd.read_csv('data/cleanMovies.csv', parse_dates = ['release_date'])
df2 = pd.read_csv('data/cleanCredits.csv')
df3 = pd.merge(df, df2, on='id', how='inner')
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
    mes : str

        mes en idioma español enero,febrero,marzo,...

    Retorno
    -------
    JSON:

        {"month":mes, "movies": int(salida) }
    
    Ejemplo
    --------
    >>> cantidad_filmaciones_mes(febrero)

        { "month": "enero", "movies": 5909 }
    """
    mesNum = meses[mes]
    salida = df[df['release_date'].dt.month == mesNum].release_date.count()
    return {"month":mes, "movies": int(salida) }

#Endpoint 2
class Dia(str, Enum):
    lunes = 'lunes'
    martes = 'martes'
    miercoles = 'miércoles'
    jueves = 'jueves'
    viernes = 'viernes'
    sabado = 'sábado'
    domingo = 'domingo'

dias = {
    'lunes': 'Monday',
    'martes': 'Tuesday',
    'miércoles': 'Wednesday',
    'jueves': 'Thursday',
    'viernes': 'Friday',
    'sábado': 'Saturday',
    'domingo': 'Sunday'
}

@app.get("/peliculas/get_day/{dia}", tags=['Peliculas'])
def cantidad_filmaciones_dia( dia: Dia ):
    """
    Cantidad de filmaciones estrenadas el dia indicado, incluyendo todos los meses.

    Esta función recibe un dia en idioma español y retorna el número de peliculas estrenadas dicho día sin importar cual es el mes o año.

    Parametros
    ----------
    dia : str

        dia en idioma español lunes, martes, miércoles, jueves, viernes, sábado, domingo

    Retorno
    -------
    JSON

        {"day":dia, "movies": int(salida) }

    Ejemplo
    --------
    >>> cantidad_filmaciones_dia(lunes)
        { "dia": "lunes", "peliculas": 3500 }
    """
    df['dia'] = df['release_date'].apply(lambda x: x.strftime('%A'))
    salida = df[df['dia'] == dias[dia]].dia.count()
    
    return {"day":dia, "movies": int(salida) }

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
    titulo = titulo.title()
    coincidencias = df[df['title'] == titulo] 

    if not(coincidencias.empty):
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
    >>> votor_titulo('Jumanji') {caso de exito} \t
    >>> {'title':'Jumanji', 'release_year:': 1995, 'vote_count': 2413, 'vote_average':6.9 } 

    >>> votor_titulo('Father of the Bride Part II') {contexto de no existencia o de votos insuficientes}  \t
    >>> {'title':'', 'release_year': '', 'vote_count': '', 'vote_average':'' }
    """
    titulo = titulo.title()

    coincidencias = df[df['title'] == titulo]

    salida_json = {'title':'', 'release_year': '', 'vote_count': '', 'vote_average':'' }
    if not(coincidencias.empty):
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

        {'actor':'Actor', 'cantidad': cantidad peliculas, 'retorno_promedio': round(avg(retorno), 2)  , 'retorno_total': round(sum(retorno), 2)}

    Ejemplo
    --------
    >>> get_actor('Tom Hanks') {caso de exito} \t 
    >>> {'actor':'Tom Hanks', 'cantidad': 71 , 'retorno_promedio':  2.52,'retorno_total': 3.96}

    >>> get_actor('Pepe El grillo') {caso error o sin existencia} \t
    >>> {'actor':'', 'cantidad': '', 'retorno_promedio': '' , 'retorno_total': ''}
    """ 
    actor = actor.title()
    indices = []
    for index, movie in df3.iterrows():
        if actor in movie['cast']:
            indices.append(index)

    if len(indices) == 0:
        salida_json = {'actor':'', 'cantidad': '', 'retorno_promedio': '' , 'retorno_total':''}
    else:
        coincidencias = df3.iloc[indices]
        retorno_promedio = coincidencias['return'].mean()
        retorno_total = coincidencias['revenue'].sum() / coincidencias['budget'].sum()
        salida_json = {'actor':actor, 'cantidad': len(indices), 'retorno_promedio': round(retorno_promedio, 2), 'retorno_total':round(retorno_total, 2)}
    return salida_json 

@app.get("/director/get_director/{director}", tags=['Directores'])
def get_director(director: str):
    """
    Éxito del director indicado medido a través del retorno. Peliculas dirigidas con fecha, costo y ganancia individual.

    Esta función recibe el nombre completo de un director y devuelve las peliculas dirigidas con fecha, costo y ganancia individual.

    Parametros
    ----------
    director : str

        nombre completo del director


    Retorno
    ----------
    JSON

        { 'director':'director', 'return': round(retorno, 2),  'movies':  [{pelicula1}, {pelicula2} ....]}.

    Ejemplo
    --------
    >>> get_director('John Lasseter') {caso de exito} \t
    >>> { "director": "John Lasseter", "return": 4.03,
        "movies": [ { "title": "Toy Story", "release_date": "1995-10-30", "budget": 30000000, "revenue": 373554033 },
        { "title": "A Bug'S Life", "release_date": "1998-11-25", "budget": 120000000, "revenue": 363258859 }, ...]

    >>> get_director('Pepe el grillo') {caso de inexistencia} \t
    >>> { 'director':'', 'return': '',  'movies': ''}
    """
    director = director.title()
    indices = []
    for index, movie in df3.iterrows():
        if director in movie['director']:
            indices.append(index)

    if len(indices) > 0:
        peliculas = df3.iloc[indices][['title', 'release_date', 'budget', 'revenue']]
        # peliculas = peliculas[['title', 'release_date', 'budget', 'revenue']]
        retorno_total = peliculas['revenue'].sum() / peliculas['budget'].sum()
        titulos = peliculas['title'].to_list()
        fechas_estreno = peliculas['release_date'].dt.date.to_list()
        presupuesto = peliculas['budget'].to_list()
        ganancia = peliculas['revenue'].to_list()
        
        #SALIDA EN VERSION LISTAS
        #salida = { 'director':director, 'return': round(retorno_total, 2),  'titles': titulos, 'release_dates': fechas_estreno, 'budgets': presupuesto, 'revenues':ganancia}
        pelis_json = [{'title': e1, 'release_date': e2, 'budget': e3, 'revenue': e4} for e1, e2, e3,e4 in zip(titulos, fechas_estreno, presupuesto, ganancia)]
        
        salida = { 'director':director, 'return': round(retorno_total, 2),  'movies': pelis_json}
    else:
        salida = { 'director':'', 'return': '',  'titles': '', 'release_dates': '', 'budgets': '', 'revenues': ''}
    return salida

@app.get("/recomendacion/get_recomendacion/{titulo}", tags=['Sistema de Recomendacion'])
def get_recomendacion(titulo: str):
    """ 
    Recomendación de las 5 peliculas más similares al titulo ingresado.
    Se recibe el titulo de una pelicula en idioma ingles y se devuelve una lista ded nombres de las 5 peliculas más similares recomendada por el sistema.

    Parametros
    ----------
    titulo : str

        Titulo en ingles de la pelicula.

    Retorno
    -------
    JSON

        { 'titles':['pelicula1', 'pelicula2', 'pelicula3', 'pelicula4' , 'pelicula5' ] }.

    Ejemplo
    --------
    >>> get_recomendacion('Jumanji') {caso de exito} \t
    >>> 

    >>> get_director('Pepe el grillo') {caso de inexistencia} \t
    >>> { 'titles': []}
    
    """
    
    titulo = titulo.title()
    coincidencias = df[df['title'] == titulo]
    if coincidencias.empty:
        salida = {'title': '',  'recommended titles': []}
    else:
        indice = coincidencias.index[0]

        entrada = df[0:2000][['title', 'genres', 'overview']]
        similitudes = matriz_similitud(entrada)
        recomendadas = obtener_recomendaciones(df = entrada, matriz_sim = similitudes, indice_pelicula = indice,top_n = 5).tolist()


        salida = {'title': titulo, 'recommended titles': recomendadas}
    return salida