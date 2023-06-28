# <h1 align=center> **API con ML incorporado para sistema de recomendación de peliculas** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>

**`Autor: Gastón Luna Paez`**

[LinkedIn](https://www.linkedin.com/in/gaston-luna-paez-a528a91a6)

## **Propuesta de trabajo**</h2>
Desarrollar y desplegar una API a partir de un *dataset* crudo con información de peliculas y su equipo de producción (por razones de espacio no se encuentran en el repositorio). La misma debe contar con 7 endpoints, siendo el ultimo un sistema de recomendación de peliculas implementeado mediante *Machine Learning* (ML).
EL objetivo del proyecto es la obtención de un **MVP** (_Minimum Viable Product_) en un tiempo y entorno reducido.

Detalles de la propuesta:

**`Transformación y limpieza del *dataset* original`**.
* Toolbox: Python, Numpy, Pandas,  Jupyter Notebook

**`Desarrollo de una API con 6 *endpoints* generales para la obtención de información básica del *dataset* ya procesado`**
* Toolbox: Python, Pandas, FastAPI, Jinja2, Render, Enum, Uvicorn, HTML, Boostrap, js

**`Analisis Exploratorio de Datos (EDA) para investigar el camino al ML`** 
* Toolbox: Python, Numpy, Pandas, Seaborn, Matplotlib

**`Desarrollar un Sistema de Recomendación de peliculas empleando ML e incorporarlo a la API como el último *endpoint*`** 
* Toolbox: Python, Numpy, Pandas, Seaborn, Matplotlib, Sklearn, RE



## **Camino recorrido al MVP**</h2>

### **`ETL (Extract, Transform and Load)`**</h3>

Partiendo de dos *dataset* asociados a peliculas y equipos de filmación (movies_dataset.csv, credits.csv) se realizo un proceso de ETL para obtener un conjunto de datos que permitiesen el desarrollo de funciones capaces de extraer información útil y procesable.

***Limpieza de movies_dataset.csv***</h4>

En dichos datos se contaba con información asociada a un abanico de peliculas, donde mucha de estas contaban con abundantes datos sin valor para el objetivo del proyecto, por ende entre los pasos realizados se encontro la elimiación de columnas redundantes así como de registros duplicados. A su vez se descartaron los registros sin fecha de lanzamiento (realease_date) dado que solo era una pequeña parte del total y dicha información era muy necesaria en pasos posteriores. Entre los pasos a destacar se encontro la "desanidación" de campos para su viabilidad de uso en el futuro, la creación de algunas columnas y el reemplazo de ciertos valores nulos.

El *DataFrame* resultante se exportó como [cleanMovies.csv](https://github.com/ramirou2/ML_MovieRecomenderSystem/blob/master/data/cleanMovies.csv).  

***Limpieza de credits.csv***</h4>

En el *dataset* en cuestión se contaba con mucha información asociada al equipo de filmación, incluyendo actores, y equipos de desarrollo, como ser directores, directores de sonido, etc. Sin embargo, para los pasos posteriores solo se requería los datos de los actores y del director general, por ende finalmente solo se extrajo esa información. Entre los pasos realizados se pueden destacar el "desanidamiento" de campos y la eliminacion de registros duplicados.

El *DataFrame* resultante se exportó como [cleanCredits.csv](https://github.com/ramirou2/ML_MovieRecomenderSystem/blob/master/data/cleanCredits.csv).

Para más detalles de estos procesos recurra a: [Movies ETL.ipynb](https://github.com/ramirou2/ML_MovieRecomenderSystem/blob/master/ETL_Inicial.ipynb)

### **`Desarrollo de API`**</h3>

****Endpoints* propuestos***</h4>

Se proponen 6 *endpoints* iniciales. Para cada uno de los *endpoint* se desarrollo una función en Python mediantes el uso del *FrameWork* FastAPI y librerias asociadas.
Funciones propuestas:

+ **cantidad_filmaciones_mes( Mes )**: Se ingresa un mes en idioma español (enero,febrero,marzo, ...) y se retorna la cantidad de películas que fueron estrenadas en el mes consultado.

+ **cantidad_filmaciones_dia( Dia )**: Se ingresa un día en idioma español(lunes, martes, miercoles, ...) y se retorna la cantidad de películas que fueron estrenadas en el día consultado.

+ **score_titulo( titulo )**: Se ingresa el título de una película y se retorna el título, año de estreno y la popularidad de la pelicula indicada.

+ **votos_titulo( titulo )**: Se ingresa el título de una película, si esta cuenta con al menos 2000 valoraciones se devuelve el título, la cantidad de votos y el promedio de las votaciones. Si la película consultada no cuenta con dicha condición se devuelve un mensaje avisando que no cumple esta condición y que por ende, no se devuelve ningun valor.

+ **get_actor( actor )**: Se ingresa el nombre de un actor y se retorna el éxito del mismo medido a través del retorno, la cantidad de películas en las que ha participado y el promedio de retorno entre esas peliculas.

+ **get_director( director )**: Se ingresa el nombre de un director y se retorna el éxito del mismo medido a través del retorno. A su vez se debe devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.

La implementación de las mismsa se puede observar en: ['main.py'](https://github.com/ramirou2/ML_MovieRecomenderSystem/blob/master/main.py)


***Implementación de la API***</h4>

Las funciones propuestas se desarrollaron de la mano de **python** y **FastAPI**, testeando su funcionamiento en el entorno local mediante **uvicorn**. En el [main.py](https://github.com/ramirou2/ML_MovieRecomenderSystem/blob/master/main.py) mencionado anteriormente se puede observar cada uno de los *endpoints* con sus respectivas rutas y el desarrollo completo de la API.


### **`Despliegue de la API`**</h3>

La API culminada fue depositada en render.com para su despligue, dado que el entorno de despliegue gratuito es muy reducido, de solo 512MB. Se opto por reducir el uso de los datos en el sistema de recomendación, dado que el procesamiento necesitado por el proceso de ML para todos los datos superaba ampliamente el limite establecido.

Otro inconveniente asiado a render fue la disminución de librerias a solo aquellas necesarias para el despliegue, así como el reversionado de las mismas a quellas versiones disponibles en el entorno de despliegue. Situación similar ocurrió con la versión de Python 3, sin embargo este punto no presento muchas dificultades.

Además de los aspectos básicos, en la ruta inicial de la API se realizo una minima interfaz gráfica amigable para ejemplificar las entradas de los endpoints y sus respuestas, sin embargo en /docs se cuenta con toda la documentación generada de manera casi automatica por FastAPI y con la posibilidad de realizar ejemplos de consultas y observar sus respuestas.

Para más detalles de la API vea :  [main.py](https://github.com/ramirou2/ML_MovieRecomenderSystem/blob/master/main.py)

***EDA***</h4>

Previo al desarrolo del sistema de recomendación se realizó un EDA para observar distribución de variables, patrones, posibles *outliers*, entre otras cosas. Dicho paso resulto de útilidad para confirmar o desechar hipotesis sobre aquellas *features* que podrían resultar útiles para las recomendaciones.
Para más información al respecto del EDA empleado dirijase a [EDA.ipynb](https://github.com/ramirou2/ML_MovieRecomenderSystem/blob/master/EDA.ipynb)

### **`Desarrollo del sistema de recomendación`**</h3>

***Idea general***</h4>

Se pleantea la creación de un sistema recomendación de películas basado en contenido, dado que no se cuenta con información de usarios para emplear otra modalidad de recomendación.

***Caso de uso:***</h4>

EL **usuario ingresa un titulo** de una pelicula. EL sistema debe responder con una **lista de las 5 peliculas** que considere más **similares**. 

En función del caso de uso general y de la información recopilada durante el EDA, se propuso la implementación de **dos modelos**.

***Modelo 1 - Matriz de Similitud de Coseno -***</h4>

Sistema de recomendación basado en contenido empleando el procesamiento de las palabras contenidas en las columnas *Overview*, *Genres* y *Title*. El uso de estas tres en conjunto disminuye las falencias del modelo en aquellos casos donde alguna de las *features* no aporte información de utilidad o cuente con los campos vacios. 

+ **Preparación de datos**

A partir del *DataFrame* de entrada, que cumpla con las condiciones especificadas en el *Docstring* asociado a las funciones del modelo en [ml_models.py](https://github.com/ramirou2/ML_MovieRecomenderSystem/blob/master/ml_models.py) se genera una columna con la unión de todos las columnas mencionadas salvando los vacios. Posteriormente se eliminan los signos de puntuación mediante expresiones regulares.

+ **Matriz TF-IDF**

Se realiza el calculo TF-IDF (*Term frequency – Inverse document frequency*) utilizando las *Stop Words* en idioma inglés. De modo de tener una representación númerica de la información de las *features* seleccionadas.
En este punto cabe mencionar que no se aplica *stemmer* para evitar afectar la capacidad de capturar similitudes semánticas precisas entre los documentos.

+ **Matriz Similitudes de Coseno**

Se calcula la matriz de sinmilitudes del coseno entre los vectores del TF-IDF obtenidos, de modo de cuantificar las similitudes en las peliculas que estan representadas.

+ **Obtención de los 5 similares**

Usando de la mantención de indices en los calculos de las martrices mencionadas, se obtiene el indice asociado al titulo ingresado como consulta, posteriormente se obtienen los 5 indices más similares, devolviendo posteriornmente los titulos más similares.

Para información más detallada y uso del modelo de manera externa a la API vea: [ml_models.py](https://github.com/ramirou2/ML_MovieRecomenderSystem/blob/master/ml_models.py)

***Modelo 2 - KNN  k-nearest neighbors -***</h4>

Empleando las *features* *genres, release_year, popularity* se empleo un modelo KNN par obtener las 5 peliculas más similares en función de un titulo de consulta. Para esta implementación se empleo aquellas variables que fueran más viables para transformación númerica y que se consideran de relevancia para la selección de proximas visualizaciones. 

+ **Preparación de datos**

Dado que el modelo debe contar con entradas númericas se debió realizar el pasaje de la *feature* *genres* a variables de este tipo, dado que es una variable categorías se empleo una metodología de *one-hot encoding*.

Por otro lado para las *features* *year_release* y *popularity* se realizo solo un escalado de datos, dada las diferencias en sus rangos de valores.

+ **Entrenamiento del modelo**

Empleando el modelo de knn otorgado por **sklearn** se entreno el modelo para todos los datos con el hiperparámetro de números de vecinos definido en k+1, siendo k el número de recomendaciones, esto es para evitar devolver la misma *query* durante la busqueda de los similares.
 
+ **Obtención de las recomendaciones**
Una vez indicado el titulo de consulta, se seleccionan las *features* empleadas en el modelo y se le aplican las mismas transformaciones de los datos.

Posteriormente mediante una función asociada al modelo se obtiene los k+1 más similares. Salteando el primer indice devuelto (el cual coincidría con la consulta) se procede la devolución de los titulos asociados.

Para información más detallada y uso del modelo de manera externa a la API vea: [ml_models.py](https://github.com/ramirou2/ML_MovieRecomenderSystem/blob/master/ml_models.py)


## **Link de Interés**</h2>


**Datos Crudos:**</h3>
[**Dataset**](https://drive.google.com/drive/folders/1nvSjC2JWUH48o3pb8xlKofi8SNHuNWeu)

**API Inicio:**</h3>
[**API --> api-moviesrecommender**](https://api-moviesrecommender.onrender.com)

<p align=center><img src=https://github.com/ramirou2/ML_MovieRecomenderSystem/tree/master/static/index.png><p>

**API Documentación:**</h3>
[**API --> documentación**](https://api-moviesrecommender.onrender.com/docs)

<p align=center><img src=https://github.com/ramirou2/ML_MovieRecomenderSystem/tree/master/static/docs.png><p>

**Video Demostrativo:**</h3>
[**Video demostración**]()