

# Proyecto Individual - Soy Henry
## Machine Learning Operations 

Se nos encomendo realizar un trabajo de Data Enineer y Data Science, donde se realizo un proyecto desde cero y se llegó a lograr un sistema de recomendación de Machine Learning.

## Descripción del problema (Contexto y rol a desarrollar)
### Contexto
Tienes tu modelo de recomendación dando unas buenas métricas 😏, y ahora, cómo lo llevas al mundo real? 👀

El ciclo de vida de un proyecto de Machine Learning debe contemplar desde el tratamiento y recolección de los datos (Data Engineer stuff) hasta el entrenamiento y mantenimiento del modelo de ML según llegan nuevos datos.

### Rol a desarrollar
Empezaste a trabajar como Data Scientist en Steam, una plataforma multinacional de videojuegos. El mundo es bello y vas a crear tu primer modelo de ML que soluciona un problema de negocio: Steam pide que te encargues de crear un sistema de recomendación de videojuegos para usuarios. 😟

Vas a sus datos y te das cuenta que la madurez de los mismos es poca (ok, es nula 😭 ): Datos anidados, de tipo raw, no hay procesos automatizados para la actualización de nuevos productos, entre otras cosas… haciendo tu trabajo imposible 😩 .

Debes empezar desde 0, haciendo un trabajo rápido de Data Engineer y tener un MVP (Minimum Viable Product) para el cierre del proyecto! Tu cabeza va a explotar 🤯, pero al menos sabes cual es, conceptualmente, el camino que debes de seguir ❗. Así que espantas los miedos y pones manos a la obra 💪

## Proceso de EDA (Análisis Exploratorio de Datos)
En esta sección se describe la exploración inicial de los datos, incluyendo la importación de librerías y la carga de los datos, así como las operaciones de limpieza y preparación de los mismos.

### Carga, Limpieza y Preparacion de Datos
Comenzamos importando las librerías necesarias y cargando los archivos en formato json.gz.
Las librerias utilizadas fueron pandas, json y gzip. 
Lo que se realizó en esta etapa fue cargar los datos en formato json.gz y transformarlos a un dataframe para que podamos trabajar de mejor forma. 
Se cargaron los archivos de steam_games, user_reviews y user_items.
Luego de la limpieza y preparacion de los datos creamos los dataframe de los respectivos datos en formato csv.

## ETL (Extracción, Transformación y Carga) de Datos
Para trabajar en los ETL, decidimos realizar un archivo para cada ETL, de esta forma nos quedaron los ETL de Games, Items y Reviews. 
Para trabajar en los ETL importamos las librerias necesarias, en estos casos utilizamos pandas, ast, numpy y nltk. 
En el proceso de ETL del archivo reviews realizamos el analisis de sentimiento donde se creo la columna 'sentiment_analysis' con la siguiente escala: debe tomar el valor '0' si es malo, '1' si es neutral y '2' si es positivo.

Una vez que dejamos listos los dataframe y transformados sus con las necesidades que consideramos importantes para nosotros, decidimos exportar un nuevo csv_final para cada uno de ellos. Los csv que luego vamos a importar y trabajar en las funciones.

## FUNCIONES

Nos encomendaron crear las siguientes funciones, que se deberan consumir en la API:

def PlayTimeGenre( genero : str ): Debe devolver año con mas horas jugadas para dicho género.
Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}

def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.
Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf, "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}

def UsersRecommend( año : int ): Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

def UsersNotRecommend( año : int ): Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

def sentiment_analysis( año : int ): Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.
Ejemplo de retorno: {Negative = 182, Neutral = 120, Positive = 278}


Para crear dichas funciones, creamos un libro de python para las 2 primeras, otro libro de python para la funcion 3 y 4 y un ultimo libro de python para crear la funcion 5. 
Se realizo de esta manera, ya que los archivos y dataframe para la funcion 1 y 2 era el mismo, para la 3 y 4 lo mismo y la quinta funcion usaba otros datos datos de daframe.

Para las funciones se utilizo la libreria de pandas, se leyeron los dataframe finales que habiamos trabajado y como necesitamos 2 dataframe en cada una de las funciones, lo que hicimos fue hacer un merge para crear un nuevo dataframe. Una vez realizado el merge, trabajamos y limpiamos ese dataframe para dejarlo listo para la funcion. 
Realizamos las funciones, las probamos con los datos y luego una vez que funcionaban las funciones, decidimos exportar los dataframe a formato parquet para que nos ocupen menos espacio en memoria, ya que van a ser los archivos consumidos por la API. 


## FASTAPI

Para la creacion de nuestra API, utilizamos FASTAPI. 
Creamos nuestro archivo main.py, instalamos fastapi y cargamos las librerias a utilizar, en nuestro caso pandas y fastapi.
Una vez credo el archivo, inicializamos FastApi de la siguiente forma: app = FastAPI()
Las funciones para la API deben tener el siguiente decorado: (@app.get(‘/’)).

Lo que hicimos en main.app fue traer las funciones finales que habiamos realizado en los libros de python y aplicar la funcion @app.get('/') para que las consuma nuestra API, aca dejamos como ejemplo la primer funcion.
Recordamos que para realizar las funciones debemos leer el archivo parquet que habiamos exportado cuando realizamos las funciones en el libro de python.


# PRIMER FUNCION 

#Traemos el archivo parquet que creamos   
data = pd.read_parquet('genres.parquet')   

# Traemos la funcion creada 
def playTimeGenre(genres):
    
    filtro = data[data['genres'] == genres] 
    if filtro.empty:
        return f"El genero {genres} no se encuentra en los registros" 
    else:
        anio_max_horas = filtro.loc[filtro['hours_game'].idxmax(), 'year']
        
    return f"El Año de lanzamiento con mas horas jugadas para el genero {genres}: {anio_max_horas}"

# Creamos la funcion para Fast Api    
@app.get("/anio_mas_horas/{genre}")
async def anio_con_mas_horas_para_genero(genero: str):
    resultado = playTimeGenre(genero)
    return resultado


Este proceso lo repetimos con todas las funciones solicitadas.
Una vez realizadas todas las funciones, se probaron en FASTAPI su funcionamiento. Al estar todas funcionando continuamos con el modelo de aprendizaje solicitado.

## Modelo de aprendizaje automático:

Una vez que toda la data es consumible por la API, está lista para consumir por los departamentos de Analytics y Machine Learning, y nuestro EDA nos permite entender bien los datos a los que tenemos acceso, es hora de entrenar nuestro modelo de machine learning para armar un sistema de recomendación. 
Elegimos la alternativa que el modelo deberá tener una relación ítem-ítem, esto es se toma un item, en base a que tan similar esa ese ítem al resto, se recomiendan similares. Aquí el input es un juego y el output es una lista de juegos recomendados, para ello recomendamos aplicar la similitud del coseno. 

def recomendacion_juego( id de producto ): Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.

Para la realizacion del modelo utilizamos las librerias de pandas y sklearn, donde nos trajimos TfidfVectorizer, cosine_similarity y TruncatedSVD.

Comenzamos cargando los dataframe de reviews y de items, donde hicimos un merge, eliminamos los archivos nulos y nos quedamos solamente con las culumnas de item_id, item_name y reviews. Ya con el dataframe listo, el sistema de recomendacion se construye utilizando los siguientes pasos:

1. **Inicializar el vectorizador TF-IDF**: Inicializamos un vectorizador TF-IDF que se utilizará para convertir las reseñas de juegos en representaciones vectoriales.

2. **Aplicar el vectorizador a la columna 'review'**: Aplicamos el vectorizador TF-IDF a la columna 'review' de nuestro conjunto de datos, lo que nos proporciona una matriz TF-IDF que representa la importancia de las palabras en cada reseña.

3. **Inicializar TruncatedSVD con el número deseado de componentes**: Utilizamos TruncatedSVD (Singular Value Decomposition) para reducir la dimensionalidad de la matriz TF-IDF. Esto es útil para reducir la memoria necesaria y acelerar los cálculos.

4. **Aplicar TruncatedSVD a la matriz TF-IDF**: Aplicamos TruncatedSVD a la matriz TF-IDF, lo que nos proporciona una representación de baja dimensionalidad de las reseñas de los juegos.

5. **Crear un diccionario que mapea los IDs de los juegos a sus nombres**: Creamos un diccionario que mapea los IDs de los juegos a sus nombres para poder mostrar los nombres de los juegos en las recomendaciones.

6. **Función de recomendación**: La función `recomendacion_juego(id_producto)` toma un ID de producto y calcula la similitud de coseno entre ese juego y todos los demás juegos en función de la matriz TF-IDF reducida. Luego, se devuelven los nombres de los juegos más similares como recomendaciones.

Una vez creada la funcion y probada en el libro de python, lo que hacemos es exportar el dataframe que usamos en la funcion a un formato tipo parquet para luego utilizarlo en la API. 


## Requisitos
Los requisitos y librerias necesarias que deben estar instaladas:

- Pandas
- Scikit-learn
- FastApi
- FastParquet
- Numpy
- Nltk
- Uvicorn


## Autor
# Bruno Mangione
Contacto: 
    Mail: brunomangione@gmail.com
    Github: brunomangione
    













