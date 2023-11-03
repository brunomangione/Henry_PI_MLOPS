from fastapi import FastAPI
import pandas as pd

app = FastAPI()


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
@app.get("/play_time_genre/{genre}")
async def most_play_time_genre(genero: str):
    resultado = playTimeGenre(genero)
    return resultado
    


# SEGUNDA FUNCION 
    
# Traemos la funcion creada
def userForGenre (genres):
    
    filtro = data[data['genres'] == genres]
    if filtro.empty:
        return f"el genero {genres} no se encuentra en los registros"
    else:
        usuario_max_horas = filtro.loc[filtro['hours_game'].idxmax()]['user_id']
        acumulacion_horas_anio = filtro.groupby('year')['hours_game'].sum().reset_index()
    
    result = {
        "Usuario con más horas jugadas para Género " + genres: usuario_max_horas,
        "Horas jugadas": [{"Año":row['year'], "Horas": row['hours_game']} for index, row in acumulacion_horas_anio.iterrows()]
    }
    
    return result

# Creamos la funcion para FastApi
@app.get("/User_for_genre/{genre}")
async def user_for_genre(genero: str):
    resultado = userForGenre(genero)
    return resultado
    
#TERCER FUNCION

# Traemos el archivo parquet para la funcion
reviews = pd.read_parquet('userrecomend.parquet')

# Traemos la funcion creada
def top_recommended_games(year):
    
    filtro_df = reviews[(reviews['year'] == year) & (reviews['recommend'] == True)]  
    if filtro_df.empty:
        return f"El año {year} no se encuentra en los registros."
    else:
        game_recommendations = filtro_df.groupby('item_name')['user_id'].count().reset_index()
        top_games = game_recommendations.sort_values(by='user_id', ascending=False)
        top_3_games = top_games.head(3)
    
    result = {"Puesto 1": top_3_games.iloc[0]['item_name']}, {"Puesto 2": top_3_games.iloc[1]['item_name']}, {"Puesto 3": top_3_games.iloc[2]['item_name']}
    
    return result

# Creamos la funcion para FastApi
@app.get("/top_games_recommended/{year}")
async def top_recommended(year: int):
    resultado = top_recommended_games(year)
    return resultado



# CUARTA FUNCION

# Traemos la funcion creada
def top_least_recommended_games(year):
    
    filtro_df = reviews[(reviews['year'] == year) & (reviews['recommend'] == False)]  
    if filtro_df.empty:
        return f"El año {year} no se encuentra en los registros."
    else:
        game_recommendations = filtro_df.groupby('item_name')['user_id'].count().reset_index()
    
        top_games = game_recommendations.sort_values(by='user_id', ascending=True)  
    
        top_3_games = top_games.head(3)
    
    result = [{"Puesto 1": top_3_games.iloc[0]['item_name']}, {"Puesto 2": top_3_games.iloc[1]['item_name']}, {"Puesto 3": top_3_games.iloc[2]['item_name']}]
    
    return result

# Creamos la funcion para FastApi
@app.get("/least_games_recommended/{year}")
async def least_recommended(year: int):
    resultado = top_least_recommended_games(year)
    return resultado


# QUINTA FUNCION

# Traemos el df de sentimiento
df_sentiment = pd.read_parquet('sentiment.parquet')

# Traemos la funcion para realizar el analisis de sentimiento
def sentiment_analysis(year):
    filtro_sentiment = df_sentiment[df_sentiment['year_x'] == year]
    
    if filtro_sentiment.empty:
        return f"El año {year} no se encuentra en los registros"
    else:
        negativo = df_sentiment.loc[df_sentiment.year_x == year, "total_negativos"].item()
        neutral = df_sentiment.loc[df_sentiment.year_x == year, "total_neutrales"].item()
        positivo = df_sentiment.loc[df_sentiment.year_x == year, "total_positivos"].item()
  
        return {"Negative" : negativo, "Neutral" : neutral, "Positive" : positivo}

# Creamos la funcion para fast api
@app.get("/sentiment_analysis/{year}")
async def get_sentiment_analysis(year: int):
    resultado = sentiment_analysis(year)    
    return resultado

# SISTEMA RECOMENDACION 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
recommend = pd.read_parquet('recommend.parquet')
# Inicializar el vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()
# Aplicar el vectorizador a la columna 'review'
tfidf_matrix = tfidf_vectorizer.fit_transform(recommend['review'])
# Inicializar TruncatedSVD con el número deseado de componentes
n_components = 100  # Ajusta este valor según tus necesidades
svd = TruncatedSVD(n_components=n_components)
# Aplicar TruncatedSVD a la matriz TF-IDF
tfidf_matrix_svd = svd.fit_transform(tfidf_matrix)
# Crear un diccionario que mapea los IDs de los juegos a sus nombres
id_to_name = recommend.set_index('item_id')['item_name'].to_dict()

def recomendacion_juego(id_producto):
    idx = recommend[recommend['item_id'] == id_producto].index[0]

    # Calcular la similitud de coseno entre los juegos basándose en la matriz TF-IDF reducida
    sim_scores = cosine_similarity([tfidf_matrix_svd[idx]], tfidf_matrix_svd)
    
    # Obtener los índices de los juegos más similares
    sim_scores = sim_scores[0]  # Desempaquetar la matriz
    similar_games_indices = sim_scores.argsort()[::-1][1:6]  # Excluyendo el propio juego

    # Recuperar los nombres de los juegos recomendados utilizando el mapeo
    recommended_games = [id_to_name[recommend['item_id'].iloc[i]] for i in similar_games_indices]

    return recommended_games

@app.get("/recommend/{item_id}")
def get_recommendations(item_id: float):
    recommendations = recomendacion_juego(item_id)
    game_name = id_to_name(item_id, "Juego no encontrado")
    return {"item_id": item_id, "game_name": game_name, "recommendations": recommendations}