
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
@app.get("/senent_analysis")
async def get_sentint_analysis(year: int):
    resultado = sentiment_analysis(year)    
    return resultado



# SISTEMA RECOMENDACION 
# Importamos las librerias para utilizar
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Traemos el df con el que vamos a trabajar 
recommend = pd.read_parquet('recomendacion.parquet')

# Inicializar el vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()

# Aplicar el vectorizador a la columna 'Review'
tfidf_matrix = tfidf_vectorizer.fit_transform(recommend['review'])

# Calcular la similitud del coseno entre los juegos
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Traemos la funcion de recomendacion
def get_recommendations(item_name, cosine_sim, recommend):
    idx_list = recommend.index[recommend['item_name'] == item_name].tolist()
    
    if idx_list:
        idx = idx_list[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Recomendar los 10 ítems más similares (excluyendo el propio ítem)
        game_indices = [i[0] for i in sim_scores]
        return recommend['item_name'].iloc[game_indices]
    else:
        return [] 

# Creamos la funcion de recomendacion para fast api
@app.get("/recommendations/{item_name}")
def get_game_recommendations(item_name: str):
    recommendations = get_recommendations(item_name, cosine_sim, recommend)
    return {"item_name": item_name, "recommendations": recommendations}
