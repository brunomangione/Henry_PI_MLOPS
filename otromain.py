
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
        sim_scores = sim_scores[1:6]  # Recomendar los 5 ítems más similares (excluyendo el propio ítem)
        game_indices = [i[0] for i in sim_scores]
        return recommend['item_name'].iloc[game_indices]
    else:
        return [] 

# Creamos la funcion de recomendacion para fast api
@app.get("/recommendations/{item_name}")
def get_game_recommendations(item_name: str):
    recommendations = get_recommendations(item_name, cosine_sim, recommend)
    return {"item_name": item_name, "recommendations": recommendations}
