
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

def load_recommendation_data(data_file):
    recommend = pd.read_parquet(data_file)
    return recommend

def initialize_recommendation_system(recommend):
    corpus = recommend['review'].values
    unique_items = recommend['item_name'].unique()

    # Crear un diccionario para mapear índices a nombres de juegos
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}

    # Crear una matriz TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Reducción de dimensionalidad con TruncatedSVD
    n_components = 100  # Ajusta este valor según tus necesidades
    svd = TruncatedSVD(n_components=n_components)
    tfidf_matrix = svd.fit_transform(tfidf_matrix)
    tfidf_matrix = normalize(tfidf_matrix)  # Normalizar la matriz

    # Calcular la similitud de coseno
    cosine_sim = np.dot(tfidf_matrix, tfidf_matrix.T)
    
    return item_to_idx, cosine_sim

def get_recommendations(item_name, item_to_idx, cosine_sim, recommend):
    idx = item_to_idx.get(item_name, -1)
    if idx != -1:
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:4]  # Recomendar los 5 ítems más similares (excluyendo el propio ítem)
        game_indices = [i[0] for i in sim_scores]
        return recommend['item_name'].iloc[game_indices].tolist()
    else:
        return []
        