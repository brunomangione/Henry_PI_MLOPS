import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

def load_recommendation_data(data_file):
    recommend = pd.read_parquet(data_file)
    return recommend

def initialize_recommendation_system(recommend):
    corpus = recommend['review'].values
    unique_items = recommend['item_name'].unique()

    # Crear un diccionario para mapear índices a nombres de juegos
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}

    # Crear una matriz TF-IDF manualmente
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Reducción de dimensionalidad con PCA
    n_components = 100  # Ajusta este valor según tus necesidades
    pca = PCA(n_components=n_components)
    tfidf_matrix = pca.fit_transform(tfidf_matrix)
    tfidf_matrix = normalize(tfidf_matrix)  # Normalizar la matriz

    # Calcular la similitud de coseno
    cosine_sim = np.dot(tfidf_matrix, tfidf_matrix.T)
    
    return item_to_idx, cosine_sim
