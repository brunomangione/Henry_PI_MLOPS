import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def load_recommendation_data(data_file):
    recommend = pd.read_parquet(data_file)
    return recommend

def initialize_recommendation_system(recommend):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(recommend['review'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

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
