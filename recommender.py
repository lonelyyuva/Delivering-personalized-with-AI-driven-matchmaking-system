import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle

# Load movie data
movies = pd.read_csv("data/movies.csv")

# Dummy user-item matrix (normally from ratings)
user_item_matrix = pd.pivot_table(movies, index='userId', columns='title', values='rating').fillna(0)

# Model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(user_item_matrix.values)

def get_recommendations(user_id):
    if user_id not in user_item_matrix.index:
        return ["No data for this user."]
    
    user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = model.kneighbors(user_vector, n_neighbors=6)
    similar_users = user_item_matrix.index[indices.flatten()[1:]]
    rec_movies = set()
    
    for u in similar_users:
        rec_movies.update(movies[movies['userId'] == u]['title'].tolist())
        
    return list(rec_movies)[:10]
