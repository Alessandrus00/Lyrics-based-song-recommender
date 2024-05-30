import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast

"""
def retrieve_query(n_query=5):
    df = pd.read_csv('./data/song_lyrics_sampled_proc.csv')
    # Randomly sample 5 rows from the DataFrame
    random_sample = df.sample(n=n_query)
    return random_sample
"""

def best_worst_recommendations(query_id,method):
    df = pd.read_csv(f'./data/{method}/song_lyrics_{method}.csv')
    
    if method in ['fasttext', 'use']:
        df['reduced_embeddings'] = df['reduced_embeddings'].apply(ast.literal_eval)
        # Find the index of the row with the specific id
        query_idx = df.index[df['id'] == query_id].tolist()[0]
        # Extract the target embedding and its cluster
        target_embedding = df.at[query_idx, 'reduced_embeddings']
        target_cluster = df.at[query_idx, 'cluster']
        
        # Filter the DataFrame to get only the embeddings in the same cluster
        cluster_df = df[df['cluster'] == target_cluster]
        cluster_embeddings = np.stack(cluster_df['reduced_embeddings'].values)

        # Calculate cosine similarities
        similarities = cosine_similarity([target_embedding], cluster_embeddings)[0]
    else:
        #df['feature_vector'] = df['feature_vector'].apply(ast.literal_eval)
        # Load the NumPy array from the .npy file
        feature_vector = np.load(f'./data/{method}/{method}.npy', allow_pickle=True)
        # Add the loaded array as a new column to the existing DataFrame
        df['feature_vector'] = feature_vector
        # Find the index of the row with the specific id
        query_idx = df.index[df['id'] == query_id].tolist()[0]
        # Extract the target embedding and its cluster
        target_embedding = df.at[query_idx, 'feature_vector']
        
        # Get all the embeddings    
        all_embeddings = np.stack(df['feature_vector'].values)

        # Calculate cosine similarities
        similarities = cosine_similarity([target_embedding], all_embeddings)[0]

    best_idx = np.argsort(similarities)[-2]
    worst_idx = np.argsort(similarities)[0]
    
    indices = np.array([best_idx,worst_idx])
    ids = df.iloc[indices]['id']
    lyrics = df.iloc[indices]['lyrics']
    similarities = similarities[indices]

    return pd.DataFrame({
        'id': ids, 
        'similarity': similarities,
        'lyrics': lyrics
    })

    
QUERY_ID = [1037597,365484,167630,1072893,784050] # IDs of 5 randomly-chosen queries

for method in ['simple_features', 'fasttext', 'use']:
    for query_id in QUERY_ID:
        print("Method:", method, "Query ID:", query_id)
        res = best_worst_recommendations(query_id, method)
        res.to_csv(f"{method}_{query_id}.csv", header='true', index=False)
        print(res)
        print()