import pandas as pd
import tensorflow_hub as hub
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from kneed import KneeLocator
import joblib

def find_optimal_pca_dimensions(df, embeddings_column, dimensions, n_clusters=5):
    
    # Extract the embeddings from the DataFrame
    embeddings = np.stack(df[embeddings_column].values)
    
    # Apply Standard Scaler
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    
    results = []
    for dim in dimensions: 
        # Apply PCA
        pca = PCA(n_components=dim)
        reduced_embeddings = pca.fit_transform(scaled_embeddings)

        # Cluster the reduced embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(reduced_embeddings)

        # Compute the Silhouette score
        silhouette_avg = silhouette_score(reduced_embeddings, cluster_labels)

        # Calculate variance preserved
        variance_preserved = np.sum(pca.explained_variance_ratio_)

        # Append results
        results.append({
            'Dimensions': dim,
            'Silhouette Score': silhouette_avg,
            'Variance Preserved': variance_preserved
        })

    return pd.DataFrame(results)


def find_best_tradeoff(results_df):
    
    # Calculate the absolute difference between the normalized metrics
    results_df['Difference'] = np.abs(results_df['Silhouette Score'] - results_df['Variance Preserved'])
    
    # Find the dimension with the minimum difference
    best_dimension = results_df.loc[results_df['Difference'].idxmin(), 'Dimensions']
    
    return best_dimension, results_df


def main():
    df = pd.read_csv('./data/lyrics_proc_train.csv')
    use = hub.load('./models/use')
    
    lyrics_embeddings = df['lyrics_proc'].apply(lambda lyrics: use([lyrics]).numpy()[0])
    df['use_embeddings'] = lyrics_embeddings
    use_embeddings_numpy = np.stack(lyrics_embeddings)

    dimensions = [2,3,5,10,15,20,35,50]

    table = find_optimal_pca_dimensions(df,'use_embeddings',dimensions)
    
    best_dim, _ = find_best_tradeoff(table)
    print("The best dimension is: ", best_dim)

    # Extract the embeddings from the DataFrame
    embeddings = np.stack(df['use_embeddings'].values)

    # Apply Standard Scaler
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    # Apply PCA
    pca = PCA(n_components=best_dim)
    reduced_embeddings = pca.fit_transform(scaled_embeddings)

    # Add reduced embedding as new column
    df['reduced_embeddings'] = reduced_embeddings.tolist()

    # Extract the embeddings from the DataFrame
    reduced_embeddings = np.stack(df['reduced_embeddings'].values)

    cluster_range=(2, 50)
    # Initialize variables to store inertia values
    cluster_range_values = range(cluster_range[0], cluster_range[1] + 1)
    inertia_values = []

    # Compute inertia for each number of clusters
    for n_clusters in cluster_range_values:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(reduced_embeddings)
        inertia_values.append(kmeans.inertia_)
    
    # Use the KneeLocator to find the elbow point
    knee = KneeLocator(cluster_range_values, inertia_values, curve='convex', direction='decreasing')
    
    n_clusters = knee.elbow
    
    # Extract the embeddings from the DataFrame
    reduced_embeddings = np.stack(df['reduced_embeddings'].values)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_embeddings)
    
    # Add cluster labels to the lyrics embeddings
    df['cluster'] = cluster_labels
    
    # Save the KMeans model
    joblib.dump(kmeans, './models/use/kmeans.pkl')
    
    # Save the PCA model
    joblib.dump(pca, './models/use/pca.pkl')
    
    # Save the Standard scaler
    joblib.dump(scaler, './models/use/scaler.pkl')
    
    # Save Fast Text embeddings original
    np.save('./data/use_embeddings.npy', use_embeddings_numpy)
    
    # Save the unified embeddings
    saved_array = np.load('./data/use_embeddings.npy')
    
    print("USE embeddings created: ", saved_array.shape)


if __name__ == "__main__":
    main()