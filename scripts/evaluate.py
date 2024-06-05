import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import joblib
import tensorflow_hub as hub
import fasttext
import fasttext.util
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim.models import LdaModel
import nltk
import proc
import os


nltk.download('punkt',quiet=True)
nltk.download('stopwords',quiet=True)
nltk.download('wordnet',quiet=True)

"""
def retrieve_query(n_query=5):
    df = pd.read_csv('./data/song_lyrics_sampled_proc.csv')
    # Randomly sample 5 rows from the DataFrame
    random_sample = df.sample(n=n_query)
    return random_sample
"""

def get_full_lyrics(df, lyrics_snippet):
    # Use the pandas Series str.contains() method to filter rows that contain the lyrics snippet
    matching_rows = df[df['lyrics'].str.contains(lyrics_snippet, case=False, na=False)]
    if len(matching_rows) > 0:
        return matching_rows['lyrics'].values[0]
    
    return None


# retrieve song index in df from lyrics snippet
def find_lyrics_index(df, lyrics_snippet):
    # Use the pandas Series str.contains() method to filter rows that contain the lyrics snippet
    matching_rows = df[df['lyrics'].str.contains(lyrics_snippet, case=False, na=False)]
    # Retrieve the indices of these rows
    matching_indices = matching_rows.index.tolist()
    return matching_indices


def apply_pca(embeddings,method):
    # Load the PCA model from the file
    joblib_file = f"./models/{method}/pca.pkl"
    pca = joblib.load(joblib_file)
    print(f"PCA model loaded from {joblib_file}")
    
    # Load the Standard Scaler from the file
    joblib_file = f"./models/{method}/scaler.pkl"
    scaler = joblib.load(joblib_file)
    print(f"Standard Scaler loaded from {joblib_file}")
    
    # Apply Standard Scaler and PCA
    scaled_embeddings = scaler.transform(embeddings)
    return pca.transform(scaled_embeddings)


def get_cluster_labels(embeddings,method):
    # Load the KMeans model from the file
    joblib_file = f"./models/{method}/kmeans.pkl"
    kmeans = joblib.load(joblib_file)
    print(f"KMeans model loaded from {joblib_file}")
    return kmeans.predict(embeddings.astype(float))


def get_use_embedding(lyrics):
    # Load the Universal Sentence Encoder module (transformer-based variant)
    model_path = "./models/use"
    use = hub.load(model_path)
    use_embedding = use([lyrics]).numpy()
    return use_embedding


def get_fasttext_embedding(lyrics):
    # Load the downloaded model
    ft = fasttext.load_model('./models/fasttext/cc.en.300.bin')
    ft_embedding = ft.get_sentence_vector(lyrics)
    return ft_embedding


def get_tfidf_embedding(lyrics):
    # Analyze the sentiment of the lyrics
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(lyrics)
    sentiment_values = list(sentiment_scores.values())

    # Load TF-IDF vectorizer
    tfidf_vectorizer_loaded = joblib.load('./models/others/tfidf_vectorizer.pkl')
    
    # Load LDA model
    lda_model_loaded = LdaModel.load('./models/others/lda_model.gensim')
    
    # Transform the document using the loaded TF-IDF vectorizer
    tfidf_vector = tfidf_vectorizer_loaded.transform([lyrics])
    
    # Extract LDA topic distribution for the document
    new_doc_bow = lda_model_loaded.id2word.doc2bow(lyrics.split())
    new_doc_lda = lda_model_loaded[new_doc_bow]
    lda_vector = np.zeros(lda_model_loaded.num_topics)
    for topic, prob in new_doc_lda:
        lda_vector[topic] = prob
    
    # Combine TF-IDF, LDA, and sentiment scores representations
    tfidf_embedding = np.hstack((sentiment_values, lda_vector, tfidf_vector.toarray()[0]))
    return tfidf_embedding


def get_embedding(lyrics, method):
    lyrics_proc = proc.preprocess_text(lyrics)
    if method == 'tfidf':
        return get_tfidf_embedding(lyrics_proc)
    if method == 'fasttext':
        return get_fasttext_embedding(lyrics_proc)
    
    return get_use_embedding(lyrics_proc)


def get_train_embeddings(method):
    if method == 'tfidf':
        return np.load('./data/tfidf_embeddings.npy')
    elif method == 'fasttext':
        return np.load('./data/fasttext_embeddings.npy')
    
    return np.load('./data/use_embeddings.npy')



def get_args_parser():
    parser = argparse.ArgumentParser('Set evaluation parameters', add_help=False)
    parser.add_argument('--dataset_train', default='./data/lyrics_proc_train.csv', type=str, help="Training dataset path")
    parser.add_argument('--dataset_test', default='./data/lyrics_proc_test.csv', type=str, help="Test dataset path")
    parser.add_argument('--eval', default='topn', type=str, choices=('test', 'topn'), help="Type of evaluation you want to execute")
    parser.add_argument('--n', default=10, type=int, help="Number of best song recommandations to retrieve")
    parser.add_argument('--method', default='tfidf', type=str, choices=('tfidf', 'fasttext', 'use'), help="Method used to compute the lyrics embedding and perform the recommendations")
    parser.add_argument('--lyrics', default='She hit me like a blinding light and I was born', type=str, help="Lyrics for which you want to retrieve similar songs. You could just provide a small chunk of it if it is in the original dataset. The full lyrics otherwise")
    parser.add_argument('--pca', action='store_true', help="Whether to reduce the size of the embeddings using PCA (just for Fast Text and USE)")
    parser.add_argument('--clustering', action='store_true', help="Whether to retrieve similar lyrics from the same cluster of the query to speed up the recommendation response (just for Fast Text and USE)")

    return parser

# used for qualitative evaluation
def best_worst_recommendations(args):
    df_test = pd.read_csv(args.dataset_test)
    df_train = pd.read_csv(args.dataset_train)
    # iterate through all the three methods and get the best and worst recomm. for 5 randomly picked lyrics
    print("Description: first row is the query itself, second is the best recommendation and third the worst\n")
    for method in ['tfidf', 'fasttext', 'use']:
        for index, row in df_test.iterrows():
            print("Method:", method, "-" ,"Song:", row['title'], '-' ,row['artist'])
            lyrics = row['lyrics']
            embedding = get_embedding(lyrics, method).reshape(1,-1)
            train_embeddings = get_train_embeddings(method)
            
            # apply PCA if specified in args
            if args.pca:
                embedding = apply_pca(embedding,args.method)
                train_embeddings = apply_pca(train_embeddings,args.method)
                
            # get cluster labels if a clustering is specified in args
            if args.clustering:
                cluster_label = get_cluster_labels(embedding,args.method)
                train_cluster_labels= get_cluster_labels(train_embeddings,args.method)
            
            # Calculate cosine similarities
            similarity_scores = cosine_similarity(train_embeddings, embedding).flatten()
            if args.clustering:
                # Identify positions where train_cluster_labels are different from cluster_label
                positions = train_cluster_labels != cluster_label
                # Set corresponding positions in similarity_scores to 0
                similarity_scores[positions] = -1
            
            similarity_scores_sorted = np.argsort(similarity_scores) # ascending order (the best at the end)
            
            # get best and worst lyrics ids
            best_idx = similarity_scores_sorted[-1]
            #bad_idx = similarity_scores_sorted[-20]
            
            bad_idx = similarity_scores_sorted[0]
            for i in range(len(similarity_scores_sorted)):
                curr_idx = similarity_scores_sorted[-i-1]
                if similarity_scores[curr_idx] < similarity_scores[best_idx] - 0.4:
                    bad_idx = curr_idx
                    break
            
            # print out the dataset
            df_out = pd.DataFrame({
                'id': [row['id'], df_train.loc[best_idx, 'id'], df_train.loc[bad_idx, 'id']],
                'similarity': [1.0, similarity_scores[best_idx], similarity_scores[bad_idx]],
                'lyrics': [lyrics, df_train.loc[best_idx, 'lyrics'], df_train.loc[bad_idx, 'lyrics']]
            })
            
            #print(df_out)
            
            # Save the results
            directory = f'./test/{method}'
            # Check if the directory exists
            if not os.path.exists(directory):
                # If it does not exist, create the directory
                os.makedirs(directory)
            print(f"Saving {os.path.join(directory,f"query_{index+1}.csv")}...")
            df_out.to_csv(os.path.join(directory,f"query_{index+1}.csv"), header='true', index=False)
    
            
# retrieve the top-n recommendations given a query lyrics
def top_n_recommendations(args):
    df_train = pd.read_csv(args.dataset_train)
    df_test = pd.read_csv(args.dataset_test)
    
    df_conc = pd.concat([df_train, df_test], axis=0)
    
    full_lyrics =  get_full_lyrics(df_conc, args.lyrics)
    
    if full_lyrics is not None:
        lyrics = full_lyrics
    else:
        lyrics = args.lyrics
    
    # get target and training embeddings according to the method in args
    embedding = get_embedding(lyrics, args.method).reshape(1,-1)
    train_embeddings = get_train_embeddings(args.method)
    
    # apply PCA if specified in args
    if args.pca:
        embedding = apply_pca(embedding,args.method)
        train_embeddings = apply_pca(train_embeddings,args.method)
        
    # get cluster labels if a clustering is specified in args
    if args.clustering:
        cluster_label = get_cluster_labels(embedding,args.method)
        train_cluster_labels= get_cluster_labels(train_embeddings,args.method)
    
    # Calculate cosine similarities
    similarity_scores = cosine_similarity(train_embeddings, embedding).flatten()
    if args.clustering:
        # Identify positions where train_cluster_labels are different from cluster_label
        positions = train_cluster_labels != cluster_label
        # Set corresponding positions in similarity_scores to 0
        similarity_scores[positions] = -1
    # sort the similarity scores indices
    similarity_scores_sorted = np.argsort(similarity_scores) # ascending order (the best at the end)
    
    # Get the top n similar embeddings
    n = args.n
    top_n_indices = similarity_scores_sorted[-n:][::-1]
    
    # print out the dataset
    df_out = pd.DataFrame({
        'id': df_train.loc[top_n_indices, 'id'],
        'similarity': similarity_scores[top_n_indices]
    })
    
    print(f"Description: Top-{n} best recommendations (only song id showed)\n")
    print(df_out)
    

def main(args):
    
    if args.eval == 'test':
        return best_worst_recommendations(args)
    
    return top_n_recommendations(args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose how to perform the evaluation.", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
