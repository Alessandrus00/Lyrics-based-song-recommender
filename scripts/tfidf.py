import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
from gensim.models import LdaModel
import numpy as np
import joblib
import os

nltk.download('vader_lexicon',quiet=True)

def main():
    # Load the DataFrame
    df = pd.read_csv('./data/lyrics_proc_train.csv')
    # 1. Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_df = df['lyrics_proc'].apply(lambda x: pd.Series(sia.polarity_scores(x)))

    # 2. Topic modeling (LDA)
    # Prepare the lyrics corpus
    lyrics_corpus = df['lyrics_proc'].apply(lambda x: x.split())
    # Create the dictionary
    dictionary = corpora.Dictionary(lyrics_corpus)
    # Create the corpus
    corpus = [dictionary.doc2bow(text) for text in lyrics_corpus]
    # Set parameters
    num_topics = 5
    passes = 15
    # Train the LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)
    # Get topic distributions
    lda_distributions = [lda_model.get_document_topics(bow) for bow in corpus]

    # Convert LDA topic distributions to a fixed-size vector
    def lda_to_vec(lda_dist, num_topics):
        vec = np.zeros(num_topics)
        for topic, prob in lda_dist:
            vec[topic] = prob
        return vec

    # 3. TF-IDF
    # Initialize TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['lyrics_proc'])

    # 4. Concatenate the three features
    # Initialize an empty numpy array to store the features
    num_features = sentiment_df.shape[1] + num_topics + tfidf_matrix.shape[1]
    features = np.empty((len(df), num_features))

    # Iterate over each row in the DataFrame
    for i in range(len(df)):
        # Extract sentiment scores
        sentiment_scores = sentiment_df.iloc[i].values
        # Extract LDA vector
        lda_vector = lda_to_vec(lda_distributions[i], num_topics)
        # Extract TF-IDF vector
        tfidf_vector = tfidf_matrix[i].toarray()[0]
        # Concatenate the vectors
        combined_vector = np.concatenate([sentiment_scores, lda_vector, tfidf_vector])
        # Assign the combined vector to the features array
        features[i] = combined_vector

    
    # Directory path you want to create
    directory = './models/others/'

    # Check if the directory exists
    if not os.path.exists(directory):
        # If it does not exist, create the directory
        os.makedirs(directory)
        
    # Save the TF-IDF vectorizer
    joblib.dump(tfidf_vectorizer, './models/others/tfidf_vectorizer.pkl')
    
    # Save the LDA model
    lda_model.save('./models/others/lda_model.gensim')
    
    # Save dictionary
    dictionary.save('./models/others/dictionary.gensim')
    
    # Save corpus
    corpora.MmCorpus.serialize('./models/others/corpus.mm', corpus)
    
    # Save the unified embeddings
    np.save('./data/tfidf_embeddings.npy', features)
    
    # Save the unified embeddings
    saved_array = np.load('./data/tfidf_embeddings.npy')
    
    print("TF-IDF embeddings created: ", saved_array.shape)


if __name__ == "__main__":
    main()
