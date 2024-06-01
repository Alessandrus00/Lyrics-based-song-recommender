import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download('punkt',quiet=True)
nltk.download('stopwords',quiet=True)
nltk.download('wordnet',quiet=True)

stop_words = stopwords.words('english')
new_stop_words = ['ooh','yeah','hey','whoa','woah', 'ohh', 'was', 'mmm', 'oooh','yah','yeh','mmm', 'hmm','deh','doh','jah']
stop_words.extend(new_stop_words)

def remove_contractions(song):
    song = re.sub(r"he's", "he is", song)
    song = re.sub(r"there's", "there is", song)
    song = re.sub(r"we're", "we are", song)
    song = re.sub(r"that's", "that is", song)
    song = re.sub(r"won't", "will not", song)
    song = re.sub(r"they're", "they are", song)
    song = re.sub(r"can't", "cannot", song)
    song = re.sub(r"wasn't", "was not", song)
    song = re.sub(r"aren't", "are not", song)
    song = re.sub(r"isn't", "is not", song)
    song = re.sub(r"what's", "what is", song)
    song = re.sub(r"haven't", "have not", song)
    song = re.sub(r"hasn't", "has not", song)
    song = re.sub(r"there's", "there is", song)
    song = re.sub(r"he's", "he is", song)
    song = re.sub(r"it's", "it is", song)
    song = re.sub(r"you're", "you are", song)
    song = re.sub(r"i'm", "i am", song)
    song = re.sub(r"shouldn't", "should not", song)
    song = re.sub(r"wouldn't", "would not", song)
    song = re.sub(r"i'm", "i am", song)
    song = re.sub(r"isn't", "is not", song)
    song = re.sub(r"here's", "here is", song)
    song = re.sub(r"you've", "you have", song)
    song = re.sub(r"we're", "we are", song)
    song = re.sub(r"what's", "what is", song)
    song = re.sub(r"couldn't", "could not", song)
    song = re.sub(r"we've", "we have", song)
    song = re.sub(r"who's", "who is", song)
    song = re.sub(r"y'all", "you all", song)
    song = re.sub(r"would've", "would have", song)
    song = re.sub(r"it'll", "it will", song)
    song = re.sub(r"we'll", "we will", song)
    song = re.sub(r"we've", "we have", song)
    song = re.sub(r"he'll", "he will", song)
    song = re.sub(r"y'all", "you all", song)
    song = re.sub(r"weren't", "were not", song)
    song = re.sub(r"didn't", "did not", song)
    song = re.sub(r"they'll", "they will", song)
    song = re.sub(r"they'd", "they would", song)
    song = re.sub(r"don't", "do n't", song)
    song = re.sub(r"they've", "they have", song)
    song = re.sub(r"i'd", "i would", song)
    song = re.sub(r"You\x89Ûªre", "You are", song)
    song = re.sub(r"where's", "where is", song)
    song = re.sub(r"we'd", "we would", song)
    song = re.sub(r"i'll", "i will", song)
    song = re.sub(r"weren't", "were not", song)
    song = re.sub(r"they're", "they are", song)
    song = re.sub(r"let's", "let us", song)
    song = re.sub(r"it's", "it is", song)
    song = re.sub(r"can't", "cannot", song)
    song = re.sub(r"don't", "do not", song)
    song = re.sub(r"you're", "you are", song)
    song = re.sub(r"i've", "I have", song)
    song = re.sub(r"that's", "that is", song)
    song = re.sub(r"i'll", "i will", song)
    song = re.sub(r"doesn't", "does not", song)
    song = re.sub(r"i'd", "i would", song)
    song = re.sub(r"didn't", "did not", song)
    song = re.sub(r"ain't", "am not", song)
    song = re.sub(r"you'll", "you will", song)
    song = re.sub(r"i've", "i have", song)
    song = re.sub(r"don't", "do not", song)
    song = re.sub(r"i'll", "i will", song)
    song = re.sub(r"i'd", "i would", song)
    song = re.sub(r"let's", "let us", song)
    song = re.sub(r"you'd", "you would", song)
    song = re.sub(r"it's", "it is", song)
    song = re.sub(r"ain't", "am not", song)
    song = re.sub(r"haven't", "have not", song)
    song = re.sub(r"could've", "could have", song)
    song = re.sub(r"youve", "you have", song)
    song = re.sub(r"ev'ry" , 'every' , song)
    song = re.sub(r"coz" , 'because' , song)
    song = re.sub(r"'cause" , 'because' , song)
    song = re.sub(r"n\'t" , 'not' , song)
    song = re.sub(r"that'll", "that will" ,song)
    
    return song

def preprocess_text(text):
    # convert all words in lower case
    text = text.lower()
    # remove contractions
    text = remove_contractions(text)
    # remove \n and words containing '
    text = text.replace('\n', ' ')
    text = re.sub(r'\b\w*\'\w*\b', '', text)
    # remove punctuation
    text = re.sub(r'[,\.\!?;]', '', text)
    #removing text in square braquet
    text = re.sub(r'\[.*?\]', ' ', text)
    #removing numbers
    text = re.sub(r'\w*\d\w*',' ', text)
    #removing bracket
    text = re.sub(r'[()]', ' ', text)
    #removing underscore at the beginning or at the end of the word
    text = re.sub(r'^_|_$', '', text)
    #removing multiple spaces (merge > 2 spaces in one space)
    text = re.sub(r' {2,}', ' ', text)
    
    # tokenize
    tokens = word_tokenize(text)
    # remove stop words
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    #remove tokens with lenght < 4
    tokens = [token for token in lemmatized_tokens if len(token) > 3]
    # Stem tokens
    stemmer = PorterStemmer()
    final_tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(final_tokens)