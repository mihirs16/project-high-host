import string
import os
import re

import pandas as pd 

from heapq import nlargest
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = re.sub("\'", "", text) 
    text = re.sub("[^a-zA-Z]"," ",text) 
    text = ' '.join(text.split()) 
    text = text.lower() 

    ps = PorterStemmer()
    _t = ""
    for t in text.split():
        _t += ps.stem(t) + " "
    text = _t

    stop_words = set(stopwords.words('english'))
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    clean_text = ' '.join(no_stopword_text)

    # return text
    return clean_text


def corpora_train():
    # corpus load as model_data.csv
    corpora_data = pd.read_csv('model-data.csv')

    # tfidf vectorizer on corpus
    tfidf_vect = TfidfVectorizer(max_df=0.8, max_features=1000)
    tfidf_vect.fit_transform(corpora_data['Text'])

    return tfidf_vect

# return list of (N% of text)-most frequent words in a 
def freq_dist(text):

    wordlist = list(word_tokenize(text))
    freq_dict = dict(Counter(wordlist))

    list_keywords = nlargest(round(0.25*len(wordlist)), freq_dict, key = freq_dict.get)
    
    return list_keywords