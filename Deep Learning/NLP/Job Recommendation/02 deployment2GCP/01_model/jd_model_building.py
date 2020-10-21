import pickle
import numpy as np
import math
import re
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


data = pd.read_csv(
    "JD5000_NLP.csv",
    engine="python",
    encoding="latin1"
)
data.drop(columns=['Unnamed: 0'], axis = 1, inplace = True)

data = data.loc[data['search_title'] != 'machine learning engineer']
#print(len(data), len(data2))

#Label Encoder from characters to numeric
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize

le = LabelEncoder()
le.fit(data["search_title"])
data['search_title_le'] = le.transform(data["search_title"])
#data = data.loc[data['search_title_le'] != 3]


jd_dict = {0: le.classes_[0], 1: le.classes_[1], 2: le.classes_[2], }
#jd_dict

n_classes = len(jd_dict)
#n_classes

y_roc = label_binarize(data['search_title_le'] , classes=[0,1,2])
#y_roc

#le.inverse_transform(list(tst))

#cleaning
def clean_text(text):
    # Removing the @
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    # Removing the URL links
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    # Keeping only letters
    text = re.sub(r"[^a-zA-Z.!?']", ' ', text)
    # change capital to lowercase
    text = text.lower()
    # Removing additional whitespaces
    text = re.sub(r" +", ' ', text)
    return text


data_clean = [clean_text(text) for text in data.description]
#len(data_clean), data_clean

data_labels = data['search_title_le']

X = data["description"] # Note that it is a Series rather than a DataFrame here
y = data['search_title_le']


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
import re


import nltk
nltk.download('stopwords')

def stem_tokenizer(text):
    stemmer = EnglishStemmer(ignore_stopwords=True)
    # Removing the URL links
    words = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    words = re.sub(r"[^A-Za-z0-9\-]", " ", words).lower().split()
    words = [stemmer.stem(word) for word in words]
    return words

tfidf = TfidfVectorizer(stop_words=stopwords.words('english'),
                        tokenizer=stem_tokenizer,
                        lowercase=True,
                        max_df=0.8,
                        min_df=2,
                        ngram_range=(1, 3)
                       )


from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report


cv = CountVectorizer(stop_words=stopwords.words('english'),
                     tokenizer=stem_tokenizer,
                     lowercase=True,
                     max_df=0.4,
                     min_df=5,
                     ngram_range=(2, 3),
                     binary=False
                    )

nbclassifier = Pipeline([('cv', cv),  

                         ('nb', MultinomialNB())
                        ])


nbclassifier = nbclassifier.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(nbclassifier, f)
    


