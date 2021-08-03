# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:44:03 2021

@author: asrinutku
"""
import numpy as np
from gensim.models import Word2Vec
import pandas as pd
import matplotlib.pyplot as plt

#dataset okuma
df = pd.read_csv('korona.csv') 

#%%
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords') #tüm dillerdeki stopwordlerin indirilmesi
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer


#türkçe stopwordların alınması
all_stopwords = stopwords.words('turkish')
toker = RegexpTokenizer(r'\w+')

cumleler = df.iloc[:, 0]
values = df.iloc[:, 2]

#tum harfler küçük hale getiriliyor 
text = [k.lower() for k in cumleler]

textwithpunct = [toker.tokenize(k) for k in text]

texts = [" ".join(k) for k in textwithpunct]
corpus = []

#stopwordlerin cümlelerden çıkarılması 
for cumle in texts:
    text_tokens = word_tokenize(cumle)
    tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]
    
    corpus.append(tokens_without_sw)

corpusseries = pd.Series(corpus)
#print(corpus[:20])

#%% pozitif ve negatif veri sayısının alınması

# pnsayisi = (df.
#      groupby(by = ['Duygu']).
#      count()
#     )

# n = pnsayisi.iloc[0]['Text'] #negatif veri sayisi
# p = pnsayisi.iloc[1]['Text'] #pozitif veri sayisi

print("Pozitif ve Negatif Data Sayısı Belirlendi..")

#%%GensimWord2VecVectorizer

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec


class GensimWord2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Word vectors are averaged across to create the document-level vectors/features.
    gensim's own gensim.sklearn_api.W2VTransformer doesn't support out of vocabulary words,
    hence we roll out our own.
    All the parameters are gensim.models.Word2Vec's parameters.
    https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
    """

    def __init__(self, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None,
                 sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5,
                 ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False,
                 callbacks=(), max_final_vocab=None):
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.sg = sg
        self.hs = hs
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.cbow_mean = cbow_mean
        self.hashfxn = hashfxn
        self.iter = iter
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.compute_loss = compute_loss
        self.callbacks = callbacks
        self.max_final_vocab = max_final_vocab

    def fit(self, X, y=None):
        self.model_ = Word2Vec(
            sentences=X, corpus_file=None,
             alpha=self.alpha, window=self.window, min_count=self.min_count,
            max_vocab_size=self.max_vocab_size, sample=self.sample, seed=self.seed,
            workers=self.workers, min_alpha=self.min_alpha, sg=self.sg, hs=self.hs,
            negative=self.negative, ns_exponent=self.ns_exponent, cbow_mean=self.cbow_mean,
            hashfxn=self.hashfxn,  null_word=self.null_word,
            trim_rule=self.trim_rule, sorted_vocab=self.sorted_vocab, batch_words=self.batch_words,
            compute_loss=self.compute_loss, callbacks=self.callbacks,
            max_final_vocab=self.max_final_vocab)
        return self

    def transform(self, X):
        X_embeddings = np.array([self._get_embedding(words) for words in X])
        return X_embeddings

    def _get_embedding(self, words):
        #valid_words = [word for word in words if word in self.model_.wv[word]]
        valid_words = []
        for word in words:
            
            exist = word in self.model_.wv[word]
            
            if(exist == True):
                valid_words.append(word)
            
        
        
        if valid_words:
            embedding = np.zeros((len(valid_words), self.size), dtype=np.float32)
            for idx, word in enumerate(valid_words):
                embedding[idx] = self.model_.wv[word]

            return np.mean(embedding, axis=0)
        else:
            return np.zeros(self.size)

#%%
from sklearn.model_selection import train_test_split

X = corpusseries 
y = values

test_size = 0.3
random_state = None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=test_size, 
    shuffle=True,
    random_state=random_state, 
    )

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#print("Data train ve test olarak ayrıldı..")

#%%


from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline


gensim_word2vec_tr = GensimWord2VecVectorizer(
                                            size=50, 
                                            window=4,
                                            min_count=1, 
                                            sg=1, 
                                            alpha=0.025, 
                                            iter=30,         
                                            workers=4,
                                            )

xgb = XGBClassifier()

w2v_xgb = Pipeline([
    ('w2v', gensim_word2vec_tr), 
    ('xgb', xgb)
])


#%%
import time

start = time.time()
w2v_xgb.fit(X_train, y_train)
elapse = time.time() - start
print('Model Fit Edilirken Geçen Süre :  ', elapse)

#%%
from sklearn.metrics import accuracy_score, confusion_matrix

y_train_pred = w2v_xgb.predict(X_train)
print('Eğitim Seti Doğruluğu %s' % accuracy_score(y_train, y_train_pred))
confusion_matrix(y_train, y_train_pred)


