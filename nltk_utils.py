import nltk

import numpy as np

#this is required only when running for the first time later can comment it.
# a package with pre trained tokenizer.
# nltk.download('punkt') 

from nltk.stem.porter import PorterStemmer # for stemming 
stemmer=PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())



def bag_of_words(tokenized_sentence,all_words):
    '''
    sentence=['Agile Manifesto','Kanban']
    words=['Agile Manifesto', 'Agile Principles', 'Daily Standup', 'Kanban']
    bog =[1,0,0,1]
    '''

    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    stemmed_words = [stem(w) for w in all_words]  # Stem the all_words list

    bag = np.zeros(len(stemmed_words), dtype=np.float32)
    for idx, w in enumerate(stemmed_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag


