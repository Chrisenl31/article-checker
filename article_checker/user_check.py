#%%
import re
from collections import Counter

import dataset_preprocessing
import keras as kr
import nltk
import pandas as pd
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from pandas import Series


def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

def label_sentence(sentence, label_dicts):
    sentence = re.sub(r'[^\w\s]', '', sentence.lower())
    words = set(sentence.split())


    stop_words = set(stopwords.words('english'))
    words = words - stop_words
    best_label = None
    best_similarity = 0

    # Compare with each label dictionary
    for label, label_words in label_dicts.items():
        label_set = set(label_words)
        similarity = jaccard_similarity(words, label_set) 


        if similarity > best_similarity:
            best_similarity = similarity
            best_label = label
    return best_label

def process_abstract(abstract, label_dicts):
    sentences = re.split(r'(?<!\w\.\w)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', abstract)
    
    labeled_sentences = []

    for sentence in sentences:
        label = label_sentence(sentence, label_dicts)  # Get the label for the sentence
        labeled_sentences.append({
            'sentence': sentence,
            'label': label
        })
    
    return labeled_sentences

article = {
    'input': "This study aims to analyze the impact of renewable energy sources on sustainability in urban areas. The methodology used involved multiple design techniques. The results show a significant improvement in energy efficiency. These findings will have major implications for future policies."
}

label_dicts = {
    'Background': Background_Label,
    'Purpose': Purpose_Label,
    'Approach': Approach_Label,
    'Finding': Finding_Label,
    'Implication': Implication_Label
}

labeled_abstract = process_abstract(article['input'], label_dicts)

for labeled_sentence in labeled_abstract:
    print(f"Sentence: {labeled_sentence['sentence']}")
    print(f"Label: {labeled_sentence['label']}")
    print("-" * 50)