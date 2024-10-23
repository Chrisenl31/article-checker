#%%
import re
from collections import Counter

import keras as kr
import pandas as pd
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from pandas import Series

#read data
data = pd.read_csv('scopus.csv')
data = data.dropna()
AbstractIs = data['Abstract']
print(AbstractIs)
pattern = re.compile(r'\w+(?=:)|\w+(?= -)|\w+(?=-)')
i=1

#%%check structure
#get structure keyword
articles = []
for Abstract in AbstractIs:
    matches_word = []
    str_AbstractIs = str(Abstract)
    matches = pattern.finditer(str_AbstractIs)
    for match in matches:
        matches_word.append(match.group())
    articles.append({
        "id":i,
        "words":matches_word
    })
    print("\n")
    i = i+1

for article in articles:
    print(f"ID: {article['id']}\nWords: {article['words']}\n")

#%%only take keywords
#BACKGROUND
pattern_key_BackgroundIs = re.compile(r'Background:\s*(.*?)(?=[.:])')
key_BackgroundIs = "Background"
backgrounds = []
for article in articles:
    str_AbstractIs = str(AbstractIs[article['id']-1])
    result_BackgroundIs = pattern_key_BackgroundIs.search(str_AbstractIs)
    if result_BackgroundIs:
        print(f"ID: {article['id']}")
        wordresult_BackgroundIs = result_BackgroundIs.group(1).strip()
        print(f"Text:{wordresult_BackgroundIs}\n")
        backgrounds.append({wordresult_BackgroundIs})
# backgrounds.append({
#         "id": article['id'],
#         "text": wordresult_BackgroundIs
# })


print(f"{backgrounds}\n")


#%%get top words w TF
#BackgroundIs
Background_Label = []
tokenizer = Tokenizer()
backgrounds_flatten = ' '.join([list(bg)[0] for bg in backgrounds])
tokenizer.fit_on_texts([backgrounds_flatten])
word_count = tokenizer.word_counts

stop_words = set(stopwords.words('english'))
filtered={word: count for word, count in word_count.items() if word not in stop_words}

if filtered:
    BackgroundIs = sorted(filtered.items(), key=lambda item: item[1], reverse=True)[:10]
    for word, count in BackgroundIs:
        print(f"'{word}': {count}")
        Background_Label.append(word)
print(f"{Background_Label}")


# %%
