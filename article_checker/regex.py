#%%
import re
from collections import Counter

import nltk

# import tensorflow
# from tensorflow import keras
# from tensorflow.keras.prepocessing.text import Tokenizer

nltk.download('stopwords')
import pandas as pd
from nltk import stopwords
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
backgroundsIs = ' '.join(backgrounds)



# %%
