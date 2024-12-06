#%%
import re
from collections import Counter

import keras as kr
import nltk
import pandas as pd
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from pandas import Series

#read data
data = pd.read_csv('dataset.csv')
data = data.dropna()
AbstractIs = data['Abstract']
print(AbstractIs)
# pattern = re.compile(r'\w+(?=:)|\w+(?= -)|\w+(?=-)|\w+(?=])|\w+(?=.)')
i=1

#%%check structure
#get structure keyword
# articles = []
# for Abstract in AbstractIs:
#     matches_word = []
#     str_AbstractIs = str(Abstract)
#     matches = pattern.finditer(str_AbstractIs)
#     for match in matches:
#         matches_word.append(match.group())
#     articles.append({
#         "id":i,
#         "words":matches_word
#     })
#     print("\n")
#     i = i+1

# for article in articles:
#     print(f"ID: {article['id']}\nAbstract: {data['Abstract']}\n")

#%%only take keywords
#BACKGROUND
articles = []
for article in articles:
    str_AbstractIs = str(AbstractIs)
    articles.append({
        "id":i,
        "abstract": str_AbstractIs
    })
    i=i+1
    print(f"{article['id']}")
pattern_key_BackgroundIs = re.compile(r'Background:\s*(.*?)(?=[.:])|Introduction:\s*(.*?)(?=[.:])|Problem:\s*(.*?)(?=[.:])', re.IGNORECASE)
backgrounds = []
for article in articles:
    str_AbstractIs = str(AbstractIs[article['id']-1])
    result_BackgroundIs = pattern_key_BackgroundIs.search(str_AbstractIs)
    if result_BackgroundIs:
        print(f"ID: {article['id']}")
        wordresult_BackgroundIs = result_BackgroundIs.group(0).strip()
        print(f"Text:{wordresult_BackgroundIs}\n")
        backgrounds.append({wordresult_BackgroundIs})
    # else: print(f"Warning: ID {article['id']} not found in AbstractIs")
# backgrounds.append({
#         "id": article['id'],
#         "text": wordresult_BackgroundIs
# })
print(f"{backgrounds}\n")
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

#%%
#PURPOSE
pattern_key_PurposeIs = re.compile(r'Purpose:\s*(.*?)(?=[.:])|Goals:\s*(.*?)(?=[.:])|Objective:\s*(.*?)(?=[.:])', re.IGNORECASE)
purposes = []
for article in articles:
    str_AbstractIs = str(AbstractIs[article['id']-1])
    result_PurposeIs = pattern_key_PurposeIs.search(str_AbstractIs)
    if result_PurposeIs:
        print(f"ID: {article['id']}")
        wordresult_PurposeIs = result_PurposeIs.group(0).strip()
        print(f"Text:{wordresult_PurposeIs}\n")
        purposes.append({wordresult_PurposeIs})
#get words
Purpose_Label = []
tokenizer = Tokenizer()
purpose_flatten = ' '.join([list(pr)[0] for pr in purposes])
tokenizer.fit_on_texts([purpose_flatten])
word_count = tokenizer.word_counts

stop_words = set(stopwords.words('english'))
filtered={word: count for word, count in word_count.items() if word not in stop_words}

if filtered:
    PurposeIs = sorted(filtered.items(), key=lambda item: item[1], reverse=True)[:10]
    for word, count in PurposeIs:
        print(f"'{word}': {count}")
        Purpose_Label.append(word)
print(f"{Purpose_Label}")

#%%
#APPROACH
pattern_key_ApproachIs = re.compile(r'approach:\s*(.*?)(?=[.:])|method:\s*(.*?)(?=[.:])|methods:\s*(.*?)(?=[.:])', re.IGNORECASE)
approaches = []
for article in articles:
    str_AbstractIs = str(AbstractIs[article['id']-1])
    result_ApproachIs = pattern_key_ApproachIs.search(str_AbstractIs)
    if result_ApproachIs:
        print(f"ID: {article['id']}")
        wordresult_ApproachIs = result_ApproachIs.group(0).strip()
        print(f"Text:{wordresult_ApproachIs}\n")
        approaches.append({wordresult_ApproachIs})
#approaches_label
Approach_Label = []
tokenizer = Tokenizer()
approach_flatten = ' '.join([list(met)[0] for met in approaches])
tokenizer.fit_on_texts([approach_flatten])
word_count = tokenizer.word_counts

stop_words = set(stopwords.words('english'))
filtered={word: count for word, count in word_count.items() if word not in stop_words}

if filtered:
    ApproachIs = sorted(filtered.items(), key=lambda item: item[1], reverse=True)[:10]
    for word, count in ApproachIs:
        print(f"'{word}': {count}")
        Approach_Label.append(word)
print(f"{Approach_Label}")

# %%FINDINGS
pattern_key_FindingIs = re.compile(r'findings:\s*(.*?)(?=[.:])|finding:\s*(.*?)(?=[.:])|result:\s*(.*?)(?=[.:])|results:\s*(.*?)(?=[.:])', re.IGNORECASE)
findings = []
for article in articles:
    str_AbstractIs = str(AbstractIs[article['id']-1])
    result_FindingIs = pattern_key_FindingIs.search(str_AbstractIs)
    if result_FindingIs:
        print(f"ID: {article['id']}")
        wordresult_FindingIs = result_FindingIs.group(0).strip()
        print(f"Text:{wordresult_FindingIs}\n")
        findings.append({wordresult_FindingIs})
#finding_label
Finding_Label = []
tokenizer = Tokenizer()
finding_flatten = ' '.join([list(result)[0] for result in findings])
tokenizer.fit_on_texts([finding_flatten])
word_count = tokenizer.word_counts

stop_words = set(stopwords.words('english'))
filtered={word: count for word, count in word_count.items() if word not in stop_words}

if filtered:
    FindingIs = sorted(filtered.items(), key=lambda item: item[1], reverse=True)[:10]
    for word, count in FindingIs:
        print(f"'{word}': {count}")
        Finding_Label.append(word)
print(f"{Finding_Label}")

#%%Implication/conclusion
pattern_key_ImplicationIs = re.compile(r'implications:\s*(.*?)(?=[.:])|conclusion:\s*(.*?)(?=[.:])|implications:\s*(.*?)(?=[.:])', re.IGNORECASE)
implications = []
for article in articles:
    str_AbstractIs = str(AbstractIs[article['id']-1])
    result_ImplicationIs = pattern_key_ImplicationIs.search(str_AbstractIs)
    if result_ImplicationIs:
        print(f"ID: {article['id']}")
        wordresult_ImplicationIs = result_ImplicationIs.group(0).strip()
        print(f"Text:{wordresult_ImplicationIs}\n")
        implications.append({wordresult_ImplicationIs})
#finding_label
Implication_Label = []
tokenizer = Tokenizer()
implications_flatten = ' '.join([list(impl)[0] for impl in implications])
tokenizer.fit_on_texts([implications_flatten])
word_count = tokenizer.word_counts

stop_words = set(stopwords.words('english'))
filtered={word: count for word, count in word_count.items() if word not in stop_words}

if filtered:
    ImplicationIs = sorted(filtered.items(), key=lambda item: item[1], reverse=True)[:10]
    for word, count in ImplicationIs:
        print(f"'{word}': {count}")
        Implication_Label.append(word)
print(f"{Implication_Label}")

# %%keywords
print(f'Background: {Background_Label}\n Purpose: {Purpose_Label}\n Method: {Approach_Label}\n Finding: {Finding_Label}\n Implication: {Implication_Label}')

# %%
