#%%Libs
import re
import string
from collections import Counter, defaultdict

import nltk
import pandas as pd
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.text import \
    Tokenizer as word_tokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pandas import Series
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Oroses data
data = pd.read_csv('new_dt2.csv')
data = data.dropna()
AbstractIs = data['Abstract']
print(AbstractIs)

#%%
#Regex & dictionary
patterns = [
    (r'(?:Background|Introduction|Introductions|Problem|Problems|Motivation|Motivations)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Background'),
    (r'(?:Purpose|Purposes|Goals|Goal|Objective|Objectives|Aims|Aim)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Objective'),
    (r'(?:Approach|Method|Methods|Design|Designs)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Methods'),
    (r'(?:Findings|Finding|Result|Results)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Results'),
    (r'(?:Implications|Implication|Conclusion|Conclusions)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Conclusions')
]
stop_words = set(stopwords.words('english'))
stop_words.update(string.punctuation)
stop_words.update(map(str,range(10)))
dictionary_content = defaultdict(list)
compiling_content = [(re.compile(pattern), label) for pattern, label in patterns]

for abstract in AbstractIs:
    for pattern, label in compiling_content:
        match = pattern.findall(abstract)
        if match:
            dictionary_content[label].extend(match)


for label, contents in dictionary_content.items(): 
    print(f"{label:} {contents}")
    print("\n")


label_sum = {}
for label, contents in dictionary_content.items():
    all_words = " ".join(contents).lower().split()
    filtered_words = [word for word in all_words if word not in stop_words and word.isalpha()]
    word_counts = Counter(filtered_words)
    top_10 = word_counts.most_common(10)
    label_sum[label] = " ".join(word for word, count in top_10)
    print(f"{label}: {top_10}")

# %%
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab')
input_paragraph = [
        "Background: Human Resource Management System (HRMS) is an important aspect of managing organizations. However, the successful integration of the system into respective roles is often associated with diverse technological challenges and trends. Some major obstacles identified in recent research include reluctance to change, lack of training, fragmented Human Resource (HR) data, rigid processes, and continuous changes in organizational needs. Exciting technology trends offer promise for next-generation HRMS solutions, including artificial intelligence (AI), machine learning, predictive analytics, and mobile accessibility. This shows the need for a systematic literature review to comprehensively map the challenges and technology trends shaping the implementation of HRMS. Objective: This research aimed to conduct a comprehensive review of existing literature to identify the main challenges faced during HRMS implementation and the latest technology trends in the space. Methods: A systematic literature review was adopted through the Kitchenham method with a focus on five databases including Scopus, Emerald, IEEE, Science Direct, and ProQuest. Results: The result was in the form of a table mapping of the challenges faced by each stakeholder in HRMS, including resistance to change, lack of management support, and limited technology infrastructure. Meanwhile, the most common technology challenges found were system integration issues, data security, and lack of technical capabilities or skills. The potential opportunities from technology trends to address the issues included training and skills development, enhanced cybersecurity, and effective change management methods. These recommendations were designed to support organizations in further optimizing HRMS utilization and leveraging the latest technologies such as AI and blockchain. Conclusion: The review used a structured method to develop a rich overview through tabular presentations summarizing problem identification and technology trend compilation of HRMS. The systematic exploration aimed to contribute valuable insights into the complexities of HRMS implementation and offer a comprehensive perspective on the emergence of relevant technology trends. The results were expected to contribute to future research directions in this important area at the nexus of Human Resource Management (HRM) and technological innovation."
]
sentences = sent_tokenize(input_paragraph[0])

labels = list(label_sum.keys())
labels_text = list(label_sum.values())
combined_texts = labels_text + sentences


vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(combined_texts)
cosine_similarities = cosine_similarity(tfidf_matrix[len(labels_text):], tfidf_matrix[:len(labels_text)])

for i, sentence in enumerate(sentences):
    similarity_scores = cosine_similarities[i]
    best_match_idx = similarity_scores.argmax()
    best_match_label = labels[best_match_idx]
    print(f"Sentence: {sentence}\nAssigned Label: {best_match_label}\n{similarity_scores}")
# %%
