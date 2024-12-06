#%%
import re
import string
from collections import defaultdict

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

# Load and preprocess data
data = pd.read_csv('new_dt2.csv')
data = data.dropna()
AbstractIs = data['Abstract']

# Define regex patterns for labels
patterns = [
    (r'(?:Background|Introduction|Introductions|Problem|Problems|Motivation|Motivations)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Background'),
    (r'(?:Purpose|Purposes|Goals|Goal|Objective|Objectives|Aims|Aim)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Objective'),
    (r'(?:Approach|Method|Methods|Design|Designs)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Methods'),
    (r'(?:Findings|Finding|Result|Results)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Results'),
    (r'(?:Implications|Implication|Conclusion|Conclusions)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Conclusions')
]

# Stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(string.punctuation)
stop_words.update(map(str, range(10)))
stemmer = PorterStemmer()

# Extract relevant content for each label
dictionary_content = defaultdict(list)
compiling_content = [(re.compile(pattern), label) for pattern, label in patterns]

for abstract in AbstractIs:
    for pattern, label in compiling_content:
        match = pattern.findall(abstract)
        if match:
            dictionary_content[label].extend(match)

# Group abstracts by label and create TF-IDF keywords
label_texts = {label: " ".join(contents) for label, contents in dictionary_content.items()}

# Apply TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=15)  # Adjust max_features if needed
tfidf_matrix = vectorizer.fit_transform(label_texts.values())
tfidf_keywords = {
    label: [word for word in vectorizer.get_feature_names_out()]
    for label, tfidf_row in zip(label_texts.keys(), tfidf_matrix)
}


domain_specific_keys = {
    "Background": ["context", "problem", "introduction", "motivation"],
    "Objective": ["goal", "aim", "purpose", "objective"],
    "Methods": ["approach", "design", "methodology", "process","applies","technique"],
    "Results": ["result", "finding", "analysis", "outcome"],
    "Conclusions": ["conclusion", "implication", "summary", "impact", "inform"]
}

enhanced_keywords = {
    label: list(set([stemmer.stem(word) for word in tfidf_keywords[label] + domain_specific_keys[label]]))
    for label in label_texts.keys()
}

#%%Tokenize input paragraph into sentences
input_paragraph = [
            "In an era where digital finance is growing rapidly, the Quick Response Code Indonesian Standard (QRIS) revolutionizes the payment system through a single unifying code. This study brings novelty in integrating TRAM and Trust in the adoption of QRIS in micro, small, and medium enterprises (MSMEs) in Indonesia, for which studies are still limited. To observe determinants of QRIS adoption by integrating the Technology Readiness Acceptance Model (TRAM) and Trust in the emerging Indonesian market where QRIS is in a growing stage. This study collects data through the survey of 210 MSME owners and staff who are familiar with and/or have used QRIS through convenience sampling. In analyzing the data, this study uses the Structural Equation Model-Partial Least Square (PLS-SEM) to examine the relationship between variables that explain influencing factors of QRIS adoption. The results show that 7 of 13 hypotheses were accepted; optimism and trust positively significantly affect perceived ease of use and perceived usefulness, while insecurity and innovation have no significant influence on perceived ease of use and perceived usefulness. Besides, this study shows unexpected positive results between discomfort, perceived ease of use, and perceived usefulness. Overall, the proposed TRAM and Trust model contributes 60.9 % in explaining QRIS adoption. This study emphasizes the importance of optimism, discomfort, trust, perceived ease of use, and perceived usefulness in influencing QRIS adoption in micro and small businesses in Indonesia. It guides QRIS providers, policymakers, financial institutions, and MSMEs in having effective QRIS adoption in business operations"
]
sentences = sent_tokenize(input_paragraph[0])

# Prepare data for TF-IDF-based similarity
combined_texts = [" ".join(words) for words in enhanced_keywords.values()] + sentences
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(combined_texts)

# Compute cosine similarity
cosine_similarities = cosine_similarity(tfidf_matrix[len(enhanced_keywords):], tfidf_matrix[:len(enhanced_keywords)])

# Assign labels based on highest similarity
labels = list(enhanced_keywords.keys())
for i, sentence in enumerate(sentences):
    similarity_scores = cosine_similarities[i]
    best_match_idx = similarity_scores.argmax()
    best_match_label = labels[best_match_idx]
    print(f"Sentence: {sentence}\nAssigned Label: {best_match_label}\nSimilarity Scores: {similarity_scores}\n")

# %%
