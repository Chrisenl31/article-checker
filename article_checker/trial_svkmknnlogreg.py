# %% Import Libraries
import re
import string
from collections import defaultdict

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# %% Load and preprocess data
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

# Stopwords and lemmatization
stop_words = set(stopwords.words('english'))
stop_words.update(string.punctuation)
stop_words.update(map(str, range(10)))
lemmatizer = WordNetLemmatizer()

# Extract relevant content for each label
dictionary_content = defaultdict(list)
compiling_content = [(re.compile(pattern), label) for pattern, label in patterns]

for abstract in AbstractIs:
    for pattern, label in compiling_content:
        match = pattern.findall(abstract)
        if match:
            dictionary_content[label].extend(match)

# Prepare training data
training_sentences = []
training_labels = []

for label, contents in dictionary_content.items():
    for sentence in contents:
        training_sentences.append(sentence)
        training_labels.append(label)

# Vectorize training sentences using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(training_sentences)

# Generate top 10 keywords per label, ensuring no overlap with domain-specific keywords
top_words = {}
all_domain_specific_keys = set()
domain_specific_keys = {
    "Background": ["background", "context", "problem", "introduction", "motivation", "increase", "trend"],
    "Objective": ["goal", "aim", "purpose", "objective", "understanding", "aimed", "observe"],
    "Methods": ["approach", "design", "methodology", "process", "technique"],
    "Results": ["result", "finding", "analysis", "outcome"],
    "Conclusions": ["conclusion", "implication", "summary", "impact", "inform", "understand", "future"]
}

# Combine all domain-specific keywords to detect overlaps
for label_keywords in domain_specific_keys.values():
    all_domain_specific_keys.update(label_keywords)

# Extract top 10 keywords per label and remove overlapping ones
for label, contents in dictionary_content.items():
    label_vector = vectorizer.transform(contents)
    tfidf_scores = label_vector.sum(axis=0).A1
    word_scores = {word: tfidf_scores[idx] for idx, word in enumerate(vectorizer.get_feature_names_out())}
    sorted_keywords = sorted(word_scores.keys(), key=word_scores.get, reverse=True)
    
    # Filter out any keywords that exist in domain-specific lists for other labels
    filtered_keywords = [word for word in sorted_keywords if word not in all_domain_specific_keys]
    top_words[label] = filtered_keywords[:10]

# Function to lemmatize text
def lemmatize_text(text):
    tokens = nltk.word_tokenize(text.lower())  # Tokenize and convert to lowercase
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(lemmatized_tokens)

# Function to check domain-specific keywords
def match_domain_keywords(sentence, keywords):
    lemmatized_sentence = lemmatize_text(sentence)
    sentence_words = set(lemmatized_sentence.split())
    for label, keyword_list in keywords.items():
        if any(word in sentence_words for word in keyword_list):
            return label  # Return the first matching label
    return None

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, training_labels, test_size=0.2, random_state=42)

# Define classifiers
svm_classifier = SVC(kernel='linear', probability=True)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
log_reg_classifier = LogisticRegression(max_iter=500)

# Create a Voting Classifier
voting_classifier = VotingClassifier(estimators=[
    ('svm', svm_classifier),
    ('knn', knn_classifier),
    ('log_reg', log_reg_classifier)
], voting='hard')

# Train the ensemble model
voting_classifier.fit(X_train, y_train)

# %% Input paragraph for prediction
input_paragraph = [
    "Background: Human Resource Management System (HRMS) is an important aspect of managing organizations. However, the successful integration of the system into respective roles is often associated with diverse technological challenges and trends. Some major obstacles identified in recent research include reluctance to change, lack of training, fragmented Human Resource (HR) data, rigid processes, and continuous changes in organizational needs. Exciting technology trends offer promise for next-generation HRMS solutions, including artificial intelligence (AI), machine learning, predictive analytics, and mobile accessibility. This shows the need for a systematic literature review to comprehensively map the challenges and technology trends shaping the implementation of HRMS. Objective: This research aimed to conduct a comprehensive review of existing literature to identify the main challenges faced during HRMS implementation and the latest technology trends in the space. Methods: A systematic literature review was adopted through the Kitchenham method with a focus on five databases including Scopus, Emerald, IEEE, Science Direct, and ProQuest. Results: The result was in the form of a table mapping of the challenges faced by each stakeholder in HRMS, including resistance to change, lack of management support, and limited technology infrastructure. Meanwhile, the most common technology challenges found were system integration issues, data security, and lack of technical capabilities or skills. The potential opportunities from technology trends to address the issues included training and skills development, enhanced cybersecurity, and effective change management methods. These recommendations were designed to support organizations in further optimizing HRMS utilization and leveraging the latest technologies such as AI and blockchain. Conclusion: The review used a structured method to develop a rich overview through tabular presentations summarizing problem identification and technology trend compilation of HRMS. The systematic exploration aimed to contribute valuable insights into the complexities of HRMS implementation and offer a comprehensive perspective on the emergence of relevant technology trends. The results were expected to contribute to future research directions in this important area at the nexus of Human Resource Management (HRM) and technological innovation."
]
sentences = sent_tokenize(input_paragraph[0])

# Predict labels with prioritization
label_sentences = defaultdict(list)
for sentence in sentences:
    keyword_label = match_domain_keywords(sentence, domain_specific_keys)
    if keyword_label:
        label_sentences[keyword_label].append(sentence)
    else:
        input_vector = vectorizer.transform([sentence])
        predicted_label = voting_classifier.predict(input_vector)[0]
        label_sentences[predicted_label].append(sentence)

# Print sentences by label
for label, sentences_in_label in label_sentences.items():
    print(f"\n{label}:")
    for sentence in sentences_in_label:
        print(f"  {sentence}")
    print("\n" + "-"*50)

# %%
