import re
import string
from collections import defaultdict

import nltk
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Constants
STOP_WORDS = set(stopwords.words('english')).union(string.punctuation, map(str, range(10)))
STEMMER = PorterStemmer()
PATTERNS = [
    (r'(?:Background|Introduction|Motivation)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Background'),
    (r'(?:Purpose|Goal|Objective)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Objective'),
    (r'(?:Approach|Method|Design)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Methods'),
    (r'(?:Finding|Result)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Results'),
    (r'(?:Implication|Conclusion)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Conclusions'),
]

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path).dropna()
    return data['Abstract']

def extract_labels(abstracts):
    content_dict = defaultdict(list)
    compiled_patterns = [(re.compile(pattern), label) for pattern, label in PATTERNS]
    for abstract in abstracts:
        for pattern, label in compiled_patterns:
            matches = pattern.findall(abstract)
            if matches:
                content_dict[label].extend(matches)
    return content_dict

def preprocess_sentences(sentences, labels):
    training_sentences = []
    training_labels = []
    for label, content in sentences.items():
        for sentence in content:
            training_sentences.append(sentence)
            training_labels.append(label)
    return training_sentences, training_labels

def augment_keywords(content_dict, vectorizer):
    augmented_sentences = []
    domain_keywords = {
        "Background": ["context", "problem", "introduction"],
        "Objective": ["goal", "aim", "purpose"],
        "Methods": ["approach", "design", "methodology"],
        "Results": ["result", "finding", "analysis"],
        "Conclusions": ["conclusion", "implication", "summary"],
    }

    for label, content in content_dict.items():
        top_words = extract_top_words(vectorizer, content)
        domain_keywords[label].extend(top_words)
        keyword_str = " ".join(domain_keywords[label])
        for sentence in content:
            augmented_sentences.append(sentence + " " + keyword_str)
    return augmented_sentences

def extract_top_words(vectorizer, content):
    label_vector = vectorizer.transform(content)
    tfidf_scores = label_vector.sum(axis=0).A1
    word_scores = {word: tfidf_scores[idx] for idx, word in enumerate(vectorizer.get_feature_names_out())}
    return sorted(word_scores, key=word_scores.get, reverse=True)[:10]

# Define classifiers
def build_voting_classifier():
    svm = SVC(kernel='linear', probability=True)
    knn = KNeighborsClassifier(n_neighbors=5)
    log_reg = LogisticRegression(max_iter=1000)
    return VotingClassifier(estimators=[('svm', svm), ('knn', knn), ('log_reg', log_reg)], voting='hard')

# Main logic
def process_abstract(title, abstract):
    # Tokenize input abstract
    sentences = sent_tokenize(abstract)

    # Predict labels for each sentence
    input_vectors = VECTORIZER.transform(sentences)
    predicted_labels = VOTING_CLASSIFIER.predict(input_vectors)

    # Group sentences by labels
    label_sentences = defaultdict(list)
    for i, sentence in enumerate(sentences):
        label_sentences[predicted_labels[i]].append(sentence)
    
    return {label: content for label, content in label_sentences.items() if content}

# Django View
def check_abstract(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        abstract = request.POST.get('abstract')

        # Process abstract (use your existing logic here)
        processed_result = process_abstract(title, abstract)

        # Example processed result (replace with actual logic)
        context = {
            'title': title,
            'abstract': abstract,
            'word_count': len(abstract.split()),
            'title_feedback': processed_result.get('title_feedback', 'N/A'),
            'abstract_relationship': processed_result.get('abstract_relationship', 0),
            'abstract_sections': processed_result.get('abstract_sections', []),
            'recommended_structure': processed_result.get('recommended_structure', {})
        }

        return render(request, 'result.html', context)
    return render(request, 'check_article.html')


# Initialize global objects
if __name__ == "__main__":
    ABSTRACTS = load_data('new_dt2.csv')
    CONTENT_DICT = extract_labels(ABSTRACTS)
    TRAIN_SENTENCES, TRAIN_LABELS = preprocess_sentences(CONTENT_DICT, PATTERNS)
    
    VECTORIZER = TfidfVectorizer(stop_words='english', max_features=1000)
    X = VECTORIZER.fit_transform(TRAIN_SENTENCES)
    
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, TRAIN_LABELS, test_size=0.2, random_state=42)
    VOTING_CLASSIFIER = build_voting_classifier()
    VOTING_CLASSIFIER.fit(X_TRAIN, Y_TRAIN)
