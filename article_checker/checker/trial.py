import re
import string
from collections import defaultdict

import nltk
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# NLTK and Stopwords Setup
# nltk.download('punkt')
# nltk.download('stopwords')

#Konfigurasi global 
STOP_WORDS = set(stopwords.words('english')).union(string.punctuation, map(str, range(10)))
STEMMER = PorterStemmer()
PATTERNS = [
    (r'(?:Background|Introduction|Motivation)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Background'),
    (r'(?:Purpose|Goal|Objective)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Objective'),
    (r'(?:Approach|Method|Design)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Methods'),
    (r'(?:Finding|Result)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Results'),
    (r'(?:Implication|Conclusion)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Conclusions'),
]

class AbstractClassifier: 
    def __init__(self, dataset_path='new_dt2.csv'):
        # Load and preprocess training data
        self.abstracts = self.load_data(dataset_path)
        self.content_dict = self.extract_labels(self.abstracts)
        self.train_sentences, self.train_labels = self.preprocess_sentences(self.content_dict)
        
        # Vectorization
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        X = self.vectorizer.fit_transform(self.train_sentences)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, self.train_labels, test_size=0.2, random_state=42)
        
        # Build and train classifier
        self.voting_classifier = self.build_voting_classifier()
        self.voting_classifier.fit(X_train, y_train)

    def load_data(self, dataset_path):
        """Load dataset from CSV"""
        data = pd.read_csv(dataset_path)
        return data['Abstract'].dropna().tolist()

    def extract_labels(self, abstracts):
        """Extract label-specific content using regex patterns"""
        content_dict = defaultdict(list)
        compiled_patterns = [(re.compile(pattern, re.IGNORECASE), label) for pattern, label in PATTERNS]
        
        for abstract in abstracts:
            for pattern, label in compiled_patterns:
                matches = pattern.findall(abstract)
                if matches:
                    content_dict[label].extend(matches)
        return content_dict

    def preprocess_sentences(self, content_dict):
        """Prepare training sentences and labels"""
        training_sentences = []
        training_labels = []
        
        for label, content in content_dict.items():
            for sentence in content:
                training_sentences.append(sentence)
                training_labels.append(label)
        
        return training_sentences, training_labels

    def build_voting_classifier(self):
        """Create ensemble voting classifier"""
        svm = SVC(kernel='linear', probability=True)
        knn = KNeighborsClassifier(n_neighbors=5)
        log_reg = LogisticRegression(max_iter=1000)
        
        return VotingClassifier(
            estimators=[('svm', svm), ('knn', knn), ('log_reg', log_reg)], 
            voting='hard'
        )

    def predict_abstract_sections(self, abstract):
        """Predict sections for a given abstract"""
        # Tokenize input abstract into sentences
        sentences = sent_tokenize(abstract)
        
        # Vectorize sentences
        input_vectors = self.vectorizer.transform(sentences)
        
        # Predict labels for each sentence
        predicted_labels = self.voting_classifier.predict(input_vectors)
        
        # Group sentences by their predicted labels
        label_sentences = defaultdict(list)
        for i, sentence in enumerate(sentences):
            label_sentences[predicted_labels[i]].append(sentence)
        
        # Return non-empty sections
        return {label: content for label, content in label_sentences.items() if content}

# Flask Application
app = Flask(__name__)

# Initialize the classifier
classifier = None
def get_classifier():
    global classifier
    if classifier is None:
        classifier = AbstractClassifier()
    return classifier

