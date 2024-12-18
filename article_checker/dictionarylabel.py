import re
import string
from collections import defaultdict

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class AbstractClassifier:
    def __init__(self, dataset_path='new_dt2.csv'):
        # Domain-specific keywords for each label
        self.DOMAIN_KEYWORDS = {
            "Background": [
                "context", "problem", "introduction", "motivation", 
                "increase", "trend", "study", "background", "rationale"
            ],
            "Objective": [
                "goal", "aim", "purpose", "objective", 
                "understanding", "targeted", "research", "intention"
            ],
            "Methods": [
                "approach", "design", "methodology", "process", 
                "technique", "analysis", "procedure", "strategy", "framework"
            ],
            "Results": [
                "result", "finding", "analysis", "outcome", 
                "discover", "reveal", "demonstrate", "evidence"
            ],
            "Conclusions": [
                "conclusion", "implication", "summary", "impact", 
                "inform", "understand", "significance", "insight"
            ]
        }

        # Regex patterns for label extraction
        self.PATTERNS = [
            (r'(?:Background|Introduction|Motivation)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Background'),
            (r'(?:Purpose|Goal|Objective)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Objective'),
            (r'(?:Approach|Method|Design)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Methods'),
            (r'(?:Finding|Result)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Results'),
            (r'(?:Implication|Conclusion)[:\-]?\s*(.*?)(?=\s*[.!?])', 'Conclusions'),
        ]

        # Stop words configuration
        self.STOP_WORDS = set(stopwords.words('english')).union(
            string.punctuation, 
            set(map(str, range(10)))
        )
        
        # Stemmer
        self.STEMMER = PorterStemmer()

        # Load and preprocess training data
        self.abstracts = self._load_data(dataset_path)
        self.content_dict = self._extract_labels(self.abstracts)
        self.train_sentences, self.train_labels = self._preprocess_sentences(self.content_dict)
        
        # Vectorization
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X = self.vectorizer.fit_transform(self.train_sentences)
        
        # Augment training data with domain keywords
        X_augmented, labels_augmented = self._augment_training_data(X, self.train_labels)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_augmented, labels_augmented, test_size=0.2, random_state=42
        )
        
        # Build and train classifier
        self.voting_classifier = self._build_voting_classifier()
        self.voting_classifier.fit(X_train, y_train)

    def _load_data(self, dataset_path):
        """Load dataset from CSV"""
        data = pd.read_csv(dataset_path)
        return data['Abstract'].dropna().tolist()

    def _extract_labels(self, abstracts):
        """Extract label-specific content using regex patterns"""
        content_dict = defaultdict(list)
        compiled_patterns = [(re.compile(pattern, re.IGNORECASE), label) for pattern, label in self.PATTERNS]
        
        for abstract in abstracts:
            for pattern, label in compiled_patterns:
                matches = pattern.findall(abstract)
                if matches:
                    content_dict[label].extend(matches)
        return content_dict

    def _preprocess_sentences(self, content_dict):
        """Prepare training sentences and labels"""
        training_sentences = []
        training_labels = []
        
        for label, content in content_dict.items():
            for sentence in content:
                training_sentences.append(sentence)
                training_labels.append(label)
        
        return training_sentences, training_labels

    def _augment_training_data(self, X, labels):
        """Augment training data with domain-specific keywords"""
        augmented_sentences = []
        augmented_labels = []
        
        for label, content in self.content_dict.items():
            # Get top words based on TF-IDF
            label_vector = self.vectorizer.transform(content)
            tfidf_scores = label_vector.sum(axis=0).A1
            word_scores = {
                word: tfidf_scores[idx] 
                for idx, word in enumerate(self.vectorizer.get_feature_names_out())
            }
            top_words = sorted(word_scores, key=word_scores.get, reverse=True)[:10]
            
            # Combine domain keywords and top words
            combined_keywords = self.DOMAIN_KEYWORDS[label] + top_words
            keyword_str = " ".join(combined_keywords)
            
            for sentence in content:
                augmented_sentence = sentence + " " + keyword_str
                augmented_sentences.append(augmented_sentence)
                augmented_labels.append(label)
        
        # Vectorize augmented sentences
        X_augmented = self.vectorizer.transform(augmented_sentences)
        
        return X_augmented, augmented_labels

    def _build_voting_classifier(self):
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

# Optional: Create a global instance if needed
abstract_classifier = AbstractClassifier()