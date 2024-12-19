# %%
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

# nltk.download("punkt")
# nltk.download("stopwords")

# Load and preprocess data
data = pd.read_csv("new_dt2.csv")
data = data.dropna()
AbstractIs = data["Abstract"]

# Define regex patterns for labels
patterns = [
    (
        r"(?:Background|Introduction|Introductions|Problem|Problems|Motivation|Motivations)[:\-]?\s*(.*?)(?=\s*[.!?])",
        "Background",
    ),
    (
        r"(?:Purpose|Purposes|Goals|Goal|Objective|Objectives|Aims|Aim)[:\-]?\s*(.*?)(?=\s*[.!?])",
        "Objective",
    ),
    (
        r"(?:Approach|Method|Methods|Design|Designs)[:\-]?\s*(.*?)(?=\s*[.!?])",
        "Methods",
    ),
    (r"(?:Findings|Finding|Result|Results)[:\-]?\s*(.*?)(?=\s*[.!?])", "Results"),
    (
        r"(?:Implications|Implication|Conclusion|Conclusions)[:\-]?\s*(.*?)(?=\s*[.!?])",
        "Conclusions",
    ),
]

# Stopwords and stemming
stop_words = set(stopwords.words("english"))
stop_words.update(string.punctuation)
stop_words.update(map(str, range(10)))
stemmer = PorterStemmer()


def process_data(data_input):
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
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(training_sentences)

    # Get top 10 words for each label based on TF-IDF scores
    top_words = {}
    for label, contents in dictionary_content.items():
        # Vectorize the label-specific content
        label_vector = vectorizer.transform(contents)
        tfidf_scores = label_vector.sum(axis=0).A1
        word_scores = {
            word: tfidf_scores[idx]
            for idx, word in enumerate(vectorizer.get_feature_names_out())
        }

        # Sort words by score and get the top 10 words
        top_words[label] = sorted(word_scores, key=word_scores.get, reverse=True)[:10]

    # Define domain-specific keywords
    domain_specific_keys = {
        "Background": [
            "context",
            "problem",
            "introduction",
            "motivation",
            "increase",
            "trend",
            "study",
            "background",
        ],
        "Objective": ["goal", "aim", "purpose", "objective", "understanding", "aimed"],
        "Methods": ["approach", "design", "methodology", "process", "technique"],
        "Results": ["result", "finding", "analysis", "outcome"],
        "Conclusions": [
            "conclusion",
            "implication",
            "summary",
            "impact",
            "inform",
            "understand",
        ],
    }

    # Augment domain-specific keywords with top 10 words for each label
    for label in domain_specific_keys.keys():
        domain_specific_keys[label].extend(top_words[label])

    # Augment training sentences with domain-specific keywords
    augmented_training_sentences = []
    for label, contents in dictionary_content.items():
        domain_keywords = " ".join(
            domain_specific_keys[label]
        )  # Combine domain-specific keywords into a string
        for sentence in contents:
            augmented_sentence = (
                sentence + " " + domain_keywords
            )  # Add domain keywords to the sentence
            augmented_training_sentences.append(augmented_sentence)

    # Tokenize input paragraph into sentences
    input_paragraph = data_input
    sentences = sent_tokenize(input_paragraph)

    # Vectorize augmented sentences using TF-IDF
    X = vectorizer.fit_transform(augmented_training_sentences)

    # Train-test split (You can adjust the ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, training_labels, test_size=0.2, random_state=42
    )

    # Define classifiers
    svm_classifier = SVC(kernel="linear")
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    log_reg_classifier = LogisticRegression(max_iter=1000)

    # Create a Voting Classifier
    voting_classifier = VotingClassifier(
        estimators=[
            ("svm", svm_classifier),
            ("knn", knn_classifier),
            ("log_reg", log_reg_classifier),
        ],
        voting="hard",
    )

    voting_classifier.fit(X_train, y_train)
    input_vectors = vectorizer.transform(sentences)
    predicted_labels = voting_classifier.predict(input_vectors)

    results = []

    for i, sentence in enumerate(sentences):
        results.append(
            {
                "Sentence": sentence,
                "Label": predicted_labels[i],
            }
        )

    return results

    # # Print predictions
    # for i, sentence in enumerate(sentences):
    #     print(f"Sentence: {sentence}\nAssigned Label: {predicted_labels[i]}\n")

    # # Group sentences by their predicted labels
    # label_sentences = defaultdict(list)
    # for i, sentence in enumerate(sentences):
    #     label_sentences[predicted_labels[i]].append(sentence)

    # # Print sentences by label, including a message for empty labels
    # for label, sentences_in_label in label_sentences.items():
    #     if sentences_in_label:
    #         print(f"\n{label}:")
    #         for sentence in sentences_in_label:
    #             print(f"{sentence}")
    #     else:
    #         print(f"Your {label} is empty.")
    #     print("\n" + "-" * 50)
