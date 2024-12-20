# CARINA: Abstract Structure Recommendation

## Project Description

CARINA (Abstract Structure Recommendation) is a project that leverages Natural Language Processing (NLP) and multiclass text classification to evaluate and classify the structure of an abstract. This tool helps ensure that academic abstracts follow a semi-structured format that consists of the following sections:
- **Background**
- **Objectives**
- **Methods**
- **Goals**
- **Conclusions**

## Features

- **Title and Abstract Input**: Users can input the title, abstract, and keywords of a research paper.
- **Abstract Structure Evaluation**: The tool analyzes the structure of the abstract and classifies each section into predefined categories: Background, Objectives, Methods, Results, and Conclusions.
- **Keyword Check**: The system checks if specific keywords are present in the abstract, indicating relevance to certain sections (e.g., "methods," "approach" for the Methods section).
- **Sentence Grouping**: Sentences are grouped under the appropriate labels (e.g., Background, Methods, Results) based on their content.
- **User-Friendly Interface**: A simple and intuitive web interface that allows users to submit abstracts and see the structured output, as well as check the presence of keywords.
- **Copy Function**: Users can copy the analysis result with the press of a button to easily share the structured abstract.

## How It Works

1. **Input**: The user enters a research paper's abstract and keywords through a form.
2. **Preprocessing**: The system tokenizes the abstract into individual sentences and processes them using NLP techniques.
3. **Multiclass Classification**: A machine learning model classifies each sentence into one of five categories: Background, Objectives, Methods, Results, or Conclusions.
4. **Keyword Matching**: The system checks if specified keywords (e.g., "methods," "result") appear in the abstract to validate the content of each section.
5. **Output**: The abstract is grouped into its respective sections, and the analysis result is displayed on the interface.

## Technologies Used

- **Python**: The core programming language.
- **Django**: For the backend web framework.
- **Scikit-learn**: Used for multiclass classification and machine learning models.
- **Natural Language Processing**: Sentence tokenization and classification.
- **HTML/CSS/JavaScript**: For the frontend user interface.
- **Bootstrap**: For responsive and clean layout.
- **SVC (Support Vector Classification)**, **KNeighborsClassifier**, **LogisticRegression**: These classifiers are part of the voting mechanism used to classify sentences into the predefined abstract categories.

## Setup Instructions

### Prerequisites

Before setting up the project, make sure you have the following installed:

- **Python** (preferably Python 3.10 or later)
- **Django**
- **Scikit-learn**
- **NLTK** (Natural Language Toolkit)

You can install the necessary Python packages using the following command:

```bash
pip install -r requirements.txt
