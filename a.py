# a.py - Logistic Regression Model

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Load data
def load_data():
    positive_file = "/Users/kaushik/Documents/coding/ml/nlp/rt-polarity.pos"
    negative_file = "/Users/kaushik/Documents/coding/ml/nlp/rt-polarity.neg"
    with open(positive_file, 'r', encoding='latin-1') as pos, open(negative_file, 'r', encoding='latin-1') as neg:
        positive_data = pos.readlines()
        negative_data = neg.readlines()
    return positive_data, negative_data

# Prepare dataset
positive_data, negative_data = load_data()
all_data = positive_data + negative_data
labels = [1] * len(positive_data) + [0] * len(negative_data)

# Split the data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(all_data, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

# Text preprocessing
vectorizer = CountVectorizer(binary=True, stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_val_vectorized = vectorizer.transform(X_val)
X_test_vectorized = vectorizer.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Model evaluation
y_pred = model.predict(X_test_vectorized)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)
