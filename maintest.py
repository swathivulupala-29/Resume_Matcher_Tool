
import streamlit as st
import pandas as pd
import numpy as np
#from utilities import extract_text_from_pdf, get_embedding  # Assuming utilities.py is in the same directory
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Load sentence transformer model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
classifier = LogisticRegression()

# Sample dataset (sentences and corresponding sentiment labels)
texts = [
    "I love this product!", "This is the worst experience ever.", 
    "It's an amazing day!", "I hate waiting in long lines.",
    "The service was fantastic!", "The food was terrible.",
    "I enjoyed the movie.", "It was a boring and dull event.",
    "Absolutely wonderful!", "Completely disappointed."
]

# positive , negative labels for the above sentences
# 1 for positive sentiment, 0 for negative sentiment
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# Convert texts to embeddings
embeddings = model.encode(texts)

# Train a simple classifier
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predict and evaluate
predictions = classifier.predict(X_test)
accuracy=accuracy_score(y_test, predictions)
print("Accuracy:", accuracy_score(y_test, predictions))

# Streamlit UI for input
st.title('Real-Time Sentiment Analysis')
st.write('Enter a sentence below and get the sentiment prediction.')

# User input box
user_input = st.text_input('Enter your sentence:', '')

# When the user enters a sentence
if user_input:
    # Convert input to embedding
    input_embedding = model.encode([user_input])

    # Predict sentiment
    prediction = classifier.predict(input_embedding)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    
    st.write(f"Sentiment: {sentiment}")