import os
import re
import pandas as pd
import warnings
import nltk
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from werkzeug.serving import run_simple
from threading import Thread
from textblob import TextBlob

nltk.download('wordnet')
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Ignore warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    # Remove URLs, mentions, special characters, and lowercase text
    text = re.sub(r'http\S+|@\S+|#\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    # Tokenize, remove stopwords, and lemmatize
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Load and preprocess the training data (assumes train.csv is already available)
train = pd.read_csv("train.csv")
train['Cleaned_Text'] = train['text'].apply(preprocess_text)

# Vectorize the data
tfidf = TfidfVectorizer(max_df=0.90, min_df=2, max_features=2000, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(train['Cleaned_Text'])

# Split data into training and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(tfidf_matrix, train['sentiment'], test_size=0.2, random_state=17)

# Train a Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(x_train, y_train)

# Validate the model and print accuracy
predictions = nb_model.predict(x_valid)
accuracy = accuracy_score(y_valid, predictions)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Cross-validate the model for a more robust accuracy estimate
cv_scores = cross_val_score(nb_model, x_train, y_train, cv=5, scoring='accuracy')
print(f"Average Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")

# Function to get sentiment from TextBlob
def get_textblob_sentiment(text):
    blob = TextBlob(text)
    # Get the polarity of the text
    polarity = blob.sentiment.polarity
    
    # Classify sentiment based on polarity score
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Flask route for main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle text analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    input_text = request.form.get('text')
    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    # Preprocess the input text
    cleaned_text = preprocess_text(input_text)
    tfidf_text = tfidf.transform([cleaned_text])

    # Get TextBlob sentiment analysis (polarity)
    sentiment_prediction = get_textblob_sentiment(input_text)

    # Display the result as predicted sentiment
    return render_template('index.html', input_text=input_text, sentiment_prediction=sentiment_prediction)

# Function to run the app in a separate thread
def run_app():
    run_simple("localhost", 5001, app)

# Start Flask app in a thread
thread = Thread(target=run_app)
thread.start()
