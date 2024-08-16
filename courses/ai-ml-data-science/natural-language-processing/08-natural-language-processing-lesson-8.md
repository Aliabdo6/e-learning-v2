### Lesson 8: Sentiment Analysis Techniques and Applications

#### Objectives:

- Understand the concept and importance of sentiment analysis.
- Learn about various sentiment analysis techniques.
- Implement sentiment analysis using both traditional and deep learning methods.

---

#### 8.1 Introduction to Sentiment Analysis

Sentiment analysis involves determining the sentiment expressed in a text, typically classifying it as positive, negative, or neutral. It is widely used in social media monitoring, customer feedback analysis, and market research.

**Common Use Cases:**

- **Social Media Monitoring:** Analyze tweets and posts to gauge public sentiment about a brand or event.
- **Customer Feedback:** Assess reviews and feedback to understand customer satisfaction.
- **Market Research:** Analyze product reviews and news articles to track market trends.

---

#### 8.2 Sentiment Analysis Techniques

**8.2.1 Traditional Methods:**

- **Rule-Based Methods:** Use predefined rules and sentiment lexicons to classify sentiment.
- **Bag of Words (BoW) with Machine Learning:** Use machine learning algorithms with BoW features to classify sentiment.

**8.2.2 Deep Learning Methods:**

- **Neural Networks:** Use simple neural networks to model sentiment.
- **Recurrent Neural Networks (RNNs):** Handle sequential data for better context understanding.
- **Transformers:** Utilize advanced models like BERT for state-of-the-art sentiment classification.

---

#### 8.3 Implementing Sentiment Analysis

**8.3.1 Rule-Based Sentiment Analysis:**

A simple rule-based method involves using sentiment lexicons like VADER (Valence Aware Dictionary and sEntiment Reasoner).

**Using VADER with NLTK:**

```python
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Sample text
text = "I love this product! It works great and exceeded my expectations."

# Analyze sentiment
sentiment = sia.polarity_scores(text)
print(sentiment)
```

**8.3.2 Sentiment Analysis with Machine Learning:**

**Using Scikit-Learn with TF-IDF and Naive Bayes:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample data
texts = ["I love this movie", "This film is terrible", "Great product", "I did not like the movie"]
labels = ["positive", "negative", "positive", "negative"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = classifier.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

**8.3.3 Deep Learning Sentiment Analysis:**

**Using LSTM with Keras:**

```python
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample data
texts = ["I love this movie", "This film is terrible", "Great product", "I did not like the movie"]
labels = [1, 0, 1, 0]  # 1: Positive, 0: Negative

# Tokenization and padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X_pad = pad_sequences(X, maxlen=10)
y = np.array(labels)

# Model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=10))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_pad, y, epochs=5)

# Predict
X_test = ["I really enjoyed the movie", "I hated the product"]
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=10)
predictions = model.predict(X_test_pad)
print(predictions)
```

**8.3.4 Using Transformers for Sentiment Analysis:**

**Using BERT for Sentiment Analysis:**

```python
from transformers import pipeline

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

# Sample text
texts = ["I love this movie", "This film is terrible"]

# Analyze sentiment
results = sentiment_pipeline(texts)
print(results)
```

---

#### 8.4 Summary and Next Steps

In this lesson, we explored various sentiment analysis techniques, including rule-based methods, traditional machine learning approaches, and advanced deep learning methods. We implemented sentiment analysis using VADER, Scikit-Learn, LSTM, and BERT.

**Next Steps:**

- Experiment with different sentiment analysis models and techniques to find the best fit for your specific application.
- Explore more advanced topics such as sentiment analysis on multilingual text and integrating sentiment analysis into larger applications.
