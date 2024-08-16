### Lesson 4: Text Classification Techniques

#### Objectives:

- Understand the concept of text classification.
- Learn about various text classification techniques.
- Implement a simple text classifier using the techniques and tools covered so far.

---

#### 4.1 Introduction to Text Classification

Text classification is the process of assigning predefined categories to text documents. It is used in various applications such as spam detection, sentiment analysis, and topic categorization.

**Common Use Cases:**

- **Spam Detection:** Classify emails as spam or not spam.
- **Sentiment Analysis:** Determine the sentiment of a review (positive, negative, neutral).
- **Topic Classification:** Categorize news articles into topics like sports, politics, and technology.

---

#### 4.2 Text Classification Techniques

**4.2.1 Traditional Machine Learning Approaches:**

- **Bag of Words (BoW):** Represents text as a vector of word counts or frequencies.
- **TF-IDF:** Weighs the importance of words based on their frequency in a document and across the corpus.
- **Naive Bayes Classifier:** A probabilistic classifier based on Bayes' theorem.
- **Support Vector Machines (SVM):** A powerful classifier that finds the hyperplane that best separates classes.

**4.2.2 Deep Learning Approaches:**

- **Neural Networks:** Use embeddings and multiple layers to capture complex patterns.
- **Recurrent Neural Networks (RNNs):** Handle sequential data by maintaining context through hidden states.
- **Convolutional Neural Networks (CNNs):** Apply convolutional layers to text data for feature extraction.

---

#### 4.3 Implementing a Text Classifier

**4.3.1 Data Preparation:**

For this example, we'll use a sample dataset to build a text classifier.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Sample data
data = {
    'text': ['I love this movie', 'This movie is terrible', 'Fantastic film', 'Not good', 'I enjoyed it', 'Horrible movie'],
    'label': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
}

df = pd.DataFrame(data)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
```

**4.3.2 Text Preprocessing:**

Preprocess the text using the techniques from Lesson 2.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

**4.3.3 Building a Classifier:**

We will use a Naive Bayes classifier for this example.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Train the classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
```

**4.3.4 Using Word Embeddings with Deep Learning:**

For a more advanced classifier, we can use word embeddings and neural networks. Here's an example using Keras and TensorFlow:

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Tokenization and padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = max([len(x) for x in X_train_seq])
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=max_len))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_pad, y_train, epochs=5, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_pad, y_test)
print("Deep Learning Model Accuracy:", accuracy)
```

---

#### 4.4 Summary and Next Steps

In this lesson, we explored text classification techniques, from traditional machine learning methods to deep learning approaches. We implemented a basic text classifier and saw how to use word embeddings with a neural network.

**Next Steps:**

- In Lesson 5, we will explore advanced topics in NLP, including sequence labeling and named entity recognition (NER).
