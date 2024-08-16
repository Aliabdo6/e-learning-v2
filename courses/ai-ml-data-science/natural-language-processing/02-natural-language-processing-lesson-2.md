### Lesson 2: Text Preprocessing Techniques in NLP

#### Objectives:

- Understand the importance of text preprocessing.
- Learn various text preprocessing techniques.
- Implement text cleaning, tokenization, and feature extraction.
- Practice with code examples.

---

#### 2.1 Importance of Text Preprocessing

Text preprocessing is a crucial step in NLP as it helps in transforming raw text into a format that is suitable for analysis. Preprocessing improves the performance of NLP models by removing noise and irrelevant information.

---

#### 2.2 Text Cleaning Techniques

**2.2.1 Lowercasing:**
Convert all text to lowercase to ensure uniformity.

```python
text = "Natural Language Processing is Fascinating!"
clean_text = text.lower()
print(clean_text)
```

**2.2.2 Removing Punctuation:**
Strip out punctuation marks to focus on the words themselves.

```python
import string

text = "Hello, world! Welcome to NLP."
clean_text = text.translate(str.maketrans('', '', string.punctuation))
print(clean_text)
```

**2.2.3 Removing Numbers:**
Remove numbers if they are not relevant for the analysis.

```python
import re

text = "There are 3 apples."
clean_text = re.sub(r'\d+', '', text)
print(clean_text)
```

**2.2.4 Removing White Spaces:**
Remove extra white spaces from the text.

```python
text = "  This is a text with extra spaces.  "
clean_text = text.strip()
print(clean_text)
```

**2.2.5 Handling Contractions:**
Expand contractions to their full forms (e.g., "isn't" to "is not").

```python
from contractions import fix

text = "I can't believe it's happening!"
expanded_text = fix(text)
print(expanded_text)
```

---

#### 2.3 Tokenization

**2.3.1 Word Tokenization:**
Split text into individual words.

```python
from nltk.tokenize import word_tokenize

text = "Natural Language Processing is fascinating!"
tokens = word_tokenize(text)
print(tokens)
```

**2.3.2 Sentence Tokenization:**
Split text into sentences.

```python
from nltk.tokenize import sent_tokenize

text = "Natural Language Processing is fascinating! It has many applications."
sentences = sent_tokenize(text)
print(sentences)
```

**2.3.3 Tokenization with SpaCy:**

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Natural Language Processing is fascinating!")
tokens = [token.text for token in doc]
print(tokens)
```

---

#### 2.4 Removing Stop Words

Stop words are common words that may be irrelevant for analysis.

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print(filtered_tokens)
```

---

#### 2.5 Stemming and Lemmatization

**2.5.1 Stemming:**
Reduces words to their root form.

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["running", "runner", "ran"]
stems = [stemmer.stem(word) for word in words]
print(stems)
```

**2.5.2 Lemmatization:**
Reduces words to their base or dictionary form.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("running runners ran")
lemmas = [token.lemma_ for token in doc]
print(lemmas)
```

---

#### 2.6 Feature Extraction

**2.6.1 Bag of Words (BoW):**
Convert text into a matrix of token counts.

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = ["I love NLP.", "NLP is fun."]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

**2.6.2 Term Frequency-Inverse Document Frequency (TF-IDF):**
Evaluate the importance of words.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

---

#### 2.7 Summary and Next Steps

In this lesson, we explored various text preprocessing techniques, including text cleaning, tokenization, stop word removal, stemming, lemmatization, and feature extraction.

**Next Steps:**

- In Lesson 3, we will delve into more advanced text preprocessing techniques and explore the concept of word embeddings.
