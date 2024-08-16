### Lesson 3: Advanced Text Preprocessing and Word Embeddings

#### Objectives:

- Learn advanced text preprocessing techniques.
- Understand the concept and applications of word embeddings.
- Implement word embeddings using popular methods.

---

#### 3.1 Advanced Text Preprocessing Techniques

**3.1.1 Handling Special Characters and HTML Tags:**
Remove or replace special characters and HTML tags from text.

```python
from bs4 import BeautifulSoup
import re

html_text = "<p>Hello, world!</p>"
soup = BeautifulSoup(html_text, "html.parser")
text = soup.get_text()

# Remove special characters
clean_text = re.sub(r'[^\w\s]', '', text)
print(clean_text)
```

**3.1.2 Normalization:**
Convert text to a consistent format, such as Unicode normalization.

```python
import unicodedata

text = "Café"
normalized_text = unicodedata.normalize('NFKD', text)
print(normalized_text)
```

**3.1.3 Spelling Correction:**
Correct spelling errors in text.

```python
from textblob import TextBlob

text = "I have a speling error."
corrected_text = TextBlob(text).correct()
print(corrected_text)
```

---

#### 3.2 Introduction to Word Embeddings

Word embeddings are dense vector representations of words that capture semantic meaning. Unlike traditional one-hot encoding, embeddings represent words in a continuous vector space where similar words have similar representations.

**3.2.1 Why Use Word Embeddings?**

- **Semantic Meaning:** Captures context and meaning.
- **Dimensionality Reduction:** Reduces the size of feature vectors compared to one-hot encoding.
- **Improved Performance:** Enhances the performance of NLP models.

**3.2.2 Popular Word Embedding Methods:**

- **Word2Vec**
- **GloVe (Global Vectors for Word Representation)**
- **FastText**

---

#### 3.3 Implementing Word Embeddings

**3.3.1 Word2Vec:**

Word2Vec is a method for learning vector representations of words from large text corpora. It has two main models:

- **Continuous Bag of Words (CBOW):** Predicts the current word based on surrounding words.
- **Skip-gram:** Predicts surrounding words based on the current word.

```python
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Sample data
sentences = [["natural", "language", "processing", "is", "fun"],
             ["word", "embeddings", "capture", "semantic", "meaning"]]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=0)

# Get vector for a word
vector = model.wv['word']
print(vector)

# Find similar words
similar_words = model.wv.most_similar('word')
print(similar_words)
```

**3.3.2 GloVe:**

GloVe is another popular word embedding method that captures global statistical information from a text corpus.

```python
from glove import Corpus, Glove

# Sample data
corpus = Corpus()
corpus.fit(sentences, window=3)

# Train GloVe model
glove = Glove(no_components=50, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

# Get vector for a word
vector = glove.word_vectors[glove.dictionary['word']]
print(vector)

# Find similar words
word = 'word'
similar_words = {k: v for k, v in glove.most_similar(word, number=5)}
print(similar_words)
```

**3.3.3 FastText:**

FastText is an extension of Word2Vec that considers subword information, which helps in handling out-of-vocabulary words better.

```python
from gensim.models import FastText

# Train FastText model
model = FastText(sentences, vector_size=50, window=3, min_count=1, sg=0)

# Get vector for a word
vector = model.wv['word']
print(vector)

# Find similar words
similar_words = model.wv.most_similar('word')
print(similar_words)
```

---

#### 3.4 Summary and Next Steps

In this lesson, we covered advanced text preprocessing techniques and explored word embeddings. We implemented word embeddings using Word2Vec, GloVe, and FastText.

**Next Steps:**

- In Lesson 4, we will explore text classification techniques and build a simple text classifier using the preprocessing and embedding techniques learned.
