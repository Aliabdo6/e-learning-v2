### Lesson 7: Text Summarization Techniques and Applications

#### Objectives:

- Understand the concept and importance of text summarization.
- Learn about different text summarization techniques.
- Implement text summarization using extractive and abstractive methods.

---

#### 7.1 Introduction to Text Summarization

Text summarization involves creating a concise summary of a larger text while preserving its essential information. It is used in applications such as news summarization, document summarization, and content aggregation.

**Common Use Cases:**

- **News Summarization:** Provide brief summaries of news articles.
- **Document Summarization:** Extract key points from lengthy reports or papers.
- **Content Aggregation:** Create summaries of multiple articles on a similar topic.

---

#### 7.2 Text Summarization Techniques

**7.2.1 Extractive Summarization:**

Extractive summarization involves selecting and combining key sentences or phrases directly from the text to form a summary. It does not generate new content but extracts relevant parts.

**Techniques for Extractive Summarization:**

- **Frequency-Based Methods:** Use word frequency to identify important sentences.
- **Graph-Based Methods:** Use algorithms like TextRank to rank sentences based on their importance.

**7.2.2 Abstractive Summarization:**

Abstractive summarization involves generating new sentences that capture the essence of the original text. It may use language models to create summaries that are not directly extracted from the text.

**Techniques for Abstractive Summarization:**

- **Sequence-to-Sequence Models:** Use encoder-decoder architectures to generate summaries.
- **Transformers:** Use models like BERT and GPT for generating summaries.

---

#### 7.3 Implementing Extractive Summarization

**7.3.1 Using Gensim for Extractive Summarization:**

Gensim provides a simple way to perform extractive summarization using the TextRank algorithm.

```python
from gensim.summarization import summarize

# Sample text
text = """
The quick brown fox jumps over the lazy dog. The quick brown fox is very agile and fast. The lazy dog, on the other hand, prefers to lie in the sun and relax.
The quick brown fox often teases the lazy dog by running circles around him. The lazy dog gets annoyed but doesn't move much.
One day, the quick brown fox ran so fast that he disappeared into the forest, leaving the lazy dog behind.
"""

# Generate summary
summary = summarize(text, ratio=0.5)
print(summary)
```

**7.3.2 Using NLTK for Extractive Summarization:**

NLTK can also be used for extractive summarization by scoring sentences based on term frequency.

```python
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re

nltk.download('punkt')
nltk.download('stopwords')

def extractive_summary(text, num_sentences=2):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\w+', text.lower())
    word_freq = Counter(word for word in words if word not in stop_words)

    sentence_scores = {}
    for sentence in sentences:
        sentence_words = re.findall(r'\w+', sentence.lower())
        sentence_scores[sentence] = sum(word_freq.get(word, 0) for word in sentence_words)

    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    return ' '.join(summary_sentences)

text = """
The quick brown fox jumps over the lazy dog. The quick brown fox is very agile and fast. The lazy dog, on the other hand, prefers to lie in the sun and relax.
The quick brown fox often teases the lazy dog by running circles around him. The lazy dog gets annoyed but doesn't move much.
One day, the quick brown fox ran so fast that he disappeared into the forest, leaving the lazy dog behind.
"""
print(extractive_summary(text))
```

---

#### 7.4 Implementing Abstractive Summarization

**7.4.1 Using Hugging Face Transformers for Abstractive Summarization:**

Hugging Face’s Transformers library provides pre-trained models for abstractive summarization.

```python
from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization")

# Sample text
text = """
The quick brown fox jumps over the lazy dog. The quick brown fox is very agile and fast. The lazy dog, on the other hand, prefers to lie in the sun and relax.
The quick brown fox often teases the lazy dog by running circles around him. The lazy dog gets annoyed but doesn't move much.
One day, the quick brown fox ran so fast that he disappeared into the forest, leaving the lazy dog behind.
"""

# Generate summary
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(summary[0]['summary_text'])
```

**7.4.2 Using BERTSUM for Abstractive Summarization:**

BERTSUM is a model specifically designed for extractive and abstractive summarization tasks using BERT.

```python
from transformers import BartTokenizer, BartForConditionalGeneration

# Load BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Encode the text
inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)

# Generate summary
summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)
```

---

#### 7.5 Summary and Next Steps

In this lesson, we explored text summarization techniques, including extractive and abstractive summarization. We implemented summarization using Gensim, NLTK, and Hugging Face Transformers.

**Next Steps:**

- In Lesson 8, we will explore sentiment analysis techniques and applications, including both traditional and deep learning approaches.
