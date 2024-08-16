### Lesson 5: Advanced NLP Topics: Sequence Labeling and Named Entity Recognition (NER)

#### Objectives:

- Understand the concept of sequence labeling and its applications.
- Learn about Named Entity Recognition (NER) and its importance.
- Implement sequence labeling and NER using popular libraries.

---

#### 5.1 Introduction to Sequence Labeling

Sequence labeling is a type of NLP task where each token in a sequence (e.g., sentence) is assigned a label. It is used in various applications such as part-of-speech tagging and named entity recognition.

**Common Sequence Labeling Tasks:**

- **Part-of-Speech (POS) Tagging:** Assign grammatical tags to words in a sentence (e.g., noun, verb).
- **Named Entity Recognition (NER):** Identify and classify entities in text (e.g., names of people, organizations).

---

#### 5.2 Named Entity Recognition (NER)

NER involves identifying and classifying entities in text into predefined categories such as person names, locations, organizations, dates, etc.

**NER Categories:**

- **PERSON:** Names of people.
- **ORG:** Names of organizations.
- **LOC:** Names of locations.
- **DATE:** Dates and times.
- **MISC:** Miscellaneous entities.

---

#### 5.3 Implementing NER with SpaCy

SpaCy is a popular library for NLP that provides pre-trained models for NER and other tasks.

**3.3.1 Installing SpaCy:**
If you haven't already installed SpaCy, do so with the following command:

```bash
pip install spacy
```

**3.3.2 Downloading SpaCy's Pre-trained Model:**

```bash
python -m spacy download en_core_web_sm
```

**3.3.3 Using SpaCy for NER:**

```python
import spacy

# Load SpaCy's pre-trained model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Barack Obama was born in Honolulu, Hawaii. He was the 44th President of the United States."

# Process text with SpaCy
doc = nlp(text)

# Extract and display entities
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
```

**Output:**

```
Barack Obama (PERSON)
Honolulu (LOC)
Hawaii (LOC)
44th (ORDINAL)
President (TITLE)
United States (GPE)
```

---

#### 5.4 Implementing Sequence Labeling with CRF

Conditional Random Fields (CRFs) are commonly used for sequence labeling tasks. We'll use the `sklearn-crfsuite` library for this.

**4.4.1 Installing sklearn-crfsuite:**

```bash
pip install sklearn-crfsuite
```

**4.4.2 Preparing Data:**
Format data for CRF by converting text into feature vectors.

```python
import nltk
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report

# Example sentence and labels
sentences = [['I', 'love', 'NLP'], ['NLP', 'is', 'fun']]
labels = [['O', 'O', 'B-NP'], ['B-NP', 'O', 'O']]

def extract_features(sentence):
    return [{'word': word} for word in sentence]

X_train = [extract_features(s) for s in sentences]
y_train = labels

# Train CRF model
crf = CRF()
crf.fit(X_train, y_train)

# Predict labels
y_pred = crf.predict(X_train)
print(flat_classification_report(y_train, y_pred))
```

**4.4.3 Feature Extraction:**
Create features for each word in a sentence.

```python
def word2features(sentence, i):
    word = sentence[i]
    features = {
        'word': word,
        'is_first': i == 0,
        'is_last': i == len(sentence) - 1,
        'prev_word': '' if i == 0 else sentence[i - 1],
        'next_word': '' if i == len(sentence) - 1 else sentence[i + 1]
    }
    return features
```

---

#### 5.5 Advanced NER with Transformers

For more advanced NER tasks, you can use transformer models like BERT.

**5.5.1 Installing Hugging Face's Transformers:**

```bash
pip install transformers
```

**5.5.2 Using BERT for NER:**

```python
from transformers import pipeline

# Load pre-trained BERT model for NER
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Sample text
text = "Barack Obama was born in Honolulu, Hawaii."

# Perform NER
results = ner_pipeline(text)

for entity in results:
    print(f"{entity['word']} ({entity['entity']})")
```

---

#### 5.6 Summary and Next Steps

In this lesson, we explored sequence labeling and Named Entity Recognition (NER). We implemented NER using SpaCy, sequence labeling using CRFs, and advanced NER with transformer models.

**Next Steps:**

- In Lesson 6, we will explore text generation techniques and applications, including language models and text generation frameworks.
