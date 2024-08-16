#### **Lesson 2: Types of Neural Networks**

**Objective:**
In this lesson, you'll learn about various types of neural networks and their applications. We'll cover Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and some advanced architectures.

---

#### **1. Convolutional Neural Networks (CNNs)**

CNNs are designed to process structured grid data, such as images. They use convolutional layers to automatically and adaptively learn spatial hierarchies of features.

**a. Key Components:**

- **Convolutional Layers:** Apply convolutional operations to detect features such as edges, textures, and patterns.
- **Pooling Layers:** Reduce the dimensionality of feature maps, often using max pooling or average pooling.
- **Fully Connected Layers:** Combine features from convolutional layers for classification or regression.

**b. Example: Building a CNN for Image Classification**

Here’s a simple example using TensorFlow and Keras:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

**c. Training the CNN:**

```python
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
```

**d. Evaluation:**

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")
```

---

#### **2. Recurrent Neural Networks (RNNs)**

RNNs are designed for sequential data, where the order of the data points is important. They have connections that loop back, allowing them to maintain a form of memory.

**a. Key Components:**

- **RNN Cells:** Basic units that process sequential data and pass information through time.
- **Long Short-Term Memory (LSTM) Cells:** Advanced RNN cells that mitigate the vanishing gradient problem and capture long-term dependencies.
- **Gated Recurrent Units (GRUs):** Simplified versions of LSTMs with fewer parameters.

**b. Example: Building an RNN for Sequence Prediction**

Here’s a simple example using LSTM cells:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(50, input_shape=(timesteps, features)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
```

**c. Training the RNN:**

```python
# Assume X_train_seq and y_train_seq are your sequential training data and labels
model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32)
```

**d. Evaluation:**

```python
test_loss = model.evaluate(X_test_seq, y_test_seq)
print(f"Test loss: {test_loss}")
```

---

#### **3. Advanced Neural Network Architectures**

**a. Generative Adversarial Networks (GANs):** Comprise two networks, a generator and a discriminator, that compete against each other. GANs are used for generating realistic images, videos, and more.

**b. Transformers:** State-of-the-art models for sequence data, especially in natural language processing. They use self-attention mechanisms to weigh the importance of different parts of the input.

**c. Autoencoders:** Used for unsupervised learning tasks like dimensionality reduction and feature learning. They consist of an encoder that compresses the data and a decoder that reconstructs it.

**d. Example: Building an Autoencoder**

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

input_data = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_data)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_data, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```

---

#### **4. Hands-On Exercise**

**Task:** Build and evaluate a CNN for the CIFAR-10 dataset (color image classification).

1. **Load the Data:**

```python
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
```

2. **Define and Train the Model:**

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
```

3. **Evaluate the Model:**

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")
```

---

#### **5. Summary and Next Steps**

In this lesson, we covered:

- Convolutional Neural Networks (CNNs) and their application to image data.
- Recurrent Neural Networks (RNNs) for sequential data.
- Advanced architectures like GANs, Transformers, and Autoencoders.

**Next Lesson Preview:**
In Lesson 3, we will explore model evaluation, hyperparameter tuning, and techniques for improving model performance.
