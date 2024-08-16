#### **Lesson 1: Introduction to Deep Learning**

**Objective:**
By the end of this lesson, you'll have a foundational understanding of deep learning, its applications, and how to set up your development environment for deep learning projects.

---

#### **1. What is Deep Learning?**

Deep learning is a subset of machine learning that involves neural networks with many layers (hence "deep") to analyze and learn from large amounts of data. It is used for tasks such as image and speech recognition, natural language processing, and more.

**Key Concepts:**

- **Neural Networks:** Composed of layers of nodes (neurons) that transform input data into output predictions.
- **Training:** The process of feeding data into the network and adjusting weights based on the error of predictions.
- **Activation Functions:** Functions like ReLU, Sigmoid, and Tanh that introduce non-linearity into the model.

**Applications of Deep Learning:**

- **Computer Vision:** Image classification, object detection, and segmentation.
- **Natural Language Processing:** Sentiment analysis, language translation, and chatbots.
- **Healthcare:** Disease prediction, medical image analysis.
- **Autonomous Vehicles:** Object detection, lane detection.

---

#### **2. Setting Up Your Development Environment**

To start with deep learning, you'll need to set up a suitable environment. Here’s a step-by-step guide:

**a. Install Python:**
Deep learning frameworks are often based on Python. Ensure you have Python installed (preferably Python 3.8 or later).

```bash
sudo apt-get update
sudo apt-get install python3 python3-pip
```

**b. Install Deep Learning Frameworks:**
We'll use TensorFlow and PyTorch, two popular deep learning frameworks. You can install them using pip.

```bash
pip install tensorflow
pip install torch torchvision
```

**c. Install Jupyter Notebook (optional but recommended):**
Jupyter Notebooks are great for interactive development and documentation.

```bash
pip install notebook
jupyter notebook
```

---

#### **3. Understanding Neural Networks**

**a. Basic Structure of a Neural Network:**

- **Input Layer:** Receives the input features.
- **Hidden Layers:** Perform computations and learn features. Each hidden layer contains neurons.
- **Output Layer:** Provides the final prediction or classification.

**b. Example: Simple Neural Network for Classification**

Here’s a basic example of how to define and train a neural network using TensorFlow and Keras:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a simple feedforward neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),  # Hidden layer with 64 neurons
    Dense(10, activation='softmax')                    # Output layer with 10 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()
```

**c. Training the Network:**

Training a network involves feeding data into the model and adjusting weights based on the error.

```python
# Assume X_train and y_train are your training data and labels
model.fit(X_train, y_train, epochs=5, batch_size=32)
```

**d. Making Predictions:**

```python
# Predict class probabilities for new data
predictions = model.predict(X_test)
```

---

#### **4. Key Concepts in Deep Learning**

**a. Loss Functions:** Measures how well the model's predictions match the actual labels (e.g., cross-entropy loss for classification).

**b. Optimizers:** Algorithms like Adam and SGD that adjust the model's weights during training.

**c. Overfitting and Regularization:** Techniques like dropout and L2 regularization prevent the model from fitting noise in the training data.

**d. Hyperparameters:** Parameters such as learning rate, number of epochs, and batch size that you tune to optimize model performance.

---

#### **5. Hands-On Exercise**

**Task:** Build a simple neural network for the MNIST dataset (handwritten digit classification).

1. **Load the Data:**

```python
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0
```

2. **Define and Train the Model:**

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
```

3. **Evaluate the Model:**

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")
```

---

#### **6. Summary and Next Steps**

In this lesson, we covered:

- The basics of deep learning and neural networks.
- Setting up your development environment.
- Building and training a simple neural network.

**Next Lesson Preview:**
In Lesson 2, we'll dive deeper into various types of neural networks (CNNs, RNNs) and their applications.
