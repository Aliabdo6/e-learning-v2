#### **Lesson 3: Model Evaluation and Hyperparameter Tuning**

**Objective:**
In this lesson, you'll learn how to evaluate the performance of your deep learning models, tune hyperparameters, and apply techniques to improve model performance.

---

#### **1. Model Evaluation Metrics**

Evaluating model performance involves using various metrics depending on the type of problem (classification or regression).

**a. Classification Metrics:**

- **Accuracy:** The proportion of correctly classified instances.

  ```python
  from sklearn.metrics import accuracy_score
  accuracy = accuracy_score(y_true, y_pred)
  ```

- **Precision, Recall, and F1-Score:** Metrics that provide insights into how well the model performs on each class.

  ```python
  from sklearn.metrics import precision_score, recall_score, f1_score
  precision = precision_score(y_true, y_pred, average='weighted')
  recall = recall_score(y_true, y_pred, average='weighted')
  f1 = f1_score(y_true, y_pred, average='weighted')
  ```

- **Confusion Matrix:** A matrix showing the true vs. predicted classifications.
  ```python
  from sklearn.metrics import confusion_matrix
  conf_matrix = confusion_matrix(y_true, y_pred)
  ```

**b. Regression Metrics:**

- **Mean Absolute Error (MAE):** The average of the absolute errors.

  ```python
  from sklearn.metrics import mean_absolute_error
  mae = mean_absolute_error(y_true, y_pred)
  ```

- **Mean Squared Error (MSE) and Root Mean Squared Error (RMSE):** Measures of the average squared error.

  ```python
  from sklearn.metrics import mean_squared_error
  mse = mean_squared_error(y_true, y_pred)
  rmse = mse ** 0.5
  ```

- **R-squared (R²):** The proportion of variance explained by the model.
  ```python
  from sklearn.metrics import r2_score
  r2 = r2_score(y_true, y_pred)
  ```

---

#### **2. Hyperparameter Tuning**

Hyperparameters are parameters set before the training process that affect model performance. Tuning these parameters can significantly impact the model's effectiveness.

**a. Common Hyperparameters:**

- **Learning Rate:** Determines the size of the steps during optimization.
- **Batch Size:** The number of samples processed before updating the model.
- **Number of Epochs:** The number of times the model is trained on the entire dataset.
- **Number of Layers and Units:** The architecture of the network, including the number of hidden layers and neurons.

**b. Techniques for Hyperparameter Tuning:**

- **Grid Search:** Exhaustively searches over a predefined set of hyperparameters.

  ```python
  from sklearn.model_selection import GridSearchCV
  param_grid = {'learning_rate': [0.001, 0.01, 0.1], 'batch_size': [16, 32, 64]}
  grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
  grid_search.fit(X_train, y_train)
  ```

- **Random Search:** Randomly samples from a range of hyperparameters.

  ```python
  from sklearn.model_selection import RandomizedSearchCV
  param_dist = {'learning_rate': [0.001, 0.01, 0.1], 'batch_size': [16, 32, 64]}
  random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3)
  random_search.fit(X_train, y_train)
  ```

- **Bayesian Optimization:** Uses probabilistic models to find optimal hyperparameters.
  ```python
  from skopt import BayesSearchCV
  param_space = {'learning_rate': (1e-5, 1e-1, 'log-uniform'), 'batch_size': (16, 128)}
  bayes_search = BayesSearchCV(estimator=model, search_spaces=param_space, n_iter=50, cv=3)
  bayes_search.fit(X_train, y_train)
  ```

---

#### **3. Techniques for Improving Model Performance**

**a. Regularization:**

- **L1 and L2 Regularization:** Adds penalties to the loss function to prevent overfitting.

  ```python
  from tensorflow.keras.regularizers import l2
  model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
  ```

- **Dropout:** Randomly drops neurons during training to prevent overfitting.
  ```python
  from tensorflow.keras.layers import Dropout
  model.add(Dropout(0.5))
  ```

**b. Data Augmentation:**

- **Image Augmentation:** Applies transformations to images to artificially increase the dataset size.
  ```python
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2)
  datagen.fit(X_train)
  ```

**c. Learning Rate Scheduling:**

- **Adjusting Learning Rate:** Reducing the learning rate during training can help fine-tune the model.
  ```python
  from tensorflow.keras.callbacks import ReduceLROnPlateau
  lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
  model.fit(X_train, y_train, callbacks=[lr_scheduler])
  ```

**d. Early Stopping:**

- **Prevent Overfitting:** Stop training when the model's performance stops improving on a validation set.
  ```python
  from tensorflow.keras.callbacks import EarlyStopping
  early_stopping = EarlyStopping(monitor='val_loss', patience=5)
  model.fit(X_train, y_train, callbacks=[early_stopping])
  ```

---

#### **4. Hands-On Exercise**

**Task:** Tune hyperparameters and evaluate the performance of a CNN on the CIFAR-10 dataset.

1. **Define and Train the Initial Model:**

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

2. **Perform Grid Search for Hyperparameter Tuning:**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'batch_size': [32, 64],
    'epochs': [10, 20]
}

# Note: GridSearchCV does not support Keras models directly, so you may need to use KerasClassifier or KerasRegressor wrappers.
```

3. **Evaluate and Compare Model Performance:**

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")
```

---

#### **5. Summary and Next Steps**

In this lesson, we covered:

- Model evaluation metrics for classification and regression.
- Techniques for hyperparameter tuning, including grid search, random search, and Bayesian optimization.
- Methods to improve model performance, such as regularization, data augmentation, and learning rate scheduling.

**Next Lesson Preview:**
In Lesson 4, we will delve into advanced topics such as transfer learning, model ensembling, and working with large-scale datasets.
