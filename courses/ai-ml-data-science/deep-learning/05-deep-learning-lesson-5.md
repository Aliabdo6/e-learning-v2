#### **Lesson 5: Model Deployment**

**Objective:**
In this lesson, you’ll learn how to deploy deep learning models effectively. We’ll cover saving and loading models, deploying models to cloud platforms, and creating APIs for model inference.

---

#### **1. Saving and Loading Models**

Saving and loading models is essential for making models available for inference without retraining.

**a. Saving Models in TensorFlow/Keras:**

**1. Save Entire Model:**

You can save the entire model, including architecture, weights, and optimizer state.

```python
model.save('my_model.h5')
```

**2. Load Saved Model:**

To load a saved model, use:

```python
from tensorflow.keras.models import load_model
model = load_model('my_model.h5')
```

**b. Saving and Loading Weights Only:**

**1. Save Weights:**

```python
model.save_weights('my_model_weights.h5')
```

**2. Load Weights:**

```python
model.load_weights('my_model_weights.h5')
```

**c. Using TensorFlow SavedModel Format:**

This format is useful for TensorFlow Serving and other deployment tools.

**1. Save in SavedModel Format:**

```python
model.save('saved_model/my_model')
```

**2. Load from SavedModel Format:**

```python
model = tf.keras.models.load_model('saved_model/my_model')
```

---

#### **2. Deploying Models to Cloud Platforms**

Cloud platforms provide scalable solutions for deploying models. Here’s an overview of deploying models to popular platforms:

**a. Google Cloud Platform (GCP):**

**1. Google AI Platform:**

- Upload your model to Google Cloud Storage.
- Use Google AI Platform to deploy and manage the model.

**2. Deploy Model Using AI Platform:**

```bash
gcloud ai-platform models create my_model
gcloud ai-platform versions create v1 --model=my_model --origin=gs://my-bucket/my_model --runtime-version=2.5 --python-version=3.7
```

**b. Amazon Web Services (AWS):**

**1. Amazon SageMaker:**

- Use SageMaker to deploy your model for batch or real-time inference.

**2. Deploy Model Using SageMaker:**

```python
import sagemaker
from sagemaker import get_execution_role

role = get_execution_role()
model = sagemaker.model.Model(model_data='s3://my-bucket/my_model.tar.gz', role=role, image_uri='my-image')
predictor = model.deploy(instance_type='ml.m5.large')
```

**c. Microsoft Azure:**

**1. Azure Machine Learning:**

- Register and deploy your model using Azure Machine Learning service.

**2. Deploy Model Using Azure:**

```python
from azureml.core import Workspace, Model

ws = Workspace.from_config()
model = Model.register(model_path='my_model.pkl', model_name='my_model', workspace=ws)
```

---

#### **3. Creating APIs for Model Inference**

APIs enable applications to interact with models in real-time.

**a. Using Flask for a Simple API:**

**1. Create a Flask API:**

```python
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('my_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([data['input']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
```

**2. Deploy Flask API:**

You can deploy the Flask application using cloud services like Heroku, AWS Elastic Beanstalk, or Google Cloud Run.

**b. Using FastAPI for a Modern API:**

**1. Create a FastAPI Application:**

```python
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model('my_model.h5')

class PredictionInput(BaseModel):
    input: list

@app.post('/predict')
def predict(data: PredictionInput):
    prediction = model.predict([data.input])
    return {'prediction': prediction.tolist()}
```

**2. Run FastAPI Application:**

```bash
uvicorn main:app --reload
```

**c. Deploy FastAPI Application:**

Deploy using cloud services such as AWS, Google Cloud, or Azure. Platforms like Heroku also support FastAPI.

---

#### **4. Hands-On Exercise**

**Task:** Deploy a model to a cloud platform and create an API for it.

1. **Save and Load a Model:**

```python
model.save('my_model.h5')
# Load the model later
model = tf.keras.models.load_model('my_model.h5')
```

2. **Deploy Model Using Flask API:**

**a. Create a Flask API File (`app.py`):**

```python
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('my_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([data['input']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
```

**b. Run the Flask API Locally:**

```bash
python app.py
```

3. **Deploy Flask API to Heroku:**

- Create a `requirements.txt` and `Procfile` for Heroku.
- Use Heroku CLI to deploy the application.

---

#### **5. Summary and Next Steps**

In this lesson, we covered:

- Saving and loading models in TensorFlow/Keras.
- Deploying models to cloud platforms such as GCP, AWS, and Azure.
- Creating APIs for model inference using Flask and FastAPI.

**Next Lesson Preview:**
In Lesson 6, we will dive into model performance monitoring and management, exploring techniques for tracking and improving model performance over time.
