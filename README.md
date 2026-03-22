# SpamShield — Federated Learning Based Email Spam Classifier

> End-to-end privacy-preserving spam detection using a Hybrid CNN-BiLSTM architecture,
> trained via Federated Averaging across 4 simulated clients and deployed as a
> containerized Flask application.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Model Architecture](#model-architecture)
- [Federated Learning Setup](#federated-learning-setup)
- [Training Results](#training-results)
- [Evaluation](#evaluation)
- [Deployment Pipeline](#deployment-pipeline)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)

---

## Problem Statement

Email spam detection is a well-studied NLP problem, but most solutions rely on
centralised training — all raw email data is sent to a single server. This poses
a significant privacy risk since emails contain sensitive personal and professional
information.

```
Traditional approach:

  Client A --[raw emails]--> Central Server --> Train Model
  Client B --[raw emails]--> Central Server
  Client C --[raw emails]--> Central Server

  Problem: raw data leaves the user's device
```

```
SpamShield approach (Federated Learning):

  Client A --> [train locally] --> [weights only] --> Server
  Client B --> [train locally] --> [weights only] --> Server (FedAvg)
  Client C --> [train locally] --> [weights only] --> Server
  Client D --> [train locally] --> [weights only] --> Server

  Raw emails never leave the client.
```

---

## Dataset

| Property             | Value                        |
|----------------------|------------------------------|
| Total emails         | 298,166                      |
| Spam (label = 1)     | 89,450  (30%)                |
| Safe (label = 0)     | 208,716 (70%)                |
| Train / Val / Test   | 208,716 / 44,725 / 44,725    |
| Unique tokens        | 682,713                      |
| Vocabulary used      | Top 20,000 tokens            |
| Max sequence length  | 300 tokens                   |
| Median email length  | 112 tokens                   |

### Label Distribution (all splits)

```
Train  ->  Safe: 69.99%  |  Spam: 30.01%
Valid  ->  Safe: 69.99%  |  Spam: 30.01%
Test   ->  Safe: 70.00%  |  Spam: 30.00%
```

Stratified splitting ensures consistent class distribution across all splits.

### Class Weights (to handle imbalance)

```
Safe (0) weight : 0.7143
Spam (1) weight : 1.6667

=> Spam errors penalised 2.33x more than safe errors during training
```

---

## System Architecture

```
Raw Email Text
      |
      v
+---------------------+
|  TextVectorization  |  --> top 20,000 tokens, max_len=300
+---------------------+
      |
      v
+---------------------+
|  Embedding Layer    |  --> (20000, 150) = 3M parameters
+---------------------+
      |
      v
+---------------------+
|  SpatialDropout1D   |  --> rate=0.2, drops entire embedding channels
+---------------------+
      |
   +--+--+--+
   |     |     |
   v     v     v
 k=3   k=5   k=7       <- parallel Conv1D branches
 128   128   128        <- filters per branch
   |     |     |
   +--+--+--+
      |
      v
+---------------------+
|  Concatenate        |  --> (batch, 300, 384)
+---------------------+
      |
      v
+---------------------+
|  MaxPooling1D       |  --> pool=2, stride=2 --> (batch, 150, 384)
+---------------------+
      |
      v
+---------------------+
|  Fusion Conv1D(1x1) |  --> (batch, 150, 128)  channel mixing
|  LayerNorm + ReLU   |
+---------------------+
      |
      v
+---------------------+
|  Bidirectional LSTM |  --> 128 units forward + 128 backward = 256
+---------------------+
      |
      v
+---------------------+
|  Dense(128) + ReLU  |
|  Dropout(0.4)       |
+---------------------+
      |
      v
+---------------------+
|  Dense(1) + Sigmoid |  --> spam probability [0.0, 1.0]
+---------------------+
      |
      v
  threshold=0.5
  >= 0.5  -->  SPAM
  <  0.5  -->  SAFE
```

---

## Model Architecture

| Layer                | Output Shape      | Parameters  |
|----------------------|-------------------|-------------|
| Input                | (None, 300)       | 0           |
| Embedding            | (None, 300, 150)  | 3,000,000   |
| SpatialDropout1D     | (None, 300, 150)  | 0           |
| Conv1D (k=3)         | (None, 300, 128)  | 57,728      |
| Conv1D (k=5)         | (None, 300, 128)  | 96,128      |
| Conv1D (k=7)         | (None, 300, 128)  | 134,528     |
| LayerNorm x3         | (None, 300, 128)  | 768         |
| Concatenate          | (None, 300, 384)  | 0           |
| MaxPooling1D         | (None, 150, 384)  | 0           |
| Fusion Conv1D (k=1)  | (None, 150, 128)  | 49,280      |
| LayerNorm            | (None, 150, 128)  | 256         |
| Bidirectional LSTM   | (None, 256)       | 263,168     |
| Dense(128)           | (None, 128)       | 32,896      |
| Dropout(0.4)         | (None, 128)       | 0           |
| Dense(1) sigmoid     | (None, 1)         | 129         |
| **Total**            |                   | **3,634,881** |

### Why Multi-Kernel CNN?

```
kernel=3  -->  captures trigrams       e.g. "click here now"
kernel=5  -->  captures 5-grams        e.g. "you have been selected as"
kernel=7  -->  captures longer context e.g. "limited time offer act now today"

Concatenating all three gives the model multi-scale feature awareness.
```

### Why Bidirectional LSTM?

```
Forward  LSTM:  "Congratulations ... winner ... click ... now"
Backward LSTM:  "now ... click ... winner ... Congratulations"

Both directions combined = richer understanding of email intent.
```

---

## Federated Learning Setup

### Client Split

| Client   | Samples | Spam    | Safe    |
|----------|---------|---------|---------|
| Client 0 | 52,179  | 29.9%   | 70.1%   |
| Client 1 | 52,179  | 30.0%   | 70.0%   |
| Client 2 | 52,179  | 30.1%   | 69.9%   |
| Client 3 | 52,179  | 30.0%   | 70.0%   |

### Training Configuration

| Parameter         | Value              |
|-------------------|--------------------|
| Algorithm         | FedAvg             |
| Rounds            | 20 (max)           |
| Local epochs      | 1 per round        |
| Batch size        | 256                |
| Early stopping    | 5 rounds patience  |
| Optimizer         | Adam (lr=1e-3)     |
| Loss              | Binary Crossentropy|
| Class weights     | Balanced           |

### FedAvg Aggregation

```
After each round:

  new_global_weight = sum( (client_size / total_size) * client_weight )
                      for each client

  Clients with more data have proportionally more influence
  on the global model update.
```

### Weight Divergence Tracking

```
Round 1  ->  C0: 25.06  C1: 25.46  C2: 25.05  C3: 25.20  (high initial divergence)
Round 6  ->  C0: 17.10  C1: 16.52  C2: 16.32  C3: 17.25  (converging)
Round 11 ->  C0: 12.30  C1: 12.61  C2: 13.14  C3: 10.95  (early stop triggered)

Decreasing divergence = clients are learning consistently.
```

---

## Training Results

### Server Validation Metrics Per Round

| Round | Val Loss | Val Accuracy | Val AUC | Note              |
|-------|----------|--------------|---------|-------------------|
| 1     | 0.1025   | 96.97%       | 0.9951  | new best          |
| 2     | 0.0582   | 97.94%       | 0.9972  | new best          |
| 3     | 0.0503   | 98.27%       | 0.9974  | new best          |
| 4     | 0.0473   | 98.49%       | 0.9971  | new best          |
| 5     | 0.0467   | 98.46%       | 0.9973  | new best          |
| 6     | 0.0463   | 98.55%       | 0.9971  | new best (saved)  |
| 7     | 0.0497   | 98.52%       | 0.9968  | no improvement 1  |
| 8     | 0.0529   | 98.63%       | 0.9957  | no improvement 2  |
| 9     | 0.0584   | 98.60%       | 0.9947  | no improvement 3  |
| 10    | 0.0590   | 98.57%       | 0.9950  | no improvement 4  |
| 11    | 0.0671   | 98.63%       | 0.9931  | early stop -> round 6 restored |

Best model: **Round 6** | Val Loss: 0.0463 | Val Accuracy: 98.55%

---

## Evaluation

### Test Set Results

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 98.55%  |
| AUC       | 0.9971  |
| Precision | 96.8%   |
| Recall    | 98.2%   |
| F1 Score  | 97.5%   |

### Confusion Matrix (Test Set, n=44,725)

```
                  Predicted SAFE    Predicted SPAM
Actual SAFE       31,091 (TN)       217   (FP)
Actual SPAM       244    (FN)       13,173 (TP)
```

### Prediction Distribution

```
Predicted SPAM : 13,634   (actual: 13,417)
Predicted SAFE : 31,091   (actual: 31,308)

Delta: 217 emails over-flagged as spam -- very low false positive rate
```

---

## Deployment Pipeline

```
Training (Jupyter + spam_env)
      |
      v
Save as SavedModel folder  -->  spam_classifier_deployment/
      |
      v
Wrap vectorizer + model     -->  deployment_model (raw text in -> prob out)
      |
      v
Convert to TFLite           -->  spam_model.tflite  (3.9 MB, 4x smaller)
  + quantization
  + SELECT_TF_OPS for LSTM
      |
      v
Flask API (app.py)
  POST /predict  -->  raw email text in, JSON response out
  GET  /health   -->  server status check
      |
      v
Docker container
  Base: python:3.10-slim
  Gunicorn: 1 worker, timeout 120s, preload
      |
      v
Render (Singapore region)
  Free tier web service
  Public URL: https://spamshield.onrender.com
```

### API Response Format

```json
{
  "probability"      : 94.2,
  "is_spam"          : true,
  "label"            : "SPAM",
  "risk_level"       : "HIGH",
  "risk_color"       : "#e74c3c",
  "found_keywords"   : ["click here", "free gift", "act now"],
  "keyword_count"    : 3,
  "highlighted_text" : "... <mark>click here</mark> ...",
  "email_length"     : 87
}
```

---

## Project Structure

```
deployment_tflite/
  |
  +-- app.py                        -> Flask backend, TFLite inference, keyword detection
  +-- spam_model.tflite             -> quantized TFLite model (3.9 MB)
  +-- vocabulary.json               -> word -> index mapping (20,000 tokens)
  +-- model_metadata.json           -> model info (accuracy, auc, rounds etc.)
  +-- requirements.txt              -> flask, tensorflow, numpy, gunicorn
  +-- Dockerfile                    -> python:3.10-slim, gunicorn entrypoint
  +-- render.yaml                   -> Render deployment config
  +-- templates/
        +-- index.html              -> frontend UI (vanilla HTML/CSS/JS)
```

---

## How to Run

### Locally (Python)

```bash
conda activate spam_env
cd deployment_tflite
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

### Locally (Docker)

```bash
docker build -t spamshield-tflite .
docker run -p 5000:5000 spamshield-tflite
# open http://localhost:5000
```

### Inference Example

```python
import requests

response = requests.post("http://localhost:5000/predict", json={
    "email": "Congratulations! You have won a free cash prize. Click here now."
})

print(response.json())
# -> { "probability": 97.3, "label": "SPAM", "risk_level": "HIGH", ... }
```

---

## Tech Stack

| Category          | Technology                              |
|-------------------|-----------------------------------------|
| Language          | Python 3.10                             |
| Deep Learning     | TensorFlow 2.13, Keras                  |
| Model Format      | TFLite (quantized, SELECT_TF_OPS)       |
| Federated Learning| Custom FedAvg implementation            |
| NLP               | TextVectorization, Embedding            |
| Web Framework     | Flask 3.0, Gunicorn                     |
| Containerisation  | Docker (python:3.10-slim)               |
| Cloud Deployment  | Render (Singapore, free tier)           |
| Environment       | Anaconda, spam_env (Python 3.10)        |

---

*Built as part of B.Tech Semester 5 project.*
