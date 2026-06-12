# Anomaly Detection System — Real-Time Network Intrusion Detection

A production-ready system that captures live network traffic, extracts statistical flow features, and classifies each flow into one of 10 attack categories (or benign) using a machine learning model trained on the **CIC-UNSW-NB15** dataset.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [The Dataset: CIC-UNSW-NB15](#2-the-dataset-cic-unsw-nb15)
   - 2.1 [What Is a "Network Flow"?](#21-what-is-a-network-flow)
   - 2.2 [What Is CICFlowMeter?](#22-what-is-cicflowmeter)
   - 2.3 [Dataset Statistics](#23-dataset-statistics)
   - 2.4 [Attack Categories (10 Classes)](#24-attack-categories-10-classes)
   - 2.5 [Why This Dataset?](#25-why-this-dataset)
3. [System Architecture](#3-system-architecture)
   - 3.1 [Pipeline A: Training](#31-pipeline-a-training)
   - 3.2 [Pipeline B: Real-Time Deployment](#32-pipeline-b-real-time-deployment)
   - 3.3 [How the Two Pipelines Connect](#33-how-the-two-pipelines-connect)
4. [Data Preprocessing](#4-data-preprocessing)
   - 4.1 [Why Do We Need Preprocessing?](#41-why-do-we-need-preprocessing)
   - 4.2 [Handling Missing Values and Infinity](#42-handling-missing-values-and-infinity)
   - 4.3 [Categorical Features](#43-categorical-features)
   - 4.4 [Feature Scaling](#44-feature-scaling)
   - 4.5 [Handling Class Imbalance](#45-handling-class-imbalance)
   - 4.6 [Train / Validation / Test Split](#46-train--validation--test-split)
5. [Machine Learning Models](#5-machine-learning-models)
   - 5.1 [XGBoost (Primary Model)](#51-xgboost-primary-model)
   - 5.2 [Multi-Layer Perceptron (Deep Learning Baseline)](#52-multi-layer-perceptron-deep-learning-baseline)
   - 5.3 [Stacking Ensemble](#53-stacking-ensemble)
   - 5.4 [Why Not Use a Transformer for Sequences?](#54-why-not-use-a-transformer-for-sequences)
6. [Evaluation Metrics](#6-evaluation-metrics)
   - 6.1 [Accuracy](#61-accuracy)
   - 6.2 [Precision, Recall, and F1-Score](#62-precision-recall-and-f1-score)
   - 6.3 [ROC-AUC](#63-roc-auc)
   - 6.4 [Confusion Matrix](#64-confusion-matrix)
   - 6.5 [Inference Latency](#65-inference-latency)
7. [Real-Time Deployment](#7-real-time-deployment)
   - 7.1 [Capturing Live Traffic](#71-capturing-live-traffic)
   - 7.2 [Extracting Flow Features](#72-extracting-flow-features)
   - 7.3 [FastAPI Inference Server](#73-fastapi-inference-server)
   - 7.4 [Example API Request & Response](#74-example-api-request--response)
8. [Project Structure](#8-project-structure)
9. [How to Run Everything](#9-how-to-run-everything)
   - 9.1 [Setup](#91-setup)
   - 9.2 [Preprocessing & Training (Local or Colab)](#92-preprocessing--training-local-or-colab)
   - 9.3 [Deploy the API](#93-deploy-the-api)
10. [Results & Discussion](#10-results--discussion)
    - 10.1 [Expected Outcomes](#101-expected-outcomes)
    - 10.2 [Model Comparison](#102-model-comparison)
11. [Future Work](#11-future-work)
12. [References](#12-references)

---

## 1. What Is This Project?

This project builds a **real-time Network Intrusion Detection System (NIDS)**. In plain English:

> Every time data travels across a network, it creates "traffic." Some of that traffic is normal (browsing the web, watching a video). Some of it is malicious — someone trying to break into a server, flood it with garbage data, or steal information. Our system watches this traffic and flags the bad stuff automatically.

The project has two main parts:

| Part | What It Does | Where It Runs |
|------|-------------|---------------|
| **Training** | Teaches a computer to recognize 10 types of network attacks using a labeled dataset | Google Colab / your laptop |
| **Deployment** | Takes the trained model and uses it on live network traffic to make real-time predictions | A server with FastAPI |

Think of Part 1 as going to school and Part 2 as starting the job.

---

## 2. The Dataset: CIC-UNSW-NB15

Machine learning needs examples. We don't have real attack data lying around, so we use a public, academic dataset called **CIC-UNSW-NB15** created by the Canadian Institute for Cybersecurity at the University of New Brunswick (UNB) [1].

### 2.1 What Is a "Network Flow"?

A **network flow** is the record of a conversation between two computers on a network.

Imagine Computer A sends a request to Computer B. The conversation might be:

1. A sends 3 packets to B (total 1,200 bytes)
2. B sends 2 packets back to A (total 800 bytes)
3. The whole thing takes 0.5 seconds
4. They use the TCP protocol on port 80 (HTTP)

A flow is exactly this: a summary of that conversation. It does **not** contain the actual content of the messages (no passwords, no credit card numbers). It only contains **statistics** about the conversation.

### 2.2 What Is CICFlowMeter?

CICFlowMeter [2] is a tool developed by UNB that:

1. Reads raw network traffic (a `.pcap` file — think of it as an audio recording of network traffic)
2. Groups packets into flows (by matching source IP, destination IP, ports, and protocol)
3. Computes **84 statistical features** for each flow

These features include things like:

| Feature Category | Examples |
|-----------------|----------|
| **Duration** | How long the flow lasted (seconds) |
| **Packet count** | Number of packets sent forward and backward |
| **Byte count** | Total bytes sent forward and backward |
| **Packet length** | Min, max, mean, std of packet sizes |
| **Packet inter-arrival time** | Time gaps between packets (min, max, mean, std) |
| **Flags** | TCP flags (SYN, ACK, FIN, RST, etc.) |
| **Idle / Active** | How long the flow was idle vs active |

All 84 features are **numerical** — no text, no images, no sequences of requests. Each flow becomes one row of numbers. This is called **tabular data** (like a spreadsheet).

### 2.3 Dataset Statistics

The CIC-UNSW-NB15 dataset is built on top of the original UNSW-NB15 [3] dataset. The original dataset was created in 2015 at the Australian Centre for Cyber Security using a tool called IXIA PerfectStorm to generate realistic normal and attack traffic. The raw packets (100 GB of `.pcap` files) were then processed through CICFlowMeter to produce the final CSV files.

| Category | Original UNSW-NB15 | CICFlowMeter Extraction | CIC-UNSW-NB15 (Final) |
|----------|------------------:|------------------------:|----------------------:|
| Benign | 2,218,764 | 3,450,658 | **358,332** |
| Analysis | 2,677 | 385 | **385** |
| Backdoor | 2,329 | 452 | **452** |
| DoS | 16,353 | 4,467 | **4,467** |
| Exploits | 44,525 | 30,951 | **30,951** |
| Fuzzers | 24,246 | 29,613 | **29,613** |
| Generic | 215,481 | 4,632 | **4,632** |
| Reconnaissance | 13,987 | 16,735 | **16,735** |
| Shellcode | 1,511 | 2,102 | **2,102** |
| Worms | 174 | 246 | **246** |
| **Total** | ~2.5M | ~3.5M | **~448,000** |

The final dataset has about **448,000 flows** with an 80:20 ratio of benign to malicious traffic (to reflect real-world conditions where most traffic is normal).

### 2.4 Attack Categories (10 Classes)

| # | Name | What It Means (Simple Explanation) | Real-World Example |
|---|------|-------------------------------------|-------------------|
| 0 | **Benign** | Normal, safe traffic | Opening google.com |
| 1 | **Analysis** | Attackers probe the system to understand its weaknesses (port scanning, vulnerability scanning) | Sending carefully crafted requests to see how the server responds |
| 2 | **Backdoor** | A hidden entry point that bypasses normal authentication | Software that lets the attacker log in without a password |
| 3 | **DoS** (Denial of Service) | Flooding a server with traffic so it can't serve real users | Millions of fake requests that make a website crash |
| 4 | **Exploits** | Taking advantage of a software bug to gain control | Sending a carefully formatted message that crashes the program and runs attacker's code |
| 5 | **Fuzzers** | Sending random, malformed data to find bugs | Sending garbage input to see if the application breaks |
| 6 | **Generic** | Attacks against encryption systems (cryptography) | Trying to break a hashing algorithm |
| 7 | **Reconnaissance** | Gathering information about the target before an attack | Finding out what software versions the server runs |
| 8 | **Shellcode** | Small piece of code injected into a running program to take control | Like Exploits but specifically about injecting executable code |
| 9 | **Worms** | Self-replicating malware that spreads automatically | A virus that copies itself to other computers without human help |

### 2.5 Why This Dataset?

There are older datasets for intrusion detection:

| Dataset | Year | Flows | Classes | Problems |
|---------|------|-------|---------|----------|
| KDD99 | 1999 | ~5M | 5 | 25+ years old, does not represent modern traffic |
| NSL-KDD | 2009 | ~150K | 5 | Reduced version of KDD99, still outdated |
| CIC-IDS2017 | 2017 | ~2.8M | 15 | Good but very large, complex preprocessing |
| **CIC-UNSW-NB15** | **2024** | **~448K** | **10** | **Modern traffic, well-balanced size, clean preprocessing, recent** |

We chose **CIC-UNSW-NB15** because:
- **Modern**: Traffic generated in 2015, features re-extracted in 2024 with CICFlowMeter
- **Size**: 448K flows is large enough to train good models but small enough to run on a laptop
- **Classes**: 10 categories cover a wide range of real attack types
- **Clean**: Comes pre-processed with `Data.csv` + `Label.csv` (80:20 ratio)
- **Recent citation**: Published with a 2024 paper [4], making it academically current

---

## 3. System Architecture

### 3.1 Pipeline A: Training

```
┌──────────────────────┐
│   CIC-UNSW-NB15 CSV  │
│   (Data.csv, 448K    │
│    flows, 84 feat.)  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    Preprocessing     │
│  ┌────────────────┐  │
│  │ • Clean NaN/Inf│  │
│  │ • Encode cats  │  │
│  │ • Scale (MLP)  │  │
│  │ • SMOTE (opt.) │  │
│  │ • Train/Val/   │  │
│  │   Test split   │  │
│  └────────────────┘  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────┐
│          Model Training              │
│                                      │
│  ┌────────┐  ┌──────┐  ┌─────────┐  │
│  │XGBoost │  │ MLP  │  │Stacking │  │
│  │(CPU)   │  │(GPU) │  │Ensemble │  │
│  └────┬───┘  └──┬───┘  └────┬────┘  │
│       │         │           │       │
│       ▼         ▼           ▼       │
│  ┌──────────────────────────────┐  │
│  │    Evaluation & Comparison   │  │
│  │  (Acc, Prec, Recall, F1,    │  │
│  │   ROC-AUC, Confusion Matrix, │  │
│  │   Latency)                   │  │
│  └──────────────┬───────────────┘  │
└─────────────────┼──────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │  Best Model    │
         │  (e.g. XGBoost │
         │   + ONNX/PKL)  │
         └────────────────┘
```

### 3.2 Pipeline B: Real-Time Deployment

```
┌──────────────────────┐
│   Live Network or    │
│   Captured PCAP File │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   CICFlowMeter       │
│   (Python library)   │
│                      │
│  • Reads packets     │
│  • Groups into flows │
│  • Computes 84       │
│    features          │
└──────────┬───────────┘
           │
           ▼ (feature vector, same format as training data)
┌──────────────────────┐
│   Preprocessing      │
│   (same transforms   │
│    as training)      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Trained Model      │
│   (load from disk)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   FastAPI Server     │
│                      │
│  POST /predict       │
│  → JSON response     │
└──────────────────────┘
```

### 3.3 How the Two Pipelines Connect

The key insight is that **both pipelines process the same 84 features in the same format**:

- **During training**: We use the pre-computed CSV from CIC-UNSW-NB15. The model learns patterns: "If feature X is high and feature Y is low, it's probably a DoS attack."
- **During deployment**: We use CICFlowMeter on live traffic to compute the **same 84 features**. The same model then applies what it learned.

This is why choosing CICFlowMeter as the feature extractor is important — it ensures the training data and the live data are in the same "language."

---

## 4. Data Preprocessing

### 4.1 Why Do We Need Preprocessing?

Raw data is messy. Think of it like cooking: you don't just throw unwashed vegetables into a pot. You clean, peel, and chop them first. Preprocessing is the same for data.

Raw CSV files may have:
- **Missing values**: Some cells are empty (NaN = "Not a Number")
- **Infinity**: Some calculations produce numbers too large to store (Inf)
- **Categorical text**: Columns like "protocol = TCP" that need to be converted to numbers
- **Different scales**: One feature ranges from 0 to 1, another from 0 to 1,000,000
- **Uneven classes**: 80% of data is "Benign," only 0.05% is "Worms"

If we skip preprocessing, the model either fails to train or learns incorrect patterns.

### 4.2 Handling Missing Values and Infinity

Machine learning models cannot process NaN or Inf values. They simply crash.

| Problem | Why It Happens | What We Do |
|---------|---------------|------------|
| NaN | Division by zero, missing sensor data | Replace with column mean (mean imputation) |
| Inf | Very large numbers from packet timing | Replace with column max |
| Negative values where not expected | Data errors | Cap at 0 or investigate |

**Code logic**: Scan every column. If NaN → fill with mean of that column. If Inf → replace with max finite value. If all values are NaN (useless column) → drop the column entirely.

### 4.3 Categorical Features

Some columns contain text, not numbers:

```
proto: TCP, UDP, ICMP, ...
state: FIN, CON, RST, ...
service: HTTP, DNS, FTP, ...
```

A model can only multiply numbers, not compare strings. We need to convert text to numbers.

**Two approaches**:

| Method | What It Does | Analogy | When to Use |
|--------|-------------|---------|-------------|
| **Label Encoding** | Assigns each category a number: TCP → 0, UDP → 1, ICMP → 2 | Giving each student a student ID | Works for tree models (XGBoost) |
| **One-Hot Encoding** | Creates a new column for each category: `is_TCP` (0/1), `is_UDP` (0/1), `is_ICMP` (0/1) | Multiple choice: pick one option | Required for neural networks (MLP) |

**Our choice**: Use **Label Encoding** for XGBoost (trees handle ordinal encoding naturally). Use **One-Hot Encoding** for MLP (neural networks need binary inputs).

**Important**: We must save the encoding mapping during training and reuse it during deployment. If the training data has TCP → 0, UDP → 1, then live data must use the same mapping.

### 4.4 Feature Scaling

Look at two features from the dataset:

| Feature | Typical Range |
|---------|-------------|
| Flow Duration | 0 – 10,000,000 (microseconds) |
| Fwd Packets/s | 0 – 1,000 |

The duration feature is **10,000x larger** than the packets-per-second feature. 

- **For tree models (XGBoost)**: This does not matter. Trees make decisions by comparing values ("is duration > 500?") regardless of scale. **No scaling needed.**
- **For neural networks (MLP)**: This matters a lot. Neural networks use gradient descent, which works best when all features are roughly the same scale (usually 0 to 1 or -1 to 1). Without scaling, the duration feature dominates the learning process. **Scaling is required.**

**Standard Scaling (Z-score)**: `x_new = (x - mean) / standard_deviation`

This transforms each feature so its mean is 0 and its standard deviation is 1. All features now live in approximately the same range.

**Important**: Compute the mean and std from the **training set only**. Apply the same transformation to validation and test sets. Never use test set statistics — that would be "cheating" (data leakage).

### 4.5 Handling Class Imbalance

Look at the class distribution again:

```
Benign:       358,332 (80.0%)
Exploits:      30,951 (6.9%)
Fuzzers:       29,613 (6.6%)
Reconnaissance: 16,735 (3.7%)
Generic:        4,632 (1.0%)
DoS:            4,467 (1.0%)
Shellcode:      2,102 (0.5%)
Backdoor:         452 (0.1%)
Analysis:         385 (0.1%)
Worms:            246 (0.05%)
```

This is called **class imbalance**. If we train a model on this raw data, it will learn to predict "Benign" for everything and still be 80% accurate. But it would completely miss Worms, Backdoor, and Analysis attacks.

**Why this happens**: The model optimizes for overall accuracy. Ignoring rare classes barely hurts the average accuracy, so the model "takes the easy way out."

**Solutions**:

| Solution | How It Works | Our Plan |
|----------|-------------|----------|
| **class_weight** | Tell the model: "Making a mistake on Worms costs 100x more than making a mistake on Benign" | ✅ Used for XGBoost (built-in) |
| **Oversampling (SMOTE)** | Create synthetic copies of rare classes by interpolating between existing examples | ✅ Used for MLP |
| **Undersampling** | Remove random samples from the majority class | ❌ We lose too much data |
| **Stratified Split** | Ensure each train/val/test set has the same class proportions as the original | ✅ Always used |

**What is SMOTE?** (Synthetic Minority Over-sampling Technique)

SMOTE creates new "fake" examples of rare classes. It works by:
1. Pick a rare-class example
2. Find its nearest neighbors of the same class
3. Draw a line between the example and its neighbor
4. Place a new point somewhere on that line

This gives the model more examples of rare classes to learn from, without simply duplicating existing data (which causes overfitting).

### 4.6 Train / Validation / Test Split

We split the data into three parts:

| Set | % of Data | Purpose |
|-----|-----------|---------|
| **Training** | 70% | The model learns from this data. It sees these examples and adjusts its parameters. |
| **Validation** | 15% | We use this to tune hyperparameters (e.g., tree depth, learning rate) and decide when to stop training. The model does NOT learn from this. |
| **Test** | 15% | We evaluate the final model here. This data is completely unseen until the very end. It simulates how the model will perform on real, new data. |

**Why not 100% training?** If you test on the same data you trained on, you get 100% accuracy — but the model just memorized the answers (overfitting). It will fail on real data. The validation and test sets are "honest" evaluations.

**Stratified Split**: Each of the three sets preserves the same class proportions. If the full dataset is 80% Benign, then training is 80% Benign, validation is 80% Benign, and test is 80% Benign. This prevents the test set from (by random chance) containing no Worms examples.

---

## 5. Machine Learning Models

### 5.1 XGBoost (Primary Model)

#### 5.1.1 What Is XGBoost?

XGBoost stands for **Extreme Gradient Boosting**. Let's break that down:

- **Decision Tree**: A flowchart. "Is feature A > 5? If yes → go left, if no → go right." Each final leaf contains a prediction.
- **Boosting**: Instead of building one perfect tree, build many small, weak trees in sequence. Each new tree focuses on fixing the mistakes of all previous trees combined.
- **Gradient**: The "mistakes" are measured using a mathematical function (gradient), which tells each new tree exactly which direction to improve.
- **Extreme**: Optimized for speed and performance (parallel processing, regularization).

Think of it like a group project:
1. Person 1 makes an initial guess (predict Benign for everything).
2. Person 2 looks at Person 1's mistakes and tries to correct them.
3. Person 3 looks at the combined output and corrects further.
4. ... continue for 500 iterations.
5. The final output is the sum of everyone's work.

#### 5.1.2 Why XGBoost?

| Reason | Explanation |
|--------|-------------|
| **Best-in-class for tabular data** | On spreadsheets of numbers, gradient-boosted trees consistently outperform neural networks [6] |
| **Handles missing values** | XGBoost learns the best "direction" when a value is missing — no need to pre-fill |
| **Feature importance** | XGBoost can tell us which features (e.g., "Packet Length Mean") are most useful for detecting attacks |
| **No scaling needed** | Tree models don't care about the scale of features |
| **Fast inference** | A single prediction takes microseconds — critical for real-time IDS |
| **Works on CPU** | No GPU required, runs on any laptop or server |

#### 5.1.3 Key Hyperparameters

Hyperparameters are settings we choose **before** training (not learned by the model). Think of them as the settings on a cooking recipe:

| Parameter | What It Controls | Typical Range | Effect of Increasing |
|-----------|-----------------|---------------|---------------------|
| `n_estimators` | Number of trees in the ensemble | 100 – 1000 | More trees = better accuracy, but slower and risk of overfitting |
| `max_depth` | How deep each tree can grow | 3 – 10 | Deeper trees = more complex patterns, but overfitting |
| `learning_rate` | How much each new tree contributes | 0.01 – 0.3 | Smaller = more careful, better generalization, needs more trees |
| `subsample` | Fraction of data used per tree | 0.5 – 1.0 | Less = more randomness, reduces overfitting |
| `colsample_bytree` | Fraction of features used per tree | 0.5 – 1.0 | Less = more randomness, reduces overfitting |
| `scale_pos_weight` | How much to penalize mistakes on rare classes | Auto-calculated | Higher = more focus on rare classes (handles imbalance) |

**Tuning strategy**: We use **GridSearchCV** or **RandomizedSearchCV** to try many combinations and pick the best based on validation F1-score.

#### 5.1.4 Multi-Class Output

XGBoost with `objective='multi:softprob'` outputs a **probability distribution** over all 10 classes:

```
{
  "Benign":         0.89,
  "Exploits":       0.05,
  "Fuzzers":        0.03,
  "Reconnaissance": 0.01,
  "DoS":            0.01,
  "Generic":        0.005,
  "Shellcode":      0.002,
  "Backdoor":       0.001,
  "Analysis":       0.001,
  "Worms":          0.000
}
```

The final prediction is the class with the highest probability (`argmax`).

### 5.2 Multi-Layer Perceptron (Deep Learning Baseline)

#### 5.2.1 What Is an MLP?

A **Multi-Layer Perceptron** is the simplest form of neural network. It is a mathematical function composed of layers:

```
Input (84 features)
    │
    ▼
Dense Layer 1 (256 neurons) → ReLU → Dropout(0.3)
    │
    ▼
Dense Layer 2 (128 neurons) → ReLU → Dropout(0.3)
    │
    ▼
Dense Layer 3 (10 neurons) → Softmax
    │
    ▼
Output (probabilities for 10 classes)
```

Each "Dense Layer" is a set of weights and biases. During training, the network adjusts these weights to minimize the error between its predictions and the true labels.

**Key components**:
- **Dense (Fully Connected)**: Every neuron connects to every neuron in the previous layer
- **ReLU** (Rectified Linear Unit): An activation function that sets negative values to 0. `f(x) = max(0, x)`. It introduces non-linearity — without it, stacking layers would be mathematically equivalent to a single layer
- **Dropout**: During training, randomly "turn off" 30% of neurons. This prevents the network from relying too heavily on any single neuron (regularization)
- **Softmax**: Converts raw scores into probabilities that sum to 1.0

#### 5.2.2 Why an MLP?

| Reason | Explanation |
|--------|-------------|
| **Baseline for deep learning** | If a simple MLP doesn't work well, a more complex architecture won't either |
| **Academic comparison** | Many papers use MLP as a baseline; we include it for completeness |
| **Different inductive bias** | Neural networks learn differently from trees — combining both in an ensemble may yield better results |
| **Can capture interactions** | MLPs can learn complex feature interactions that trees might miss |

#### 5.2.3 Training Details

MLPs require more care than XGBoost:

| Requirement | Why |
|-------------|-----|
| **Feature scaling** | Without scaling, large-valued features dominate gradient updates |
| **GPU training** | Neural networks benefit massively from GPU parallelism |
| **More hyperparameters** | Learning rate, batch size, number of layers, layer sizes, dropout rate, optimizer choice |
| **Early stopping** | Stop training when validation loss stops improving (prevents overfitting) |

We train the MLP on **Google Colab** (free GPU) using PyTorch.

### 5.3 Stacking Ensemble

#### 5.3.1 What Is Stacking?

Stacking is a technique where we combine multiple different models to make better predictions. The idea:

1. Train several different models (the "base models"): XGBoost, LightGBM, MLP
2. Train a small "meta-model" (usually a simple Logistic Regression) that learns how to best combine the predictions from all base models

```
          ┌──────────┐
          │ XGBoost  │────┐
          └──────────┘    │
                          ├──► ┌────────────┐    ┌──────────┐
          ┌──────────┐    │    │ Concatenate│───►│Meta-Model│───► Final Prediction
          │ LightGBM │────┘    │ Predictions│    └──────────┘
          └──────────┘    └──► └────────────┘
                          │
          ┌──────────┐    │
          │   MLP    │────┘
          └──────────┘
```

#### 5.3.2 Why Stacking?

- **Diversity**: Different models make different kinds of mistakes. XGBoost might be great at detecting DoS but bad at Worms. MLP might be better at Worms. The meta-model learns "when to trust which model."
- **Robustness**: Stacking almost always outperforms any single model, though the improvement may be small (0.5–2%).
- **Standard practice**: Stacking won the Netflix Prize and many Kaggle competitions.

#### 5.3.3 Train/Val Split for Stacking

To prevent "data leakage" (the meta-model cheating by seeing the same data the base models trained on), we use **k-fold cross-validation** for stacking:

1. Split training data into 5 folds
2. For each fold: train base models on 4 folds, predict on the 1 held-out fold
3. The out-of-fold predictions become the training data for the meta-model
4. Meta-model learns to combine these predictions

### 5.4 Why Not Use a Transformer for Sequences?

This question is worth addressing because many modern cybersecurity papers use Transformers.

**The short answer**: Our dataset (CIC-UNSW-NB15) is **tabular**, not sequential. Each flow is an independent row, not part of a time-ordered sequence with dependencies between rows.

| Aspect | Sequence Data (e.g., sentences, web logs) | Tabular Data (CIC-UNSW-NB15) |
|--------|-------------------------------------------|------------------------------|
| Structure | Ordered list of tokens | Independent rows and columns |
| Relationships | Position matters (word before/after) | Each row stands alone |
| Example | "GET /login → POST /login → POST /login" (brute force pattern) | `{duration=0.5, packets=12, bytes=3000}` |
| Best model | Transformer, LSTM | XGBoost, Random Forest |

**If we had web access logs** (URLs, timestamps, IPs), a Transformer would be ideal to learn attack patterns across sequences of requests. But the CIC-UNSW-NB15 dataset was created by extracting flow statistics from raw network packets, which removes the sequential structure.

**Could we reconstruct sequences?** Yes, by grouping flows by source IP and sorting by timestamp. This would create sequences of flow features that a Transformer could process. This is an interesting research direction listed in [Future Work](#11-future-work), but it requires flows to have reliable timestamps and sufficient density per source IP to form meaningful sequences.

---

## 6. Evaluation Metrics

### 6.1 Accuracy

```
Accuracy = (Correct Predictions) / (Total Predictions)
```

**Simple, but misleading for imbalanced data.**

If 80% of data is Benign, a model that predicts "Benign" for everything achieves 80% accuracy — but it's completely useless. It misses every single attack.

**Our use**: Report accuracy, but do NOT rely on it as the primary metric.

### 6.2 Precision, Recall, and F1-Score

These three metrics are computed **per class** and then averaged.

#### For a Single Class (e.g., "Worms"):

| | Predicted: Worms | Predicted: Not Worms |
|---|---|---|
| **Actual: Worms** | True Positive (TP) | False Negative (FN) |
| **Actual: Not Worms** | False Positive (FP) | True Negative (TN) |

- **Precision** = TP / (TP + FP): "Of everything I labeled as Worms, how many were actually Worms?"  
  High precision → few false alarms.
- **Recall** = TP / (TP + FN): "Of all actual Worms, how many did I catch?"  
  High recall → few missed attacks.
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall): Harmonic mean of precision and recall.  
  Balances both. **This is our primary metric.**

#### Averaging Methods

| Method | How It Works | When to Use |
|--------|-------------|-------------|
| **Macro F1** | Compute F1 for each class, then average. Every class has equal weight. | When rare classes matter — our main choice |
| **Weighted F1** | Average weighted by the number of samples per class | Gives more weight to Benign (the majority) |
| **Micro F1** | Aggregate TP/FP/FN across all classes | Equivalent to accuracy for multi-class |

**Why Macro F1?** Because "Worms" (0.05% of data) matters just as much as "Benign" (80%). A model that misses all Worms should get a low score.

### 6.3 ROC-AUC

**ROC Curve**: A graph that shows the trade-off between True Positive Rate and False Positive Rate as the classification threshold changes.

- **AUC = 1.0**: Perfect model
- **AUC = 0.5**: Random guessing

For multi-class, we compute **One-vs-Rest ROC-AUC** (one curve per class, comparing that class vs all others).

ROC-AUC tells us: "How well can the model rank examples? Does it assign higher probabilities to correct classes than incorrect ones?"

### 6.4 Confusion Matrix

A matrix where:
- Rows: actual classes
- Columns: predicted classes
- Diagonal: correct predictions
- Off-diagonal: mistakes

```
               Predicted
              Ben  Exp  Fuz  Rec ...
Actual Ben   [350K   2K   1K  ...]
       Exp   [  1K  28K   1K  ...]
       Fuz   [  1K   1K  27K  ...]
       ...
```

A confusion matrix helps us understand **which classes the model confuses**. For example, if the model frequently confuses "Backdoor" with "Exploits," we know those two attack types have similar network patterns.

### 6.5 Inference Latency

For real-time detection, speed matters. We measure:

- **Average prediction time per flow** (microseconds)
- **Throughput**: Flows per second

A target: **< 1 ms per prediction** (to handle high traffic volumes).

---

## 7. Real-Time Deployment

### 7.1 Capturing Live Traffic

Live traffic capture requires **administrative privileges** (root/admin). Methods:

| Method | Tool | Pros | Cons |
|--------|------|------|------|
| **tcpdump** | `tcpdump -i eth0 -w capture.pcap` | Standard, efficient, writes to file | Requires post-processing |
| **pyshark** | Python wrapper for Wireshark/TShark | Python-native, easy integration | Slower, memory-intensive |
| **scapy** | Python packet manipulation library | Very flexible | Slow for high-volume traffic |

**Our recommendation**: Use tcpdump to capture short `.pcap` segments (e.g., 60-second windows), then process with CICFlowMeter. This avoids overloading the CPU with real-time feature extraction.

### 7.2 Extracting Flow Features

We use the Python `cicflowmeter` library, which implements CICFlowMeter in pure Python:

```python
from cicflowmeter import FlowMeter
from scapy.all import rdpcap

# Read packets from pcap
packets = rdpcap("capture.pcap")

# Create flow meter
meter = FlowMeter()

# Process each packet
for pkt in packets:
    meter.process(pkt)

# Get flow features as a DataFrame
flows = meter.flows_to_dataframe()
```

The output is a DataFrame with the same 84 features used during training. This ensures compatibility with the trained model.

**Important preprocessing for live data**: Apply the **same** transformations learned during training:
- Load the saved LabelEncoder for categorical features
- Load the saved StandardScaler parameters
- Apply the same NaN/Inf handling

### 7.3 FastAPI Inference Server

We use **FastAPI**, a modern Python web framework, to serve predictions:

```
POST /predict
  Body: JSON with flow features
  Response: JSON with predicted class, confidence, and per-class probabilities

POST /predict_pcap
  Body: pcap file upload
  Response: List of predictions for each flow in the pcap
```

The server:
1. Receives feature data
2. Applies the same preprocessing as training
3. Runs the trained model
4. Returns the result as JSON

### 7.4 Example API Request & Response

**Request** (individual flow):
```json
POST /predict
{
  "flow_duration": 1250000,
  "fwd_packet_length_max": 1460,
  "fwd_packet_length_min": 40,
  "fwd_packet_length_mean": 450.5,
  "fwd_packet_length_std": 320.1,
  "bwd_packet_length_max": 1400,
  "bwd_packet_length_min": 40,
  "bwd_packet_length_mean": 380.2,
  "flow_packets_s": 25.3,
  "flow_bytes_s": 14500.0,
  ... (all 84 features)
}
```

**Response**:
```json
{
  "prediction": "DoS",
  "confidence": 0.94,
  "probabilities": {
    "Benign": 0.01,
    "Analysis": 0.001,
    "Backdoor": 0.001,
    "DoS": 0.94,
    "Exploits": 0.02,
    "Fuzzers": 0.01,
    "Generic": 0.005,
    "Reconnaissance": 0.01,
    "Shellcode": 0.002,
    "Worms": 0.001
  },
  "processing_time_ms": 0.45
}
```

---

## 8. Project Structure

```
Anomaly-detection/
│
├── README.md
├── requirements.txt
├── setup.py
│
├── data/
│   ├── raw/                          # Download CIC-UNSW-NB15 here
│   │   ├── Data.csv
│   │   └── Label.csv
│   └── processed/                    # Output of preprocessing script
│       ├── X_train.npy
│       ├── X_val.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_val.npy
│       └── y_test.npy
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── clean.py                 # Handle NaN, Inf, drop useless columns
│   │   ├── encode.py                # Label encoding / one-hot encoding
│   │   ├── scale.py                 # StandardScaler wrapper
│   │   ├── balance.py               # SMOTE oversampling
│   │   └── split.py                 # Stratified train/val/test split
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── cic_features.py          # Feature names, metadata, groupings
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── xgboost_model.py         # XGBoost training + tuning
│   │   ├── mlp_model.py             # PyTorch MLP definition + training
│   │   ├── ensemble.py              # Stacking ensemble
│   │   └── utils.py                 # Model saving/loading, prediction
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # All evaluation metrics
│   │   ├── visualize.py             # Confusion matrix, ROC curves
│   │   └── compare.py               # Compare all models side-by-side
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── capture.py               # Live traffic capture (tcpdump wrapper)
│   │   └── flow_extractor.py        # CICFlowMeter integration
│   │
│   └── deployment/
│       ├── __init__.py
│       └── api.py                    # FastAPI server
│
├── models_saved/                    # Trained model artifacts
│   ├── xgboost_model.json
│   ├── mlp_model.pth
│   ├── stacking_model.pkl
│   ├── label_encoders.pkl
│   └── scaler.pkl
│
├── colab/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_train_xgboost.ipynb
│   └── 03_train_mlp.ipynb
│
├── notebooks/
│   └── exploration.ipynb
│
└── tests/
    ├── test_preprocessing.py
    ├── test_models.py
    └── test_api.py
```

---

## 9. How to Run Everything

### 9.1 Setup

```bash
# 1. Clone the repository
git clone https://github.com/Huymatip123/Anomaly-detection.git
cd Anomaly-detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**requirements.txt** includes:
- `numpy`, `pandas` — data manipulation
- `scikit-learn` — preprocessing, metrics, train/test split
- `xgboost`, `lightgbm` — gradient boosting models
- `torch` — neural networks (install separately for GPU)
- `imbalanced-learn` — SMOTE oversampling
- `fastapi`, `uvicorn` — API server
- `cicflowmeter` — flow extraction from pcap/live traffic
- `matplotlib`, `seaborn` — visualization
- `jupyter` — notebooks

### 9.2 Preprocessing & Training (Local or Colab)

**Option A: Run Python scripts locally**

```bash
# Step 1: Preprocess
python src/preprocessing/clean.py
python src/preprocessing/encode.py
python src/preprocessing/scale.py
python src/preprocessing/balance.py
python src/preprocessing/split.py

# Step 2: Train models
python src/models/xgboost_model.py
python src/models/mlp_model.py          # Recommended: run on Colab
python src/models/ensemble.py

# Step 3: Evaluate
python src/evaluation/compare.py
```

**Option B: Run notebooks on Google Colab (recommended for GPU training)**

1. Upload the `colab/` folder to Google Drive
2. Open `colab/01_eda_and_preprocessing.ipynb` in Colab
3. Run cells sequentially
4. For `02_train_xgboost.ipynb`: runs on CPU (free Colab)
5. For `03_train_mlp.ipynb`: enable GPU runtime (Runtime → Change runtime type → GPU)
6. Download the trained models from Colab to `models_saved/`

### 9.3 Deploy the API

```bash
# Start the FastAPI server
uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000

# Test with curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"flow_duration": 1250000, "fwd_packet_length_max": 1460, ...}'
```

---

## 10. Results & Discussion

### 10.1 Expected Outcomes

Based on published research using CIC-UNSW-NB15:

| Model | Macro F1 | ROC-AUC | Inference Time |
|-------|----------|---------|---------------|
| Random Forest | ~0.75 | ~0.92 | 50 μs |
| XGBoost | ~0.82 | ~0.95 | 30 μs |
| MLP | ~0.78 | ~0.93 | 100 μs (GPU) / 2 ms (CPU) |
| **Stacking Ensemble** | **~0.84** | **~0.96** | **200 μs** |

*These are approximate; actual results depend on hyperparameters and preprocessing.*

### 10.2 Model Comparison

We will compare models on:

1. **Overall performance**: Macro F1, Weighted F1, ROC-AUC
2. **Per-class performance**: Which models handle rare classes (Worms, Analysis, Backdoor) best?
3. **Efficiency**: Training time, inference latency, model size
4. **Explainability**: Can we understand why the model makes certain predictions? (XGBoost feature importance vs MLP black-box)

---

## 11. Future Work

This project is a foundation. Future improvements could include:

### 11.1 Temporal / Sequence Modeling
As discussed in Section 5.4, reconstructing flow sequences by grouping source IPs over time windows would allow Transformer-based sequence models. This would capture multi-step attack patterns like:
- Reconnaissance → Exploit → Backdoor (intrusion chain)
- Gradual increase in request rate (slow DDoS)

### 11.2 Graph Neural Networks
Represent the network as a graph (computers = nodes, flows = edges). GNNs can learn attack patterns that involve communication patterns between multiple hosts.

### 11.3 BERT for Payload Analysis
If we have access to raw packet payloads (not just flow statistics), a BERT-like model could analyze the actual content of network packets — detecting SQL injection strings, malicious scripts, etc.

### 11.4 Online Learning
Attack patterns evolve. An online learning system would continuously update the model with new traffic data, adapting to new attack types without full retraining.

### 11.5 Self-Supervised Pre-Training
Train a model on unlabeled traffic data (which is abundant) using self-supervised learning, then fine-tune on the labeled dataset. This often improves performance when labeled data is limited.

### 11.6 Adversarial Robustness
Test the model against adversarial attacks (small perturbations to traffic designed to evade detection). This is crucial for real-world deployment where attackers actively try to hide.

### 11.7 Explainable AI (XAI)
Implement SHAP or LIME to provide human-readable explanations for each prediction:
```
Why was this flow flagged as "DoS"?
- Flow Duration: 0.1s (very short) → contributed +0.3 to DoS probability
- Fwd Packets/s: 5000 (extremely high) → contributed +0.4 to DoS probability
- Packet Length Mean: 64 bytes (very small) → contributed +0.2 to DoS probability
```

---

## 12. References

1. **CIC-UNSW-NB15 Dataset** — Canadian Institute for Cybersecurity, UNB.  
   https://www.unb.ca/cic/datasets/cic-unsw-nb15.html

2. **CICFlowMeter** — Lashkari, A. H., et al. Network traffic flow generator and analyzer.  
   https://github.com/ahlashkari/CICFlowMeter

3. **UNSW-NB15 Dataset** — Moustafa, N., Slay, J. "UNSW-NB15: a comprehensive data set for network intrusion detection systems." MilCIS, 2015.  
   https://research.unsw.edu.au/projects/unsw-nb15-dataset

4. **Poisoning and Evasion: Deep Learning-Based NIDS under Adversarial Attacks** — Mohammadian, H., Lashkari, A. H., Ghorbani, A. PST, 2024.

5. **CICFlowMeter Python Port** — Tanwir Ahmad.  
   https://gitlab.abo.fi/tahmad/cicflowmeter-py

6. **Why Tree-Based Models Beat Neural Networks on Tabular Data** — Grinsztajn, L., Oyallon, E., Varoquaux, G. "Why do tree-based models still outperform deep learning on typical tabular data?" NeurIPS 2022.

7. **SMOTE** — Chawla, N. V., et al. "SMOTE: Synthetic Minority Over-sampling Technique." JAIR, 2002.

8. **XGBoost** — Chen, T., Guestrin, C. "XGBoost: A Scalable Tree Boosting System." KDD, 2016.

---

*This README was written for educational purposes. Each concept is explained from first principles, assuming the reader is familiar with basic Python programming but new to machine learning and cybersecurity.*
