# **Machine Learning Projects**

This repository showcases a collection of machine learning projects focused on different algorithms, including classification, regression, reinforcement learning, and clustering. The goal of these projects is to build models from scratch, experiment with real-world datasets, and strengthen the understanding of machine learning principles.

## **Project Overview**

### 1. **Neural Networks Implementation**
- **Goal**: Gain insights into feedforward neural networks and backpropagation.
- **Key Aspects**:
  - Built a fully connected neural network from scratch.
  - Implemented weight updates using gradient descent.
  - Tested the model on sample datasets to observe its learning process.

---

### 2. **Income Prediction Model**
- **Goal**: Develop and compare classification models to predict whether an individual earns more than $50K annually.
- **Key Aspects**:
  - Implemented and compared logistic regression, decision trees, and SVM models.
  - Performed exploratory data analysis and feature engineering to refine predictions.
  - Evaluated model performance using accuracy, precision, and recall metrics.

---

### 3. **Fundamental Machine Learning Algorithms from Scratch**
- **Goal**: Strengthen understanding of fundamental machine learning techniques by coding algorithms from scratch.
- **Algorithms Implemented**:
  - **Regression**:
    - Ordinary Least Squares (OLS) Regression.
    - Ridge Regression.
    - Gradient Descent Optimization.
  - **Classification**:
    - Linear Discriminant Analysis (LDA).
    - Quadratic Discriminant Analysis (QDA).
    - Logistic Regression (one-vs-all and multi-class softmax).
    - Support Vector Machines (SVMs) with different kernels.
- **Datasets Used**:
  - A 2D classification dataset for visualizing decision boundaries.
  - The MNIST dataset for multi-class classification.
  - A medical dataset for regression, predicting diabetic conditions.
- **Evaluation Metrics**:
  - Classification models assessed using accuracy and error rates.
  - Regression models evaluated with Mean Squared Error (MSE).
- **Key Aspects**:
  - Implemented one-vs-all logistic regression using gradient descent.
  - Visualized decision boundaries for classification models.
  - Analyzed performance using error plots and model comparisons.

---

### 4. **Reinforcement Learning: Environment Design & Solving**
- **Assignment Overview**:
  - **Part 1**: Defined deterministic and stochastic RL environments (Gymnasium-compliant) for scenarios like Warehouse Robot, Traffic Light Control, or Drone Delivery.
  - **Part 2**: Solved environments using tabular methods (Q-learning and SARSA).
  - **Part 3**: Applied Q-learning to a stock-trading environment using NVIDIA stock data.
- **Key Files**:
  - `fall24_rl_assignment_1.ipynb`: Jupyter notebook with code for all three parts.
  - Trained models (Q-tables) saved as `.pkl` or `.h5` files.

---

### 5. **K-means Clustering: Deterministic Centroid Initialization**
- **Report Overview**:
  - Improved K-means by initializing centroids as the farthest points from existing centroids.
  - Achieved **5x faster convergence** (avg. iterations reduced from 12.6 to 2.5).
- **Key Files**:
  - `kmeans_improved.py`: Code for deterministic centroid initialization.
  - `report.pdf`: Detailed analysis of results and methodology.

---

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Tharun2104/Machine-Learning-Projects.git
