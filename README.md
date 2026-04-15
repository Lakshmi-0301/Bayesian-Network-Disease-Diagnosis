# Bayesian Network for Breast Cancer Diagnosis

This project implements a Bayesian Network to predict the probability of breast cancer diagnosis (Malignant vs. Benign) based on several clinical features. The model uses the **Breast Cancer Wisconsin (Diagnostic) Dataset** and leverages probabilistic graphical models for inference.

##  Project Overview

The core objective is to compute the posterior probability:
$$P(D \mid X_1, X_2, X_3, X_4)$$
where:
- $D$: Diagnosis (Malignant/Benign)
- $X_i$: Clinical features (e.g., radius_mean, texture_mean, smoothness_mean, concavity_mean)

The implementation utilizes the `pgmpy` library to construct the network, estimate parameters, and perform variable elimination for inference.

##  Methodology

### 1. Data Processing
- **Dataset**: `data.csv` containing features for breast cancer diagnosis.
- **Discretization**: Continuous variables are transformed into discrete states (`Low`, `High`) based on their median values to satisfy the requirements of a Discrete Bayesian Network.

### 2. Model Architecture
The Bayesian Network is structured as follows:
- `diagnosis` $\rightarrow$ `radius_mean`
- `diagnosis` $\rightarrow$ `texture_mean`
- `diagnosis` $\rightarrow$ `smoothness_mean`
- `radius_mean` $\rightarrow$ `concavity_mean`

This structure captures the conditional dependencies between the nodes, allowing for robust probabilistic inference.

### 3. Parameter Estimation
The model parameters (Conditional Probability Distributions - CPDs) are estimated from the data using the **Maximum Likelihood Estimator (MLE)**.

### 4. Inference
Using **Variable Elimination**, the model can calculate:
- **Prior Probabilities**: $P(D)$
- **Posterior Probabilities**: $P(D \mid Evidence)$, such as $P(Diagnosis \mid All\ Features = High)$.

##  Getting Started

### Prerequisites
Ensure you have the following Python libraries installed:
```bash
pip install pandas pgmpy scikit-learn matplotlib networkx
```

### Usage
1. Clone this repository.
2. Ensure `data.csv` is in the project root.
3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook Bayesian_Network_for_Disease_Diagnosis.ipynb
   ```

##  Results

The model achieves strong predictive performance on the dataset:

| Metric    | Benign (B) | Malignant (M) | Overall |
|-----------|------------|---------------|---------|
| Precision | 0.92       | 0.78          | -       |
| Recall    | 0.85       | 0.88          | -       |
| F1-Score  | 0.89       | 0.83          | -       |
| **Accuracy**| -          | -             | **86%** |

##  Project Structure
- `Bayesian_Network_for_Disease_Diagnosis.ipynb`: Main implementation notebook.
- `data.csv`: Input dataset.
- `bn_structure.png`: Visualization of the Bayesian Network structure.
- `README.md`: Project documentation.

##  Mathematical Foundation
The project is built on the foundation of **Bayes’ Theorem**:
$$P(D \mid T) = \frac{P(T \mid D) P(D)}{P(T)}$$

And the **Joint Distribution** for this specific network:
$$P(D, X_1, X_2, X_3, X_4) = P(D)P(X_1|D)P(X_2|D)P(X_3|D)P(X_4|X_1)$$

