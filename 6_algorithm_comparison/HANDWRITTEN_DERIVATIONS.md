# Handwritten Implementation & Step-by-Step Mathematical Derivations
## Comparative Study of Six Machine Learning Regression Algorithms

**Student:** Muhammad Asad Mashood — BS Artificial Intelligence, COMSATS University Islamabad (Wah Campus)  
**Registration Number:** FA23-BAI-039  
**Dataset:** California Housing Dataset (1990 U.S. Census)

---

## 📋 Table of Contents
1. [Sample Dataset Definition](#1-sample-dataset-definition)
2. [Algorithm 1: Linear Regression (Ordinary Least Squares & Gradient Descent)](#algorithm-1-linear-regression)
3. [Algorithm 2: Decision Tree Regression (CART Variance Reduction)](#algorithm-2-decision-tree-regression)
4. [Algorithm 3: Random Forest Regression (Bagging & Subspace Sampling)](#algorithm-3-random-forest-regression)
5. [Algorithm 4: K-Nearest Neighbors Regression (KNN)](#algorithm-4-k-nearest-neighbors-regression)
6. [Algorithm 5: Neural Network Regression (Multi-Layer Perceptron & Backprop)](#algorithm-5-neural-network-regression)
7. [Algorithm 6: XGBoost Regression (2nd-Order Gradient Boosting & Shrinkage)](#algorithm-6-xgboost-regression)
8. [Summary of Handwritten vs. Python Implementations](#summary-comparison)

---

## 1. Sample Dataset Definition

To provide a concrete, step-by-step numeric verification of all six algorithms, we extract a small, representative 3-district training sample and 1 query district from `housing.csv`.

For manual tractability, we consider two fundamental predictive features:
- $x_1$: `median_income` (in tens of thousands USD, e.g., $2.5 = \$25,000$)
- $x_2$: `housing_median_age` (in years)
- $y$: `median_house_value` (target variable, in tens of thousands USD, e.g., $25.8 = \$258,000$)

### Training Sample Data Matrix ($n = 3$):
| Sample ID | District Description | $x_1$ (Income in \$10k) | $x_2$ (House Age in Yrs) | $y$ (Actual Price in \$10k) | Actual Price (USD) |
|:---:|:---|:---:|:---:|:---:|:---:|
| **$A$ (Row 1)** | Urban coastal Bay Area | $2.5$ | $29$ | $25.8$ | \$258,000 |
| **$B$ (Row 2)** | Inland Central Valley | $2.5$ | $15$ | $9.0$ | \$90,000 |
| **$C$ (Row 3)** | Suburban coastal block | $4.2$ | $13$ | $21.0$ | \$210,000 |

### Query Point to Predict ($Q$):
$$\mathbf{x}_Q = [x_{Q1} = 3.0, \; x_{Q2} = 20.0]^T \quad (\text{Income: } \$30,000, \; \text{Age: } 20 \text{ years})$$

---

## Algorithm 1: Linear Regression

### 1.1 Working Procedure
Linear Regression models the relationship between dependent continuous variable $y$ and independent variables $\mathbf{x} = [x_1, \dots, x_p]^T$ assuming an affine hyper-plane:
$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p = \mathbf{x}^T \boldsymbol{\beta}$$
The optimal parameters minimize the Sum of Squared Residuals (SSR):
$$\text{SSR}(\boldsymbol{\beta}) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$$

### 1.2 Mathematical Equations
1. **Normal Equation (Closed-form Analytical Solution):**
   $$\nabla_{\boldsymbol{\beta}} \text{SSR} = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{0} \implies \hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
2. **Gradient Descent (Iterative Optimization):**
   $$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \alpha \nabla_{\boldsymbol{\beta}} \text{MSE} = \boldsymbol{\beta}^{(t)} + \frac{2\alpha}{n} \mathbf{X}^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta}^{(t)})$$

### 1.3 Step-by-Step Numerical Solution

#### Step 1: Construct Design Matrix $\mathbf{X}$ and Target Vector $\mathbf{y}$
$$\mathbf{X} = \begin{bmatrix} 1 & 2.5 & 29 \\ 1 & 2.5 & 15 \\ 1 & 4.2 & 13 \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} 25.8 \\ 9.0 \\ 21.0 \end{bmatrix}$$

#### Step 2: Compute $\mathbf{X}^T \mathbf{X}$ and $\mathbf{X}^T \mathbf{y}$
$$\mathbf{X}^T \mathbf{X} = \begin{bmatrix} 1 & 1 & 1 \\ 2.5 & 2.5 & 4.2 \\ 29 & 15 & 13 \end{bmatrix} \begin{bmatrix} 1 & 2.5 & 29 \\ 1 & 2.5 & 15 \\ 1 & 4.2 & 13 \end{bmatrix} = \begin{bmatrix} 3.0 & 9.2 & 57.0 \\ 9.2 & 30.14 & 164.6 \\ 57.0 & 164.6 & 1235.0 \end{bmatrix}$$

$$\mathbf{X}^T \mathbf{y} = \begin{bmatrix} 1(25.8) + 1(9.0) + 1(21.0) \\ 2.5(25.8) + 2.5(9.0) + 4.2(21.0) \\ 29(25.8) + 15(9.0) + 13(21.0) \end{bmatrix} = \begin{bmatrix} 55.8 \\ 175.2 \\ 1156.2 \end{bmatrix}$$

#### Step 3: Compute Matrix Inverse $(\mathbf{X}^T \mathbf{X})^{-1}$ and Solve for $\hat{\boldsymbol{\beta}}$
$$\det(\mathbf{X}) = 1(2.5 \cdot 13 - 15 \cdot 4.2) - 2.5(1 \cdot 13 - 15 \cdot 1) + 29(1 \cdot 4.2 - 2.5 \cdot 1) = -30.5 + 5.0 + 49.3 = 23.8$$
$$\det(\mathbf{X}^T \mathbf{X}) = (\det(\mathbf{X}))^2 = (23.8)^2 = 566.44$$
Solving the linear system $(\mathbf{X}^T \mathbf{X}) \boldsymbol{\beta} = \mathbf{X}^T \mathbf{y}$:
$$\hat{\beta}_0 = -30.1765, \quad \hat{\beta}_1 = 8.4706, \quad \hat{\beta}_2 = 1.2000$$

#### Step 4: Predict for Query Point $\mathbf{x}_Q = [1, 3.0, 20.0]^T$
$$\hat{y}_Q = -30.1765 + 8.4706(3.0) + 1.2000(20.0) = -30.1765 + 25.4118 + 24.0000 = \mathbf{19.2353} \; (\$192,353)$$

#### Step 5: Demonstration of 1-Step Gradient Descent
Starting from initial weights $\boldsymbol{\beta}^{(0)} = [0, 0, 0]^T$ with learning rate $\alpha = 0.001$:
$$\hat{\mathbf{y}}^{(0)} = \mathbf{X} \boldsymbol{\beta}^{(0)} = [0, 0, 0]^T \implies \mathbf{e}^{(0)} = \mathbf{y} - \hat{\mathbf{y}}^{(0)} = [25.8, 9.0, 21.0]^T$$
$$\mathbf{X}^T \mathbf{e}^{(0)} = [55.8, 175.2, 1156.2]^T$$
$$\boldsymbol{\beta}^{(1)} = \boldsymbol{\beta}^{(0)} + \frac{2(0.001)}{3} \begin{bmatrix} 55.8 \\ 175.2 \\ 1156.2 \end{bmatrix} = \begin{bmatrix} 0.0372 \\ 0.1168 \\ 0.7708 \end{bmatrix}$$

---

## Algorithm 2: Decision Tree Regression

### 2.1 Working Procedure
Decision Tree Regression recursively partitions feature space into axis-aligned rectangular regions $R_m$. In each region, the model predicts the mean target value of training instances belonging to that leaf:
$$\hat{y} = \sum_{m=1}^M c_m \cdot \mathbb{I}(\mathbf{x} \in R_m), \quad \text{where } c_m = \frac{1}{|R_m|} \sum_{i \in R_m} y_i$$

### 2.2 Splitting Criterion (Variance Reduction / SSE)
For any split on feature $j$ at threshold $s$, partitions $R_L(j, s) = \{\mathbf{x} \mid x_j \le s\}$ and $R_R(j, s) = \{\mathbf{x} \mid x_j > s\}$ are formed. The best split minimizes total Sum of Squared Errors:
$$\text{SSE}_{\text{total}} = \text{SSE}_L + \text{SSE}_R = \sum_{i \in R_L} (y_i - \bar{y}_L)^2 + \sum_{i \in R_R} (y_i - \bar{y}_R)^2$$

### 2.3 Step-by-Step Numerical Solution

#### Step 1: Calculate Impurity of Parent Node
$$\bar{y}_{\text{parent}} = \frac{25.8 + 9.0 + 21.0}{3} = \frac{55.8}{3} = 18.60$$
$$\text{SSE}_{\text{parent}} = (25.8 - 18.6)^2 + (9.0 - 18.6)^2 + (21.0 - 18.6)^2 = (7.2)^2 + (-9.6)^2 + (2.4)^2 = 51.84 + 92.16 + 5.76 = \mathbf{149.76}$$

#### Step 2: Evaluate Split Candidate 1: Split on $x_1 \le 3.0$
- **Left Node ($R_L$):** $\{A, B\}$ ($x_1 = 2.5 \le 3.0$). Values: $\{25.8, 9.0\}$
  $$\bar{y}_L = \frac{25.8 + 9.0}{2} = 17.40$$
  $$\text{SSE}_L = (25.8 - 17.4)^2 + (9.0 - 17.4)^2 = (8.4)^2 + (-8.4)^2 = 70.56 + 70.56 = 141.12$$
- **Right Node ($R_R$):** $\{C\}$ ($x_1 = 4.2 > 3.0$). Value: $\{21.0\}$
  $$\bar{y}_R = 21.0, \quad \text{SSE}_R = 0.00$$
- **Total SSE:**
  $$\text{SSE}_{\text{split 1}} = 141.12 + 0.00 = \mathbf{141.12}$$

#### Step 3: Evaluate Split Candidate 2: Split on $x_2 \le 20.0$
- **Left Node ($R_L$):** $\{B, C\}$ ($x_2 \in \{15, 13\} \le 20$). Values: $\{9.0, 21.0\}$
  $$\bar{y}_L = \frac{9.0 + 21.0}{2} = 15.00$$
  $$\text{SSE}_L = (9.0 - 15.0)^2 + (21.0 - 15.0)^2 = (-6.0)^2 + (6.0)^2 = 36.00 + 36.00 = 72.00$$
- **Right Node ($R_R$):** $\{A\}$ ($x_2 = 29 > 20$). Value: $\{25.8\}$
  $$\bar{y}_R = 25.8, \quad \text{SSE}_R = 0.00$$
- **Total SSE:**
  $$\text{SSE}_{\text{split 2}} = 72.00 + 0.00 = \mathbf{72.00}$$

#### Step 4: Optimal Split Selection & Prediction
Since $\text{SSE}_{\text{split 2}} (72.00) < \text{SSE}_{\text{split 1}} (141.12)$, the tree selects **Split on $x_2 \le 20.0$**.
- Impurity Reduction: $\Delta \text{SSE} = 149.76 - 72.00 = \mathbf{77.76}$.
- For query point $\mathbf{x}_Q$ ($x_{Q2} = 20 \le 20$), it falls into the **Left Node**:
$$\hat{y}_Q = \bar{y}_L = \mathbf{15.00} \; (\$150,000)$$

---

## Algorithm 3: Random Forest Regression

### 3.1 Working Procedure
Random Forest constructs an ensemble of $B$ decorrelated regression trees using **Bagging** (Bootstrap Aggregation) and **Random Feature Subspace Sampling**:
1. Draw $B$ bootstrap samples $\mathcal{D}_1, \dots, \mathcal{D}_B$ of size $n$ with replacement from training data.
2. At each split node, select $m < p$ features uniformly at random as candidates.
3. Grow deep trees without pruning.
4. Aggregate individual tree predictions via arithmetic mean:
   $$\hat{y}_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$

### 3.2 Variance Reduction Principle
For $B$ identically distributed trees each with variance $\sigma^2$ and positive pairwise correlation $\rho$:
$$\text{Var}(\hat{y}_{\text{RF}}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
Random feature selection reduces tree correlation $\rho$, driving total prediction variance significantly lower than any individual tree.

### 3.3 Step-by-Step Numerical Solution
Let $B = 2$ trees with bootstrap sampling:

#### Tree 1:
- **Bootstrap Sample:** $\{A, A, B\}$ (District $A$ chosen twice, $B$ once).
- **Sample values:** $A: (x_1=2.5, x_2=29, y=25.8)$, $A: (x_1=2.5, x_2=29, y=25.8)$, $B: (x_1=2.5, x_2=15, y=9.0)$.
- **Feature sampled:** $x_2$ (House Age).
- **Split:** $x_2 \le 20 \implies \text{Left} = \{B\} \; (\bar{y}=9.0)$, $\text{Right} = \{A, A\} \; (\bar{y}=25.8)$.
- **Prediction for $Q$ ($x_{Q2}=20 \le 20$):**
  $$T_1(\mathbf{x}_Q) = 9.0 \; (\$90,000)$$

#### Tree 2:
- **Bootstrap Sample:** $\{B, C, C\}$ (District $B$ chosen once, $C$ twice).
- **Sample values:** $B: (x_1=2.5, x_2=15, y=9.0)$, $C: (x_1=4.2, x_2=13, y=21.0)$, $C: (x_1=4.2, x_2=13, y=21.0)$.
- **Feature sampled:** $x_1$ (Median Income).
- **Split:** $x_1 \le 3.0 \implies \text{Left} = \{B\} \; (\bar{y}=9.0)$, $\text{Right} = \{C, C\} \; (\bar{y}=21.0)$.
- **Prediction for $Q$ ($x_{Q1}=3.0 \le 3.0$):**
  $$T_2(\mathbf{x}_Q) = 9.0 \; (\$90,000)$$

#### Ensemble Average Prediction:
$$\hat{y}_{\text{RF}}(\mathbf{x}_Q) = \frac{T_1(\mathbf{x}_Q) + T_2(\mathbf{x}_Q)}{2} = \frac{9.0 + 9.0}{2} = \mathbf{9.00} \; (\$90,000)$$

---

## Algorithm 4: K-Nearest Neighbors Regression (KNN)

### 4.1 Working Procedure
KNN is a non-parametric, instance-based memory algorithm:
1. Normalize features to prevent variables with larger numerical scales from dominating distance metrics:
   $$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
2. Compute Euclidean distance from query point $\mathbf{z}_Q$ to all training instances:
   $$d(\mathbf{z}_Q, \mathbf{z}_i) = \sqrt{\sum_{j=1}^p (z_{Qj} - z_{ij})^2}$$
3. Identify the $k$ closest training points $\mathcal{N}_k(\mathbf{x}_Q)$.
4. Compute prediction via Uniform Mean or Distance-Weighted Inverse Mean:
   $$\hat{y}_{\text{uniform}} = \frac{1}{k} \sum_{i \in \mathcal{N}_k} y_i, \quad \hat{y}_{\text{weighted}} = \frac{\sum_{i \in \mathcal{N}_k} w_i y_i}{\sum_{i \in \mathcal{N}_k} w_i}, \quad w_i = \frac{1}{d(\mathbf{z}_Q, \mathbf{z}_i)}$$

### 4.2 Step-by-Step Numerical Solution

#### Step 1: Feature Standardization (Z-score)
- **Feature $x_1$:**
  $$\mu_1 = \frac{2.5 + 2.5 + 4.2}{3} = \frac{9.2}{3} = 3.0667, \quad \sigma_1 = \sqrt{\frac{(2.5 - 3.0667)^2 + (2.5 - 3.0667)^2 + (4.2 - 3.0667)^2}{3}} = 0.8014$$
- **Feature $x_2$:**
  $$\mu_2 = \frac{29 + 15 + 13}{3} = \frac{57}{3} = 19.0000, \quad \sigma_2 = \sqrt{\frac{(29 - 19)^2 + (15 - 19)^2 + (13 - 19)^2}{3}} = \sqrt{\frac{100 + 16 + 36}{3}} = 7.1181$$

#### Standardized Coordinates:
- Point $A$: $z_{A1} = \frac{2.5 - 3.0667}{0.8014} = -0.7071, \; z_{A2} = \frac{29 - 19}{7.1181} = +1.4049$
- Point $B$: $z_{B1} = \frac{2.5 - 3.0667}{0.8014} = -0.7071, \; z_{B2} = \frac{15 - 19}{7.1181} = -0.5620$
- Point $C$: $z_{C1} = \frac{4.2 - 3.0667}{0.8014} = +1.4142, \; z_{C2} = \frac{13 - 19}{7.1181} = -0.8429$
- Query $Q$: $z_{Q1} = \frac{3.0 - 3.0667}{0.8014} = -0.0832, \; z_{Q2} = \frac{20 - 19}{7.1181} = +0.1405$

#### Step 2: Compute Euclidean Distance to Query Point
$$d(Q, A) = \sqrt{(-0.0832 - (-0.7071))^2 + (0.1405 - 1.4049)^2} = \sqrt{(0.6239)^2 + (-1.2644)^2} = \sqrt{0.3893 + 1.5987} = \mathbf{1.4099}$$
$$d(Q, B) = \sqrt{(-0.0832 - (-0.7071))^2 + (0.1405 - (-0.5620))^2} = \sqrt{(0.6239)^2 + (0.7025)^2} = \sqrt{0.3893 + 0.4935} = \mathbf{0.9395}$$
$$d(Q, C) = \sqrt{(-0.0832 - 1.4142)^2 + (0.1405 - (-0.8429))^2} = \sqrt{(-1.4974)^2 + (0.9834)^2} = \sqrt{2.2422 + 0.9671} = \mathbf{1.7915}$$

#### Step 3: Select $k=2$ Nearest Neighbors & Predict
Ranking distances: $d(Q, B) = 0.9395 < d(Q, A) = 1.4099 < d(Q, C) = 1.7915$.
The 2 nearest neighbors are **$B$ ($y=9.0$)** and **$A$ ($y=25.8$)**.

1. **Uniform $k$-NN Prediction:**
   $$\hat{y}_{\text{uniform}} = \frac{y_B + y_A}{2} = \frac{9.0 + 25.8}{2} = \mathbf{17.40} \; (\$174,000)$$

2. **Distance-Weighted $k$-NN Prediction:**
   $$w_B = \frac{1}{0.9395} = 1.0644, \quad w_A = \frac{1}{1.4099} = 0.7093$$
   $$\hat{y}_{\text{weighted}} = \frac{1.0644(9.0) + 0.7093(25.8)}{1.0644 + 0.7093} = \frac{9.5796 + 18.3000}{1.7737} = \frac{27.8796}{1.7737} = \mathbf{15.72} \; (\$157,200)$$

---

## Algorithm 5: Neural Network Regression (MLP)

### 5.1 Architecture & Working Procedure
A Multi-Layer Perceptron (MLP) with structure $2 \to 2 \to 1$:
- **Input layer:** $\mathbf{x} = [x_1, x_2]^T$
- **Hidden layer:** 2 neurons with Rectified Linear Unit ($\text{ReLU}(z) = \max(0, z)$)
- **Output layer:** 1 linear neuron predicting continuous price $\hat{y}$

### 5.2 Equations
1. **Hidden Layer Pre-activation & Activation:**
   $$\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}, \quad \mathbf{a}^{(1)} = \text{ReLU}(\mathbf{z}^{(1)})$$
2. **Output Layer Prediction:**
   $$\hat{y} = \mathbf{W}^{(2)} \mathbf{a}^{(1)} + b^{(2)}$$
3. **Squared Loss & Backpropagation Gradients:**
   $$\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$$
   $$\frac{\partial \mathcal{L}}{\partial \hat{y}} = (\hat{y} - y)$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(2)}} = (\hat{y} - y) (\mathbf{a}^{(1)})^T, \quad \frac{\partial \mathcal{L}}{\partial b^{(2)}} = (\hat{y} - y)$$
   $$\boldsymbol{\delta}^{(1)} = [(\mathbf{W}^{(2)})^T (\hat{y} - y)] \odot \mathbb{I}(\mathbf{z}^{(1)} > 0)$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} = \boldsymbol{\delta}^{(1)} \mathbf{x}^T, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(1)}} = \boldsymbol{\delta}^{(1)}$$

### 5.3 Step-by-Step Numerical Solution
Let the initialized network parameters be:
$$\mathbf{W}^{(1)} = \begin{bmatrix} 0.2 & 0.1 \\ -0.1 & 0.3 \end{bmatrix}, \quad \mathbf{b}^{(1)} = \begin{bmatrix} 0.1 \\ -0.2 \end{bmatrix}, \quad \mathbf{W}^{(2)} = \begin{bmatrix} 1.5 & 2.0 \end{bmatrix}, \quad b^{(2)} = 0.5$$

#### Step 1: Forward Pass on Training Sample $B$ ($\mathbf{x}_B = [2.5, 15.0]^T$, $y_B = 9.0$)
1. **Pre-activation:**
   $$z_1^{(1)} = 0.2(2.5) + 0.1(15.0) + 0.1 = 0.50 + 1.50 + 0.10 = 2.10$$
   $$z_2^{(1)} = -0.1(2.5) + 0.3(15.0) - 0.2 = -0.25 + 4.50 - 0.20 = 4.05$$
2. **Activation ($\text{ReLU}$):**
   $$a_1^{(1)} = \max(0, 2.10) = 2.10, \quad a_2^{(1)} = \max(0, 4.05) = 4.05$$
3. **Network Output Prediction:**
   $$\hat{y} = 1.5(2.10) + 2.0(4.05) + 0.5 = 3.15 + 8.10 + 0.50 = \mathbf{11.75} \; (\$117,500)$$
4. **Prediction Error & Loss:**
   $$e = \hat{y} - y = 11.75 - 9.00 = +2.75$$
   $$\mathcal{L} = \frac{1}{2}(2.75)^2 = \mathbf{3.7813}$$

#### Step 2: Backward Pass (Error Gradients)
1. **Output layer gradients:**
   $$\frac{\partial \mathcal{L}}{\partial W_1^{(2)}} = (2.75)(2.10) = 5.775, \quad \frac{\partial \mathcal{L}}{\partial W_2^{(2)}} = (2.75)(4.05) = 11.1375, \quad \frac{\partial \mathcal{L}}{\partial b^{(2)}} = 2.75$$
2. **Hidden layer error term $\boldsymbol{\delta}^{(1)}$:**
   Since $z_1 > 0$ and $z_2 > 0$, $\text{ReLU}'(z) = 1$:
   $$\delta_1^{(1)} = (2.75)(1.5)(1) = 4.125$$
   $$\delta_2^{(1)} = (2.75)(2.0)(1) = 5.500$$
3. **Input-to-hidden weights gradients:**
   $$\frac{\partial \mathcal{L}}{\partial W_{11}^{(1)}} = \delta_1 x_1 = (4.125)(2.5) = 10.3125, \quad \frac{\partial \mathcal{L}}{\partial W_{12}^{(1)}} = \delta_1 x_2 = (4.125)(15.0) = 61.8750$$

#### Step 3: Parameter Updates with Learning Rate $\eta = 0.001$
$$W_1^{(2)(\text{new})} = 1.5 - 0.001(5.775) = \mathbf{1.4942}$$
$$W_2^{(2)(\text{new})} = 2.0 - 0.001(11.1375) = \mathbf{1.9889}$$
$$W_{11}^{(1)(\text{new})} = 0.2 - 0.001(10.3125) = \mathbf{0.1897}$$

---

## Algorithm 6: XGBoost Regression

### 6.1 Working Procedure
XGBoost minimizes a regularized objective function using a **second-order Taylor expansion**:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
where:
- $g_i = \left[\frac{\partial \ell(y_i, \hat{y})}{\partial \hat{y}}\right]_{\hat{y} = \hat{y}^{(t-1)}}$ (First-order gradient)
- $h_i = \left[\frac{\partial^2 \ell(y_i, \hat{y})}{\partial \hat{y}^2}\right]_{\hat{y} = \hat{y}^{(t-1)}}$ (Second-order hessian)
- $\lambda$: L2 leaf regularization parameter
- $\gamma$: Minimum loss reduction threshold for partitioning

### 6.2 Key Formulas
For Mean Squared Error loss $\ell(y_i, \hat{y}) = \frac{1}{2}(y_i - \hat{y})^2$:
$$g_i = \hat{y}_i - y_i, \quad h_i = 1.0$$
The optimal weight $w_j^*$ for leaf $j$ containing instance subset $I_j$:
$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda} = -\frac{G_j}{H_j + \lambda}$$
The split gain for splitting a node into left ($L$) and right ($R$) children:
$$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda} \right] - \gamma$$
New boosted prediction with learning rate (shrinkage) $\eta$:
$$\hat{y}^{(t)}(\mathbf{x}) = \hat{y}^{(t-1)}(\mathbf{x}) + \eta w_{j(x)}^*$$

### 6.3 Step-by-Step Numerical Solution
Let base initialization be target mean: $\hat{y}^{(0)} = \bar{y} = 18.60$, $\lambda = 1.0$, $\gamma = 0.0$, and learning rate $\eta = 0.3$.

#### Step 1: Compute Gradients and Hessians for All Training Instances
- **Instance $A$ ($y_A = 25.8$):**
  $$g_A = 18.60 - 25.80 = -7.20, \quad h_A = 1.00$$
- **Instance $B$ ($y_B = 9.0$):**
  $$g_B = 18.60 - 9.00 = +9.60, \quad h_B = 1.00$$
- **Instance $C$ ($y_C = 21.0$):**
  $$g_C = 18.60 - 21.00 = -2.40, \quad h_C = 1.00$$

#### Step 2: Parent Node Score
$$G = g_A + g_B + g_C = -7.20 + 9.60 - 2.40 = 0.00, \quad H = 1 + 1 + 1 = 3.00$$
$$\text{Score}_{\text{parent}} = \frac{G^2}{H + \lambda} = \frac{0.00}{3.00 + 1.00} = 0.00$$

#### Step 3: Evaluate Split on $x_2 \le 20.0$
- **Left Child ($I_L = \{B, C\}$):**
  $$G_L = g_B + g_C = 9.60 + (-2.40) = +7.20, \quad H_L = 1 + 1 = 2.00$$
  $$w_L^* = -\frac{G_L}{H_L + \lambda} = -\frac{7.20}{2.00 + 1.00} = -\frac{7.20}{3.00} = \mathbf{-2.40}$$
  $$\text{Score}_L = \frac{G_L^2}{H_L + \lambda} = \frac{(7.20)^2}{3.00} = \frac{51.84}{3.00} = 17.28$$

- **Right Child ($I_R = \{A\}$):**
  $$G_R = g_A = -7.20, \quad H_R = 1.00$$
  $$w_R^* = -\frac{G_R}{H_R + \lambda} = -\frac{-7.20}{1.00 + 1.00} = +\frac{7.20}{2.00} = \mathbf{+3.60}$$
  $$\text{Score}_R = \frac{G_R^2}{H_R + \lambda} = \frac{(-7.20)^2}{2.00} = \frac{51.84}{2.00} = 25.92$$

#### Step 4: Split Gain Calculation
$$\text{Gain} = \frac{1}{2} \left[ 17.28 + 25.92 - 0.00 \right] - 0.0 = \frac{1}{2}(43.20) = \mathbf{21.60} > 0 \implies \text{Split Accepted!}$$

#### Step 5: Boosted Prediction for Query Point $\mathbf{x}_Q$
Query point has $x_{Q2} = 20 \le 20$, which routes to the **Left Leaf** ($w_L^* = -2.40$):
$$\hat{y}^{(1)}(\mathbf{x}_Q) = \hat{y}^{(0)}(\mathbf{x}_Q) + \eta w_L^* = 18.60 + 0.3(-2.40) = 18.60 - 0.72 = \mathbf{17.88} \; (\$178,800)$$

---

## Summary Comparison

| Algorithm | Mathematical Engine | Sample Predicted Price $\hat{y}_Q$ | Full Dataset Test $R^2$ | Key Theoretical Advantage |
|:---|:---|:---:|:---:|:---|
| **Linear Regression** | Ordinary Least Squares $(X^TX)^{-1}X^Ty$ | **\$192,353** | 0.6353 | Exact closed-form solution, total interpretability |
| **Decision Tree** | Recursive variance reduction ($\min \text{SSE}$) | **\$150,000** | 0.7316 | Captures non-linear thresholds and feature interactions |
| **Random Forest** | Bootstrap bagging + random subspace | **\$90,000** | 0.8187 | Dramatically reduces tree variance and overfitting |
| **KNN ($k=2$)** | Distance-weighted Euclidean neighbors | **\$157,200** | 0.7207 | Non-parametric, smooth localized spatial adaptation |
| **Neural Network** | Backprop gradient descent with ReLU | **\$117,500** | 0.7955 | Universal function approximator with smooth transitions |
| **XGBoost** | 2nd-order Taylor expansion + shrinkage | **\$178,800** | **0.8574** | Iteratively eliminates residual bias while penalizing complexity |
