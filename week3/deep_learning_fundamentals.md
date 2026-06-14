# Week 3: Deep Learning Fundamentals & Architectures

This document structures the core mathematical concepts and practical mechanisms of deep learning in a logical, textbook-style progression. It covers data processing, the optimization engine, various network architectures, and concludes with a practical overview of essential PyTorch APIs.

---

## Part 1: Deep Learning Core Concepts

### 1. Data Processing: Train Before You Clean
A counter-intuitive but highly effective approach in deep learning is to **train a baseline model before comprehensively cleaning the data**.

*   **Top Losses Analysis:** A trained model can identify anomalies in your dataset. By plotting "top losses" (predictions where the model was highly confident but incorrect, or correct but entirely unconfident), the model acts as a diagnostic tool. Using utilities like `ImageClassifierCleaner`, you can isolate noisy, mislabeled, or corrupted data and systematically correct it.

#### Image Normalization and Augmentation
Neural networks require all inputs in a mini-batch to have uniform dimensions. For image data, this requires transformation:
*   **Squish:** Alters the aspect ratio, which can distort spatial features.
*   **Pad:** Adds zero-padding (black borders) to preserve aspect ratio, though it wastes computational cycles on empty space.
*   **Crop:** Extracts a central region, risking the loss of peripheral features.
*   **RandomResizedCrop (Data Augmentation):** Extracts a different, randomly sized and positioned crop during each epoch. This prevents the model from memorizing exact pixel layouts, drastically improving generalization.

### 2. The Optimization Engine: Gradient Descent
At its core, a neural network is an arbitrarily complex mathematical function—primarily nested matrix multiplications interspersed with non-linear activation functions (e.g., ReLU). To optimize this function, we rely on **Gradient Descent** using a canonical 7-step loop:

1.  **Initialize Parameters:** Instantiate network weights with random values.
2.  **Calculate Predictions (Forward Pass):** Propagate inputs through the network via matrix multiplications and activations.
3.  **Calculate Loss:** Use a differentiable loss function (e.g., Mean Squared Error or Cross-Entropy) to quantify the divergence between predictions and ground-truth labels.
4.  **Calculate Gradients (Backward Pass):** Compute the partial derivatives of the loss with respect to every weight in the network.
5.  **Step the Weights:** Update the parameters by stepping in the opposite direction of the gradient, scaled by the **learning rate**.
6.  **Repeat:** Iterate over mini-batches for multiple epochs.
7.  **Stop:** Terminate training upon convergence or when validation metrics degrade (overfitting).

### 3. Architectures & Modalities

#### 3.1 Transfer Learning & Deployment (Computer Vision)
Training deep architectures from scratch is computationally prohibitive. **Transfer Learning** bypasses this by leveraging models pre-trained on massive datasets (e.g., ResNet on ImageNet).
*   **The Foundation:** The early layers of the network act as generalized feature extractors (detecting edges, gradients, and semantic textures).
*   **The Head:** We discard the final classification layer and replace it with a randomly initialized matrix mapped to our specific target classes.
*   **Training (Freezing):** We initially "freeze" the foundational layers and only apply gradient descent to the new head. Later, we may unfreeze the entire network for a fine-tuning pass with a highly suppressed learning rate.
*   **Deployment:** The finalized architecture and state dictionary are exported to a `.pkl` (pickle) file. This artifact can be deployed into production using frameworks like **Gradio** hosted on Hugging Face Spaces.

#### 3.2 Natural Language Processing (NLP)
For sequence data, the industry standard relies on Transformer architectures (via the Hugging Face ecosystem).
*   **Tokenization:** Text is parsed into sub-word units (tokens).
*   **Numericalization:** Tokens are mapped to integer IDs based on a pre-trained vocabulary, yielding 1D tensors.
*   **Fine-Tuning:** Leveraging pre-trained language models (e.g., DeBERTa), which already encapsulate rich grammatical and contextual representations, we attach a sequence-classification head. The model is then fine-tuned via gradient descent for downstream tasks like sentiment analysis or document similarity.

#### 3.3 Collaborative Filtering & Embeddings
Recommender systems (e.g., Netflix, Amazon) frequently utilize collaborative filtering to approximate missing entries in sparse matrices (Users vs. Items).
*   **The Math (Dot Product):** We assign latent factor vectors to each user ($\mathbf{u}$) and item ($\mathbf{v}$). The predicted rating $\hat{r}_{ui}$ is the dot product of these vectors plus user and item biases:
    $$ \hat{r}_{ui} = \mathbf{u} \cdot \mathbf{v} + b_u + b_i $$
*   **Embeddings:** An embedding is a computational shortcut. Mathematically, it is equivalent to multiplying a one-hot encoded vector $\mathbf{x}$ by a weight matrix $W$. Because $\mathbf{x}^T W = W_{i,*}$ (where $i$ is the active index), PyTorch bypasses the matrix multiplication and performs a direct memory lookup.
    *   **Example:** Imagine a vocabulary of 5 words and a weight matrix $W$ of size $5 \times 3$ (each word has a 3-dimensional embedding).
    To get the embedding for word index 2, the mathematical formulation is:
    $$ \begin{bmatrix} 0 & 0 & 1 & 0 & 0 \end{bmatrix} \times \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.6 \\ \mathbf{0.7} & \mathbf{0.8} & \mathbf{0.9} \\ 1.0 & 1.1 & 1.2 \\ 1.3 & 1.4 & 1.5 \end{bmatrix} = \begin{bmatrix} 0.7 & 0.8 & 0.9 \end{bmatrix} $$
    Instead of performing 15 floating-point multiplications and additions (most of which are multiplied by 0), an `nn.Embedding` layer simply fetches row 2 directly from memory: `W[2]`. It is an $O(1)$ index lookup rather than an $O(N)$ matrix multiplication.
*   **Weight Decay (L2 Regularization):** To prevent latent factors from unbounded growth (overfitting), an $L_2$ penalty $\lambda$ is added to the loss function:
    $$ \text{Loss} = \sum(r_{ui} - \hat{r}_{ui})^2 + \lambda(\|\mathbf{U}\|^2 + \|\mathbf{V}\|^2) $$
    *   **Why we do this:** Without constraints, gradient descent might find that multiplying a user factor of $1000$ by an item factor of $0.005$ perfectly fits a specific training rating of $5.0$. However, these extreme, highly specialized numbers will fail disastrously on unseen data (overfitting).
    *   **Example:** By adding the squared magnitude of the weights to the loss, the optimizer is penalized for large numbers. It forces the network to distribute the "concept" of the rating across all latent factors (e.g., $u=[1.1, 2.0, -0.5]$ and $v=[2.0, 1.0, -1.6]$) resulting in smaller, more generalized weights that perform better on validation data.

#### 3.4 Tabular Data & Random Forests
Deep learning is not universally optimal; for tabular (spreadsheet) data, **Random Forests** generally serve as the superior baseline.
*   **The Math (Splitting & Impurity):** Decision trees partition data by searching for binary splits that maximize information gain, typically measured by Gini Impurity. For $C$ classes, where $p_i$ is the fraction of items in class $i$:
    $$ G = 1 - \sum_{i=1}^{C} p_i^2 $$
    The algorithm greedily selects the split that yields the largest drop in $G$.
    *   **Impurity Split Diagram:**
    ```text
                         [ Root Node ]
                      10 Cats, 10 Dogs (Mixed)
                      Gini Impurity: High (0.5)
                               |
                   Feature: "Weight > 20 lbs?"
                               |
                -------------------------------
               |                               |
          [ Left Node ]                 [ Right Node ]
        True: 9 Dogs, 1 Cat          False: 9 Cats, 1 Dog
      Gini Impurity: Low (0.18)    Gini Impurity: Low (0.18)
      
      Conclusion: Splitting by weight successfully separated the 
      classes, drastically reducing overall impurity.
    ```
*   **Bagging:** A Random Forest is an ensemble of unpruned, high-variance decision trees trained on random bootstrapped subsets of data. The final prediction $\hat{y}$ is the average of all $M$ tree predictions:
    $$ \hat{y} = \frac{1}{M} \sum_{j=1}^{M} f_j(x) $$
    Because the individual trees are unbiased but structurally diverse, their uncorrelated errors cancel out upon averaging.
*   **Advantages:** They are immune to outliers, do not require feature normalization, handle non-linearities natively, and provide explicit feature importance metrics.

---

## Part 2: Essential PyTorch APIs

PyTorch automates the calculus required for backpropagation. Below are the foundational functions used in modern deep learning pipelines, complete with graduate-level, self-contained examples.

### 1. `torch.tensor()`
Tensors are the fundamental data structures in PyTorch, heavily optimized for GPU acceleration.

```python
import torch

# Creating a 2D tensor (Matrix)
weights = torch.tensor([[0.5, -0.2], 
                        [0.1,  0.8]])
```

### 2. `requires_grad_()` & `backward()`
The `requires_grad_()` method flags a tensor so PyTorch's autograd engine constructs a computation graph tracking every operation applied to it. `backward()` triggers the reverse-mode automatic differentiation, computing the calculus and populating the `.grad` attribute.

```python
# Initialize a scalar with gradient tracking enabled
x = torch.tensor(3.0).requires_grad_()

# Forward pass: y = x^2 + 4x
y = (x ** 2) + (4 * x)

# Compute gradients (dy/dx)
y.backward()

# The derivative of x^2 + 4x at x=3 is 2(3) + 4 = 10
print(x.grad) # Output: tensor(10.)
```

### 3. Matrix Multiplication (`@`)
The `@` operator (standardized in Python 3.5) performs matrix multiplication. It is the primary engine of neural network forward passes.

```python
inputs = torch.tensor([[1.0, 2.0]])          # 1x2 Matrix
weights = torch.tensor([[0.5, 0.1], 
                        [-0.2, 0.8]])        # 2x2 Matrix

# Compute the dot products
activations = inputs @ weights 
print(activations) # Output: tensor([[0.1000, 1.7000]])
```

### 4. Non-linearities: `F.relu()` and `torch.sigmoid()`
Without non-linear activation functions, a neural network, regardless of depth, collapses mathematically into a single linear transformation.
*   **`F.relu()`:** The Rectified Linear Unit clamps negative values to zero. It allows networks to learn complex, non-linear mappings without suffering from vanishing gradients.
*   **`torch.sigmoid()`:** Compresses any real-valued number into the $(0, 1)$ domain, making it the standard activation for binary classification probabilities.

```python
import torch.nn.functional as F

raw_outputs = torch.tensor([-2.5, 0.0, 3.1])

# ReLU removes negative features
hidden_state = F.relu(raw_outputs)
print(hidden_state) # Output: tensor([0.0000, 0.0000, 3.1000])

# Sigmoid converts unbounded logits into probabilities
probabilities = torch.sigmoid(raw_outputs)
print(probabilities) # Output: tensor([0.0759, 0.5000, 0.9569])
```