# Tech Interview Preparation: Deep Learning & Tabular Data

This document contains precise, high-signal Q&A pairs for data science and machine learning interviews, covering recommendation systems, regularization, tree-based ensembles, embeddings, transfer learning, and NLP preprocessing.

---

### Topic 1: Collaborative Filtering & L2 Regularization

**Q: How does Matrix Factorization work in the context of Collaborative Filtering?**
**A:** Matrix Factorization decomposes a large, sparse user-item interaction matrix into two smaller, dense matrices: one for Users and one for Items. Each user and item is represented by a dense vector of latent factors (an embedding). To predict a rating or interaction, the model computes the dot product of the specific user's embedding and the item's embedding, optionally adding user and item biases.

**Q: What happens if L2 Regularization (Weight Decay) is NOT applied when training these embeddings?**
**A:** The model will severely overfit the training data. Because interaction matrices are highly sparse, gradient descent can easily inflate the embedding weights to extreme values (e.g., assigning a factor of `1000` to a user and `0.005` to an item) to achieve a perfect 0 loss on a specific training sample. 

**Q: How does this unregularized behavior affect performance on unseen data?**
**A:** It results in catastrophic failure. The model memorizes the noise of the sparse training set rather than learning generalizable latent features. When an unregularized, inflated user vector is multiplied by a new item vector during inference, the dot product will yield wild, nonsensical predictions.

**Q: How exactly does L2 Regularization solve this overfitting problem?**
**A:** L2 Regularization adds an penalty term to the loss function equal to the sum of the squared magnitude of all embedding weights (multiplied by a hyperparameter $\lambda$). This forces the optimizer to minimize both the prediction error *and* the size of the weights. The model is forced to distribute the "concept" of a rating across all latent factors rather than relying on massive outliers, resulting in smaller, generalized weights that perform robustly on unseen data.

---

### Topic 2: Tabular Data & Random Forests

**Q: In a Decision Tree, how does the algorithm determine where to split the data?**
**A:** The algorithm searches across all features for a binary split that maximizes Information Gain or minimizes Impurity (commonly measured via Gini Impurity). For example, if a node contains an equal mix of two classes, its impurity is high. The algorithm tests threshold values (e.g., `feature_X > 2.5`) and selects the split that results in two child nodes that are as homogeneous (pure) as possible.

**Q: Decision trees are known to have high variance and overfit easily. How does a Random Forest resolve this?**
**A:** A Random Forest relies on an ensemble technique called **Bagging** (Bootstrap Aggregating):
1. **Bootstrap:** It generates multiple subsets of the training data by sampling with replacement.
2. **Train:** It trains an independent, unpruned decision tree on each subset. Furthermore, at each split, it only considers a random subset of features.
3. **Aggregate:** Because the individual trees are trained on different data and features, they overfit in distinct, uncorrelated ways. By averaging their predictions (or taking a majority vote), the uncorrelated errors cancel out, leaving a robust, low-variance prediction.

**Q: When preparing categorical data (like "City") for a Random Forest, is One-Hot Encoding necessary?**
**A:** Generally, no. Unlike linear models or neural networks that rely on dot products, decision trees only require ordinal or numeric values to define split thresholds. Categorical variables can simply be mapped to arbitrary integer codes (e.g., NY=1, LA=2, Chicago=3). The tree will numerically partition the data (e.g., `City Code <= 1.5`). In fact, One-Hot Encoding can negatively impact tree performance by creating highly sparse, imbalanced splits.

---

### Topic 3: Embeddings

**Q: What is an embedding in the context of deep learning?**
**A:** An embedding is a dense, low-dimensional vector representation of categorical data or discrete items (like words, users, or products). It captures semantic relationships and latent traits in a continuous vector space, allowing similar items to have similar vector representations.

**Q: Mathematically, how does an embedding layer differ from multiplying by a one-hot encoded vector?**
**A:** Mathematically, they are exactly equivalent. Multiplying a one-hot encoded vector by a weight matrix isolates a specific row of that matrix. However, an embedding layer acts as an architectural and computational shortcut; instead of performing expensive matrix multiplications where the vast majority of terms are zero, it performs a direct $O(1)$ memory index lookup to fetch the corresponding row.

---

### Topic 4: Transfer Learning (Computer Vision & General)

**Q: What is Transfer Learning and why is it essential in modern deep learning?**
**A:** Transfer Learning involves taking a model pre-trained on a massive, general dataset (like ImageNet for vision) and fine-tuning it for a specific, narrower task. It is essential because training deep networks from scratch requires massive amounts of data and computational resources, whereas transfer learning drastically reduces both while achieving higher accuracy and faster convergence.

**Q: Explain the process of adapting a pre-trained model for a new classification task.**
**A:** First, we remove the model's "head" (the final classification layer) and replace it with a newly initialized weight matrix matching the number of target classes in our specific dataset. We then "freeze" the foundational layers (preventing their weights from updating during backpropagation) and train only the new head using gradient descent. Once the head is stable, we may unfreeze the entire network and fine-tune it with a highly suppressed learning rate.

---

### Topic 5: Natural Language Processing (NLP) Preprocessing

**Q: Before feeding text into a Transformer model, what preprocessing steps are fundamentally required?**
**A:** Raw text must undergo two main steps: Tokenization and Numericalization.
1. **Tokenization:** The text is parsed into smaller, discrete sub-word units called tokens (e.g., separating punctuation, prefixes, and root words).
2. **Numericalization:** Each token is mapped to a unique integer ID based on the pre-trained model's specific vocabulary. This converts the sequence of text into a 1D tensor of integers that the neural network can mathematically process via its embedding layers.