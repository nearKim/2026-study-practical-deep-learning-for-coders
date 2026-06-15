# Mock Tech Interview: Junior Data Scientist

**Interviewer (Senior ML Engineer):** Welcome! Today we are going to dive into some fundamental machine learning concepts. We'll be covering recommendation systems, regularization, and tree-based models. Let's start with Collaborative Filtering.

---

### Topic 1: Collaborative Filtering & L2 Regularization

**Interviewer:** Can you explain how Matrix Factorization (using embeddings) works in Collaborative Filtering?

**Candidate:** Sure. In a recommendation system, we have a massive, sparse matrix of Users and Items (like movies). Matrix Factorization breaks this giant matrix down into two smaller, dense matrices: one for Users and one for Items. We represent each user and item as an "embedding"—a vector of latent factors. To predict a rating, we just take the dot product of the user's embedding and the item's embedding.

**Interviewer:** Good. Now, what happens if we *don't* apply L2 Regularization (Weight Decay) when training those embeddings?

**Candidate:** If we don't constrain the model, it will heavily overfit the training data. Because the matrix is sparse, a user might have only rated one or two niche movies. Without L2 regularization, the gradient descent algorithm will inflate the embedding weights to extreme values (e.g., assigning a factor of `1000` to the user and `0.005` to the movie) just to perfectly output a 5-star rating for that specific training example. 

**Interviewer:** And how would that behave on a real dataset?

**Candidate:** Disastrously. It would memorize the noise in the training set. When that user logs in and we try to predict their rating for a *new* movie they haven't seen, that inflated factor of `1000` will multiply with the new movie's factors, resulting in wild, nonsensical predictions. 

**Interviewer:** Exactly. So how does L2 Regularization fix that?

**Candidate:** L2 Regularization adds the squared magnitude of all the embedding weights to the loss function. It essentially tells the optimizer: *"Minimize the error, but do it using the smallest numbers possible."* This forces the model to distribute the learning across *all* the latent factors rather than relying on one massive outlier, resulting in a generalized model that handles unseen data much better.

---

### Topic 2: Tabular Data & Random Forests

**Interviewer:** Let's switch gears to tabular data. You're building a Random Forest classifier. Walk me through how a single Decision Tree decides where to split the data.

**Candidate:** A Decision Tree looks at all the features and tries to find a binary split that maximizes "Information Gain" or minimizes "Impurity." It usually measures this using Gini Impurity. 
For example, if a node has 10 cats and 10 dogs, it's highly impure (mixed). The tree will test a feature, say "Weight > 20 lbs". If that split results in one node being almost all dogs, and the other node being almost all cats, the Gini Impurity drops significantly. The tree greedily picks the split that drops the impurity the most.

**Interviewer:** That makes sense for one tree. But decision trees are notoriously prone to overfitting. How does a Random Forest solve this?

**Candidate:** It uses a technique called **Bagging** (Bootstrap Aggregating). 
1. **Bootstrap:** It creates many subsets of the training data by sampling *with replacement*. 
2. **Train:** It trains a full, unpruned decision tree on each of these subsets. It also randomly restricts the features each tree is allowed to look at during a split.
3. **Aggregate:** Because each tree sees slightly different data and features, they all overfit in slightly different, uncorrelated ways. When we average their predictions together, those individual errors cancel out. We're left with a highly accurate, robust prediction.

**Interviewer:** Great explanation. Final question: In pandas, if you have a categorical column like "City", do you need to One-Hot Encode it before feeding it into a Random Forest?

**Candidate:** Usually, no. Unlike linear models or neural networks (which require numbers to perform matrix multiplication), decision trees just need to find a split point. You can simply map the cities to numeric codes (e.g., NY=1, LA=2, Chicago=3). The tree will just split the data numerically (e.g., "City code < 2.5"). While One-Hot Encoding *works*, it can actually degrade the performance of a tree by making the data overly sparse.

**Interviewer:** Spot on. You clearly understand the mechanics under the hood. Great job!