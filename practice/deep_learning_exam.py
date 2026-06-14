"""
================================================================================
PRACTICAL DEEP LEARNING CODING EXAM
================================================================================

Instructions:
  - Fill in the missing code where indicated by `# TODO` comments.
  - You MUST use ALL of the following somewhere in your answers:
      1.  torch.tensor(), requires_grad_(), and backward()
      2.  The Python matrix-multiplication operator @
      3.  torch.sigmoid() and F.relu()
      4.  sklearn RandomForestClassifier + feature importance extraction
      5.  torch.nn.Embedding, user/item dot product, and Weight Decay (L2)
      6.  The 7-step PyTorch training loop (labeled below)

Sections:
  A. Tensor fundamentals & manual backprop
  B. Forward pass with @, sigmoid, and ReLU
  C. Tabular data with scikit-learn
  D. Collaborative filtering with embeddings
  E. Full 7-step training loop

Run this file after filling in all TODOs:
    python practice/deep_learning_exam.py

Good luck!
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# ==============================================================================
# SECTION A — PyTorch Core: tensor(), requires_grad_(), backward()
# ==============================================================================

def manual_gradient_step():
    """
    Compute y = (x ** 2) + 3*x using a tensor that tracks gradients,
    then run backward() and return x's gradient.
    """
    # Create a scalar tensor with value 4.0 and enable gradient tracking
    # TODO: Initialize `x` as a PyTorch tensor with value 4.0
    x = None

    # TODO: Enable gradient tracking on x (in-place method)
    pass

    y = (x ** 2) + (3 * x)
    
    # TODO: Backpropagate to compute gradients
    pass

    return x.grad.item()


# ==============================================================================
# SECTION B — @ operator, torch.sigmoid(), F.relu()
# ==============================================================================

def two_layer_forward(x, W1, b1, W2, b2):
    """
    Two-layer network:
      hidden = F.relu(x @ W1 + b1)
      output = torch.sigmoid(hidden @ W2 + b2)
    Returns the output tensor.
    """
    # TODO: Compute the hidden layer (linear transformation + ReLU)
    hidden = None
    
    # TODO: Compute the output layer (linear transformation + sigmoid)
    output = None
    
    return output


# ==============================================================================
# SECTION C — scikit-learn RandomForestClassifier + feature importance
# ==============================================================================

def train_forest_and_get_importance(X, y, n_estimators=50, random_state=42):
    """
    Train a RandomForestClassifier and return (trained_model, feature_importances).
    feature_importances should be a 1-D numpy array aligned with X's columns.
    """
    # TODO: Initialize the RandomForestClassifier
    clf = None
    
    # TODO: Fit the classifier to X and y
    pass

    # TODO: Extract the feature importances
    importances = None
    
    return clf, importances


# ==============================================================================
# SECTION D — Collaborative Filtering: Embedding, dot product, Weight Decay
# ==============================================================================

class CollaborativeFilteringModel(nn.Module):
    """
    Latent-factor model: predict rating = dot(user_embedding, item_embedding).
    """

    def __init__(self, n_users, n_items, n_factors=8):
        super().__init__()
        # TODO: Initialize user and item embeddings
        self.user_embedding = None
        self.item_embedding = None

    def forward(self, user_ids, item_ids):
        user_vecs = self.user_embedding(user_ids)
        item_vecs = self.item_embedding(item_ids)
        
        # Element-wise multiply then sum across factors = dot product per pair
        # TODO: Compute predictions (dot product of user_vecs and item_vecs)
        preds = None
        
        return preds


def cf_loss_with_weight_decay(model, user_ids, item_ids, ratings, weight_decay=0.01):
    """
    MSE loss plus L2 penalty on all embedding weights (conceptual weight decay).
    """
    preds = model(user_ids, item_ids)
    mse = F.mse_loss(preds, ratings)
    l2_penalty = weight_decay * sum(
        (p ** 2).sum() for p in model.parameters()
    )
    return mse + l2_penalty


def make_cf_optimizer(model, lr=0.05, weight_decay=0.01):
    """
    Return an Adam optimizer that also applies Weight Decay (L2 regularization)
    via the optimizer's weight_decay parameter.
    """
    # TODO: Create and return a torch.optim.Adam optimizer
    raise NotImplementedError("TODO: Create and return an Adam optimizer")


# ==============================================================================
# SECTION E — The 7-Step Training Loop
# ==============================================================================

def seven_step_training_loop(
    model,
    user_ids,
    item_ids,
    ratings,
    n_epochs=5,
    lr=0.1,
    weight_decay=0.01,
):
    """
    Train `model` using the canonical 7-step loop:

      Step 1 — Initialize parameters   (model already created; set up optimizer)
      Step 2 — Calculate predictions   (forward pass)
      Step 3 — Calculate loss          (MSE + weight decay)
      Step 4 — Calculate gradients     (zero_grad + backward)
      Step 5 — Step the weights        (optimizer.step)
      Step 6 — Repeat                  (loop over epochs)
      Step 7 — Stop                    (return when done)

    Returns a list of per-epoch loss values (floats).
    """
    epoch_losses = []

    # ── Step 1: Initialize parameters ──────────────────────────────────────
    optimizer = make_cf_optimizer(model, lr=lr, weight_decay=weight_decay)

    # ── Step 6: Repeat ─────────────────────────────────────────────────────
    for epoch in range(n_epochs):

        # ── Step 2: Calculate predictions ────────────────────────────────────
        # TODO: Calculate predictions
        preds = None

        # ── Step 3: Calculate loss ───────────────────────────────────────────
        loss = cf_loss_with_weight_decay(
            model, user_ids, item_ids, ratings, weight_decay=weight_decay
        )

        # ── Step 4: Calculate gradients ──────────────────────────────────────
        # TODO: Zero gradients and backpropagate loss
        pass

        # ── Step 5: Step the weights ─────────────────────────────────────────
        # TODO: Update weights using the optimizer
        pass

        epoch_losses.append(loss.item())

    # ── Step 7: Stop ─────────────────────────────────────────────────────────
    return epoch_losses


# ==============================================================================
# TEST SUITE & SCORING  (do not modify below this line)
# ==============================================================================

def _record(result, name, passed):
    result["total"] += 1
    if passed:
        result["passed"] += 1
    else:
        result["failures"].append(name)


def run_exam_tests():
    results = {"passed": 0, "total": 0, "failures": []}
    torch.manual_seed(42)
    np.random.seed(42)

    # --- Section A: manual_gradient_step ---
    try:
        grad = manual_gradient_step()
        # d/dx (x^2 + 3x) at x=4 => 2*4 + 3 = 11
        _record(results, "A: manual_gradient_step returns correct gradient", grad == 11.0)
        _record(results, "A: gradient is a float", isinstance(grad, float))
    except Exception as e:
        _record(results, f"A: manual_gradient_step runs without error ({e})", False)

    # --- Section B: two_layer_forward ---
    try:
        x = torch.tensor([[1.0, 2.0]])
        W1 = torch.tensor([[0.5, -0.3], [0.1, 0.8]])
        b1 = torch.tensor([0.0, 0.0])
        W2 = torch.tensor([[0.6], [-0.4]])
        b2 = torch.tensor([0.1])

        out = two_layer_forward(x, W1, b1, W2, b2)

        hidden_manual = F.relu(x @ W1 + b1)
        expected = torch.sigmoid(hidden_manual @ W2 + b2)

        _record(results, "B: two_layer_forward output shape", out.shape == expected.shape)
        _record(
            results,
            "B: two_layer_forward matches manual computation",
            torch.allclose(out, expected, atol=1e-6),
        )
        _record(
            results,
            "B: output values are in (0, 1) from sigmoid",
            bool((out > 0).all() and (out < 1).all()),
        )
    except Exception as e:
        _record(results, f"B: two_layer_forward runs without error ({e})", False)

    # --- Section C: train_forest_and_get_importance ---
    try:
        X = np.array(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 50.0],
                [3.0, 30.0, 80.0],
                [4.0, 40.0, 60.0],
                [5.0, 50.0, 90.0],
                [1.5, 15.0, 70.0],
            ]
        )
        y = np.array([0, 0, 1, 1, 1, 0])

        clf, importances = train_forest_and_get_importance(X, y)

        _record(
            results,
            "C: returns a fitted RandomForestClassifier",
            isinstance(clf, RandomForestClassifier) and hasattr(clf, "estimators_"),
        )
        _record(
            results,
            "C: feature importances length matches n_features",
            len(importances) == X.shape[1],
        )
        _record(
            results,
            "C: feature importances sum to ~1",
            np.isclose(importances.sum(), 1.0, atol=1e-6),
        )
        _record(
            results,
            "C: model can predict on training data",
            clf.predict(X).shape == y.shape,
        )
    except Exception as e:
        _record(results, f"C: train_forest_and_get_importance runs without error ({e})", False)

    # --- Section D: CollaborativeFilteringModel ---
    try:
        n_users, n_items, n_factors = 6, 8, 4
        cf_model = CollaborativeFilteringModel(n_users, n_items, n_factors)

        user_ids = torch.tensor([0, 1, 2, 3])
        item_ids = torch.tensor([1, 2, 3, 4])

        preds = cf_model(user_ids, item_ids)

        _record(
            results,
            "D: forward returns correct shape",
            preds.shape == torch.Size([4]),
        )

        # Dot-product sanity check with manual lookup
        u0 = cf_model.user_embedding(torch.tensor([0]))
        i1 = cf_model.item_embedding(torch.tensor([1]))
        manual_dot = (u0 * i1).sum()
        _record(
            results,
            "D: prediction equals dot product of embeddings",
            torch.isclose(preds[0], manual_dot.squeeze(), atol=1e-5),
        )

        ratings = torch.tensor([3.5, 2.0, 4.5, 1.0])
        loss = cf_loss_with_weight_decay(cf_model, user_ids, item_ids, ratings)
        _record(results, "D: cf loss is a scalar tensor", loss.dim() == 0)
        _record(results, "D: cf loss is positive", loss.item() > 0)

        opt = make_cf_optimizer(cf_model, weight_decay=0.05)
        _record(
            results,
            "D: optimizer uses weight_decay",
            opt.param_groups[0]["weight_decay"] == 0.05,
        )
    except Exception as e:
        _record(results, f"D: collaborative filtering runs without error ({e})", False)

    # --- Section E: seven_step_training_loop ---
    try:
        torch.manual_seed(0)
        cf_train = CollaborativeFilteringModel(5, 5, n_factors=3)
        u = torch.tensor([0, 1, 2, 3, 4])
        i = torch.tensor([4, 3, 2, 1, 0])
        r = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])

        losses = seven_step_training_loop(
            cf_train, u, i, r, n_epochs=10, lr=0.2, weight_decay=0.01
        )

        _record(results, "E: returns one loss per epoch", len(losses) == 10)
        _record(
            results,
            "E: losses decrease over training",
            losses[-1] < losses[0],
        )
        _record(
            results,
            "E: all recorded losses are positive floats",
            all(isinstance(v, float) and v > 0 for v in losses),
        )

        # Verify embeddings actually changed (weights were stepped)
        cf_fresh = CollaborativeFilteringModel(5, 5, n_factors=3)
        torch.manual_seed(0)
        cf_fresh.user_embedding.weight.data.copy_(
            torch.randn_like(cf_fresh.user_embedding.weight)  # dummy init marker
        )
        _record(
            results,
            "E: model parameters changed after training",
            not torch.equal(
                cf_train.user_embedding.weight.data,
                CollaborativeFilteringModel(5, 5, n_factors=3).user_embedding.weight.data,
            ),
        )
    except Exception as e:
        _record(results, f"E: seven_step_training_loop runs without error ({e})", False)

    # --- Final score ---
    pct = (results["passed"] / results["total"] * 100) if results["total"] else 0.0
    print("=" * 60)
    print("DEEP LEARNING EXAM — TEST RESULTS")
    print("=" * 60)
    print(f"Tests passed: {results['passed']} / {results['total']}")
    if results["failures"]:
        print("\nFailed tests:")
        for name in results["failures"]:
            print(f"  ✗ {name}")
    print(f"\nScore: {pct:.0f}%")
    print("=" * 60)
    return pct


if __name__ == "__main__":
    run_exam_tests()