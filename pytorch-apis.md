# PyTorch APIs

These notes collect PyTorch-specific APIs and behaviors that are useful across lessons.

## `torch.relu` / `Tensor.relu`

ReLU means `relu(x) = max(0, x)`. In PyTorch, it is the common nonlinearity placed between matrix multiplies or layer calls.

Conceptually:

- Positive inputs pass through unchanged.
- Negative inputs become zero.
- The kink gives the network nonlinearity.

During backpropagation, ReLU's local derivative is a gate: gradients flow through positive activations and stop at non-positive activations.

## `requires_grad`

`requires_grad=True` tells PyTorch to record operations involving a tensor into a computation graph.

Use it for tensors whose derivatives you need, especially model parameters. If a tensor does not require gradients, PyTorch can skip recording graph history for operations that only involve non-gradient tensors.

## `backward()`

`backward()` runs backpropagation from a scalar loss through the recorded graph.

Each operation contributes its local derivative. PyTorch multiplies those local derivatives along the graph using the chain rule, producing `d loss / d parameter` for each parameter involved in the loss.

## `.grad`

After `backward()`, PyTorch stores each parameter's computed gradient in `.grad`.

Important behavior: `.grad` accumulates across `backward()` calls. It adds rather than overwrites, so real training loops clear gradients each step before computing new ones.

## `optimizer.zero_grad()`

`optimizer.zero_grad()` clears accumulated gradients before the next backward pass.

A typical training step is:

1. Run the model forward.
2. Compute one scalar loss.
3. Clear old gradients with `optimizer.zero_grad()`.
4. Run `loss.backward()`.
5. Update parameters with `optimizer.step()`.

## `torch.no_grad()`

`torch.no_grad()` tells PyTorch not to build a computation graph inside its block.

Use it for inference and manual parameter updates because those operations do not need gradients. This makes them cheaper and lower-memory.
