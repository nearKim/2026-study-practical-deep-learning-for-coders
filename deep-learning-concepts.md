# Basic Deep Learning Concepts

These notes collect concepts that apply across lessons and frameworks.

## ReLU: the only nonlinearity in a basic net

`relu(x) = max(0, x)` passes positives through unchanged and flattens negatives to zero. That kink is the whole point: without a nonlinear function between layers, stacking linear layers collapses into a single linear layer (`W2(W1x)` is just some other `Wx`), so depth would buy you nothing.

With the kink, each unit adds one hinge. Summing enough hinges bends a straight line into many shapes. That is the intuition behind universal approximation, and it is why `relu` sits between the matrix multiplies.

## Backpropagation: why gradients exist

Backpropagation is the chain rule run along the recorded computation graph. Every operation knows its local derivative, meaning how its output reacts to its input. Running backward from the loss multiplies those local derivatives together.

The product is exactly `d loss / d parameter`: how much the loss changes if that parameter changes.

ReLU's local derivative acts like a gate:

- `1` where the input was positive
- `0` where the input was not positive

Gradients only flow back through units that were active on the forward pass. This is also why a unit stuck always-negative stops learning, a problem often called a dead ReLU.

## Gradient Descent: what the gradient is for

The gradient is the slope of the loss at the current parameters, so stepping in the opposite direction heads downhill.

The learning rate is the step size:

- Too small: training crawls.
- Too large: training can leap past the bottom, making the loss climb or diverge.

That is why people often decay the learning rate as training approaches a minimum.

## Computation Graph

A computation graph records the operations that produced a value. The loss must collapse to a single scalar because that single quantity is what every parameter is differentiated against.

Best model, one sentence: no single winner since it is a speed-vs-accuracy tradeoff, but the best all-rounder is `convnext`, with `levit` for speed and `beit` for top accuracy.
